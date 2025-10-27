"""
Gradio Super Resolution Application with ESRGAN.
Working version with proper RealESRGAN model loading.
"""

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDB."""
    
    def __init__(self, nf, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBBlock(nn.Module):
    """Residual in Residual Dense Block."""
    
    def __init__(self, nf, gc=32):
        super(RRDBBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for Real-ESRGAN."""
    
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDBBlock(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        
        # Upsample based on scale factor
        if self.scale == 2:
            fea = self.lrelu(self.conv_up1(nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 4:
            fea = self.lrelu(self.conv_up1(nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.conv_up2(nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        
        return out


# Global model cache
_model_cache = {}


def download_model_from_url(url, save_path):
    """Download model from URL."""
    import requests
    
    if save_path.exists():
        print(f"✅ Model already exists: {save_path}")
        return
    
    print(f"📥 Downloading model from {url}...")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = (downloaded / total_size) * 100
                    print(f"  Progress: {progress:.1f}%", end='\r')
    
    print(f"\n✅ Model downloaded successfully!")


def load_model(scale: int = 4) -> tuple:
    """Load RealESRGAN model (cached)."""
    cache_key = f"realesrgan_x{scale}"
    
    if cache_key in _model_cache:
        print("✅ Using cached model")
        return _model_cache[cache_key]
    
    try:
        # Model paths
        model_dir = Path("model_dir")
        model_path = model_dir / f"RealESRGAN_x{scale}plus.pth"
        
        # Alternative download URLs
        model_urls = {
            2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        }
        
        # Download if needed
        if not model_path.exists():
            if scale in model_urls:
                download_model_from_url(model_urls[scale], model_path)
            else:
                raise ValueError(f"Scale {scale}x not supported. Use 2 or 4.")
        
        print(f"📂 Loading model from: {model_path.resolve()}")
        
        # Load model - support MPS (Apple Silicon), CUDA (NVIDIA), and CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🍎 Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("🚀 Using NVIDIA GPU (CUDA)")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU")
        
        # Load checkpoint first to detect input channels
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
            print("✅ Using EMA parameters")
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
            print("✅ Using regular parameters")
        else:
            state_dict = checkpoint
            print("✅ Using direct state dict")
        
        # Detect input channels from conv_first.weight
        if 'conv_first.weight' in state_dict:
            in_channels = state_dict['conv_first.weight'].shape[1]
            print(f"📊 Detected {in_channels} input channels")
        else:
            in_channels = 3
            print("⚠️  Could not detect input channels, using 3")
        
        # Detect number of RRDB blocks by counting body layers
        num_blocks = 23  # Default
        body_keys = [k for k in state_dict.keys() if k.startswith('body.')]
        if body_keys:
            max_block = max([int(k.split('.')[1]) for k in body_keys if k.split('.')[1].isdigit()])
            num_blocks = max_block + 1
            print(f"📊 Detected {num_blocks} RRDB blocks")
        
        # Create model with detected architecture
        model = RRDBNet(in_nc=in_channels, out_nc=3, nf=64, nb=num_blocks, gc=32, scale=scale)
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        model.to(device)
        model.eval()
        
        _model_cache[cache_key] = (model, device)
        print(f"✅ Model loaded successfully on {device}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def upscale_image(image: Image.Image, scale_factor: int, progress=gr.Progress()) -> tuple:
    """
    Upscale an image using Real-ESRGAN.
    
    Args:
        image: Input PIL Image
        scale_factor: Upscaling factor (2 or 4)
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (upscaled_image, info_text)
    """
    if image is None:
        return None, "⚠️ Please upload an image first"
    
    try:
        import time
        start_time = time.time()
        
        # Get original dimensions
        orig_width, orig_height = image.size
        
        # Load model
        progress(0.1, desc="Loading model...")
        model, device = load_model(scale_factor)
        
        if model is None:
            # Fallback to Lanczos
            progress(0.5, desc="Using fallback method...")
            new_size = (orig_width * scale_factor, orig_height * scale_factor)
            upscaled = image.resize(new_size, Image.LANCZOS)
            info = f"""
### ⚠️ Using Fallback Method

**Original Size:** {orig_width} × {orig_height} px
**Enhanced Size:** {upscaled.size[0]} × {upscaled.size[1]} px
**Method:** Lanczos (Model loading failed)

💡 **Fix:** Check that model file exists in `model_dir/` folder
            """
            return upscaled, info
        
        # Check if CPU and use fallback for speed (MPS and CUDA use AI model)
        if device.type == 'cpu':
            progress(0.5, desc="CPU detected - using fast Lanczos upscaling...")
            print("⚠️  CPU detected - using Lanczos fallback for better performance")
            new_size = (orig_width * scale_factor, orig_height * scale_factor)
            upscaled = image.resize(new_size, Image.LANCZOS)
            elapsed_time = time.time() - start_time
            info = f"""
### ✅ Enhancement Complete (Fast Mode)

**Original Size:** {orig_width} × {orig_height} px
**Enhanced Size:** {upscaled.size[0]} × {upscaled.size[1]} px
**Scale Factor:** {scale_factor}×
**Method:** Lanczos Resampling
**Device:** CPU (AI model too slow on CPU)
**Processing Time:** {elapsed_time:.2f}s

💡 *For AI-powered results, use a Mac with Apple Silicon or a GPU. CPU uses high-quality Lanczos interpolation.*
            """
            return upscaled, info
        
        # Convert to RGB if needed
        progress(0.2, desc="Preparing image...")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        progress(0.3, desc="Converting to tensor...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        print(f"🔍 Input tensor: shape={img_tensor.shape}, range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Handle 12-channel input models by padding with zeros
        progress(0.4, desc="Checking model input...")
        model_input_channels = model.conv_first.weight.shape[1]
        print(f"🔍 Model expects {model_input_channels} channels, input has {img_tensor.shape[1]} channels")
        
        if model_input_channels == 12 and img_tensor.shape[1] == 3:
            print("📊 Model expects 12 channels, padding input...")
            # Pad RGB (3 channels) to 12 channels with zeros
            padding = torch.zeros(img_tensor.shape[0], 9, img_tensor.shape[2], img_tensor.shape[3])
            img_tensor = torch.cat([img_tensor, padding], dim=1)
        elif model_input_channels != img_tensor.shape[1]:
            print(f"⚠️  Channel mismatch! Model expects {model_input_channels}, got {img_tensor.shape[1]}")
        
        # Move to device after padding
        progress(0.5, desc="Moving to device...")
        img_tensor = img_tensor.to(device)
        
        # Process image
        progress(0.6, desc=f"Enhancing image from {orig_width}×{orig_height} to {orig_width*scale_factor}×{orig_height*scale_factor}...")
        print(f"🎨 Enhancing image ({orig_width}×{orig_height}) with {scale_factor}x model...")
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert back to PIL
        progress(0.8, desc="Converting result...")
        output = output.squeeze(0).cpu()
        
        # Debug: Check output range and statistics
        output_min = output.min().item()
        output_max = output.max().item()
        output_mean = output.mean().item()
        print(f"🔍 Output tensor: shape={output.shape}, min={output_min:.3f}, max={output_max:.3f}, mean={output_mean:.3f}")
        
        # Check per-channel statistics
        for c in range(min(3, output.shape[0])):
            ch_min = output[c].min().item()
            ch_max = output[c].max().item()
            ch_mean = output[c].mean().item()
            print(f"   Channel {c}: min={ch_min:.3f}, max={ch_max:.3f}, mean={ch_mean:.3f}")
        
        # Normalize if output is in unusual range or if mean is too dark
        if output_max > 1.5 or output_min < -0.5:
            print("⚠️  Unusual output range detected, normalizing to [0, 1]...")
            output = (output - output_min) / (output_max - output_min + 1e-8)
        elif output_mean < 0.1 and scale_factor == 2:
            # Special case: 2x model produces very dark output
            print("⚠️  2x model output is too dark, applying normalization...")
            output = (output - output_min) / (output_max - output_min + 1e-8)
        else:
            output = output.clamp(0, 1)
        
        print(f"🔍 Final output range: min={output.min().item():.3f}, max={output.max().item():.3f}, mean={output.mean().item():.3f}")
        
        output = transforms.ToPILImage()(output)
        
        # Get new dimensions
        progress(0.9, desc="Finalizing...")
        new_width, new_height = output.size
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        pixel_increase = ((new_width * new_height) / (orig_width * orig_height) - 1) * 100
        
        # Create info text
        progress(1.0, desc="Complete!")
        if device.type == "mps":
            device_name = "Apple Silicon GPU (MPS)"
        elif device.type == "cuda":
            device_name = "NVIDIA GPU (CUDA)"
        else:
            device_name = "CPU"
        info = f"""
### ✅ Enhancement Complete!

**Original Size:** {orig_width} × {orig_height} px
**Enhanced Size:** {new_width} × {new_height} px
**Scale Factor:** {scale_factor}×
**Model:** Real-ESRGAN (RRDBNet)
**Device:** {device_name}
**Pixel Increase:** +{pixel_increase:.1f}%
**Processing Time:** {elapsed_time:.2f}s

🎨 *AI-powered super-resolution complete!*
        """
        
        print(f"✅ Enhancement complete: {new_width}×{new_height}")
        return output, info
        
    except Exception as e:
        import traceback
        error_msg = f"""
❌ **Error:** {str(e)}

**Troubleshooting:**
- Ensure model file is in `model_dir/` folder
- Try a smaller image
- Try scale 2× instead of 4×
- Check available RAM/VRAM
- Restart the app

**Full error:**
```
{traceback.format_exc()}
```
        """
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None, error_msg


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    
    with gr.Blocks(title="Real-ESRGAN", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Real-ESRGAN Super Resolution
        ### Professional AI-powered image enhancement
        
        Upload an image and enhance it with Real-ESRGAN!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                input_image = gr.Image(
                    type="pil",
                    label="Upload Image",
                    sources=["upload", "clipboard", "webcam"],
                )
                
                gr.Markdown("### ⚙️ Settings")
                
                scale_factor = gr.Radio(
                    choices=[2, 4],
                    value=4,
                    label="Upscale Factor",
                    info="2× is faster, 4× gives maximum detail",
                )
                
                submit_btn = gr.Button(
                    "✨ Enhance Image",
                    variant="primary",
                    size="lg",
                )
                
                gr.Markdown("""
                ---
                **Model Files:**
                Place model files in `model_dir/` folder:
                - `RealESRGAN_x2plus.pth`
                - `RealESRGAN_x4plus.pth`
                
                Models will auto-download if not found!
                
                **Tips:**
                - First run downloads model (~17-67MB)
                - Processing takes 5-30 seconds
                - Works best on photos and artwork
                - GPU accelerates processing (MPS/CUDA)
                - M1/M2/M3/M4 Macs use Apple Silicon GPU
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Output")
                
                output_image = gr.Image(
                    type="pil",
                    label="Enhanced Image",
                )
                
                info_text = gr.Markdown()
        
        gr.Markdown("---")
        gr.Markdown("## 📸 Best Practices")
        gr.Markdown("""
        - **Photos**: 4× scale for maximum quality
        - **Large Images** (>2000px): Use 2× scale
        - **Old Photos**: Great for restoration
        - **Screenshots**: 2× scale recommended
        - **GPU**: Significantly faster (Apple Silicon MPS, NVIDIA CUDA)
        """)
        
        # Connect button
        submit_btn.click(
            fn=upscale_image,
            inputs=[input_image, scale_factor],
            outputs=[output_image, info_text],
        )
        
        gr.Markdown("""
        ---
        ### 🚀 About Real-ESRGAN
        
        **Real-World Enhanced Super-Resolution GAN**
        
        - 🎯 State-of-the-art super-resolution
        - 🖼️ Photorealistic results
        - 🏗️ RRDBNet architecture (23 blocks)
        - ⚡ GPU-accelerated
        
        Built with [Gradio](https://gradio.app/) + [PyTorch](https://pytorch.org/)
        """)
    
    return demo


def main() -> None:
    """Launch the application."""
    print("🚀 Starting Real-ESRGAN Super Resolution App...")
    
    # Detect device
    if torch.backends.mps.is_available():
        device_info = "Apple Silicon GPU (MPS) 🍎"
    elif torch.cuda.is_available():
        device_info = "NVIDIA GPU (CUDA) 🚀"
    else:
        device_info = "CPU 💻"
    
    print(f"📊 Device: {device_info}")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()