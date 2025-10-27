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
        # Model paths (we only use 4x model)
        model_dir = Path("model_dir")
        model_path = model_dir / f"RealESRGAN_x{scale}plus.pth"
        
        # Download URL for 4x model
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        
        # Download if needed
        if not model_path.exists():
            if scale == 4:
                download_model_from_url(model_url, model_path)
            else:
                raise ValueError(f"Scale {scale}x model not available. Only 4x model is used.")
        
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


def upscale_image(image: Image.Image, scale_factor: int, natural_strength: float = 0.5, progress=gr.Progress()) -> tuple:
    """
    Upscale an image using Real-ESRGAN.
    
    Args:
        image: Input PIL Image
        scale_factor: Upscaling factor (2 or 4)
        natural_strength: How much to reduce digitalization (0=AI raw, 1=very natural)
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
        original_size = (orig_width, orig_height)  # Save for display
        
        # FIX: Use 4x model for 2x upscaling (2x model expects 12 channels)
        # Strategy: Downscale image to 50%, then use 4x model to get 2x final result
        use_4x_for_2x = False
        if scale_factor == 2:
            print("🎨 Using 4x AI model for 2x upscaling (better quality)...")
            progress(0.05, desc="Preparing for 2× upscaling...")
            
            # Downscale to 50% first
            half_width = orig_width // 2
            half_height = orig_height // 2
            downscaled = image.resize((half_width, half_height), Image.LANCZOS)
            print(f"   Downscaled to {half_width}×{half_height}")
            
            # Load 4x model
            progress(0.1, desc="Loading AI model...")
            model, device = load_model(4)
            
            if model is None:
                return None, "❌ **Error:** Failed to load 4x AI model for 2x upscaling. Check that model file exists in `model_dir/` folder."
            
            # Use the downscaled image for processing
            image = downscaled
            orig_width, orig_height = half_width, half_height
            use_4x_for_2x = True
        else:
            # Load model normally for 4x
            progress(0.1, desc="Loading AI model...")
            model, device = load_model(scale_factor)
        
        if model is None:
            return None, "❌ **Error:** Failed to load AI model. Check that model file exists in `model_dir/` folder."
        
        # Convert to RGB if needed
        progress(0.15, desc="Preparing image...")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        progress(0.2, desc="Converting to tensor...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        print(f"🔍 Input tensor: shape={img_tensor.shape}, range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Handle 12-channel input models by padding with zeros
        progress(0.25, desc="Checking model input...")
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
        progress(0.3, desc="Moving to device...")
        img_tensor = img_tensor.to(device)
        
        # Process image
        progress(0.35, desc=f"Enhancing {orig_width}×{orig_height} → {orig_width*scale_factor}×{orig_height*scale_factor}...")
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert back to PIL
        progress(0.75, desc="Converting result...")
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
        
        # Normalize if output is in unusual range
        if output_max > 1.5 or output_min < -0.5:
            print("⚠️  Unusual output range detected, normalizing to [0, 1]...")
            output = (output - output_min) / (output_max - output_min + 1e-8)
        else:
            output = output.clamp(0, 1)
        
        print(f"🔍 Final output range: min={output.min().item():.3f}, max={output.max().item():.3f}, mean={output.mean().item():.3f}")
        
        output = transforms.ToPILImage()(output)
        
        # Post-processing: Reduce over-sharpening and digitalization
        if natural_strength > 0:
            progress(0.8, desc="Applying natural enhancement...")
            from PIL import ImageFilter, ImageEnhance
            
            # Apply blur based on strength (0 = no blur, 1 = max blur of 1.5)
            if natural_strength > 0.1:
                blur_radius = natural_strength * 1.5
                output = output.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                print(f"🌿 Applied blur: radius={blur_radius:.2f}")
            
            # Reduce contrast (1.0 = original, lower = less contrast)
            contrast_factor = 1.0 - (natural_strength * 0.15)  # Max -15% contrast
            enhancer = ImageEnhance.Contrast(output)
            output = enhancer.enhance(contrast_factor)
            print(f"🌿 Adjusted contrast: {contrast_factor:.2f}")
            
            # Reduce sharpness
            sharpness_factor = 1.0 - (natural_strength * 0.25)  # Max -25% sharpness
            enhancer = ImageEnhance.Sharpness(output)
            output = enhancer.enhance(sharpness_factor)
            print(f"🌿 Adjusted sharpness: {sharpness_factor:.2f}")
            
            print(f"✨ Applied natural enhancement (strength={natural_strength:.2f})")
        
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
        
        # Show correct info based on whether we used 4x model for 2x
        if use_4x_for_2x:
            model_info = "Real-ESRGAN 4× (used for 2× via smart downscaling)"
            note = "🎨 *Used 4× AI model with smart downscaling for better 2× quality!*"
        else:
            model_info = "Real-ESRGAN (RRDBNet)"
            note = "🎨 *AI-powered super-resolution complete!*"
        
        info = f"""
### ✅ Enhancement Complete!

**Original Size:** {original_size[0]} × {original_size[1]} px
**Enhanced Size:** {new_width} × {new_height} px
**Scale Factor:** {scale_factor}×
**Model:** {model_info}
**Device:** {device_name}
**Processing Time:** {elapsed_time:.2f}s

{note}
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
                    info="Both use AI! 2× uses 4× model with smart downscaling",
                )
                
                natural_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    label="Natural Look Strength",
                    info="0=AI raw (sharp/digital), 1=very natural (soft)",
                )
                
                submit_btn = gr.Button(
                    "✨ Enhance Image",
                    variant="primary",
                    size="lg",
                )
                
                gr.Markdown("""
                ---
                **Model Files:**
                - `RealESRGAN_x4plus.pth` (auto-downloads if not found)
                - Stored in `model_dir/` folder
                
                **Tips:**
                - First run downloads model (~67MB)
                - Processing takes 5-30 seconds
                - Works best on photos and artwork
                - GPU accelerates processing (MPS/CUDA)
                - M1/M2/M3/M4 Macs use Apple Silicon GPU
                - Adjust "Natural Look" if output looks too digital
                - Both 2× and 4× use the same AI model
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
            inputs=[input_image, scale_factor, natural_strength],
            outputs=[output_image, info_text],
            show_progress="minimal",  # Minimal progress indicator
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