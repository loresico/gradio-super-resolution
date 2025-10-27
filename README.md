# 🎨 Real-ESRGAN Super Resolution App

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-5.49+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional AI-powered image super-resolution using Real-ESRGAN with an intuitive Gradio web interface. Enhance your images up to 4x resolution with state-of-the-art deep learning!

![Demo](demo.gif)

## ✨ Features

- 🤖 **Real-ESRGAN AI Model** - State-of-the-art super-resolution (RRDBNet with 23 blocks)
- 🖼️ **Easy Upload** - Drag & drop, paste, or webcam capture
- 📈 **Smart Upscaling** - 2× and 4× scaling (both use AI)
- 🍎 **Apple Silicon Support** - GPU acceleration on M1/M2/M3/M4 Macs (MPS)
- 🚀 **NVIDIA GPU Support** - CUDA acceleration
- 🌿 **Natural Look Control** - Adjustable post-processing to reduce over-digitalization
- 🌐 **Modern Web Interface** - Beautiful, responsive Gradio UI
- ⚡ **Auto-Download Models** - Automatic model download (~67MB) on first run
- 📊 **Processing Stats** - See dimensions, device, and timing info

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/loresico/gradio-super-resolution
cd gradio-super-resolution

# 2. Run setup (installs Python 3.12 and dependencies with uv)
./setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Run the app
python src/main.py
```

Open your browser to: **http://localhost:7860**

The first run will automatically download the Real-ESRGAN model (~67MB).

## 📦 Installation

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- macOS, Linux, or Windows

### Setup with uv

This project uses `uv` for fast, reliable dependency management:

```bash
# First time setup
./setup.sh

# Or manually with uv
uv sync --all-extras

# Clean rebuild
./setup.sh --force-clean
```

## 🎯 Usage

### Web Interface

```bash
# Start the app
python src/main.py
```

Then open **http://localhost:7860** in your browser and:

1. **Upload an image** - Drag & drop or click to upload
2. **Choose scale** - Select 2× or 4× upscaling
3. **Adjust Natural Look** - Control AI sharpness (0=sharp, 1=natural)
4. **Click "Enhance Image"** - Process with GPU acceleration
5. **Download result** - Right-click the output image to save

### As a Module

```python
from src.main import upscale_image
from PIL import Image

# Load image
img = Image.open("photo.jpg")

# Upscale 4x with natural look strength of 0.5
enhanced, info = upscale_image(img, scale_factor=4, natural_strength=0.5)

# Save result
enhanced.save("photo_4x.jpg")
print(info)  # See processing details
```

## 🔧 Configuration

Edit `src/main.py` to customize:

```python
demo.launch(
    server_name="127.0.0.1",  # Listen on all interfaces
    server_port=7860,        # Change port
    share=True,              # Create public link (Gradio)
    show_error=True,         # Show error messages
)
```

## 🎨 How It Works

### AI Architecture

- **Model:** Real-ESRGAN (RRDBNet)
- **Architecture:** 23 Residual-in-Residual Dense Blocks
- **Input:** RGB images (3 channels)
- **Output:** High-resolution enhanced images
- **Training:** Pre-trained on diverse real-world images

### Smart 2× Upscaling

For 2× upscaling, we use an intelligent approach:
1. Downscale input image to 50% using Lanczos
2. Apply 4× AI model (doubles dimensions twice)
3. Result: 2× of original size with AI quality

This avoids using the incompatible 2× model (which requires 12 channels).

### Natural Look Processing

Post-processing adjustments to reduce AI over-sharpening:
- **Gaussian blur** - Softens edges (configurable 0-1.5 radius)
- **Contrast reduction** - Less harsh contrast (up to -15%)
- **Sharpness reduction** - Removes artificial edges (up to -25%)

## 🧪 Testing & Development

### Install Dev Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras
```

### Run Tests

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov=src --cov-report=term-missing

# With HTML coverage report
uv run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code with Black
uv run black src/ tests/

# Check formatting
uv run black --check src/ tests/

# Lint with flake8
uv run flake8 src/ tests/
```

## 🚀 Performance

### GPU Acceleration

- **M1/M2/M3/M4 Macs:** Uses MPS (Metal Performance Shaders) - ~5-15 seconds
- **NVIDIA GPUs:** Uses CUDA - ~3-10 seconds  
- **CPU:** Slower but works - ~30-120 seconds

### Image Size Recommendations

- **Small images** (<1000px): Use 4× scale
- **Medium images** (1000-2000px): Use 4× or 2× scale
- **Large images** (>2000px): Use 2× scale to avoid memory issues

## 🎨 Use Cases

- 📸 **Photo Enhancement** - Upscale old or low-res photos
- 🎮 **Game Assets** - Improve texture quality
- 🖼️ **Art Restoration** - Enhance artwork resolution
- 📱 **Social Media** - Prepare images for printing
- 🎬 **Video Frames** - Upscale individual frames

## 📁 Project Structure

```
gradio-super-resolution/
├── src/
│   ├── __init__.py
│   └── main.py              # Main Gradio app with Real-ESRGAN
├── tests/
│   ├── __init__.py
│   └── test_main.py         # Unit tests
├── model_dir/               # Auto-downloaded models (gitignored)
│   └── RealESRGAN_x4plus.pth
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD with uv
├── pyproject.toml           # Dependencies and dev tools
├── uv.lock                  # Locked dependencies
├── setup.sh                 # Setup script
├── verify_python_version.sh # Python version check
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Install dev dependencies (`uv sync --all-extras`)
4. Make your changes
5. Run tests (`uv run pytest`)
6. Format code (`uv run black src/ tests/`)
7. Commit with conventional commits (`feat: add new feature`)
8. Push and open a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Change port in src/main.py
demo.launch(server_port=7861)
```

### Model Download Issues

```bash
# Delete cached model and retry
rm -rf model_dir/
python src/main.py  # Will re-download
```

### Out of Memory (OOM)

- Use 2× scale instead of 4×
- Downscale your input image first
- Close other GPU-intensive applications

### CPU Performance

The app works on CPU but is much slower. For best performance:
- Use Apple Silicon Mac (M1/M2/M3/M4) → MPS acceleration
- Use NVIDIA GPU → CUDA acceleration
- Intel Macs will use CPU (slower but works)

### Image Quality Issues

If output looks too digital/sharp:
- Increase "Natural Look Strength" slider to 0.7-1.0
- This applies smoothing to reduce AI artifacts

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - State-of-the-art super-resolution
- [Gradio](https://gradio.app/) - Amazing ML web interfaces
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Pillow](https://python-pillow.org/) - Python imaging library
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

## 📧 Contact

- GitHub: [@loresico](https://github.com/loresico)
- Issues: [GitHub Issues](https://github.com/loresico/gradio-super-resolution/issues)

---

⭐ If you find this useful, please star the repo!

**Ready to enhance your images with AI?** Run `./setup.sh` and get started!