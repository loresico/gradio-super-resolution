# 🎨 Gradio Super Resolution App

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered image super-resolution with an intuitive Gradio web interface. Enhance your images up to 4x resolution!

![Demo](demo.gif)

## ✨ Features

- 🖼️ **Easy Upload** - Drag & drop, paste, or webcam capture
- 📈 **Multiple Scale Factors** - 2x, 3x, or 4x upscaling
- 🌐 **Web Interface** - Beautiful, responsive Gradio UI
- ⚡ **Fast Processing** - Quick results with bicubic interpolation
- 📊 **Detailed Stats** - See before/after dimensions and improvements
- 🎯 **Ready to Extend** - Easy to add AI models

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/gradio-super-resolution
cd gradio-super-resolution

# 2. Run setup (downloads portable Python, installs dependencies)
./setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Run the app
python src/main.py
```

Open your browser to: http://localhost:7860

## 📦 Installation

### Using the Template

This project uses portable Python - no system Python required!

```bash
./setup.sh                    # First time setup
./setup.sh --force-clean      # Clean rebuild
```

## 🎯 Usage

### Basic Usage

```bash
# Start the app
python src/main.py

# Or with Poetry
poetry run python src/main.py
```

### As a Module

```python
from src.main import upscale_image
from PIL import Image

# Load image
img = Image.open("photo.jpg")

# Upscale 2x
enhanced, info = upscale_image(img, scale_factor=2)

# Save result
enhanced.save("photo_2x.jpg")
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

## 🚀 Adding AI Models

Currently uses basic bicubic interpolation. To add AI models:

### Option 1: Real-ESRGAN

```bash
# Add to pyproject.toml dependencies
uv pip install realesrgan

# Update src/main.py
from realesrgan import RealESRGANer
# ... implement model loading and inference
```

### Option 2: Use Hugging Face Models

```bash
uv pip install transformers torch

# Use pre-trained models from HF Hub
```

### Option 3: Custom Model

See `src/main.py` - replace the `upscale_image` function with your model inference.

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

## 📊 Current Approach

**Method:** Bicubic Interpolation
- ✅ Fast and simple
- ✅ No model download needed
- ✅ Works offline
- ❌ Limited quality improvement

**For Better Results:**
- Real-ESRGAN: Best for photos
- SwinIR: Transformer-based, excellent quality
- EDSR/WDSR: Lightweight and fast

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
│   └── main.py              # Main Gradio app
├── tests/
│   ├── __init__.py
│   └── test_main.py         # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD
├── pyproject.toml           # Dependencies
├── setup.sh                 # Setup script
└── README.md
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit with conventional commits (`feat: add new feature`)
6. Push and open a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Change port in src/main.py
demo.launch(server_port=7861)
```

### Gradio Not Installing

```bash
./setup.sh --force-clean
source .venv/bin/activate
uv pip install gradio
```

### Image Upload Issues

- Check file format (JPG, PNG supported)
- Try smaller images first
- Check browser console for errors

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [Gradio](https://gradio.app/) - Amazing ML web interfaces
- [Pillow](https://python-pillow.org/) - Python imaging library
- Built with portable Python template

## 📧 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Issues: [GitHub Issues](https://github.com/yourusername/gradio-super-resolution/issues)

---

⭐ If you find this useful, please star the repo!

**Ready to enhance your images?** Run `./setup.sh` and get started!