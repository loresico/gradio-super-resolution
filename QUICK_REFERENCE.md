# Quick Reference Card - Gradio Super Resolution

## 🚀 Common Commands

```bash
# Setup (installs Python 3.12 and dependencies with uv)
./setup.sh

# Clean rebuild (removes everything, starts fresh)
./setup.sh --force-clean

# Run the app
python src/main.py

# Show help
./setup.sh --help
```

## 📊 Quick Decision Guide

```
First time?
└─ ./setup.sh

Something broken?
└─ ./setup.sh --force-clean

Changed dependencies?
├─ Edit pyproject.toml
└─ uv sync --all-extras

Model download failed?
└─ rm -rf model_dir/ && python src/main.py
```

## 🎯 Common Workflows

| Task | Commands |
|------|----------|
| **Initial setup** | `./setup.sh` |
| **Run app** | `source .venv/bin/activate` → `python src/main.py` |
| **Add dependency** | Edit `pyproject.toml` → `uv sync --all-extras` |
| **Run tests** | `uv run pytest` |
| **Format code** | `uv run black src/ tests/` |
| **Fix issues** | `./setup.sh --force-clean` |

## 📁 What Gets Created

```
.python/          # Portable Python 3.12 (~80MB, gitignored)
.venv/            # Virtual environment (~200MB with PyTorch, gitignored)
model_dir/        # Real-ESRGAN model (~67MB, gitignored)
uv.lock           # Locked dependencies (committed)
```

## 🎨 Color Guide

- ✅ Green = Success, found existing
- 📥 Blue = Downloading/installing
- 🧹 Yellow = Cleaning
- ❌ Red = Error
- 💡 Light blue = Info/tips

## ⚡ Speed Guide

| Operation | Time | Network |
|-----------|------|---------|
| First setup (Python + deps) | 2-3 min | Required |
| Model download (first run) | 30-60 sec | Required |
| Reuse existing Python | 10-30 sec | Optional |
| Clean rebuild | 2-3 min | Required |
| Image processing (GPU) | 5-15 sec | Not required |
| Image processing (CPU) | 30-120 sec | Not required |

## 💾 Disk Space

| Item | Size |
|------|------|
| Portable Python 3.12 | ~80 MB |
| Virtual environment (with PyTorch) | ~200 MB |
| Real-ESRGAN model | ~67 MB |
| Total | ~350 MB |

## 🚫 .gitignore

Always ignore these:
```gitignore
.python/         # Portable Python installation
.venv/           # Virtual environment
model_dir/       # Downloaded AI models
__pycache__/     # Python cache
*.pyc            # Compiled Python
uv.lock          # Lock file (commit for reproducibility)
```

## 🔧 Quick Fixes

```bash
# Virtual environment broken
rm -rf .venv/ && ./setup.sh

# Everything broken
./setup.sh --force-clean

# Model download failed
rm -rf model_dir/ && python src/main.py

# Dependencies out of sync
uv sync --all-extras

# Port already in use
# Edit src/main.py: demo.launch(server_port=7861)
```

## 🎯 Key Features

1. **Real-ESRGAN AI** - State-of-the-art super-resolution
2. **GPU Accelerated** - MPS (Apple Silicon) and CUDA support
3. **Auto-download** - Models download on first run
4. **Natural Look** - Adjustable post-processing
5. **Smart 2x** - Uses 4x model with downscaling

## 📞 Need Help?

```bash
# Show help
./setup.sh --help

# Check Python version
python --version  # Should be 3.12+

# Check GPU availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"

# Test model loading
python -c "from src.main import load_model; print(load_model(4))"

# View logs
python src/main.py  # Check console for debug output
```
