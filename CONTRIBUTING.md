# Contributing Guide

Thank you for considering contributing to this project! This guide will help you understand our development process and standards.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Style](#code-style)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/gradio-super-resolution.git
cd gradio-super-resolution

# Add upstream remote
git remote add upstream https://github.com/loresico/gradio-super-resolution.git
```

### 2. Create a Branch

```bash
# Always create a new branch for your changes
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Make Your Changes

Follow the project structure and coding standards (see below).

## 📝 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

Must be one of:

- **feat**: New feature for the user
- **fix**: Bug fix for the user
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, whitespace)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to build process or auxiliary tools
- **ci**: Changes to CI configuration files and scripts
- **revert**: Reverts a previous commit

### Scope (Optional)

The scope should specify the place of the commit change:

- `model` - AI model changes
- `ui` - Gradio interface changes
- `processing` - Image processing logic
- `deps` - Dependency updates
- `docs` - Documentation
- `config` - Configuration files
- `ci` - CI/CD related
- `setup` - Setup scripts

### Subject

- Use imperative, present tense: "add" not "added" nor "adds"
- Don't capitalize the first letter
- No period (.) at the end
- Maximum 50 characters

### Body (Optional)

- Use imperative, present tense
- Include motivation for the change
- Contrast with previous behavior

### Footer (Optional)

- Reference issues: `Closes #123`, `Fixes #456`
- Note breaking changes: `BREAKING CHANGE: description`

### Examples

#### Simple Feature
```
feat(ui): add natural look strength slider
```

#### Bug Fix with Details
```
fix(model): correct 2x upscaling dark output issue

The 2x model was producing dark images due to incompatible
channel count. Now using 4x model with smart downscaling.

Fixes #42
```

#### Documentation Update
```
docs(readme): update Real-ESRGAN usage instructions

Added clarification about GPU acceleration and model
auto-download process.
```

#### Breaking Change
```
feat(model): remove Lanczos fallback, AI-only approach

BREAKING CHANGE: CPU users no longer get Lanczos fallback.
AI model will run on CPU (slower). GPU strongly recommended.

Closes #89
```

#### Dependency Update
```
chore(deps): upgrade gradio to 5.49.1

- Updated gradio from 5.0.x to 5.49.1
- Includes UI improvements and bug fixes
- No breaking changes
```

#### Refactoring
```
refactor(processing): extract post-processing to separate function

Extracted natural enhancement logic into separate function
for better readability and testability.
```

## 🔄 Pull Request Process

### 1. Sync with Upstream

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Run Tests and Checks

```bash
# Format code
uv run black src/ tests/

# Check formatting
uv run black --check src/ tests/

# Lint code
uv run flake8 src/ tests/

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Test the setup script
./setup.sh --force-clean
```

### 3. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

Go to GitHub and create a Pull Request with:

**Title:** Follow commit message convention
```
feat(setup): add Python 3.14 support
```

**Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the documentation accordingly
- [ ] My commits follow the conventional commits specification

## Related Issues
Closes #(issue number)
```

### 5. Code Review

- Be open to feedback
- Respond to comments promptly
- Make requested changes in new commits
- Don't force-push after review has started

## 🛠️ Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Git
- (Optional) NVIDIA GPU with CUDA or Apple Silicon Mac for GPU acceleration

### Setup Development Environment

```bash
# Run setup (installs Python and dependencies)
./setup.sh

# Activate virtual environment
source .venv/bin/activate

# Install all dependencies including dev tools
uv sync --all-extras

# Verify installation
python src/main.py
```

### Project Structure

```
gradio-super-resolution/
├── src/                     # Source code
│   ├── __init__.py
│   └── main.py             # Main Gradio app with Real-ESRGAN
├── tests/                   # Test files
│   ├── __init__.py
│   └── test_main.py        # Unit tests
├── model_dir/              # Downloaded AI models (gitignored)
│   └── RealESRGAN_x4plus.pth
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD with uv
├── setup.sh                # Setup script
├── verify_python_version.sh # Python version verification
├── pyproject.toml          # Project config & dependencies
├── uv.lock                 # Locked dependencies
├── CONTRIBUTING.md         # This file
└── README.md               # Documentation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run with terminal coverage report
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_main.py

# Run with verbose output
uv run pytest -v

# Run and show print statements
uv run pytest -s
```

### Writing Tests

- Use `pytest` framework
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive names
- Test edge cases
- Add docstrings
- Mock heavy operations (model loading, GPU operations)

Example:
```python
def test_upscale_image_with_2x_scale():
    """Test upscale_image function with 2x scale factor."""
    from PIL import Image
    from src.main import upscale_image
    
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Test upscaling (mock model loading for speed)
    result, info = upscale_image(img, scale_factor=2, natural_strength=0.5)
    
    assert result is not None
    assert "2×" in info
```

## 🎨 Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting (88 char line length)
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and testable

```python
def upscale_image(
    image: Image.Image, 
    scale_factor: int, 
    natural_strength: float = 0.5
) -> tuple[Image.Image, str]:
    """
    Upscale an image using Real-ESRGAN AI model.
    
    Args:
        image: Input PIL Image
        scale_factor: Upscaling factor (2 or 4)
        natural_strength: Natural look strength (0.0-1.0)
        
    Returns:
        Tuple of (enhanced_image, info_text)
    """
    # Implementation here
    pass
```

### Bash Scripts

- Use `shellcheck` for linting
- Use `set -e` for error handling
- Add comments for complex logic
- Use meaningful variable names
- Quote variables: `"$var"` not `$var`

```bash
#!/usr/bin/env bash
set -e

# Configuration
PYTHON_VERSION="3.13.9"

echo "Setting up Python ${PYTHON_VERSION}"
```

### Documentation

- Use Markdown for documentation
- Include code examples
- Keep it concise and clear
- Update README when adding features
- Add comments for complex code

## 📦 Release Process

(For Maintainers)

### Version Bump

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit with conventional commit
git commit -m "chore(release): bump version to 1.0.0"
```

### Tagging

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## 🐛 Reporting Bugs

### Before Reporting

- Check existing issues
- Try latest version
- Verify it's reproducible

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run '...'
2. See error

**Expected behavior**
What you expected to happen

**Environment**
- OS: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.12.0]
- GPU: [e.g., Apple M4, NVIDIA RTX 4090, CPU only]
- Browser: [e.g., Chrome 120, Safari 17]
- Error output: [paste here]

**Additional context**
Any other relevant information (image size, model loading status, etc.)
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem

**Describe the solution you'd like**
A clear description of what you want to happen

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other context or screenshots
```

## 📞 Questions?

- Open a [Discussion](https://github.com/loresico/gradio-super-resolution/discussions)
- Check the [README](README.md)
- Review [existing issues](https://github.com/loresico/gradio-super-resolution/issues)
- Check Real-ESRGAN documentation for model-specific questions

## 🙏 Thank You!

Your contributions make this project better for everyone!

---

**Happy Contributing! 🎉**