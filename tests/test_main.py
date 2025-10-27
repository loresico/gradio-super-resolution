"""
Tests for Real-ESRGAN super resolution module.

Note: These tests require downloading the Real-ESRGAN model (~67MB) on first run.
For faster tests without model downloads, mock the load_model function.
"""

import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
import torch
from src.main import upscale_image, create_interface, load_model


class TestUpscaleImage:
    """Tests for the upscale_image function."""

    def test_upscale_none_image(self):
        """Test upscale with None image."""
        result, info = upscale_image(None, 2, natural_strength=0.5)
        assert result is None
        assert "Please upload an image" in info

    @pytest.mark.slow
    def test_upscale_2x_real_model(self):
        """Test 2x upscaling with real AI model (slow, requires model download)."""
        # Create a small test image
        img = Image.new("RGB", (50, 50), color="red")

        result, info = upscale_image(img, 2, natural_strength=0.5)

        assert result is not None
        assert result.size == (100, 100)
        assert "100 × 100" in info
        assert "2×" in info

    @pytest.mark.slow
    def test_upscale_4x_real_model(self):
        """Test 4x upscaling with real AI model (slow, requires model download)."""
        img = Image.new("RGB", (25, 25), color="green")

        result, info = upscale_image(img, 4, natural_strength=0.5)

        assert result is not None
        assert result.size == (100, 100)
        assert "100 × 100" in info
        assert "4×" in info

    def test_upscale_converts_rgba_to_rgb(self):
        """Test that RGBA images are converted to RGB."""
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))

        # Mock model loading to avoid download
        with patch("src.main.load_model") as mock_load:
            mock_model = MagicMock()
            mock_device = torch.device("cpu")
            mock_load.return_value = (mock_model, mock_device)
            
            # Mock model output
            mock_model.return_value = torch.rand(1, 3, 100, 100)
            mock_model.conv_first.weight.shape = [64, 3, 3, 3]
            
            result, info = upscale_image(img, 2, natural_strength=0.0)

            # Image should be converted to RGB internally
            assert result is not None
            assert result.mode == "RGB"

    def test_natural_strength_parameter(self):
        """Test that natural_strength parameter is accepted."""
        img = Image.new("RGB", (50, 50), color="blue")

        with patch("src.main.load_model") as mock_load:
            mock_model = MagicMock()
            mock_device = torch.device("cpu")
            mock_load.return_value = (mock_model, mock_device)
            mock_model.return_value = torch.rand(1, 3, 100, 100)
            mock_model.conv_first.weight.shape = [64, 3, 3, 3]

            # Test with different natural strengths
            result1, _ = upscale_image(img, 2, natural_strength=0.0)
            result2, _ = upscale_image(img, 2, natural_strength=1.0)

            assert result1 is not None
            assert result2 is not None

    @pytest.mark.parametrize(
        "width,height,scale",
        [
            (50, 50, 2),
            (25, 25, 4),
        ],
    )
    def test_upscale_various_sizes_mocked(self, width, height, scale):
        """Test upscaling with various image sizes (mocked for speed)."""
        img = Image.new("RGB", (width, height))

        with patch("src.main.load_model") as mock_load:
            mock_model = MagicMock()
            mock_device = torch.device("cpu")
            mock_load.return_value = (mock_model, mock_device)
            
            expected_width = width * scale
            expected_height = height * scale
            mock_model.return_value = torch.rand(1, 3, expected_height, expected_width)
            mock_model.conv_first.weight.shape = [64, 3, 3, 3]

            result, info = upscale_image(img, scale, natural_strength=0.5)

            assert result is not None
            assert f"{scale}×" in info


class TestInterface:
    """Tests for the Gradio interface."""

    def test_create_interface_returns_blocks(self):
        """Test that create_interface returns a Gradio Blocks object."""
        import gradio as gr

        demo = create_interface()

        assert isinstance(demo, gr.Blocks)

    def test_interface_has_correct_title(self):
        """Test that interface has the correct title."""
        demo = create_interface()

        # Gradio Blocks objects have title in config
        assert demo.title == "Real-ESRGAN"


class TestModelLoading:
    """Tests for model loading functionality."""

    @pytest.mark.slow
    def test_load_model_4x(self):
        """Test loading the 4x model (slow, downloads model)."""
        model, device = load_model(4)

        assert model is not None
        assert device is not None
        assert device.type in ["cpu", "cuda", "mps"]

    def test_model_caching(self):
        """Test that models are cached after first load."""
        # Clear cache first
        from src.main import _model_cache
        _model_cache.clear()

        with patch("src.main.torch.load") as mock_load:
            # Setup mock
            mock_load.return_value = {
                "params_ema": {
                    "conv_first.weight": torch.rand(64, 3, 3, 3),
                    "body.0.rdb1.conv1.weight": torch.rand(32, 64, 3, 3),
                }
            }

            # First load
            model1, device1 = load_model(4)
            
            # Second load should use cache (torch.load not called again)
            model2, device2 = load_model(4)

            assert model1 is model2
            assert device1 == device2
            assert mock_load.call_count == 1  # Only called once
