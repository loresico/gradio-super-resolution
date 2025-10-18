"""
Tests for super resolution module.
"""

import pytest
from PIL import Image
from src.main import upscale_image, create_interface


class TestUpscaleImage:
    """Tests for the upscale_image function."""

    def test_upscale_none_image(self):
        """Test upscale with None image."""
        result, info = upscale_image(None, 2)
        assert result is None
        assert "Please upload an image" in info

    def test_upscale_2x(self):
        """Test 2x upscaling."""
        # Create a small test image
        img = Image.new("RGB", (100, 100), color="red")

        result, info = upscale_image(img, 2)

        assert result is not None
        assert result.size == (200, 200)
        assert "200 × 200" in info
        assert "2×" in info

    def test_upscale_3x(self):
        """Test 3x upscaling."""
        img = Image.new("RGB", (50, 50), color="blue")

        result, info = upscale_image(img, 3)

        assert result is not None
        assert result.size == (150, 150)
        assert "150 × 150" in info

    def test_upscale_4x(self):
        """Test 4x upscaling."""
        img = Image.new("RGB", (25, 25), color="green")

        result, info = upscale_image(img, 4)

        assert result is not None
        assert result.size == (100, 100)
        assert "100 × 100" in info

    def test_upscale_preserves_mode(self):
        """Test that upscaling preserves image mode."""
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))

        result, info = upscale_image(img, 2)

        assert result is not None
        assert result.mode == "RGBA"

    @pytest.mark.parametrize("width,height,scale", [
        (100, 100, 2),
        (50, 75, 3),
        (200, 150, 4),
    ])
    def test_upscale_various_sizes(self, width, height, scale):
        """Test upscaling with various image sizes."""
        img = Image.new("RGB", (width, height))

        result, info = upscale_image(img, scale)

        expected_width = width * scale
        expected_height = height * scale

        assert result is not None
        assert result.size == (expected_width, expected_height)


class TestInterface:
    """Tests for the Gradio interface."""

    def test_create_interface_returns_blocks(self):
        """Test that create_interface returns a Gradio Blocks object."""
        import gradio as gr

        demo = create_interface()

        assert isinstance(demo, gr.Blocks)

    def test_interface_has_title(self):
        """Test that interface has a title."""
        demo = create_interface()

        # Gradio Blocks objects have title in config
        assert demo.title == "Super Resolution"
