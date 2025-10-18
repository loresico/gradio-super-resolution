"""
Gradio Super Resolution Application.

A simple image super-resolution app using bicubic interpolation.
Replace with actual AI model for better results.
"""

import gradio as gr
from PIL import Image


def upscale_image(image: Image.Image, scale_factor: int) -> tuple[Image.Image, str]:
    """
    Upscale an image using bicubic interpolation.

    Args:
        image: Input PIL Image
        scale_factor: Upscaling factor (2, 3, or 4)

    Returns:
        Tuple of (upscaled_image, info_text)
    """
    if image is None:
        return None, "⚠️ Please upload an image first"

    try:
        # Get original dimensions
        orig_width, orig_height = image.size

        # Calculate new dimensions
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)

        # Upscale using bicubic interpolation
        # TODO: Replace with actual AI model (Real-ESRGAN, SwinIR, etc.)
        upscaled = image.resize((new_width, new_height), Image.BICUBIC)

        # Create info text
        info = f"""
### ✅ Enhancement Complete!

**Original Size:** {orig_width} × {orig_height} px
**Enhanced Size:** {new_width} × {new_height} px
**Scale Factor:** {scale_factor}×
**Method:** Bicubic Interpolation
**Pixel Increase:** +{((new_width * new_height) / (orig_width * orig_height) - 1) * 100:.1f}%

💡 *Using basic bicubic upscaling. For better results, integrate an AI model!*
        """

        return upscaled, info

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        title="Super Resolution",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🎨 Image Super Resolution
        ### Enhance your images with AI-powered upscaling
        
        Upload an image and increase its resolution up to 4x!
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
                    choices=[2, 3, 4],
                    value=2,
                    label="Upscale Factor",
                    info="Higher = more detail but slower",
                )

                submit_btn = gr.Button(
                    "✨ Enhance Image",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("""
                ---
                **Tips:**
                - Works best on photos and artwork
                - Larger images take longer to process
                - Try different scale factors for best results
                """)

            with gr.Column(scale=1):
                gr.Markdown("### 📥 Output")

                output_image = gr.Image(
                    type="pil",
                    label="Enhanced Image",
                )

                info_text = gr.Markdown()

        # Connect button to function
        submit_btn.click(
            fn=upscale_image,
            inputs=[input_image, scale_factor],
            outputs=[output_image, info_text],
        )

        gr.Markdown("""
        ---
        ### 🚀 About This App
        
        This is a simple super-resolution demo using bicubic interpolation.
        For production use, integrate AI models like:
        - Real-ESRGAN (photo enhancement)
        - SwinIR (transformer-based)
        - EDSR/WDSR (efficient models)
        
        Built with [Gradio](https://gradio.app/) 🎉
        """)

    return demo


def main() -> None:
    """Launch the Gradio application."""
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,  # Set to True to create public link
    )


if __name__ == "__main__":
    main()