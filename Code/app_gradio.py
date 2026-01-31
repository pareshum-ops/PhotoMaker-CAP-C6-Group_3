"""
Gradio UI for PhotoMaker V2 with Optional Image Upload
Automatically loads existing input image if present.
"""

import gradio as gr
from pathlib import Path
import glob
import shutil

from PhotoMaker_Extensions.cli import main as run_photomaker
from PhotoMaker_Extensions import config


# Load existing image OR save uploaded one
def get_or_save_input_image(uploaded_file):
    input_dir = Path("/teamspace/studios/this_studio/PhotoMaker/Data/Input")
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_path = input_dir / "uploaded_input_image.png"

    # Case 1: User uploaded a new image → replace existing
    if uploaded_file is not None:
        shutil.copy(uploaded_file, saved_path)
        return str(saved_path)

    # Case 2: No upload → check if an image already exists
    existing_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if existing_images:
        return str(existing_images[0])

    # Case 3: No upload AND no existing image
    return None


# Return existing image path for UI preview
def get_existing_input_image():
    input_dir = Path("/teamspace/studios/this_studio/PhotoMaker/Data/Input")
    existing_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if existing_images:
        return str(existing_images[0])
    return None


def generate_images(uploaded_image, left_prompt, right_prompt, seed_value):
    # 1. Get existing image OR save uploaded one
    image_path = get_or_save_input_image(uploaded_image)
    if image_path is None:
        return "No input image found. Please upload one.", [], []

    # 2. Update config
    config.INPUT_IMAGES = [image_path]
    config.PROMPTS_FACE_LEFT = [left_prompt]
    config.PROMPTS_FACE_RIGHT = [right_prompt]

    if seed_value in ["", None]:
        config.SEED = None
    else:
        try:
            config.SEED = int(seed_value)
        except:
            config.SEED = None

    # 3. Run the PhotoMaker pipeline
    run_photomaker()

    # 4. Load output images
    out_dir = Path(config.OUTPUT_DIR)

    left_imgs = sorted(glob.glob(str(out_dir / "left_*.png")))
    right_imgs = sorted(glob.glob(str(out_dir / "right_*.png")))

    return "Generation complete.", left_imgs, right_imgs


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎨 PhotoMaker V2 — Gradio UI (Optional Upload + Auto‑Load Input Image)")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Settings")

                uploaded_image = gr.Image(
                    label="Upload Input Image (optional)",
                    type="filepath",
                    value=get_existing_input_image()  # show existing image if present
                )

                left_prompt = gr.Textbox(
                    label="Left Face Prompt",
                    value=config.PROMPTS_FACE_LEFT[0],
                )

                right_prompt = gr.Textbox(
                    label="Right Face Prompt",
                    value=config.PROMPTS_FACE_RIGHT[0],
                )

                seed = gr.Textbox(
                    label="Seed (leave empty for random)",
                    value=str(config.SEED) if config.SEED is not None else "",
                )

                generate_btn = gr.Button("Generate Images")

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status")

                left_gallery = gr.Gallery(
                    label="Left Face Results",
                    columns=2,
                    height="auto",
                )

                right_gallery = gr.Gallery(
                    label="Right Face Results",
                    columns=2,
                    height="auto",
                )

        generate_btn.click(
            fn=generate_images,
            inputs=[uploaded_image, left_prompt, right_prompt, seed],
            outputs=[status, left_gallery, right_gallery],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=True,
        allowed_paths=[
            "/teamspace/studios/this_studio/PhotoMaker/Data/Output",
            "/teamspace/studios/this_studio/PhotoMaker/Data/Input"
        ]
    )
