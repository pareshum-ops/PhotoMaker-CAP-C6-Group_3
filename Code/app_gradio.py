# app_gradio.py

import gradio as gr
from pathlib import Path
import glob
import shutil

from PhotoMaker_Extensions.cli import main as run_photomaker
from PhotoMaker_Extensions import config
from PhotoMaker_Extensions.extract_dct_watermark import extract_from_image


def get_or_save_input_image(uploaded_file):
    input_dir = Path("/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input")
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_path = input_dir / "uploaded_input_image.png"

    if uploaded_file is not None:
        shutil.copy(uploaded_file, saved_path)
        return str(saved_path)

    existing = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    return str(existing[0]) if existing else None


def get_existing_input_image():
    input_dir = Path("/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input")
    existing = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    return str(existing[0]) if existing else None


def generate_images(uploaded_image, left_prompt, right_prompt, seed_value):
    image_path = get_or_save_input_image(uploaded_image)
    if image_path is None:
        return "No input image found. Please upload one.", [], []

    try:
        seed = int(seed_value) if seed_value else None
    except:
        seed = None

    run_photomaker(
        input_image=image_path,
        left_prompt=left_prompt,
        right_prompt=right_prompt,
        seed=seed,
    )

    out_dir = Path(config.OUTPUT_DIR)
    left_imgs = sorted(glob.glob(str(out_dir / "left_*.png")))
    right_imgs = sorted(glob.glob(str(out_dir / "right_*.png")))

    return "Generation complete.", left_imgs, right_imgs


def build_ui():
    with gr.Blocks() as demo:

        gr.Markdown("## 🎨 PhotoMaker V2 — Gradio UI (Dynamic Prompts + Auto‑Load Input Image)")

        with gr.Row():
            with gr.Column(scale=1):
                uploaded_image = gr.Image(
                    label="Upload Input Image (optional)",
                    type="filepath",
                    value=get_existing_input_image()
                )

                left_prompt = gr.Textbox(label="Left Face Prompt")
                right_prompt = gr.Textbox(label="Right Face Prompt")
                seed = gr.Textbox(label="Seed (optional)")

                generate_btn = gr.Button("Generate Images")
                status = gr.Textbox(label="Status")

            with gr.Column(scale=1):
                left_gallery = gr.Gallery(label="Left Face Results", columns=2)
                right_gallery = gr.Gallery(label="Right Face Results", columns=2)

            with gr.Column(scale=1):
                extract_image = gr.Image(
                    label="Select Image to Extract Watermark",
                    type="filepath",
                    #height=150,          # controls vertical size
                    #width=150,           # controls horizontal size
                    #image_mode="contain" # preserves aspect ratio
                )

                gr.Markdown("### 🔍 Extract Invisible Watermark")
                num_bits = gr.Number(
                    label="Number of Bits",
                    value=48,   # Group3 = 6 chars × 8 bits
                    precision=0
                )
                extract_btn = gr.Button("Extract Watermark")
                extracted_text = gr.Textbox(label="Recovered Text")
                extracted_bits = gr.Textbox(label="Recovered Bitstring")


        generate_btn.click(
            fn=generate_images,
            inputs=[uploaded_image, left_prompt, right_prompt, seed],
            outputs=[status, left_gallery, right_gallery],
        )
        
        extract_btn.click(
            fn=lambda img, bits: extract_from_image(img, int(bits)),
            inputs=[extract_image, num_bits],
            outputs=[extracted_text, extracted_bits],
        )


    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=True,
        allowed_paths=[
            "/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Output",
            "/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input"
                    ]
    )
