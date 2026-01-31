# cli.py

from pathlib import Path
from .pipeline_loader import load_pipeline, get_device
from .face_utils import load_face_detector
from .generation import generate_images
from .watermark import add_watermark
from .config import *


def main():
    print("=" * 50)
    print("PhotoMaker V2 CLI")
    print("=" * 50)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)

    config = globals()

    left_imgs, right_imgs, seed = generate_images(pipe, face_detector, config)

    print(f"\nSaving outputs to {output_dir}/")

    # LEFT FACE
    for prompt_text, imgs in left_imgs:
        safe = prompt_text.replace(" ", "_").replace(",", "")
        for i, img in enumerate(imgs):
            img = add_watermark(img)
            filename = f"left_{safe}_seed{seed}_{i+1}.png"
            img.save(output_dir / filename)
            print(f"Saved: {filename}")

    # RIGHT FACE
    for prompt_text, imgs in right_imgs:
        safe = prompt_text.replace(" ", "_").replace(",", "")
        for i, img in enumerate(imgs):
            img = add_watermark(img)
            filename = f"right_{safe}_seed{seed}_{i+1}.png"
            img.save(output_dir / filename)
            print(f"Saved: {filename}")

    print(f"\nDone! Generated {len(left_imgs)} left-face and {len(right_imgs)} right-face images with seed {seed}")


if __name__ == "__main__":
    main()
