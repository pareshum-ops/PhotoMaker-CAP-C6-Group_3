# extract_dct_watermark.py
#
# Drop‑in script for extracting invisible DCT watermarks
# from PhotoMaker V2 output images.
#
# Usage:
#   python extract_dct_watermark.py --image path/to/image.png --num_bits 48
#
# For watermark text "PARESH":
#   6 characters × 8 bits = 48 bits

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# Import your DCT extraction function
from PhotoMaker_Extensions.dct_watermark import extract_watermark


def bits_to_text(bits):
    """Convert list of 0/1 bits into ASCII text."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)


def extract_from_image(image_path, num_bits):
    """Load image → extract bits → decode text."""
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)

    bits = extract_watermark(np_img, num_bits)
    text = bits_to_text(bits)

    return text, ''.join(map(str, bits))


def main(image_path, num_bits):
    print("=" * 60)
    print("DCT Invisible Watermark Extraction")
    print("=" * 60)

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found → {image_path}")
        return

    print(f"Extracting watermark from: {image_path}")
    print(f"Expecting {num_bits} bits")

    text, bitstring = extract_from_image(image_path, num_bits)

    print("\nRecovered Bitstring:")
    print(bitstring)

    print("\nRecovered Text:")
    print(text)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True,
                        help="Path to the watermarked image")
    parser.add_argument("--num_bits", type=int, required=True,
                        help="Number of bits to extract (e.g., 48 for 'PARESH')")

    args = parser.parse_args()
    main(args.image, args.num_bits)
