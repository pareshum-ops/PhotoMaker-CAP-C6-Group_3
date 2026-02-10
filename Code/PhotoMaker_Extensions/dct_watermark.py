import numpy as np
import cv2

# -----------------------------
# Utility: 8x8 DCT and IDCT
# -----------------------------
def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))

# ---------------------------------------------------
# Embed watermark bits into mid‑frequency DCT coeffs
# ---------------------------------------------------
def embed_watermark(image, watermark_bits, strength=5):
    """
    image: HxWx3 uint8
    watermark_bits: list/array of 0/1 bits
    strength: magnitude added to DCT coefficient
    """
    img = image.copy().astype(np.float32)
    h, w, _ = img.shape

    # Convert to Y channel (luminance)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]

    # Ensure divisible by 8
    H = h - (h % 8)
    W = w - (w % 8)
    Y = Y[:H, :W]

    bit_idx = 0
    total_bits = len(watermark_bits)

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            if bit_idx >= total_bits:
                break

            block = Y[i:i+8, j:j+8]
            dct_block = dct2(block)

            # Choose mid‑frequency coefficient
            # (2,3) is a common robust choice
            if watermark_bits[bit_idx] == 1:
                dct_block[2, 3] += strength
            else:
                dct_block[2, 3] -= strength

            Y[i:i+8, j:j+8] = idct2(dct_block)
            bit_idx += 1

    # Put Y back
    ycrcb[:H, :W, 0] = Y
    watermarked = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return np.clip(watermarked, 0, 255).astype(np.uint8)

# ---------------------------------------------------
# Extract watermark bits
# ---------------------------------------------------
def extract_watermark(image, num_bits):
    img = image.astype(np.float32)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]

    h, w = Y.shape
    H = h - (h % 8)
    W = w - (w % 8)
    Y = Y[:H, :W]

    bits = []

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            if len(bits) >= num_bits:
                break

            block = Y[i:i+8, j:j+8]
            dct_block = dct2(block)

            # Read the sign of the coefficient
            bits.append(1 if dct_block[2, 3] > 0 else 0)

    return bits
