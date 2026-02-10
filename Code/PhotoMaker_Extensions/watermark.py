from PIL import Image, ImageDraw, ImageFont
import math

def add_watermark(image, text="© AI-Generated image by CAP-C6-Group_3", opacity=80, spacing=250, angle=30):
    """
    Adds a repeating diagonal watermark pattern across the entire image.
    - text: watermark text
    - opacity: transparency (0–255)
    - spacing: distance between repeated watermarks
    - angle: rotation angle for diagonal effect
    """

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    W, H = image.size

    # Transparent layer for watermark
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Dynamic font size based on image width
    font_size = max(42, W // 8)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Create a temporary image to measure text size
    temp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Create a tile for the watermark text
    tile = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.text((0, 0), text, fill=(255, 255, 255, opacity), font=font)

    # Rotate tile for diagonal effect
    rotated_tile = tile.rotate(angle, expand=True)
    tile_w, tile_h = rotated_tile.size

    # Tile the watermark across the entire image
    for y in range(-tile_h, H + tile_h, spacing):
        for x in range(-tile_w, W + tile_w, spacing):
            layer.alpha_composite(rotated_tile, dest=(x, y))

    # Merge watermark layer with original image
    return Image.alpha_composite(image, layer).convert("RGB")
