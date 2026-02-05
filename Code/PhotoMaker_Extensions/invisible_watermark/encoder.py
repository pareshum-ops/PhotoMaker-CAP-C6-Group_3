# PhotoMaker_Extensions/invisible_watermark/encoder.py

import torch
import torch.nn as nn

class WatermarkEncoder(nn.Module):
    def __init__(self, bit_length=64):
        super().__init__()
        self.bit_length = bit_length

        # Map watermark bits → feature map
        self.embed = nn.Sequential(
            nn.Linear(bit_length, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64),
            nn.ReLU()
        )

        # Simple U-Net style encoder
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, image, bits):
        B, C, H, W = image.shape

        # Expand watermark to spatial map
        wm = self.embed(bits).view(B, 1, 64, 64)
        wm = torch.nn.functional.interpolate(wm, size=(H, W), mode="bilinear")

        # Concatenate image + watermark map
        x = torch.cat([image, wm], dim=1)

        # Predict residual
        residual = self.conv(x)

        # Add residual to image
        return torch.clamp(image + 0.01 * residual, 0, 1)
