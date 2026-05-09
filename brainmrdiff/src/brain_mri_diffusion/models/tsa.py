"""Tumor + Structure Aggregation (TSA) module.

Fuses 5 anatomical conditioning masks into a spatial feature map
that is injected into the UNet at multiple resolutions.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class TSAModule(nn.Module):
    """
    Input : (B, 5, H, W)  – [tumor_mask, brain_mask, wmt, cgm, lv]
    Output: (B, out_channels, H, W)

    The output is a conditioning feature map that is concatenated with
    the noisy image before entering the UNet encoder.
    """

    def __init__(
        self,
        in_channels: int = 5,
        mid_channels: int = 64,
        out_channels: int = 256,
        n_res_blocks: int = 3,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
        )

        res_blocks = [ResBlock(mid_channels) for _ in range(n_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )

        # Per-structure attention weights (learned importance weighting)
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=-1),
        )

    def forward(self, masks: Tensor) -> Tensor:
        # Compute learned per-structure importance
        w = self.attn_weights(masks)                    # (B, 5)
        masks = masks * w.unsqueeze(-1).unsqueeze(-1)   # (B, 5, H, W)

        x = self.stem(masks)
        x = self.res_blocks(x)
        return self.out_proj(x)                         # (B, out_channels, H, W)
