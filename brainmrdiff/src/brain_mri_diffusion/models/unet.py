"""Conditional UNet backbone for BrainMRDiff.

Architecture:
  - Encoder / Decoder with skip connections
  - Sinusoidal timestep embedding injected via AdaGN
  - TSA conditioning feature map concatenated at input
  - Self-attention at configurable resolutions
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, base_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.base_dim = base_dim

    def forward(self, t: Tensor) -> Tensor:
        emb = get_timestep_embedding(t, self.base_dim)
        return self.net(emb)


# ---------------------------------------------------------------------------
# Normalization with timestep conditioning (AdaGN)
# ---------------------------------------------------------------------------


class AdaGroupNorm(nn.Module):
    """Adaptive Group Norm conditioned on timestep + optional modality embedding."""

    def __init__(self, num_groups: int, channels: int, emb_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * channels)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = rearrange(h, "b c h w -> b (h w) c")
        h, _ = self.attn(h, h, h)
        h = rearrange(h, "b (h w) c -> b c h w", h=H, w=W)
        return x + h


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = AdaGroupNorm(num_groups, in_channels, emb_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = AdaGroupNorm(num_groups, out_channels, emb_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.dropout(self.act(self.norm2(h, emb)))
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        has_attn: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, emb_dim, dropout)
        self.attn = SelfAttention(out_ch) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x: Tensor, emb: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.res(x, emb)
        h = self.attn(h) if not isinstance(self.attn, nn.Identity) else self.attn(h)
        skip = h
        h = self.down(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        emb_dim: int,
        has_attn: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, emb_dim, dropout)
        self.attn = SelfAttention(out_ch) if has_attn else nn.Identity()

    def forward(self, x: Tensor, skip: Tensor, emb: Tensor) -> Tensor:
        x = self.up(x)
        # Handle potential size mismatch due to odd dims
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)
        return self.attn(x) if not isinstance(self.attn, nn.Identity) else self.attn(x)


# ---------------------------------------------------------------------------
# Main UNet
# ---------------------------------------------------------------------------


class ConditionalUNet(nn.Module):
    """
    Conditional UNet for BrainMRDiff.

    Inputs:
        x        : (B, in_channels, H, W)  – noisy MRI slice (1 channel)
        t        : (B,)                    – diffusion timestep
        cond_feat: (B, cond_channels, H, W) – TSA output

    Output:
        (B, in_channels, H, W)  – predicted noise
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 1,
        cond_channels: int = 256,
        base_channels: int = 64,
        channel_mults: Optional[List[int]] = None,
        attn_resolutions: Optional[List[int]] = None,
        dropout: float = 0.1,
        num_modalities: int = 4,
    ) -> None:
        super().__init__()

        channel_mults = channel_mults or [1, 2, 4, 8]
        attn_resolutions = attn_resolutions or [16]

        emb_dim = base_channels * 4
        self.emb_dim = emb_dim

        # Timestep + modality embeddings
        self.time_emb = TimestepEmbedding(base_channels, emb_dim)
        self.mod_emb = nn.Embedding(num_modalities, emb_dim)

        # Fuse modality into time embedding
        self.emb_fuse = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.SiLU())

        # Input conv: noisy image + TSA features
        self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, 1, 1)

        # Build encoder
        self.downs = nn.ModuleList()
        cur_res = image_size
        ch_in = base_channels
        skip_channels = [base_channels]  # track channels for each skip

        for mult in channel_mults:
            ch_out = base_channels * mult
            has_attn = cur_res in attn_resolutions
            self.downs.append(DownBlock(ch_in, ch_out, emb_dim, has_attn, dropout))
            skip_channels.append(ch_out)
            cur_res //= 2
            ch_in = ch_out

        # Bottleneck
        self.mid_res1 = ResidualBlock(ch_in, ch_in, emb_dim, dropout)
        self.mid_attn = SelfAttention(ch_in)
        self.mid_res2 = ResidualBlock(ch_in, ch_in, emb_dim, dropout)

        # Build decoder
        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            ch_out = base_channels * mult
            skip_ch = skip_channels.pop()
            has_attn = (cur_res * 2) in attn_resolutions
            self.ups.append(UpBlock(ch_in, skip_ch, ch_out, emb_dim, has_attn, dropout))
            cur_res *= 2
            ch_in = ch_out

        # Final output
        self.out_norm = nn.GroupNorm(8, ch_in)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch_in, in_channels, 3, 1, 1)

    def forward(self, x: Tensor, t: Tensor, cond_feat: Tensor, modality: Tensor) -> Tensor:
        # Fuse time + modality
        t_emb = self.time_emb(t)
        m_emb = self.mod_emb(modality)
        emb = self.emb_fuse(torch.cat([t_emb, m_emb], dim=-1))

        # Concatenate TSA features with noisy image
        h = torch.cat([x, cond_feat], dim=1)
        h = self.input_conv(h)

        # Encoder
        skips = [h]
        for down in self.downs:
            h, skip = down(h, emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        # Decoder
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, emb)

        return self.out_conv(self.out_act(self.out_norm(h)))
