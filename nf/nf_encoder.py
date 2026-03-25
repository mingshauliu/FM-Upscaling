"""
3D ResNet Encoder with FiLM astrophysics conditioning.

Extracts a summary vector from 3D gas fields, conditioned on astrophysical
parameters via FiLM layers at every encoder stage.

Key features:
1. FiLM layers modulate features based on astrophysical parameters
2. Pre-activation residual blocks with circular padding
3. Squeeze-Excitation for channel attention
4. Gradient checkpointing for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.net = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        self.net.bias.data[:feature_dim] = 1.0  # gamma = 1

    def forward(self, x, cond):
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(x.shape[0], -1, 1, 1, 1)
        beta = beta.view(x.shape[0], -1, 1, 1, 1)
        return gamma * x + beta


class SEBlock3D(nn.Module):
    """Squeeze-Excitation with SiLU."""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 8), bias=False),
            nn.SiLU(),
            nn.Linear(max(ch // reduction, 8), ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class EncoderBlockFiLM(nn.Module):
    """Encoder block with FiLM conditioning."""
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=0, bias=False)

        self.norm2 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=0, bias=False)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout)

        self.residual_conv = None
        if in_ch != out_ch:
            self.residual_conv = nn.Conv3d(in_ch, out_ch, 1, bias=False)

        # Fix 1: learnable strided conv with circular padding instead of AvgPool3d
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=0, bias=False)
        # Fix 2: SE inside residual branch (before add)
        self.se = SEBlock3D(out_ch)

        self.film1 = FiLMLayer(cond_dim, out_ch)
        self.film2 = FiLMLayer(cond_dim, out_ch)

    def forward(self, x, cond):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        h = self.norm1(x)
        h = self.act(h)
        h = F.pad(h, (1, 1, 1, 1, 1, 1), mode='circular')
        h = self.conv1(h)
        h = self.film1(h, cond)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = F.pad(h, (1, 1, 1, 1, 1, 1), mode='circular')
        h = self.conv2(h)
        h = self.film2(h, cond)

        h = self.se(h)
        h = h + residual

        # circular pad before strided conv to preserve periodicity
        return self.down(F.pad(h, (1, 1, 1, 1, 1, 1), mode='circular'))


class ResidualBlock3DFiLM(nn.Module):
    """Pre-activation residual block with FiLM."""
    def __init__(self, ch, cond_dim, expansion=2, dropout=0.1):
        super().__init__()
        mid = ch * expansion
        self.norm1 = nn.GroupNorm(1, ch)
        self.conv1 = nn.Conv3d(ch, mid, 3, padding=0, bias=False)
        self.norm2 = nn.GroupNorm(1, mid)
        self.conv2 = nn.Conv3d(mid, ch, 3, padding=0, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout)

        self.film1 = FiLMLayer(cond_dim, mid)
        self.film2 = FiLMLayer(cond_dim, ch)

    def forward(self, x, cond):
        h = self.norm1(x)
        h = self.act(h)
        h = F.pad(h, (1, 1, 1, 1, 1, 1), mode='circular')
        h = self.conv1(h)
        h = self.film1(h, cond)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = F.pad(h, (1, 1, 1, 1, 1, 1), mode='circular')
        h = self.conv2(h)
        h = self.film2(h, cond)

        return x + h


class Encoder3DFiLM(nn.Module):
    """
    3D Encoder with FiLM astrophysics conditioning.

    Extracts a summary vector from gas field volumes, conditioned on
    astrophysical parameters via FiLM modulation at every stage.

    Args:
        in_ch: Input channels (1 for gas field)
        base: Base channel multiplier
        num_astro: Number of astrophysical conditioning params
        cond_embed_dim: Embedded conditioning dimension
        dropout: Dropout rate
    """
    def __init__(self, in_ch=1, base=16, num_astro=33, cond_embed_dim=128, dropout=0.2):
        super().__init__()

        self.cond_embed = nn.Sequential(
            nn.Linear(num_astro, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        self.stem = nn.Conv3d(in_ch, base, 3, padding=0, bias=False)

        self.enc1 = EncoderBlockFiLM(base, 2*base, cond_embed_dim, dropout=dropout)
        self.enc2 = EncoderBlockFiLM(2*base, 4*base, cond_embed_dim, dropout=dropout)
        self.enc3 = EncoderBlockFiLM(4*base, 8*base, cond_embed_dim, dropout=dropout)

        self.bottleneck_res1 = ResidualBlock3DFiLM(8*base, cond_embed_dim, expansion=2, dropout=dropout)
        self.bottleneck_res2 = ResidualBlock3DFiLM(8*base, cond_embed_dim, expansion=2, dropout=dropout)
        self.bottleneck_se = SEBlock3D(8*base)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output_norm = nn.LayerNorm(8*base)

    @property
    def output_dim(self):
        """Dimension of the output summary vector."""
        return self.output_norm.normalized_shape[0]

    def forward(self, x, astro_params):
        """
        Args:
            x: Input volume (B, 1, D, H, W)
            astro_params: Astrophysical params (B, num_astro)

        Returns:
            summary: (B, 8*base)
        """
        cond = self.cond_embed(astro_params)

        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='circular')
        x = self.stem(x)

        x = checkpoint(self.enc1, x, cond, use_reentrant=False)
        x = checkpoint(self.enc2, x, cond, use_reentrant=False)
        x = checkpoint(self.enc3, x, cond, use_reentrant=False)

        x = checkpoint(self.bottleneck_res1, x, cond, use_reentrant=False)
        x = checkpoint(self.bottleneck_res2, x, cond, use_reentrant=False)
        x = self.bottleneck_se(x)

        x = self.pool(x).flatten(1)

        return self.output_norm(x)
