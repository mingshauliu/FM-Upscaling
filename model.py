"""3D UNet with FiLM conditioning for flow matching.

Fix 1: MaxPool3d -> learnable strided conv with circular padding before stride.
Fix 2: Bottleneck uses stacked residual blocks with FiLM (baryon bridge MidBlock style).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x, c):
        s, b = self.proj(c).chunk(2, dim=1)
        return x * (1 + s.reshape(-1, x.size(1), 1, 1, 1)) + b.reshape(-1, x.size(1), 1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, circular):
        super().__init__()
        self.pad_mode = "circular" if circular else "constant"
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3)
        self.film1 = FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3)
        self.film2 = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, c):
        h = self.act(self.norm1(x))
        h = self.film1(self.conv1(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        h = self.act(self.norm2(h))
        h = self.film2(self.conv2(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        return h + self.skip(x)


# Fix 2: residual bottleneck block (norm->act->conv->FiLM->norm->act->conv + residual)
class BottleneckBlock(nn.Module):
    def __init__(self, channels, cond_dim, circular):
        super().__init__()
        self.pad_mode = "circular" if circular else "constant"
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3)
        self.film = FiLM(cond_dim, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3)
        self.act = nn.SiLU()

    def forward(self, x, c):
        h = self.act(self.norm1(x))
        h = self.film(self.conv1(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        h = self.act(self.norm2(h))
        h = self.conv2(F.pad(h, [1]*6, mode=self.pad_mode))
        return h + x


class UNet(nn.Module):
    def __init__(self, in_channels=2, base_channels=128, out_channels=1,
                 param_dim=6, circular_padding=True, num_blocks=2):
        super().__init__()
        bc = base_channels
        circ = circular_padding
        self.pad_mode = "circular" if circ else "constant"
        self._ckpt = False

        # conditioning: time (64) + params (64) -> 128 -> fused 128
        self.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        self.param_mlp = nn.Sequential(nn.Linear(param_dim, 128), nn.SiLU(), nn.Linear(128, 64))
        cd = 128
        self.cond_fuse = nn.Sequential(nn.Linear(cd, cd * 2), nn.SiLU(), nn.Linear(cd * 2, cd))

        # encoder
        self.enc1 = nn.ModuleList([ResBlock(in_channels if i == 0 else bc, bc, cd, circ) for i in range(num_blocks)])
        self.enc2 = nn.ModuleList([ResBlock(bc, bc, cd, circ) for _ in range(num_blocks)])
        self.enc3 = nn.ModuleList([ResBlock(bc if i == 0 else 2*bc, 2*bc, cd, circ) for i in range(num_blocks)])

        # Fix 1: learnable strided conv downsampling instead of MaxPool3d
        self.down1 = nn.Conv3d(bc, bc, kernel_size=3, stride=2, padding=0)
        self.down2 = nn.Conv3d(bc, bc, kernel_size=3, stride=2, padding=0)
        self.down3 = nn.Conv3d(2*bc, 2*bc, kernel_size=3, stride=2, padding=0)

        # Fix 2: stacked residual bottleneck blocks with FiLM
        self.bottleneck = nn.ModuleList([BottleneckBlock(2*bc, cd, circ) for _ in range(2)])

        # decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up3_conv = nn.Conv3d(2*bc, 2*bc, 3)
        self.dec3 = nn.ModuleList([ResBlock(4*bc if i == 0 else 2*bc, 2*bc, cd, circ) for i in range(num_blocks)])

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up2_conv = nn.Conv3d(2*bc, bc, 3)
        self.dec2 = nn.ModuleList([ResBlock(2*bc if i == 0 else bc, bc, cd, circ) for i in range(num_blocks)])

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up1_conv = nn.Conv3d(bc, bc // 2, 3)
        self.dec1 = nn.ModuleList([ResBlock(bc // 2 + bc if i == 0 else bc // 2, bc // 2, cd, circ) for i in range(num_blocks)])

        self.out_norm = nn.GroupNorm(1, bc // 2)
        self.out_conv = nn.Conv3d(bc // 2, out_channels, 1)

    def enable_gradient_checkpointing(self):
        self._ckpt = True

    def _blocks(self, blocks, x, c):
        for b in blocks:
            x = checkpoint(b, x, c, use_reentrant=False) if (self._ckpt and self.training) else b(x, c)
        return x

    def forward(self, x, t, params):
        c = self.cond_fuse(torch.cat([
            self.time_mlp(sinusoidal_embedding(t, 64)),
            self.param_mlp(params),
        ], dim=1))

        # encoder — skip from pre-downsample, then learnable strided conv down
        s1 = self._blocks(self.enc1, x, c)
        s2 = self._blocks(self.enc2, self.down1(F.pad(s1, [1]*6, mode=self.pad_mode)), c)
        s3 = self._blocks(self.enc3, self.down2(F.pad(s2, [1]*6, mode=self.pad_mode)), c)
        x = self._blocks(self.bottleneck, self.down3(F.pad(s3, [1]*6, mode=self.pad_mode)), c)

        x = self.up3_conv(F.pad(self.up3(x), [1]*6, mode=self.pad_mode))
        x = self._blocks(self.dec3, torch.cat([x, s3], 1), c)
        x = self.up2_conv(F.pad(self.up2(x), [1]*6, mode=self.pad_mode))
        x = self._blocks(self.dec2, torch.cat([x, s2], 1), c)
        x = self.up1_conv(F.pad(self.up1(x), [1]*6, mode=self.pad_mode))
        x = self._blocks(self.dec1, torch.cat([x, s1], 1), c)

        return self.out_conv(F.silu(self.out_norm(x)))
