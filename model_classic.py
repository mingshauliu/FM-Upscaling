"""Classic 3D UNet — faithful reproduction of the original pipeline architecture.

3 levels, 1 block per level, 128/128/256 channels, MaxPool3d downsampling,
non-residual bottleneck with a single FiLM, post-activation residual blocks.

Supports norm_type="group" (GroupNorm, default) or "pixel" (PixelNorm).
PixelNorm was the best-performing variant in the original pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PixelNorm(nn.Module):
    """Normalize across channels at each spatial position (resolution-invariant)."""
    def __init__(self, num_channels=None, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


def _make_norm(norm_type, num_channels):
    if norm_type == "pixel":
        return PixelNorm(num_channels)
    return nn.GroupNorm(1, num_channels)


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


class UNetBlock(nn.Module):
    """Single encoder/decoder block: norm->act->conv->FiLM x2, residual, optional pool.

    Post-activation residual: h = act(h + residual), matching the original pipeline.
    """

    def __init__(self, in_ch, out_ch, cond_dim, circular, down=True, norm_type="group"):
        super().__init__()
        self.down = down
        self.pad_mode = "circular" if circular else "constant"
        self.norm1 = _make_norm(norm_type, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3)
        self.film1 = FiLM(cond_dim, out_ch)
        self.norm2 = _make_norm(norm_type, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3)
        self.film2 = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if self.down:
            self.pool = nn.MaxPool3d(2)

    def forward(self, x, c):
        res = self.skip(x)
        h = self.act(self.norm1(x))
        h = self.film1(self.conv1(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        h = self.act(self.norm2(h))
        h = self.film2(self.conv2(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        h = self.act(h + res)
        if self.down:
            return h, self.pool(h)
        return h


class ClassicUNet(nn.Module):
    """Original pipeline architecture: 3 levels, 128/128/256 channels.

    Channels: enc1(in->bc) -> enc2(bc->bc) -> enc3(bc->2bc) -> bottleneck(2bc) ->
              dec3(4bc->2bc) -> dec2(2bc->bc) -> dec1(bc+bc//2->bc//2) -> out(bc//2->1)
    """

    def __init__(self, in_channels=2, base_channels=128, out_channels=1,
                 param_dim=6, circular_padding=True, norm_type="group"):
        super().__init__()
        bc = base_channels
        circ = circular_padding
        nt = norm_type
        self.pad_mode = "circular" if circ else "constant"
        self._ckpt = False
        cd = 128  # condition dim

        # conditioning
        self.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        self.param_mlp = nn.Sequential(nn.Linear(param_dim, 128), nn.SiLU(), nn.Linear(128, 64))
        self.cond_fuse = nn.Sequential(nn.Linear(cd, cd * 2), nn.SiLU(), nn.Linear(cd * 2, cd))

        # encoder — 1 block per level with MaxPool inside
        self.enc1 = UNetBlock(in_channels, bc, cd, circ, down=True, norm_type=nt)
        self.enc2 = UNetBlock(bc, bc, cd, circ, down=True, norm_type=nt)
        self.enc3 = UNetBlock(bc, 2*bc, cd, circ, down=True, norm_type=nt)

        # bottleneck — 2 convolutions + 1 FiLM, no residual (matches original)
        self.bn_norm1 = _make_norm(nt, 2*bc)
        self.bn_conv1 = nn.Conv3d(2*bc, 2*bc, 3)
        self.bn_norm2 = _make_norm(nt, 2*bc)
        self.bn_conv2 = nn.Conv3d(2*bc, 2*bc, 3)
        self.bn_film = FiLM(cd, 2*bc)

        # decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up3_conv = nn.Conv3d(2*bc, 2*bc, 3)
        self.dec3 = UNetBlock(4*bc, 2*bc, cd, circ, down=False, norm_type=nt)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up2_conv = nn.Conv3d(2*bc, bc, 3)
        self.dec2 = UNetBlock(2*bc, bc, cd, circ, down=False, norm_type=nt)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up1_conv = nn.Conv3d(bc, bc // 2, 3)
        self.dec1 = UNetBlock(bc + bc // 2, bc // 2, cd, circ, down=False, norm_type=nt)

        self.out_conv = nn.Conv3d(bc // 2, out_channels, 1)

    def enable_gradient_checkpointing(self):
        self._ckpt = True

    def _ckpt_call(self, module, *args):
        if self._ckpt and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x, t, params):
        c = self.cond_fuse(torch.cat([
            self.time_mlp(sinusoidal_embedding(t, 64)),
            self.param_mlp(params),
        ], dim=1))

        # encoder
        s1, x = self._ckpt_call(self.enc1, x, c)
        s2, x = self._ckpt_call(self.enc2, x, c)
        s3, x = self._ckpt_call(self.enc3, x, c)

        # bottleneck
        x = F.silu(self.bn_norm1(x))
        x = self.bn_conv1(F.pad(x, [1]*6, mode=self.pad_mode))
        x = F.silu(self.bn_norm2(x))
        x = self.bn_conv2(F.pad(x, [1]*6, mode=self.pad_mode))
        x = self.bn_film(x, c)

        # decoder
        x = self.up3_conv(F.pad(self.up3(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec3, torch.cat([x, s3], 1), c)

        x = self.up2_conv(F.pad(self.up2(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec2, torch.cat([x, s2], 1), c)

        x = self.up1_conv(F.pad(self.up1(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec1, torch.cat([x, s1], 1), c)

        return self.out_conv(x)
