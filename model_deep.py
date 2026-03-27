"""Deep 3D UNet with FiLM conditioning for flow matching.

Prioritises depth (4 encoder/decoder levels) over width (narrower channels).
Each level has a single ResBlock to keep skip connections tight between
corresponding encoder and decoder levels. MaxPool3d downsampling for stability.
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


class DeepUNet(nn.Module):
    """4-level UNet: deeper spatial hierarchy, narrower channels.

    Default channel progression (base_channels=64):
        enc1: in -> 64   (128^3)    skip1 -> dec1
        enc2: 64 -> 128  (64^3)     skip2 -> dec2
        enc3: 128 -> 192 (32^3)     skip3 -> dec3
        enc4: 192 -> 256 (16^3)     skip4 -> dec4
        bottleneck: 256  (8^3)
    """

    def __init__(self, in_channels=2, base_channels=64, out_channels=1,
                 param_dim=6, circular_padding=True):
        super().__init__()
        bc = base_channels
        ch = [bc, 2*bc, 3*bc, 4*bc]  # [64, 128, 192, 256]
        circ = circular_padding
        self.pad_mode = "circular" if circ else "constant"
        self._ckpt = False

        # conditioning
        self.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        self.param_mlp = nn.Sequential(nn.Linear(param_dim, 128), nn.SiLU(), nn.Linear(128, 64))
        cd = 128
        self.cond_fuse = nn.Sequential(nn.Linear(cd, cd * 2), nn.SiLU(), nn.Linear(cd * 2, cd))

        # encoder — 1 ResBlock per level, skip at each
        self.enc1 = ResBlock(in_channels, ch[0], cd, circ)
        self.enc2 = ResBlock(ch[0], ch[1], cd, circ)
        self.enc3 = ResBlock(ch[1], ch[2], cd, circ)
        self.enc4 = ResBlock(ch[2], ch[3], cd, circ)

        self.down = nn.MaxPool3d(2)

        # bottleneck — 2 stacked residual blocks
        self.bottleneck = nn.ModuleList([BottleneckBlock(ch[3], cd, circ) for _ in range(2)])

        # decoder — mirror of encoder, concat skips at each level
        self.up4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up4_conv = nn.Conv3d(ch[3], ch[3], 3)
        self.dec4 = ResBlock(ch[3] + ch[3], ch[3], cd, circ)   # cat(up, skip4)

        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up3_conv = nn.Conv3d(ch[3], ch[2], 3)
        self.dec3 = ResBlock(ch[2] + ch[2], ch[2], cd, circ)   # cat(up, skip3)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up2_conv = nn.Conv3d(ch[2], ch[1], 3)
        self.dec2 = ResBlock(ch[1] + ch[1], ch[1], cd, circ)   # cat(up, skip2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up1_conv = nn.Conv3d(ch[1], ch[0], 3)
        self.dec1 = ResBlock(ch[0] + ch[0], ch[0], cd, circ)   # cat(up, skip1)

        self.out_norm = nn.GroupNorm(1, ch[0])
        self.out_conv = nn.Conv3d(ch[0], out_channels, 1)

    def enable_gradient_checkpointing(self):
        self._ckpt = True

    def _block(self, block, x, c):
        return checkpoint(block, x, c, use_reentrant=False) if (self._ckpt and self.training) else block(x, c)

    def forward(self, x, t, params):
        c = self.cond_fuse(torch.cat([
            self.time_mlp(sinusoidal_embedding(t, 64)),
            self.param_mlp(params),
        ], dim=1))

        # encoder
        s1 = self._block(self.enc1, x, c)            # 128^3
        s2 = self._block(self.enc2, self.down(s1), c) # 64^3
        s3 = self._block(self.enc3, self.down(s2), c) # 32^3
        s4 = self._block(self.enc4, self.down(s3), c) # 16^3

        # bottleneck                                    # 8^3
        x = self.down(s4)
        for b in self.bottleneck:
            x = self._block(b, x, c)

        # decoder — each level receives skip from its corresponding encoder level
        x = self.up4_conv(F.pad(self.up4(x), [1]*6, mode=self.pad_mode))
        x = self._block(self.dec4, torch.cat([x, s4], 1), c)  # 16^3

        x = self.up3_conv(F.pad(self.up3(x), [1]*6, mode=self.pad_mode))
        x = self._block(self.dec3, torch.cat([x, s3], 1), c)  # 32^3

        x = self.up2_conv(F.pad(self.up2(x), [1]*6, mode=self.pad_mode))
        x = self._block(self.dec2, torch.cat([x, s2], 1), c)  # 64^3

        x = self.up1_conv(F.pad(self.up1(x), [1]*6, mode=self.pad_mode))
        x = self._block(self.dec1, torch.cat([x, s1], 1), c)  # 128^3

        return self.out_conv(F.silu(self.out_norm(x)))
