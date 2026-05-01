"""
Model-parallel inference: splits UNet across 2 GPUs + torch.compile.

Standalone script — does not modify any existing pipeline code.
Encoder+bottleneck on GPU0, decoder on GPU1. Skip connections transferred
via NVLink. Effective memory: ~160 GB across 2 GPUs (handles 512³+).

Usage:
    python infer_mp.py --config config/config_infer_l25_on_SB35.yaml
    python infer_mp.py --config config/config_infer_l25_on_SB35.yaml --compile
    python infer_mp.py --config config/config_infer_l25_on_SB35.yaml --compile --n-samples 1
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchdiffeq import odeint
from tqdm import tqdm

from model_classic import ClassicUNet, sinusoidal_embedding
from train import FlowMatchingModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Model-parallel wrapper
# ---------------------------------------------------------------------------

class ModelParallelUNet(torch.nn.Module):
    """Wraps a ClassicUNet with encoder on dev0, decoder on dev1."""

    def __init__(self, net: ClassicUNet, dev0: str = "cuda:0", dev1: str = "cuda:1"):
        super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.pad_mode = net.pad_mode

        # --- GPU 0: conditioning + encoder + bottleneck ---
        self.time_mlp = net.time_mlp.to(dev0)
        self.param_mlp = net.param_mlp.to(dev0)
        self.cond_fuse = net.cond_fuse.to(dev0)

        self.enc1 = net.enc1.to(dev0)
        self.enc2 = net.enc2.to(dev0)
        self.enc3 = net.enc3.to(dev0)

        self.bn_norm1 = net.bn_norm1.to(dev0)
        self.bn_conv1 = net.bn_conv1.to(dev0)
        self.bn_norm2 = net.bn_norm2.to(dev0)
        self.bn_conv2 = net.bn_conv2.to(dev0)
        self.bn_film = net.bn_film.to(dev0)

        # --- GPU 1: decoder ---
        self.up3 = net.up3  # no params, stateless
        self.up3_conv = net.up3_conv.to(dev1)
        self.dec3 = net.dec3.to(dev1)

        self.up2 = net.up2
        self.up2_conv = net.up2_conv.to(dev1)
        self.dec2 = net.dec2.to(dev1)

        self.up1 = net.up1
        self.up1_conv = net.up1_conv.to(dev1)
        self.dec1 = net.dec1.to(dev1)

        self.out_conv = net.out_conv.to(dev1)

    def forward(self, x, t, params):
        # x on dev0
        x = x.to(self.dev0)
        t = t.to(self.dev0)
        params = params.to(self.dev0)

        c = self.cond_fuse(torch.cat([
            self.time_mlp(sinusoidal_embedding(t, 64)),
            self.param_mlp(params),
        ], dim=1))

        # encoder on GPU 0
        s1, x = self.enc1(x, c)
        s2, x = self.enc2(x, c)
        s3, x = self.enc3(x, c)

        # bottleneck on GPU 0
        x = F.silu(self.bn_norm1(x))
        x = self.bn_conv1(F.pad(x, [1]*6, mode=self.pad_mode))
        x = F.silu(self.bn_norm2(x))
        x = self.bn_conv2(F.pad(x, [1]*6, mode=self.pad_mode))
        x = self.bn_film(x, c)

        # transfer to GPU 1
        x = x.to(self.dev1)
        c = c.to(self.dev1)
        s1 = s1.to(self.dev1)
        s2 = s2.to(self.dev1)
        s3 = s3.to(self.dev1)

        # decoder on GPU 1
        x = self.up3_conv(F.pad(self.up3(x), [1]*6, mode=self.pad_mode))
        x = self.dec3(torch.cat([x, s3], 1), c)
        del s3

        x = self.up2_conv(F.pad(self.up2(x), [1]*6, mode=self.pad_mode))
        x = self.dec2(torch.cat([x, s2], 1), c)
        del s2

        x = self.up1_conv(F.pad(self.up1(x), [1]*6, mode=self.pad_mode))
        x = self.dec1(torch.cat([x, s1], 1), c)
        del s1

        return self.out_conv(x)


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------

ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}


def odeint_kwargs(method, num_steps, rtol, atol):
    if method in ADAPTIVE:
        return {'rtol': rtol, 'atol': atol}
    return {'options': {'step_size': 1.0 / num_steps}}


def sample_mp(mp_net, cdm, params, noise_std, num_steps=100, method='euler',
              rtol=1e-4, atol=1e-4):
    """Run ODE sampling with model-parallel UNet.

    ODE state lives on dev0. Each forward pass sends data to dev0->dev1->dev0.
    """
    dev0 = mp_net.dev0
    B = cdm.size(0)
    x0 = cdm + torch.randn_like(cdm) * noise_std if noise_std > 0 else cdm.clone()
    buf = torch.empty(B, 2, *cdm.shape[2:], device=dev0, dtype=cdm.dtype)
    buf[:, 1:2] = cdm
    t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=dev0)

    def f(t, x):
        buf[:, 0:1] = x
        # forward: dev0 -> dev1, output on dev1
        out = mp_net(buf, t.expand(B), params)
        # move result back to dev0 where ODE state lives
        return out.to(dev0)

    with torch.no_grad():
        trajectory = odeint(f, x0, t_span, method=method,
                            **odeint_kwargs(method, num_steps, rtol, atol))
    return trajectory[-1]


# ---------------------------------------------------------------------------
# Data loading helpers (reuse logic from infer.py without importing)
# ---------------------------------------------------------------------------

def pbc_crop(vol, size, starts=None):
    D = vol.shape[0]
    if starts is None:
        starts = tuple(random.randint(0, D - 1) for _ in range(3))
    idx = [np.arange(s, s + size) % D for s in starts]
    return vol[np.ix_(*idx)], starts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_infer_l25_on_SB35.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--n-samples", type=int, default=None, help="override n_samples")
    parser.add_argument("--dev0", default="cuda:0")
    parser.add_argument("--dev1", default="cuda:1")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inf = cfg["inference"]
    ckpt = args.checkpoint or inf.get("checkpoint")
    print(f"Checkpoint: {ckpt}")
    print(f"Devices: {args.dev0} (encoder) + {args.dev1} (decoder)")

    # Load base model on CPU
    base = FlowMatchingModel.load_from_checkpoint(ckpt, cfg=cfg, strict=False)
    noise_std = base.noise_std

    # Wrap in model-parallel
    mp_net = ModelParallelUNet(base.net, dev0=args.dev0, dev1=args.dev1)
    mp_net.eval()
    del base

    # Optional torch.compile
    if args.compile:
        print("Compiling model with torch.compile (reduce-overhead)...")
        mp_net = torch.compile(mp_net, mode="reduce-overhead")
        # Warmup compile with small tensor
        with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
            d = torch.randn(1, 1, 32, 32, 32, device=args.dev0)
            p = torch.randn(1, cfg["model"]["param_dim"], device=args.dev0)
            buf = torch.empty(1, 2, 32, 32, 32, device=args.dev0, dtype=torch.bfloat16)
            buf[:, 1:2] = d
            mp_net(buf, torch.tensor([0.5], device=args.dev0), p)
            del d, p, buf
        print("Compile warmup done.")

    method = inf.get("method", "euler")
    rtol = inf.get("rtol", 1e-4)
    atol = inf.get("atol", 1e-4)
    num_steps = inf["num_steps"]
    param_dim = cfg["model"]["param_dim"]

    for src in inf["sources"]:
        name = src["name"]
        print(f"\n{'='*50}\n{name}\n{'='*50}")

        cdm_all = np.load(src["cdm_path"], mmap_mode="r")
        gas_all = np.load(src["gas_path"], mmap_mode="r") if src.get("gas_path") else None
        params_all = np.loadtxt(src["param_path"])

        n = args.n_samples or src.get("n_samples") or cdm_all.shape[0]
        n = min(n, cdm_all.shape[0])
        if len(params_all) > n:
            params_all = params_all[:n]
        if params_all.ndim == 2 and params_all.shape[1] > param_dim:
            params_all = params_all[:, :param_dim]

        n_stoch = src.get("n_stochastic", 1)
        res = src.get("resolution", cdm_all.shape[1])
        crop = src.get("crop_size")

        out_dir = Path(inf["output_dir"]) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        gas_dir = None
        if gas_all is not None:
            gas_dir = Path(inf["output_dir"]) / f"{name}_true_gas"
            gas_dir.mkdir(parents=True, exist_ok=True)

        # Build job list (skip existing)
        jobs = []
        for idx in range(n):
            if crop and res > crop:
                dm_np, starts = pbc_crop(cdm_all[idx], crop)
                dm_np = dm_np.astype(np.float32)
                if gas_dir is not None:
                    gp = gas_dir / f"sample_{idx:03d}.npy"
                    if not gp.exists():
                        np.save(gp, pbc_crop(gas_all[idx], crop, starts)[0])
            else:
                dm_np = cdm_all[idx].astype(np.float32)
                if gas_dir is not None:
                    gp = gas_dir / f"sample_{idx:03d}.npy"
                    if not gp.exists():
                        np.save(gp, gas_all[idx])

            p_np = params_all[idx].astype(np.float32)
            for k in range(n_stoch):
                sfx = f"_{k}" if n_stoch > 1 else ""
                sp = out_dir / f"sample_{idx:03d}{sfx}.npy"
                if not sp.exists():
                    jobs.append((dm_np, p_np, sp))

        total = n * n_stoch
        pbar = tqdm(total=total, initial=total - len(jobs), desc=name)

        # Process sequentially (model spans 2 GPUs, so 1 sample at a time)
        for dm_np, p_np, path in jobs:
            d = torch.from_numpy(dm_np).unsqueeze(0).unsqueeze(0).to(args.dev0)
            pt = torch.from_numpy(p_np).unsqueeze(0).to(args.dev0)
            with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
                out = sample_mp(mp_net, d, pt, noise_std,
                                num_steps=num_steps, method=method,
                                rtol=rtol, atol=atol)
            np.save(path, out.squeeze().float().cpu().numpy())
            del d, pt, out
            torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        print(f"  -> {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
