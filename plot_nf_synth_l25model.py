"""Truth vs pred plot for NF run on FM-synthesized gas maps from the
2iky8gr1 L25 model. Configurable via CLI for L25 LH or L50 SB35 dirs.

Usage:
  python plot_nf_synth_l25model.py \
      --synth-dir <dir of sample_NNN.npy> \
      --params <param txt path> \
      --nf-ckpt <path/to/nf.ckpt> \
      --out <output png path> \
      --max-samples 30 \
      [--num-cosmo 2]
"""

import argparse
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nf.nf_module import LitNFRegressor

N_POSTERIOR = 2000


class IndexedGasDataset(Dataset):
    def __init__(self, files, indices, params_all, num_cosmo):
        self.files = files
        self.indices = indices
        self.targets = params_all[indices, :num_cosmo]
        self.astro = params_all[indices, num_cosmo:]
        print(f"IndexedGasDataset: {len(files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = np.load(self.files[idx]).astype(np.float32)
        vol = torch.from_numpy(vol).unsqueeze(0)
        return vol, torch.from_numpy(self.targets[idx]), torch.from_numpy(self.astro[idx])


def run_inference(model, dataset, device, n_posterior):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    y_true_all, y_mean_all, y_std_all = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y, astro) in enumerate(loader):
            x, astro = x.to(device), astro.to(device)
            summary, aux = model(x, astro)
            samples = model.flow.sample(summary, num_samples=n_posterior).permute(1, 0, 2)
            y_true_all.append(y.numpy())
            y_mean_all.append(samples.mean(1).cpu().numpy())
            y_std_all.append(samples.std(1).cpu().numpy())
            print(f"  batch {i+1}: {(i+1)*x.shape[0]}/{len(dataset)}  ({time.time()-t0:.1f}s)")
    return (np.concatenate(y_true_all),
            np.concatenate(y_mean_all),
            np.concatenate(y_std_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--nf-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--num-cosmo", type=int, default=2)
    ap.add_argument("--title", default="NF on FM Synthetic Gas")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.synth_dir, "sample_*.npy")))
    if len(files) < args.max_samples:
        print(f"WARN: only {len(files)} synth files in {args.synth_dir}, expected {args.max_samples}")
    files = files[:args.max_samples]
    if not files:
        print(f"ERR: no samples in {args.synth_dir}")
        sys.exit(1)
    indices = [int(re.search(r"sample_(\d+)", os.path.basename(f)).group(1)) for f in files]
    params_all = np.loadtxt(args.params).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"NF ckpt: {args.nf_ckpt}")
    print(f"Synth dir: {args.synth_dir}  ({len(files)} samples)")

    model = LitNFRegressor.load_from_checkpoint(args.nf_ckpt, map_location=device)
    model.eval().to(device)

    ds = IndexedGasDataset(files, indices, params_all, args.num_cosmo)
    y_true, y_mean, y_std = run_inference(model, ds, device, N_POSTERIOR)

    param_names = [r"$\Omega_m$", r"$\sigma_8$"][:args.num_cosmo]
    fig, axes = plt.subplots(1, args.num_cosmo, figsize=(6 * args.num_cosmo, 5))
    if args.num_cosmo == 1:
        axes = [axes]
    for j, (ax, label) in enumerate(zip(axes, param_names)):
        x = y_true[:, j]; yp = y_mean[:, j]; err = y_std[:, j]
        xmin, xmax = x.min(), x.max()
        m = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
        line = np.linspace(xmin - m, xmax + m, 100)
        rmse = float(np.sqrt(np.mean((x - yp) ** 2)))
        r2 = 1 - float(np.sum((x - yp) ** 2) / (np.sum((x - x.mean()) ** 2) + 1e-12))
        ax.errorbar(x, yp, yerr=err, fmt="o", ms=4, alpha=0.5,
                    color="tab:blue", elinewidth=0.5, capsize=1)
        ax.plot(line, line, "r--", lw=2)
        ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
        ax.set_title(f"{label}  RMSE={rmse:.4f}  R²={r2:.3f}")
        ax.set_xlim(xmin - m, xmax + m); ax.set_ylim(xmin - m, xmax + m)
        ax.grid(alpha=0.3)
    plt.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
