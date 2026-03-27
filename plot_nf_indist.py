"""Quick in-distribution truth vs pred plot for trained NF model."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from nf.nf_module import LitNFRegressor, CosmoVolumeDataset
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "/mnt/home/mliu1/pipeline_v3/nf_checkpoints/l25/nf-epoch=242-val_loss=0.0000.ckpt"
GAS = "/mnt/home/mliu1/ceph/CAMELS-L25n256/cached/Grids_Mgas_IllustrisTNG_LH_128_z=0.0_log1p_normed.npy"
PARAMS = "/mnt/home/mliu1/ceph/CAMELS-L25n256/cached/param_IllustrisTNG_LH_L25n256_normed.txt"
N_POSTERIOR = 2000
OUT = "nf_truth_vs_pred_LH.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = LitNFRegressor.load_from_checkpoint(CKPT, map_location=device)
model.eval().to(device)

ds = CosmoVolumeDataset(gas_path=GAS, param_path=PARAMS, num_cosmo=2, crop_size=None)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
print(f"Samples: {len(ds)}")

y_true_all, y_mean_all, y_std_all = [], [], []
with torch.no_grad():
    for i, (x, y, astro) in enumerate(loader):
        x, astro = x.to(device), astro.to(device)
        summary, aux = model(x, astro)
        samples = model.flow.sample(summary, num_samples=N_POSTERIOR)
        samples = samples.permute(1, 0, 2)
        y_true_all.append(y.numpy())
        y_mean_all.append(samples.mean(1).cpu().numpy())
        y_std_all.append(samples.std(1).cpu().numpy())
        if (i + 1) % 25 == 0:
            print(f"  {(i+1)*2}/{len(ds)}")

y_true = np.concatenate(y_true_all)
y_mean = np.concatenate(y_mean_all)
y_std = np.concatenate(y_std_all)

param_names = [r"$\Omega_m$", r"$\sigma_8$"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for j, (ax, label) in enumerate(zip(axes, param_names)):
    x = y_true[:, j]
    yp = y_mean[:, j]
    err = y_std[:, j]
    xmin, xmax = x.min(), x.max()
    margin = 0.05 * (xmax - xmin)
    line = np.linspace(xmin - margin, xmax + margin, 100)
    rmse = np.sqrt(np.mean((x - yp) ** 2))
    r2 = 1 - np.sum((x - yp) ** 2) / (np.sum((x - np.mean(x)) ** 2) + 1e-12)
    ax.errorbar(x, yp, yerr=err, fmt="o", ms=4, alpha=0.5,
                color="tab:blue", elinewidth=0.5, capsize=1)
    ax.plot(line, line, "r--", lw=2)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{label}  RMSE={rmse:.4f}  R²={r2:.3f}")
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(xmin - margin, xmax + margin)
    ax.grid(alpha=0.3)

plt.suptitle("NF In-Distribution (LH L25)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT}")
