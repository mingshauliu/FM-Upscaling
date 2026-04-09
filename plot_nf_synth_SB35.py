"""Truth vs pred plot for NF run on FM-synthesized SB35 gas maps (L50).

Loads per-sample synthetic gas files from synth_v3/L50_SB35_full,
runs NF inference, and plots true vs predicted cosmo params.
Also runs inference on the corresponding true gas maps for comparison.
"""

import sys, os, glob, re
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from nf.nf_module import LitNFRegressor, CosmoVolumeDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NF_CKPT   = "/mnt/home/mliu1/pipeline_v3/nf_checkpoints/l50/nf-epoch=150-val_loss=0.0000.ckpt"
SYNTH_DIR = "/mnt/home/mliu1/ceph/CAMELS-L50n512/synth_v3/L50_SB35_full"
TRUE_DIR  = "/mnt/home/mliu1/ceph/CAMELS-L50n512/synth_v3/L50_SB35_full_true_gas"
PARAMS    = "/mnt/home/mliu1/ceph/CAMELS-L50n512/cached/param_IllustrisTNG_SB35_L50n512_normed.txt"
N_POSTERIOR = 2000
OUT = "nf_truth_vs_pred_SB35_synth.png"


class IndexedGasDataset(Dataset):
    """Dataset that loads individual sample_NNN.npy files with matched params."""

    def __init__(self, sample_dir, param_path, num_cosmo=2):
        files = sorted(glob.glob(os.path.join(sample_dir, "sample_*.npy")))
        self.files = files
        self.indices = [int(re.search(r"sample_(\d+)", f).group(1)) for f in files]
        params_all = np.loadtxt(param_path).astype(np.float32)
        self.targets = params_all[self.indices, :num_cosmo]
        self.astro = params_all[self.indices, num_cosmo:]
        print(f"IndexedGasDataset: {len(files)} samples from {sample_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = np.load(self.files[idx]).astype(np.float32)
        vol = torch.from_numpy(vol).unsqueeze(0)  # (1, D, D, D)
        target = torch.from_numpy(self.targets[idx])
        astro = torch.from_numpy(self.astro[idx])
        return vol, target, astro


def run_inference(model, dataset, device, n_posterior):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    y_true_all, y_mean_all, y_std_all = [], [], []
    with torch.no_grad():
        for i, (x, y, astro) in enumerate(loader):
            x, astro = x.to(device), astro.to(device)
            summary, aux = model(x, astro)
            samples = model.flow.sample(summary, num_samples=n_posterior)
            samples = samples.permute(1, 0, 2)
            y_true_all.append(y.numpy())
            y_mean_all.append(samples.mean(1).cpu().numpy())
            y_std_all.append(samples.std(1).cpu().numpy())
            if (i + 1) % 25 == 0:
                print(f"  {(i+1)*2}/{len(dataset)}")

    return (np.concatenate(y_true_all),
            np.concatenate(y_mean_all),
            np.concatenate(y_std_all))


# --- Main ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = LitNFRegressor.load_from_checkpoint(NF_CKPT, map_location=device)
model.eval().to(device)

# Find common sample indices between synth and true
synth_files = sorted(glob.glob(os.path.join(SYNTH_DIR, "sample_*.npy")))
true_files = sorted(glob.glob(os.path.join(TRUE_DIR, "sample_*.npy")))
synth_ids = {int(re.search(r"sample_(\d+)", f).group(1)) for f in synth_files}
true_ids = {int(re.search(r"sample_(\d+)", f).group(1)) for f in true_files}
common_ids = sorted(synth_ids & true_ids)
print(f"Synth samples: {len(synth_ids)}, True samples: {len(true_ids)}, Common: {len(common_ids)}")

# Run on synth
print("\n--- Synth gas inference ---")
ds_synth = IndexedGasDataset(SYNTH_DIR, PARAMS)
y_true_s, y_mean_s, y_std_s = run_inference(model, ds_synth, device, N_POSTERIOR)

# Run on true gas (only common indices)
print("\n--- True gas inference ---")
ds_true = IndexedGasDataset(TRUE_DIR, PARAMS)
y_true_t, y_mean_t, y_std_t = run_inference(model, ds_true, device, N_POSTERIOR)

# Plot: 2 rows (synth, true) x 2 cols (Omega_m, sigma_8)
param_names = [r"$\Omega_m$", r"$\sigma_8$"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for row, (y_true, y_mean, y_std, tag) in enumerate([
    (y_true_s, y_mean_s, y_std_s, "Synth Gas"),
    (y_true_t, y_mean_t, y_std_t, "True Gas"),
]):
    for j, (label) in enumerate(param_names):
        ax = axes[row, j]
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
        ax.set_title(f"{tag} — {label}  RMSE={rmse:.4f}  R²={r2:.3f}")
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(xmin - margin, xmax + margin)
        ax.grid(alpha=0.3)

plt.suptitle("NF on SB35 L50 — Synth vs True Gas", fontsize=14)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT}")
