"""
Inference script for normalizing flow parameter inference.

Loads a trained NF checkpoint, runs inference on 1P sources defined in
config.yaml, and generates pred vs true scatter plots with error bars.

Usage:
    python nf_infer.py                              # uses config.yaml
    python nf_infer.py --config my.yaml
    python nf_infer.py --checkpoint path.ckpt        # override checkpoint
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except (ImportError, OSError):
    pass

from .nf_module import LitNFRegressor, CosmoVolumeDataset


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_with_uncertainty(model, dataloader, n_samples=2000, device="cuda"):
    """
    Run inference with uncertainty quantification.

    Returns:
        y_true: (N, num_cosmo)
        y_mean: (N, num_cosmo) posterior mean
        y_std: (N, num_cosmo) posterior std
        aux_pred: (N, num_cosmo) auxiliary head predictions
    """
    y_true_list, y_mean_list, y_std_list, aux_list = [], [], [], []

    for x, y, astro in dataloader:
        x = x.to(device)
        astro = astro.to(device)

        summary, aux = model(x, astro)

        samples = model.flow.sample(summary, num_samples=n_samples)
        samples = samples.permute(1, 0, 2)  # (B, n_samples, num_cosmo)

        y_true_list.append(y.numpy())
        y_mean_list.append(samples.mean(dim=1).cpu().numpy())
        y_std_list.append(samples.std(dim=1).cpu().numpy())
        aux_list.append(aux.cpu().numpy())

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_mean_list),
        np.concatenate(y_std_list),
        np.concatenate(aux_list),
    )


def plot_truth_vs_pred(y_true, y_pred, y_std, out_path, title="", dpi=200):
    """Plot true vs predicted with error bars for each cosmo parameter."""
    num_params = y_true.shape[1]
    param_names = [r"$\Omega_m$", r"$\sigma_8$"][:num_params]

    fig, axes = plt.subplots(1, num_params, figsize=(6 * num_params, 5))
    if num_params == 1:
        axes = [axes]

    for j, (ax, label) in enumerate(zip(axes, param_names)):
        x = y_true[:, j]
        y = y_pred[:, j]
        err = y_std[:, j]

        xmin, xmax = x.min(), x.max()
        margin = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
        line = np.linspace(xmin - margin, xmax + margin, 100)

        mse = np.mean((x - y) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((x - y) ** 2) / (np.sum((x - np.mean(x)) ** 2) + 1e-12)
        mean_std = np.mean(err)

        ax.errorbar(
            x, y, yerr=err,
            fmt='o', markersize=6, alpha=0.7,
            color='tab:blue', elinewidth=0.8, capsize=2,
            label=title if title else "NF"
        )
        ax.plot(line, line, 'r--', linewidth=2)

        ax.legend(fontsize=9)
        ax.set_xlabel("Truth", fontsize=12)
        ax.set_ylabel("Prediction", fontsize=12)
        ax.set_title(f"{label}  RMSE={rmse:.4f}  R²={r2:.3f}  <σ>={mean_std:.4f}",
                      fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(xmin - margin, xmax + margin)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_calibration(y_true, y_pred, y_std, out_path, dpi=200):
    """Plot uncertainty calibration (expected vs observed coverage)."""
    from scipy.stats import norm

    num_params = y_true.shape[1]
    param_names = [r"$\Omega_m$", r"$\sigma_8$"][:num_params]

    fig, axes = plt.subplots(1, num_params, figsize=(6 * num_params, 5))
    if num_params == 1:
        axes = [axes]

    for j, (ax, label) in enumerate(zip(axes, param_names)):
        z = np.abs(y_true[:, j] - y_pred[:, j]) / (y_std[:, j] + 1e-12)

        confidence_levels = np.linspace(0.1, 0.99, 50)
        observed = [np.mean(z < norm.ppf((1 + c) / 2)) for c in confidence_levels]

        ax.plot(confidence_levels, observed, 'b-o', markersize=4, label='Observed')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
        ax.fill_between(confidence_levels,
                        confidence_levels - 0.1, confidence_levels + 0.1,
                        alpha=0.2, color='red', label='±10% band')

        ax.set_xlabel("Expected coverage", fontsize=12)
        ax.set_ylabel("Observed coverage", fontsize=12)
        ax.set_title(f"Calibration: {label}", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_residuals(y_true, y_pred, y_std, out_path, dpi=200):
    """Plot normalized residual distributions."""
    from scipy.stats import norm

    num_params = y_true.shape[1]
    param_names = [r"$\Omega_m$", r"$\sigma_8$"][:num_params]

    fig, axes = plt.subplots(1, num_params, figsize=(6 * num_params, 5))
    if num_params == 1:
        axes = [axes]

    for j, (ax, label) in enumerate(zip(axes, param_names)):
        residuals = (y_true[:, j] - y_pred[:, j]) / (y_std[:, j] + 1e-12)

        ax.hist(residuals, bins=30, density=True, alpha=0.7,
                color='tab:blue', label='Normalized residuals')

        x = np.linspace(-4, 4, 100)
        ax.plot(x, norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')

        mean_r = np.mean(residuals)
        std_r = np.std(residuals)
        ax.axvline(mean_r, color='green', linestyle='--', label=f'Mean={mean_r:.2f}')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, f'\u03bc={mean_r:.3f}\n\u03c3={std_r:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_xlabel("Normalized residual", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Residuals: {label}", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-4, 4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def process_source(source_cfg, model, nf_cfg, output_dir, device):
    """Run inference on one 1P source and generate plots."""
    name = source_cfg["name"]
    print(f"\n{'='*60}")
    print(f"Processing source: {name}")
    print(f"{'='*60}")

    ds = CosmoVolumeDataset(
        gas_path=source_cfg["gas_path"],
        param_path=source_cfg["param_path"],
        num_cosmo=nf_cfg["model"]["num_cosmo"],
        crop_size=source_cfg.get("crop_size"),
    )

    loader = DataLoader(
        ds, batch_size=nf_cfg.get("inference", {}).get("batch_size", 2),
        shuffle=False, num_workers=0,
    )

    n_samples = nf_cfg.get("inference", {}).get("n_posterior_samples", 2000)
    print(f"  Running inference ({len(ds)} samples, {n_samples} posterior draws)...")

    y_true, y_mean, y_std, aux_pred = predict_with_uncertainty(
        model, loader, n_samples=n_samples, device=device
    )

    # Print metrics
    param_names = ["Omega_m", "sigma_8"][:nf_cfg["model"]["num_cosmo"]]
    for j, pname in enumerate(param_names):
        rmse = np.sqrt(np.mean((y_true[:, j] - y_mean[:, j]) ** 2))
        mae = np.mean(np.abs(y_true[:, j] - y_mean[:, j]))
        mean_std = np.mean(y_std[:, j])
        print(f"  {pname}: RMSE={rmse:.4f}, MAE={mae:.4f}, <sigma>={mean_std:.4f}")

    source_dir = output_dir / name
    source_dir.mkdir(parents=True, exist_ok=True)

    # Save raw predictions first so data is never lost if plotting fails
    np.savez(
        source_dir / "predictions.npz",
        y_true=y_true, y_mean=y_mean, y_std=y_std, aux_pred=aux_pred,
    )
    print(f"  Saved predictions to {source_dir / 'predictions.npz'}")

    dpi = nf_cfg.get("inference", {}).get("dpi", 200)

    plot_truth_vs_pred(
        y_true, y_mean, y_std,
        source_dir / "truth_vs_pred.png",
        title=name, dpi=dpi,
    )

    plot_calibration(
        y_true, y_mean, y_std,
        source_dir / "calibration.png", dpi=dpi,
    )

    plot_residuals(
        y_true, y_mean, y_std,
        source_dir / "residuals.png", dpi=dpi,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path from config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    nf_cfg = cfg["nf"]
    infer_cfg = nf_cfg.get("inference", {})

    ckpt = args.checkpoint or infer_cfg.get("checkpoint")
    if ckpt is None:
        import glob
        ckpt_dir = nf_cfg.get("training", {}).get("checkpoint_dir", "nf_checkpoints")
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        if not paths:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        ckpt = max(paths, key=os.path.getmtime)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from {ckpt}")

    model = LitNFRegressor.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    output_dir = Path(infer_cfg.get("output_dir", "outputs/nf_inference"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in infer_cfg.get("sources", []):
        process_source(source, model, nf_cfg, output_dir, device)

    print(f"\nAll inference complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
