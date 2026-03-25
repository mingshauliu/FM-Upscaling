"""
Prepare synthetic gas fields for NF inference.

Stacks per-sample .npy files from synthesis output into single arrays
that CosmoVolumeDataset can load. Uses the first stochastic realisation
(k=0) from each sample.

Usage:
    python prepare_nf_synth.py --synth_dir outputs/synth_l50crop/L50crop_1P_L25 \
        --param_path /path/to/params.txt \
        --out_dir outputs/nf_synth_ready/L50crop_1P_L25 \
        --param_cols 6
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth_dir", required=True,
                        help="Directory with sample_XXX_0.npy files")
    parser.add_argument("--param_path", required=True,
                        help="Original param file for this 1P set")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for stacked files")
    parser.add_argument("--param_cols", type=int, default=None,
                        help="Truncate params to this many columns (default: all)")
    args = parser.parse_args()

    synth_dir = Path(args.synth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find unique sample indices, use k=0 realisation
    files = sorted(synth_dir.glob("sample_*_0.npy"))
    if not files:
        # Try without stochastic suffix
        files = sorted(synth_dir.glob("sample_*.npy"))
        files = [f for f in files if len(f.stem.split("_")) == 2]

    if not files:
        print(f"No samples found in {synth_dir}")
        return

    print(f"Stacking {len(files)} samples from {synth_dir}")
    arrays = [np.load(f) for f in files]
    stacked = np.stack(arrays, axis=0)
    print(f"  Stacked shape: {stacked.shape}")

    gas_out = out_dir / "gas_synth.npy"
    np.save(gas_out, stacked)
    print(f"  Saved: {gas_out}")

    # Copy/truncate params
    params = np.loadtxt(args.param_path)
    if len(params) > len(files):
        params = params[:len(files)]
    if args.param_cols and params.ndim == 2 and params.shape[1] > args.param_cols:
        print(f"  Truncating params from {params.shape[1]} to {args.param_cols} columns")
        params = params[:, :args.param_cols]

    param_out = out_dir / "params.txt"
    np.savetxt(param_out, params)
    print(f"  Saved: {param_out} (shape {params.shape})")


if __name__ == "__main__":
    main()
