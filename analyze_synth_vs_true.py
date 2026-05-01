"""Quick pred-vs-true comparison for L50 SB35 synth gas vs true gas.

Loads paired sample_NNN.npy files from the synth dir and the matching
true-gas dir, computes per-sample MSE / L1 / Pearson-r / xcorr(k<=15),
and produces a summary .npz + diagnostic PNG.
"""

import os
import sys
import glob
import re
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import xcorr_metric

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNTH_DIR = "/mnt/home/mliu1/ceph/CAMELS-L50n512/synth_v3_l25model/L50_SB35_from_l25model"
TRUE_CACHE = "/mnt/home/mliu1/ceph/CAMELS-L50n512/cached/Grids_Mgas_IllustrisTNG_SB35_256_z=0.0_log1p_normed.npy"
BOX_SIZE = 50.0
OUT_NPZ = "/mnt/home/mliu1/pipeline_v3/synth_vs_true_L50_SB35.npz"
OUT_PNG = "/mnt/home/mliu1/pipeline_v3/synth_vs_true_L50_SB35.png"


def pearson(a, b):
    am = a - a.mean()
    bm = b - b.mean()
    return float((am * bm).sum() / (np.sqrt((am * am).sum() * (bm * bm).sum()) + 1e-30))


def main():
    files = sorted(glob.glob(os.path.join(SYNTH_DIR, "sample_*.npy")))
    print(f"Found {len(files)} synth samples")
    if not files:
        print("No synth files — aborting.")
        return

    idx = [int(re.search(r"sample_(\d+)", os.path.basename(f)).group(1)) for f in files]

    true_all = np.load(TRUE_CACHE, mmap_mode="r")
    print(f"True cache shape: {true_all.shape}, dtype: {true_all.dtype}")

    mse, l1, r, xcorr = [], [], [], []
    true_mean, pred_mean, true_std, pred_std = [], [], [], []

    t0 = time.time()
    for k, (i, fp) in enumerate(zip(idx, files)):
        pred = np.load(fp).astype(np.float32)
        true = np.asarray(true_all[i]).astype(np.float32)
        if pred.shape != true.shape:
            print(f"  shape mismatch idx {i}: {pred.shape} vs {true.shape}")
            continue
        d = pred - true
        mse.append(float((d * d).mean()))
        l1.append(float(np.abs(d).mean()))
        r.append(pearson(pred, true))
        xcorr.append(float(xcorr_metric(true, pred, BOX_SIZE)))
        true_mean.append(float(true.mean())); pred_mean.append(float(pred.mean()))
        true_std.append(float(true.std()));   pred_std.append(float(pred.std()))
        if (k + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  {k+1}/{len(files)}  ({dt:.1f}s elapsed, {dt/(k+1):.2f}s/sample)")

    mse = np.array(mse); l1 = np.array(l1); r = np.array(r); xcorr = np.array(xcorr)
    tm = np.array(true_mean); pm = np.array(pred_mean)
    ts = np.array(true_std);  ps = np.array(pred_std)

    print("\n=== Summary over", len(mse), "pairs ===")
    for name, v in [("MSE", mse), ("L1", l1), ("Pearson r", r), ("xcorr(k<=15)", xcorr)]:
        print(f"  {name:16s} mean={v.mean():.4f}  median={np.median(v):.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")
    print(f"  true mean/std (avg): {tm.mean():.4f} / {ts.mean():.4f}")
    print(f"  pred mean/std (avg): {pm.mean():.4f} / {ps.mean():.4f}")

    np.savez(OUT_NPZ, idx=np.array(idx[:len(mse)]), mse=mse, l1=l1, pearson=r, xcorr=xcorr,
             true_mean=tm, pred_mean=pm, true_std=ts, pred_std=ps)
    print(f"Saved: {OUT_NPZ}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0,0].hist(mse, bins=20); ax[0,0].set_title(f"MSE (mean={mse.mean():.4f})"); ax[0,0].set_xlabel("MSE")
    ax[0,1].hist(r, bins=20);   ax[0,1].set_title(f"Pearson r (mean={r.mean():.3f})"); ax[0,1].set_xlabel("r")
    ax[0,2].hist(xcorr, bins=20); ax[0,2].set_title(f"xcorr k<=15 (mean={xcorr.mean():.3f})"); ax[0,2].set_xlabel("xcorr")
    ax[1,0].scatter(tm, pm, s=12); ax[1,0].plot([tm.min(), tm.max()], [tm.min(), tm.max()], 'k--', lw=0.8)
    ax[1,0].set_xlabel("true mean"); ax[1,0].set_ylabel("pred mean"); ax[1,0].set_title("Per-sample means")
    ax[1,1].scatter(ts, ps, s=12); ax[1,1].plot([ts.min(), ts.max()], [ts.min(), ts.max()], 'k--', lw=0.8)
    ax[1,1].set_xlabel("true std"); ax[1,1].set_ylabel("pred std"); ax[1,1].set_title("Per-sample stds")
    ax[1,2].scatter(r, xcorr, s=12); ax[1,2].set_xlabel("Pearson r"); ax[1,2].set_ylabel("xcorr(k<=15)")
    ax[1,2].set_title("voxel-r vs k-space xcorr")
    fig.suptitle(f"L50 SB35 synth (l25 model, lrbstziw) vs true — N={len(mse)}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
