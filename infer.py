"""
Synthesize gas maps from a trained checkpoint.

    python infer.py                              # uses config.yaml
    python infer.py --config my.yaml
    python infer.py --checkpoint path.ckpt       # override checkpoint
"""

import argparse, copy, glob, os, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from train import FlowMatchingModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)


def find_best_ckpt(log_dir="lightning_logs"):
    paths = sorted(glob.glob(os.path.join(log_dir, "*/checkpoints/*.ckpt")))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {log_dir}")
    return max(paths, key=os.path.getmtime)


def pbc_crop(vol, size, starts=None):
    D = vol.shape[0]
    if starts is None:
        starts = tuple(random.randint(0, D - 1) for _ in range(3))
    idx = [np.arange(s, s + size) % D for s in starts]
    return vol[np.ix_(*idx)], starts


def build_models(ckpt, num_gpus, cfg):
    base = FlowMatchingModel.load_from_checkpoint(ckpt, cfg=cfg, strict=False)
    models = {}
    for i in range(num_gpus):
        m = copy.deepcopy(base).to(f"cuda:{i}").eval()
        m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
        models[i] = m
    del base
    return models


def warmup(models, spatial, param_dim, method):
    for i, m in models.items():
        dev = f"cuda:{i}"
        with torch.no_grad(), torch.amp.autocast("cuda", torch.float16):
            d = torch.randn(1, 1, spatial, spatial, spatial, device=dev)
            p = torch.randn(1, param_dim, device=dev)
            for _ in range(2):
                m.sample(d, p, num_steps=2, method=method)
            del d, p
        torch.cuda.synchronize(dev)


def process_source(src, models, num_steps, param_dim, out_base, method='euler', rtol=1e-4, atol=1e-4):
    name = src["name"]
    print(f"\n{'='*50}\n{name}\n{'='*50}")

    cdm_all = np.load(src["cdm_path"], mmap_mode="r")
    gas_all = np.load(src["gas_path"], mmap_mode="r") if src.get("gas_path") else None
    params_all = np.loadtxt(src["param_path"])

    n = src.get("n_samples") or cdm_all.shape[0]
    n = min(n, cdm_all.shape[0])
    if len(params_all) > n:
        params_all = params_all[:n]
    if params_all.ndim == 2 and params_all.shape[1] > param_dim:
        params_all = params_all[:, :param_dim]

    n_stoch = src.get("n_stochastic", 1)
    res = src.get("resolution", cdm_all.shape[1])
    crop = src.get("crop_size")
    spatial = crop if crop else res

    out_dir = Path(out_base) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    gas_dir = None
    if gas_all is not None:
        gas_dir = Path(out_base) / f"{name}_true_gas"
        gas_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = len(models)
    warmup(models, spatial, param_dim, method)

    # build job list
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

    def _run(gpu, dm, p, path):
        dev = f"cuda:{gpu}"
        d = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0).to(dev)
        pt = torch.from_numpy(p).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", torch.float16):
            out = models[gpu].sample(d, pt, num_steps=num_steps, method=method, rtol=rtol, atol=atol)
        np.save(path, out.squeeze().float().cpu().numpy())
        del d, pt, out
        torch.cuda.synchronize(dev)

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        for start in range(0, len(jobs), num_gpus):
            batch = jobs[start:start + num_gpus]
            futs = []
            for i, (dm, p, sp) in enumerate(batch):
                futs.append(pool.submit(_run, i % num_gpus, dm, p, sp))
            for f in as_completed(futs):
                f.result()
                pbar.update(1)
    pbar.close()
    print(f"  -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inf = cfg["inference"]
    ckpt = args.checkpoint or inf.get("checkpoint") or find_best_ckpt()
    num_gpus = min(inf.get("num_gpus", 4), torch.cuda.device_count())
    print(f"Checkpoint: {ckpt}  |  GPUs: {num_gpus}")

    models = build_models(ckpt, num_gpus, cfg)

    method = inf.get("method", "euler")
    rtol = inf.get("rtol", 1e-4)
    atol = inf.get("atol", 1e-4)
    for src in inf["sources"]:
        process_source(src, models, inf["num_steps"], cfg["model"]["param_dim"], inf["output_dir"],
                       method=method, rtol=rtol, atol=atol)

    for i in range(num_gpus):
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
