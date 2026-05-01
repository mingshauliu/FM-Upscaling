"""
Train a flow matching model.

    python train.py                      # uses config.yaml
    python train.py --config my.yaml     # custom config
"""

import argparse, random, os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import UNet
from model_deep import DeepUNet
from model_classic import ClassicUNet

try:
    import Pk_library as PKL
    HAS_PKL = True
except ImportError:
    HAS_PKL = False


# ── dataset ──────────────────────────────────────────────────────────────────

class AstroDataset(Dataset):
    """Mmap-backed CDM/gas dataset with optional PBC cropping."""

    def __init__(self, cdm, gas, params, indices=None, crop_size=None):
        self.cdm, self.gas = cdm, gas
        self.params = torch.FloatTensor(params)
        self.indices = indices
        self.crop_size = crop_size

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.cdm)

    def __getitem__(self, idx):
        i = self.indices[idx] if self.indices is not None else idx
        cdm = torch.from_numpy(self.cdm[i].copy())
        gas = torch.from_numpy(self.gas[i].copy())
        D = cdm.shape[0]
        if self.crop_size and D > self.crop_size:
            for ax in range(3):
                s = random.randint(0, D - 1)
                ix = torch.arange(s, s + self.crop_size) % D
                cdm = cdm.index_select(ax, ix)
                gas = gas.index_select(ax, ix)
        else:
            shifts = (random.randint(0, D - 1),
                      random.randint(0, D - 1),
                      random.randint(0, D - 1))
            cdm = torch.roll(cdm, shifts, dims=(0, 1, 2))
            gas = torch.roll(gas, shifts, dims=(0, 1, 2))
        return cdm.unsqueeze(0), gas.unsqueeze(0), self.params[idx]


# ── augmentation ─────────────────────────────────────────────────────────────

class RandomRotateFlip3D:
    PAIRS = [(2, 3), (3, 4), (2, 4)]

    def __call__(self, *tensors):
        k = torch.randint(0, 4, (1,)).item()
        axes = self.PAIRS[torch.randint(0, 3, (1,)).item()]
        tensors = tuple(torch.rot90(t, k, axes) for t in tensors)
        for d in (2, 3, 4):
            if torch.rand(1).item() < 0.5:
                tensors = tuple(torch.flip(t, [d]) for t in tensors)
        return tensors


# ── xcorr metric ─────────────────────────────────────────────────────────────

def xcorr_metric(d1, d2, box_size):
    d1 = (d1 - d1.mean()) / d1.std()
    d2 = (d2 - d2.mean()) / d2.std()
    Pk = PKL.XPk([d1, d2], box_size, 0, MAS=["CIC", "CIC"], threads=1)
    k = Pk.k1D
    xpk = Pk.PkX1D[:, 0] / np.sqrt(Pk.Pk1D[:, 0] * Pk.Pk1D[:, 1])
    m = k <= 15
    return np.trapz(xpk[m], k[m]) / (k[m].max() - k[m].min())


def pk_ratio_metric(synth, true, box_size, k_max=15.0):
    """Mean of P_synth(k)/P_true(k) over k <= k_max on raw (unnormalized) fields."""
    Pk_s = PKL.Pk(synth.astype(np.float32), box_size, MAS="CIC", threads=1)
    Pk_t = PKL.Pk(true.astype(np.float32), box_size, MAS="CIC", threads=1)
    k = Pk_s.k3D
    m = k <= k_max
    ratio = Pk_s.Pk[:, 0] / (Pk_t.Pk[:, 0] + 1e-30)
    return float(np.nanmean(ratio[m]))


# ── lightning module ─────────────────────────────────────────────────────────

class FlowMatchingModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        m = cfg["model"]
        t = cfg["training"]
        self.lr = t["lr"]
        self.wd = t["weight_decay"]
        self.noise_std = t["noise_std"]
        self.scheduler = t.get("scheduler", "plateau")
        self.warmup_epochs = t.get("warmup_epochs", 0)
        self.max_epochs = t["max_epochs"]
        self.xcorr_every = t["xcorr_every_n_epochs"]
        self.xcorr_steps = t["xcorr_num_steps"]
        self.xcorr_method = t.get("xcorr_method", "euler")
        self.xcorr_rtol = t.get("xcorr_rtol", 1e-4)
        self.xcorr_atol = t.get("xcorr_atol", 1e-4)
        d = cfg["data"]
        crop = d.get("crop_size")
        res = d["resolution"]
        self.box_size = d["box_size"] * (crop / res) if crop and crop < res else d["box_size"]

        arch = m.get("architecture", "unet")
        common = dict(
            in_channels=m["in_channels"],
            base_channels=m["base_channels"],
            out_channels=m["out_channels"],
            param_dim=m["param_dim"],
            circular_padding=m["circular_padding"],
        )
        if arch == "deep_unet":
            self.net = DeepUNet(**common)
        elif arch == "classic":
            self.net = ClassicUNet(**common, norm_type=m.get("norm_type", "group"))
        else:
            self.net = UNet(**common, num_blocks=m.get("num_blocks", 2))
        if t.get("gradient_checkpointing", False):
            self.net.enable_gradient_checkpointing()
        self.aug = RandomRotateFlip3D()

        # optional EMA (exponential moving average of weights)
        ema_cfg = t.get("ema") or {}
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema_decay = float(ema_cfg.get("decay", 0.9999))
        self.ema_warmup_steps = int(ema_cfg.get("warmup_steps", 0))
        self._ema_shadow = None   # dict[name -> tensor] of EMA weights
        self._ema_backup = None   # dict[name -> tensor] of raw weights during swap

        # optional NF evaluation during validation (L25 only)
        self.nf_eval_ckpt = t.get("nf_eval_checkpoint")
        self.nf_model = None
        self._nf_preds = []  # collects (y_true, y_mean, y_std) across val batches

    def forward(self, x, t, p):
        return self.net(x, t, p)

    def _step(self, batch, augment=False):
        cdm, gas, params = batch
        if augment:
            cdm, gas = self.aug(cdm, gas)
        B = cdm.size(0)
        t = torch.rand(B, device=cdm.device)
        x0 = (cdm + torch.randn_like(cdm) * self.noise_std) if self.noise_std > 0 else cdm
        x1 = gas
        t_exp = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_exp) * x0 + t_exp * x1
        pred = self(torch.cat([x_t, cdm], 1), t, params)
        return F.mse_loss(pred, x1 - x0)

    def training_step(self, batch, _):
        loss = self._step(batch, augment=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_enabled:
            self._ema_update()

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_epoch=True, prog_bar=True)
        is_xcorr_epoch = (not self.trainer.sanity_checking
                          and self.xcorr_every > 0
                          and self.current_epoch % self.xcorr_every == 0)
        if HAS_PKL and is_xcorr_epoch and batch_idx == 0:
            self._log_xcorr(batch)
        if self.nf_eval_ckpt and is_xcorr_epoch:
            self._nf_eval_step(batch)
        return loss

    def on_validation_epoch_start(self):
        if self.ema_enabled and self._ema_shadow is not None:
            self._ema_swap_in()

    def on_validation_epoch_end(self):
        if self.ema_enabled and self._ema_backup is not None:
            self._ema_swap_out()
        if self._nf_preds:
            self._log_nf_plot()
            self._nf_preds.clear()

    # ── EMA helpers ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ema_update(self):
        """Update EMA shadow weights after each optimizer step."""
        if self.global_step < self.ema_warmup_steps:
            return
        if self._ema_shadow is None:
            self._ema_shadow = {n: p.detach().clone().float()
                                for n, p in self.net.named_parameters() if p.requires_grad}
            return
        for n, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            if self._ema_shadow[n].device != p.device:
                self._ema_shadow[n] = self._ema_shadow[n].to(p.device)
            self._ema_shadow[n].mul_(self.ema_decay).add_(p.detach().float(),
                                                          alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def _ema_swap_in(self):
        """Temporarily replace net weights with EMA weights for validation."""
        self._ema_backup = {}
        for n, p in self.net.named_parameters():
            if n in self._ema_shadow:
                if self._ema_shadow[n].device != p.device:
                    self._ema_shadow[n] = self._ema_shadow[n].to(p.device)
                self._ema_backup[n] = p.detach().clone()
                p.data.copy_(self._ema_shadow[n].to(p.dtype))

    @torch.no_grad()
    def _ema_swap_out(self):
        """Restore raw training weights after validation."""
        for n, p in self.net.named_parameters():
            if n in self._ema_backup:
                p.data.copy_(self._ema_backup[n])
        self._ema_backup = None

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"]
        # nf_model is a lazy-loaded frozen evaluator — exclude from checkpoint
        # so we don't bloat ckpts and so resume doesn't hit strict-load errors.
        for k in [k for k in sd if k.startswith("nf_model.")]:
            del sd[k]
        if self.ema_enabled and self._ema_shadow is not None:
            # Preserve live (student) weights for training resume.
            checkpoint["live_state_dict"] = {
                k: v.detach().cpu().clone()
                for k, v in sd.items() if k.startswith("net.")
            }
            # Overwrite net.* in state_dict with EMA weights so `load_from_checkpoint`
            # at inference time picks up the EMA (teacher) model directly.
            for n, ema_t in self._ema_shadow.items():
                key = f"net.{n}"
                if key in sd:
                    sd[key] = ema_t.to(sd[key].dtype).detach().cpu().clone()
            checkpoint["ema_shadow"] = {n: t.detach().cpu() for n, t in self._ema_shadow.items()}

    def on_load_checkpoint(self, checkpoint):
        # Old checkpoints may contain lazy-loaded nf_model.* keys — strip them
        # so strict state_dict loading succeeds on resume.
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if k.startswith("nf_model.")]:
            del sd[k]
        if self.ema_enabled and "ema_shadow" in checkpoint:
            self._ema_shadow = {n: t for n, t in checkpoint["ema_shadow"].items()}
        # On training resume, swap live weights back into state_dict so the student
        # continues from its own trajectory (EMA stays in self._ema_shadow).
        try:
            is_resume = self.trainer is not None and \
                        getattr(self.trainer, "state", None) is not None and \
                        getattr(self.trainer.state, "fn", None) == "fit"
        except RuntimeError:
            is_resume = False
        if is_resume and "live_state_dict" in checkpoint:
            for k, v in checkpoint["live_state_dict"].items():
                checkpoint["state_dict"][k] = v

    def _ode_func(self, cdm, params, buf):
        """Return an ODE function f(t, x) for use with torchdiffeq."""
        B = cdm.size(0)
        def f(t, x):
            buf[:, 0:1] = x
            return self(buf, t.expand(B), params)
        return f

    def _offload_ode_func(self, cdm, params, buf):
        """ODE function that offloads skip connections to CPU to save GPU memory."""
        B = cdm.size(0)
        def f(t, x):
            buf[:, 0:1] = x
            return self.net.forward_offload(buf, t.expand(B), params)
        return f

    def _tiled_ode_func(self, cdm, params, buf, tile_size):
        """ODE function that evaluates the model on non-overlapping tiles.

        Disables circular padding so tiles use zero-padded boundaries.
        """
        B = cdm.size(0)
        spatial = list(cdm.shape[2:])
        self.net.set_pad_mode("constant")

        def f(t, x):
            buf[:, 0:1] = x
            out = torch.zeros_like(x)
            t_exp = t.expand(B)

            for iz in range(0, spatial[0], tile_size):
                for iy in range(0, spatial[1], tile_size):
                    for ix in range(0, spatial[2], tile_size):
                        sz = min(tile_size, spatial[0] - iz)
                        sy = min(tile_size, spatial[1] - iy)
                        sx = min(tile_size, spatial[2] - ix)
                        patch = buf[:, :, iz:iz+sz, iy:iy+sy, ix:ix+sx]
                        p_out = self(patch, t_exp, params)
                        out[:, :, iz:iz+sz, iy:iy+sy, ix:ix+sx] = p_out
                        del patch, p_out
                        torch.cuda.empty_cache()
            return out

        return f

    @torch.no_grad()
    def _log_xcorr(self, batch):
        cdm, gas, params = batch
        # Run ODE integration in float32 to avoid fp16 overflow/NaN
        cdm32, params32 = cdm.float(), params.float()
        B, dev = cdm32.size(0), cdm32.device
        # x starts noisy (matches training x0), conditioning channel stays clean
        x0 = cdm32 + torch.randn_like(cdm32) * self.noise_std if self.noise_std > 0 else cdm32.clone()
        buf = torch.empty(B, 2, *cdm32.shape[2:], device=dev, dtype=torch.float32)
        buf[:, 1:2] = cdm32
        t_span = torch.linspace(0.0, 1.0, self.xcorr_steps + 1, device=dev)
        with torch.amp.autocast("cuda", enabled=False):
            x = odeint(self._ode_func(cdm32, params32, buf), x0, t_span,
                        method=self.xcorr_method,
                        **self._odeint_kwargs(self.xcorr_method,
                                              self.xcorr_steps,
                                              self.xcorr_rtol,
                                              self.xcorr_atol))[-1]
        d1 = x[0, 0].cpu().numpy()
        d2 = gas[0, 0].float().cpu().numpy()
        if np.std(d1) < 1e-8 or np.std(d2) < 1e-8 or not np.isfinite(d1).all():
            return  # skip if output is constant or NaN (early training)
        val = xcorr_metric(d1, d2, self.box_size)
        self.log("xcorr", val, prog_bar=True, on_step=False, on_epoch=True)
        try:
            pkr = pk_ratio_metric(d1, d2, self.box_size)
            self.log("pk_ratio", pkr, prog_bar=True, on_step=False, on_epoch=True)
        except Exception as e:
            print(f"  pk_ratio failed: {e}")

    def _load_nf_model(self):
        """Lazy-load the NF model on first use."""
        if self.nf_model is not None:
            return
        from nf.nf_module import LitNFRegressor
        self.nf_model = LitNFRegressor.load_from_checkpoint(
            self.nf_eval_ckpt, map_location=self.device)
        self.nf_model.eval().to(self.device)
        for p in self.nf_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _nf_eval_step(self, batch):
        """Generate synthetic gas via FM ODE, then run NF to predict cosmo params."""
        self._load_nf_model()
        cdm, gas, params = batch
        cdm32, params32 = cdm.float(), params.float()
        B, dev = cdm32.size(0), cdm32.device
        x0 = cdm32 + torch.randn_like(cdm32) * self.noise_std if self.noise_std > 0 else cdm32.clone()
        buf = torch.empty(B, 2, *cdm32.shape[2:], device=dev, dtype=torch.float32)
        buf[:, 1:2] = cdm32
        t_span = torch.linspace(0.0, 1.0, self.xcorr_steps + 1, device=dev)
        with torch.amp.autocast("cuda", enabled=False):
            synth_gas = odeint(
                self._ode_func(cdm32, params32, buf), x0, t_span,
                method=self.xcorr_method,
                **self._odeint_kwargs(self.xcorr_method, self.xcorr_steps,
                                      self.xcorr_rtol, self.xcorr_atol))[-1]
        # NF expects (B, 1, D, H, W) gas input + astro params
        num_cosmo = self.nf_model.hparams.get("num_cosmo", 2)
        y_true = params[:, :num_cosmo].cpu().numpy()
        astro = params[:, num_cosmo:].to(dev)
        summary, _ = self.nf_model(synth_gas, astro)
        samples = self.nf_model.flow.sample(summary, num_samples=500)
        samples = samples.permute(1, 0, 2)  # (B, n_samples, num_cosmo)
        y_mean = samples.mean(1).cpu().numpy()
        y_std = samples.std(1).cpu().numpy()
        self._nf_preds.append((y_true, y_mean, y_std))

    def _log_nf_plot(self):
        """Create truth vs pred scatter plot and log to wandb."""
        y_true = np.concatenate([p[0] for p in self._nf_preds])
        y_mean = np.concatenate([p[1] for p in self._nf_preds])
        y_std = np.concatenate([p[2] for p in self._nf_preds])
        num_cosmo = y_true.shape[1]
        param_names = [r"$\Omega_m$", r"$\sigma_8$"][:num_cosmo]
        fig, axes = plt.subplots(1, num_cosmo, figsize=(6 * num_cosmo, 5))
        if num_cosmo == 1:
            axes = [axes]
        for j, (ax, label) in enumerate(zip(axes, param_names)):
            x, yp, err = y_true[:, j], y_mean[:, j], y_std[:, j]
            xmin, xmax = x.min(), x.max()
            margin = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
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
        fig.suptitle(f"NF on FM synth gas (epoch {self.current_epoch})", fontsize=13)
        plt.tight_layout()
        if hasattr(self.logger, "experiment"):
            import wandb
            self.logger.experiment.log(
                {"nf_truth_vs_pred": wandb.Image(fig)},
                step=self.global_step)
        plt.close(fig)

    @staticmethod
    def _odeint_kwargs(method, num_steps, rtol, atol):
        ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}
        if method in ADAPTIVE:
            return {'rtol': rtol, 'atol': atol}
        return {'options': {'step_size': 1.0 / num_steps}}

    def sample(self, cdm, params, num_steps=100, method='euler', rtol=1e-4, atol=1e-4,
               tile_size=None, offload_skips=False):
        self.eval()
        B, dev = cdm.size(0), cdm.device
        x0 = cdm + torch.randn_like(cdm) * self.noise_std if self.noise_std > 0 else cdm.clone()
        buf = torch.empty(B, 2, *cdm.shape[2:], device=dev, dtype=cdm.dtype)
        buf[:, 1:2] = cdm
        ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}
        if method in ADAPTIVE:
            t_span = torch.tensor([0.0, 1.0], device=dev)
        else:
            t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=dev)
        use_tiling = tile_size is not None and any(s > tile_size for s in cdm.shape[2:])
        if offload_skips:
            ode_fn = self._offload_ode_func(cdm, params, buf)
        elif use_tiling:
            orig_pad_mode = self.net.pad_mode
            ode_fn = self._tiled_ode_func(cdm, params, buf, tile_size)
        else:
            ode_fn = self._ode_func(cdm, params, buf)
        with torch.no_grad():
            trajectory = odeint(ode_fn, x0, t_span,
                                method=method,
                                **self._odeint_kwargs(method, num_steps, rtol, atol))
        if use_tiling and not offload_skips:
            self.net.set_pad_mode(orig_pad_mode)
        return trajectory[-1]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler == "cosine":
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6)
            if self.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.01, total_iters=self.warmup_epochs)
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt, [warmup, cosine], milestones=[self.warmup_epochs])
            else:
                sched = cosine
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, patience=10)
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    def on_train_epoch_start(self):
        # Manual linear warmup for plateau scheduler (cosine handles its own via SequentialLR).
        if self.scheduler == "plateau" and self.warmup_epochs > 0 \
                and self.current_epoch < self.warmup_epochs:
            scale = (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr * scale


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    d, t = cfg["data"], cfg["training"]

    seed = t.get("seed")
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # load mmap data
    cdm = np.load(d["cdm_path"], mmap_mode="r")
    gas = np.load(d["gas_path"], mmap_mode="r")
    params = np.loadtxt(d["param_path"])
    print(f"CDM {cdm.shape}  Gas {gas.shape}  Params {params.shape}")

    crop = d.get("crop_size")
    if crop and crop >= d["resolution"]:
        crop = None

    # train/val split
    n = len(cdm)
    n_val = int(n * d["val_split"])
    idx = np.random.permutation(n)
    tr_idx, va_idx = idx[:n - n_val], idx[n - n_val:]

    tr_ds = AstroDataset(cdm, gas, params[tr_idx], tr_idx, crop)
    va_ds = AstroDataset(cdm, gas, params[va_idx], va_idx, crop)
    pw = t["num_workers"] > 0
    kw = dict(pin_memory=True, persistent_workers=pw, prefetch_factor=1 if pw else None, drop_last=True)
    tr_dl = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], **kw)
    kw["drop_last"] = False
    va_dl = DataLoader(va_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], **kw)

    model = FlowMatchingModel(cfg)

    trainer = pl.Trainer(
        logger=WandbLogger(log_model="False"),
        max_epochs=t["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=t["devices"],
        strategy=t["strategy"],
        precision=t["precision"],
        gradient_clip_val=t["gradient_clip"],
        accumulate_grad_batches=t["accumulate_grad"],
        log_every_n_steps=t["log_every_n_steps"],
        check_val_every_n_epoch=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", filename="best-{epoch:03d}-{val_loss:.6f}",
                                   save_top_k=1, mode="min", save_last=True)],
        num_sanity_val_steps=2,
    )
    trainer.fit(model, tr_dl, va_dl, ckpt_path=t.get("resume_from"))
    print(f"Best: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
