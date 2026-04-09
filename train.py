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

    def on_validation_epoch_end(self):
        if self._nf_preds:
            self._log_nf_plot()
            self._nf_preds.clear()

    def _ode_func(self, cdm, params, buf):
        """Return an ODE function f(t, x) for use with torchdiffeq."""
        B = cdm.size(0)
        def f(t, x):
            buf[:, 0:1] = x
            return self(buf, t.expand(B), params)
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

    def sample(self, cdm, params, num_steps=100, method='euler', rtol=1e-4, atol=1e-4):
        self.eval()
        B, dev = cdm.size(0), cdm.device
        x0 = cdm + torch.randn_like(cdm) * self.noise_std if self.noise_std > 0 else cdm.clone()
        buf = torch.empty(B, 2, *cdm.shape[2:], device=dev, dtype=cdm.dtype)
        buf[:, 1:2] = cdm
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=dev)
        with torch.no_grad():
            trajectory = odeint(self._ode_func(cdm, params, buf), x0, t_span,
                                method=method,
                                **self._odeint_kwargs(method, num_steps, rtol, atol))
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
