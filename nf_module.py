"""
Lightning module, dataset, and data module for normalizing flow inference.

Combines the FiLM-conditioned encoder with a conditional normalizing flow
to learn p(cosmo_params | gas_field, astro_params).
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from nf_encoder import Encoder3DFiLM
from nf_flow import ConditionalFlow


# ---------------------------
# Data Augmentation
# ---------------------------
class RandomRotateFlip3D:
    """Random 90-degree rotations and flips for 4D tensors (C,D,H,W)."""

    def __init__(self, dims=(1, 2, 3)):
        self.dims = dims
        self.axis_pairs = [(1, 2), (2, 3), (1, 3)]

    def __call__(self, *tensors):
        k = torch.randint(0, 4, (1,)).item()
        axes = self.axis_pairs[torch.randint(0, len(self.axis_pairs), (1,)).item()]
        tensors = tuple(torch.rot90(t, k, axes) for t in tensors)

        for dim in self.dims:
            if torch.rand(1).item() < 0.5:
                tensors = tuple(torch.flip(t, [dim]) for t in tensors)
        return tensors


class TransformSubset(Dataset):
    """Wrapper to apply augmentation transforms to a Subset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        vol, target, astro = self.subset[idx]
        if self.transform is not None:
            (vol,) = self.transform(vol)
        return vol, target, astro


# ---------------------------
# Dataset
# ---------------------------
class CosmoVolumeDataset(Dataset):
    """
    Memory-mapped dataset for NF inference training.

    Loads cached log1p-normalised gas fields and pre-normalised parameters.
    First `num_cosmo` columns are target cosmo params (Omega_m, sigma_8).
    Remaining columns are astrophysical conditioning params.

    Args:
        gas_path: Path to cached gas field .npy (N, D, D, D)
        param_path: Path to cached normalised params .txt
        num_cosmo: Number of target cosmo parameters (default 2)
        crop_size: Optional PBC crop size (None = use full grid)
    """

    def __init__(self, gas_path, param_path, num_cosmo=2, crop_size=None):
        self.gas = np.load(gas_path, mmap_mode="r")
        params_all = np.loadtxt(param_path).astype(np.float32)
        n_grids = self.gas.shape[0]

        # Truncate params to match grid count
        if len(params_all) > n_grids:
            params_all = params_all[:n_grids]

        self.targets = params_all[:, :num_cosmo]       # cosmo params
        self.num_cosmo = num_cosmo
        self.crop_size = crop_size
        self.resolution = self.gas.shape[1]

        # Astrophysical conditioning params (remaining columns)
        if params_all.shape[1] > num_cosmo:
            self.astro = params_all[:, num_cosmo:]
        else:
            # No astro params (e.g. if param file only has cosmo)
            self.astro = np.zeros((n_grids, 1), dtype=np.float32)

        self.num_astro = self.astro.shape[1]

        print(f"NF Dataset: gas={self.gas.shape}, targets={self.targets.shape}, "
              f"astro={self.astro.shape}, crop={crop_size}")

    def __len__(self):
        return self.gas.shape[0]

    def _pbc_crop(self, vol):
        """Random PBC crop from (D,D,D) to (crop_size,crop_size,crop_size)."""
        D = vol.shape[0]
        cs = self.crop_size
        starts = tuple(np.random.randint(0, D) for _ in range(3))
        idx = [np.arange(s, s + cs) % D for s in starts]
        return vol[np.ix_(*idx)]

    def __getitem__(self, idx):
        vol = np.asarray(self.gas[idx], dtype=np.float32)

        if self.crop_size and self.crop_size < self.resolution:
            vol = self._pbc_crop(vol)

        vol = np.expand_dims(vol, axis=0)  # (1, D, D, D)
        target = self.targets[idx].copy()
        astro = self.astro[idx].copy()

        return (
            torch.from_numpy(vol),
            torch.from_numpy(target),
            torch.from_numpy(astro),
        )


# ---------------------------
# Data Module
# ---------------------------
class NFDataModule(pl.LightningDataModule):
    """Lightning DataModule for NF inference."""

    def __init__(
        self,
        gas_path,
        param_path,
        num_cosmo=2,
        crop_size=None,
        val_split=0.2,
        batch_size=2,
        num_workers=4,
        seed=42,
    ):
        super().__init__()
        self.gas_path = gas_path
        self.param_path = param_path
        self.num_cosmo = num_cosmo
        self.crop_size = crop_size
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.target_mean = None
        self.target_std = None
        self.num_astro = None

    def setup(self, stage=None):
        full_ds = CosmoVolumeDataset(
            self.gas_path, self.param_path,
            num_cosmo=self.num_cosmo, crop_size=self.crop_size,
        )
        self.num_astro = full_ds.num_astro

        val_len = max(1, int(len(full_ds) * self.val_split))
        train_len = len(full_ds) - val_len

        train_ds, val_ds = random_split(
            full_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Compute target stats from training set
        train_targets = full_ds.targets[list(train_ds.indices)]
        self.target_mean = train_targets.mean(axis=0).astype(np.float32)
        self.target_std = train_targets.std(axis=0).astype(np.float32)
        print(f"Target stats: mean={self.target_mean}, std={self.target_std}")

        self.train_ds = TransformSubset(train_ds, transform=RandomRotateFlip3D())
        self.val_ds = val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


# ---------------------------
# Lightning Module
# ---------------------------
class LitNFRegressor(pl.LightningModule):
    """
    Lightning module combining FiLM encoder and conditional normalizing flow.

    The encoder receives gas fields and astrophysical parameters,
    producing a summary vector that conditions the flow for parameter inference.
    An auxiliary regression head provides additional supervision.
    """

    def __init__(
        self,
        lr_encoder=3e-4,
        lr_flow=1e-4,
        weight_decay=1e-4,
        base_channels=16,
        cond_embed_dim=128,
        num_astro=33,
        flow_hidden=128,
        flow_transforms=8,
        flow_type="nsf",
        num_cosmo=2,
        dropout=0.15,
        warmup_epochs=15,
        max_epochs=1000,
        aux_loss_weight=0.5,
        aux_loss_decay=0.98,
        gradient_clip_val=1.0,
        target_mean=None,
        target_std=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # FiLM-conditioned encoder
        self.encoder = Encoder3DFiLM(
            in_ch=1,
            base=base_channels,
            num_astro=num_astro,
            cond_embed_dim=cond_embed_dim,
            dropout=dropout,
        )
        encoder_out_dim = 8 * base_channels

        # Auxiliary regression head
        self.aux_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_out_dim // 2, num_cosmo),
        )

        # Flow
        target_mean_t = torch.tensor(target_mean) if target_mean is not None else None
        target_std_t = torch.tensor(target_std) if target_std is not None else None

        self.flow = ConditionalFlow(
            num_params=num_cosmo,
            context_dim=encoder_out_dim,
            hidden_dim=flow_hidden,
            num_transforms=flow_transforms,
            flow_type=flow_type,
            target_mean=target_mean_t,
            target_std=target_std_t,
        )

        self.aux_weight = aux_loss_weight

    def forward(self, x, astro):
        summary = self.encoder(x, astro)
        aux_pred = self.aux_head(summary)
        return summary, aux_pred

    def _compute_loss(self, batch, stage="train"):
        x, y, astro = batch

        summary, aux_pred = self(x, astro)

        # NLL loss from flow
        log_prob = self.flow.log_prob(summary, y)

        # Handle NaN/Inf
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            self.log(f"{stage}/nan_count", torch.isnan(log_prob).sum().float())
            log_prob = torch.where(
                torch.isnan(log_prob) | torch.isinf(log_prob),
                torch.tensor(-100.0, device=log_prob.device),
                log_prob
            )

        nll_loss = -log_prob.mean()

        # Auxiliary MSE loss (on normalized targets)
        y_norm = (y - self.flow.target_mean) / self.flow.target_std
        aux_pred_norm = (aux_pred - self.flow.target_mean) / self.flow.target_std
        aux_loss = F.mse_loss(aux_pred_norm, y_norm)

        # Decaying auxiliary weight
        current_epoch = self.current_epoch if self.current_epoch is not None else 0
        effective_aux_weight = self.aux_weight * (self.hparams.aux_loss_decay ** current_epoch)

        total_loss = nll_loss + effective_aux_weight * aux_loss

        self.log(f"{stage}/nll_loss", nll_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/aux_loss", aux_loss, sync_dist=True)
        self.log(f"{stage}/aux_weight", effective_aux_weight, sync_dist=True)
        self.log(f"{stage}/loss", total_loss, prog_bar=True, sync_dist=True)

        if stage == "val":
            with torch.no_grad():
                pred_mean, pred_std = self.flow.get_posterior_stats(summary, num_samples=500)
                mae = (pred_mean - y).abs().mean(dim=0)
                param_names = ["Omega_m", "sigma_8"][:self.hparams.num_cosmo]
                for i, name in enumerate(param_names):
                    self.log(f"{stage}/mae_{name}", mae[i], sync_dist=True)
                    self.log(f"{stage}/std_{name}", pred_std[:, i].mean(), sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        encoder_params = list(self.encoder.parameters())
        flow_params = list(self.flow.parameters())

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.hparams.lr_encoder, 'name': 'encoder'},
            {'params': flow_params, 'lr': self.hparams.lr_flow, 'name': 'flow'},
        ], weight_decay=self.hparams.weight_decay)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01, end_factor=1.0,
            total_iters=self.hparams.warmup_epochs
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
