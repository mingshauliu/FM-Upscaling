"""
Training entry point for normalizing flow parameter inference.

Reads NF settings from config.yaml and trains the FiLM-conditioned
encoder + conditional normalizing flow model.

Usage:
    python nf_train.py                       # uses config.yaml
    python nf_train.py --config my.yaml
"""

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
import yaml

from nf_module import LitNFRegressor, NFDataModule


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    nf_cfg = cfg["nf"]
    nf_data = nf_cfg["data"]
    nf_model = nf_cfg["model"]
    nf_train = nf_cfg["training"]

    pl.seed_everything(nf_train.get("seed", 42), workers=True)

    # Data module
    dm = NFDataModule(
        gas_path=nf_data["gas_path"],
        param_path=nf_data["param_path"],
        num_cosmo=nf_model["num_cosmo"],
        crop_size=nf_data.get("crop_size"),
        val_split=nf_data.get("val_split", 0.2),
        batch_size=nf_train["batch_size"],
        num_workers=nf_train.get("num_workers", 4),
        seed=nf_train.get("seed", 42),
    )
    dm.setup()

    # Model
    model = LitNFRegressor(
        lr_encoder=nf_train["lr_encoder"],
        lr_flow=nf_train["lr_flow"],
        weight_decay=nf_train.get("weight_decay", 1e-4),
        base_channels=nf_model["base_channels"],
        cond_embed_dim=nf_model.get("cond_embed_dim", 128),
        num_astro=dm.num_astro,
        flow_hidden=nf_model.get("flow_hidden", 128),
        flow_transforms=nf_model.get("flow_transforms", 8),
        flow_type=nf_model.get("flow_type", "nsf"),
        num_cosmo=nf_model["num_cosmo"],
        dropout=nf_model.get("dropout", 0.15),
        warmup_epochs=nf_train.get("warmup_epochs", 15),
        max_epochs=nf_train["max_epochs"],
        aux_loss_weight=nf_train.get("aux_loss_weight", 0.5),
        aux_loss_decay=nf_train.get("aux_loss_decay", 0.98),
        gradient_clip_val=nf_train.get("gradient_clip_val", 1.0),
        target_mean=dm.target_mean,
        target_std=dm.target_std,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    ckpt_dir = nf_train.get("checkpoint_dir", "nf_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            filename="nf-{epoch:03d}-{val_loss:.4f}",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=nf_train["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=nf_train.get("devices", 4),
        strategy=nf_train.get("strategy", "ddp"),
        precision=nf_train.get("precision", "16-mixed"),
        gradient_clip_val=nf_train.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=nf_train.get("accumulate_grad_batches", 8),
        log_every_n_steps=nf_train.get("log_every_n_steps", 10),
        logger=WandbLogger(
            project=nf_train.get("wandb_project", "cosmo-nf"),
            log_model=False,
            save_dir=ckpt_dir,
        ),
        callbacks=callbacks,
        deterministic=False,
    )

    resume_from = nf_train.get("resume_from")
    if resume_from is None:
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_from = last_ckpt

    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.fit(model, dm, ckpt_path=resume_from)
    else:
        trainer.fit(model, dm)


if __name__ == "__main__":
    main()
