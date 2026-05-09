"""CLI entry point: train the BrainMRDiff diffusion model."""

import argparse
import logging
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from ..data.dataset import get_dataloaders
from ..models.diffusion import GaussianDiffusion
from ..models.tsa import TSAModule
from ..models.unet import ConditionalUNet
from ..training.trainer import DiffusionTrainer
from ..utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BrainMRDiff")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--processed_dir", default=None, help="Override processed dir")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint dir")
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg) -> GaussianDiffusion:
    tsa = TSAModule(
        in_channels=5,
        mid_channels=cfg.unet_base_channels,
        out_channels=cfg.tsa_out_channels,
    )
    unet = ConditionalUNet(
        image_size=cfg.image_size,
        in_channels=1,
        cond_channels=cfg.tsa_out_channels,
        base_channels=cfg.unet_base_channels,
        channel_mults=list(cfg.unet_channel_mults),
        attn_resolutions=list(cfg.unet_attn_resolutions),
        dropout=cfg.unet_dropout,
        num_modalities=len(cfg.modalities),
    )
    model = GaussianDiffusion(
        unet=unet,
        tsa=tsa,
        num_steps=cfg.num_diffusion_steps,
        beta_schedule=cfg.beta_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        lambda_tgap=cfg.lambda_tgap,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger(__name__).info(f"Model parameters: {n_params:,}")
    return model


def _resolve(base, p: str) -> str:
    from pathlib import Path
    path = Path(p)
    if not path.is_absolute():
        path = (Path(base) / path).resolve()
    return str(path)


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    from pathlib import Path
    cfg = OmegaConf.load(args.config)
    cfg_dir = Path(args.config).resolve().parent

    # Apply CLI overrides; resolve relative paths from config directory
    if args.device:
        cfg.device = args.device
    cfg.processed_dir = args.processed_dir or _resolve(cfg_dir, cfg.processed_dir)
    cfg.checkpoint_dir = args.checkpoint_dir or _resolve(cfg_dir, cfg.checkpoint_dir)
    cfg.output_dir = args.output_dir or _resolve(cfg_dir, cfg.output_dir)

    set_seed(cfg.get("seed", 42))
    logger.info(OmegaConf.to_yaml(cfg))

    # Data
    train_loader, val_loader = get_dataloaders(
        processed_dir=cfg.processed_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.get("num_workers", 4),
        modalities=list(cfg.modalities),
        image_size=cfg.image_size,
        seed=cfg.get("seed", 42),
        cache_in_memory=cfg.get("cache_in_memory", False),
    )

    # Model
    model = build_model(cfg)

    # Trainer
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=cfg.output_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        resume=args.resume or cfg.get("resume"),
    )

    trainer.train()


if __name__ == "__main__":
    main()
