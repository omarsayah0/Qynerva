"""CLI entry point: generate MRI samples from a trained BrainMRDiff model."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..data.dataset import BraTSDataset
from ..models.diffusion import GaussianDiffusion
from ..models.tsa import TSAModule
from ..models.unet import ConditionalUNet
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MRI samples with BrainMRDiff")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: best.pt)")
    parser.add_argument("--output_dir", default=None, help="Where to save generated images")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Steps for DDIM sampling")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_npy", action="store_true", help="Also save raw .npy files")
    return parser.parse_args()


def build_model(cfg) -> GaussianDiffusion:
    from ..scripts.train import build_model as _build
    return _build(cfg)


def save_images(samples: torch.Tensor, output_dir: Path, prefix: str = "sample") -> None:
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_np = samples.float().cpu().numpy()

    for i, img in enumerate(samples_np):
        img = img.squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output_dir / f"{prefix}_{i:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def _resolve(base: Path, p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = OmegaConf.load(args.config)
    cfg_dir = Path(args.config).resolve().parent

    if args.device:
        cfg.device = args.device

    cfg.processed_dir = _resolve(cfg_dir, cfg.processed_dir)
    cfg.checkpoint_dir = _resolve(cfg_dir, cfg.checkpoint_dir)
    cfg.output_dir = _resolve(cfg_dir, cfg.output_dir)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or cfg.output_dir) / "generated"

    # Load model
    model = build_model(cfg).to(device)
    ckpt_manager = CheckpointManager(cfg.checkpoint_dir)
    state = ckpt_manager.load(args.checkpoint) if args.checkpoint else ckpt_manager.load_best()

    if state is None:
        logger.error("No checkpoint found. Train the model first.")
        return

    model.load_state_dict(state["model"])
    model.eval()
    logger.info(f"Model loaded from checkpoint (epoch={state.get('epoch', '?')})")

    # Load conditioning data from val set
    dataset = BraTSDataset(
        processed_dir=cfg.processed_dir,
        modalities=list(cfg.modalities),
        image_size=cfg.image_size,
        augment=False,
    )

    indices = torch.randperm(len(dataset))[: args.num_samples].tolist()

    batch_images, batch_conds, batch_modalities = [], [], []
    for idx in indices:
        sample = dataset[idx]
        batch_images.append(sample["image"])
        batch_conds.append(sample["cond"])
        batch_modalities.append(sample["modality"])

    images = torch.stack(batch_images).to(device)
    conds = torch.stack(batch_conds).to(device)
    modalities = torch.stack(batch_modalities).to(device)

    logger.info(f"Generating {args.num_samples} samples using {args.sampler.upper()}...")

    with torch.no_grad():
        if args.sampler == "ddim":
            generated = model.ddim_sample(
                cond=conds,
                modality=modalities,
                num_steps=args.ddim_steps,
            )
        else:
            shape = (args.num_samples, 1, cfg.image_size, cfg.image_size)
            generated = model.sample(cond=conds, modality=modalities, shape=shape)

    save_images(generated, output_dir, prefix="generated")
    save_images(images, output_dir, prefix="real")

    if args.save_npy:
        np.save(output_dir / "generated.npy", generated.cpu().numpy())
        np.save(output_dir / "real.npy", images.cpu().numpy())
        np.save(output_dir / "cond.npy", conds.cpu().numpy())

    logger.info(f"Saved {args.num_samples} samples to {output_dir}")


if __name__ == "__main__":
    main()
