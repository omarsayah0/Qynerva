"""CLI entry point: evaluate a trained BrainMRDiff model."""

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..data.dataset import get_dataloaders
from ..evaluation.metrics import aggregate_metrics, evaluate_batch
from ..models.diffusion import GaussianDiffusion
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BrainMRDiff model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint (default: best.pt)")
    parser.add_argument("--num_batches", type=int, default=None, help="Limit eval batches")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


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
    cfg.output_dir = args.output_dir or _resolve(cfg_dir, cfg.output_dir)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    from ..scripts.train import build_model
    model = build_model(cfg).to(device)

    ckpt_manager = CheckpointManager(cfg.checkpoint_dir)
    state = ckpt_manager.load(args.checkpoint) if args.checkpoint else ckpt_manager.load_best()

    if state is None:
        logger.error("No checkpoint found.")
        return

    model.load_state_dict(state["model"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {state.get('epoch', '?')}")

    _, val_loader = get_dataloaders(
        processed_dir=cfg.processed_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.get("num_workers", 4),
        modalities=list(cfg.modalities),
        image_size=cfg.image_size,
    )

    all_metrics = []
    n_batches = args.num_batches or len(val_loader)

    logger.info(f"Evaluating {n_batches} batches...")
    pbar = tqdm(val_loader, total=n_batches, desc="Evaluating")

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if i >= n_batches:
                break

            images = batch["image"].to(device)
            conds = batch["cond"].to(device)
            modalities = batch["modality"].to(device)

            generated = model.ddim_sample(
                cond=conds,
                modality=modalities,
                num_steps=args.ddim_steps,
            )

            gen_norm = (generated + 1) / 2
            img_norm = (images + 1) / 2

            metrics = evaluate_batch(
                pred=gen_norm,
                target=img_norm,
                pred_mask=(gen_norm > 0.5).float(),
                target_mask=conds[:, 0:1],
            )
            all_metrics.append(metrics)
            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})

    final = aggregate_metrics(all_metrics)
    logger.info("=" * 40)
    logger.info("Evaluation Results:")
    for k, v in final.items():
        logger.info(f"  {k.upper():8s}: {v:.4f}")
    logger.info("=" * 40)

    output_dir = Path(args.output_dir or cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
