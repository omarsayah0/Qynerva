"""
CLI entry point for training.

Usage
-----
    qynerva_classification_train [OPTIONS]

Options
-------
    --data-dir       PATH    Path to raw Data/ folder          (default: Data)
    --output-dir     PATH    Where to save models/logs/plots   (default: outputs)
    --batch-size     INT     Batch size                        (default: 32)
    --stage1-epochs  INT     Max epochs for stage 1            (default: 10)
    --stage2-epochs  INT     Max epochs for stage 2            (default: 20)
    --stage1-lr      FLOAT   Learning rate for stage 1         (default: 1e-3)
    --stage2-lr      FLOAT   Learning rate for stage 2         (default: 1e-5)
    --no-pretrained          Skip ImageNet pre-training
    --device         STR     "cpu" | "cuda"                    (default: auto)
    --seed           INT     Random seed                       (default: 42)
    --no-plots               Skip saving training-history plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from SRC.config.config import Config
from SRC.training.trainer import run_training
from SRC.utils.logger import setup_logging
from SRC.utils.visualization import load_and_plot


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qynerva_classification_train",
        description="Train EfficientNetB3 brain-tumor MRI classifier (two-stage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-dir", type=Path, default=Path("Data"),
                        help="Root data directory (contains one folder per class).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Directory for saved models, logs, and plots.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train backbone from scratch (not recommended).")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'. Auto-detects if omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip saving training-history plots.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Build config from defaults, override with CLI arguments
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        pretrained=not args.no_pretrained,
        random_seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.device is not None:
        config.device = args.device

    config.create_output_dirs()

    setup_logging(
        log_level=logging.INFO,
        log_file=config.logs_dir / "train.log",
    )

    logger = logging.getLogger(__name__)
    logger.info("Configuration: %s", config)

    # Run the two-stage training pipeline
    run_training(config)

    # Generate plots (unless suppressed)
    if not args.no_plots and config.history_path.exists():
        try:
            load_and_plot(config.history_path, config.plots_dir)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)

    logger.info("Training complete. Outputs: %s", config.output_dir.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
