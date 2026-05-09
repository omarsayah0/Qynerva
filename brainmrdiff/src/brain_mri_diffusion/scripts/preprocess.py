"""CLI entry point: preprocess BraTS dataset."""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from ..data.preprocessing import BraTSPreprocessor
from ..utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BraTS dataset for BrainMRDiff")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--processed_dir", default=None, help="Override processed directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed data")
    parser.add_argument("--use_synthseg", action="store_true", help="Run SynthSeg (requires FreeSurfer)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _resolve(base: Path, p: str) -> str:
    """Resolve p relative to base if it's a relative path."""
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    cfg = OmegaConf.load(args.config)
    cfg_dir = Path(args.config).resolve().parent

    # Resolve relative paths from the config file's directory
    data_dir = args.data_dir or _resolve(cfg_dir, cfg.data_dir)
    processed_dir = args.processed_dir or _resolve(cfg_dir, cfg.processed_dir)

    logger.info(f"Data directory  : {data_dir}")
    logger.info(f"Processed output: {processed_dir}")
    logger.info(f"Image size      : {cfg.image_size}")
    logger.info(f"Modalities      : {cfg.modalities}")
    logger.info(f"SynthSeg        : {args.use_synthseg}")

    preprocessor = BraTSPreprocessor(
        data_dir=data_dir,
        processed_dir=processed_dir,
        image_size=cfg.image_size,
        modalities=list(cfg.modalities),
        use_synthseg=args.use_synthseg,
    )
    preprocessor.run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
