"""
CLI entry point for full MRI volume classification.

Pipeline:
  1. Load a .nii / .nii.gz volume and extract all 2-D slices.
  2. Run the existing 2-D classifier on every slice (unchanged).
  3. Aggregate slice predictions via majority voting → patient diagnosis.
  4. Select the top-N highest-confidence slices for the winning class.
  5. Run HiResCAM XAI on those slices (existing XAI pipeline, unchanged).
  6. Save a JSON report, per-slice CSV, individual overlays, and a grid PNG.

Usage
-----
Single volume:
    python -m SRC.main_volume --volume scan.nii.gz

Multiple volumes in a folder:
    python -m SRC.main_volume --folder /path/to/nii_files

Options:
    --volume      PATH   Path to a single .nii / .nii.gz file.
    --folder      PATH   Folder containing NIfTI files (processed in batch).
    --output-dir  PATH   Where to save all outputs.
                         (default: outputs/volume_results)
    --model       PATH   Path to .pth checkpoint.
                         (default: <output-dir>/../../models/best_model.pth
                          i.e. outputs/models/best_model.pth)
    --class-map   PATH   Path to class_to_idx.json.
    --axis        STR    Slicing axis: axial | coronal | sagittal.
                         (default: axial)
    --top-n       INT    Number of top slices for XAI.
                         (default: 5)
    --keep-blank         Include near-blank slices (excluded by default).
    --device      STR    Force device: cpu | cuda.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from SRC.config.config import Config
from SRC.prediction.predictor import Predictor
from SRC.utils.logger import setup_logging

from SRC.volume.loader import MRIVolumeLoader
from SRC.volume.inference import VolumeInference
from SRC.volume.aggregator import aggregate
from SRC.volume.xai_runner import run_xai_on_top_slices
from SRC.volume.reporter import save_report

_NII_EXTENSIONS = {".nii", ".gz"}   # .nii.gz has two suffixes; we check .nii too


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main_volume",
        description="Classify a full MRI volume using the existing 2-D classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--volume", type=Path, metavar="PATH",
                      help="Path to a single .nii or .nii.gz file.")
    mode.add_argument("--folder", type=Path, metavar="PATH",
                      help="Folder containing NIfTI files.")

    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/volume_results"),
                        help="Root directory for all output files.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to .pth model checkpoint.")
    parser.add_argument("--class-map", type=Path, default=None,
                        help="Path to class_to_idx.json.")
    parser.add_argument("--axis", type=str, default="axial",
                        choices=["axial", "coronal", "sagittal"],
                        help="Axis along which to slice the volume.")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top-confidence slices used for XAI.")
    parser.add_argument("--keep-blank", action="store_true",
                        help="Do not discard near-blank slices.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'.")

    return parser.parse_args(argv)


def _collect_nii_files(folder: Path) -> list[Path]:
    """Return all .nii and .nii.gz files under *folder*."""
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and (
            p.suffix.lower() == ".nii"
            or (p.suffix.lower() == ".gz" and p.stem.endswith(".nii"))
        )
    )
    return files


def process_volume(
    nii_path: Path,
    predictor: Predictor,
    output_dir: Path,
    axis: str = "axial",
    top_n: int = 5,
    skip_blank: bool = True,
) -> None:
    """Run the full pipeline for one NIfTI file.

    Args:
        nii_path:   Path to the NIfTI volume.
        predictor:  Initialised :class:`~SRC.prediction.predictor.Predictor`.
        output_dir: Where to write this patient's outputs.
        axis:       Slicing axis.
        top_n:      Number of top-N slices for XAI.
        skip_blank: Whether to skip near-blank slices.
    """
    logger = logging.getLogger(__name__)
    patient_id = nii_path.stem.replace(".nii", "")   # strip both .nii and .gz

    logger.info("=" * 60)
    logger.info("Processing: %s", nii_path.name)

    # ------------------------------------------------------------------ #
    # Step 1 — load volume and extract 2-D slices
    # ------------------------------------------------------------------ #
    loader = MRIVolumeLoader(nii_path, axis=axis, skip_blank=skip_blank)
    pil_slices = loader.get_slices()

    if not pil_slices:
        logger.error("No usable slices found in %s — skipping.", nii_path.name)
        return

    logger.info("Extracted %d slices from %s", len(pil_slices), nii_path.name)

    # ------------------------------------------------------------------ #
    # Steps 2 & 3 — classify every slice with the existing 2-D classifier
    # ------------------------------------------------------------------ #
    volume_inf = VolumeInference(predictor)
    slice_results = volume_inf.classify_slices(pil_slices)

    # ------------------------------------------------------------------ #
    # Step 4 — majority voting + top-N slice selection
    # ------------------------------------------------------------------ #
    report = aggregate(
        patient_id=patient_id,
        slice_results=slice_results,
        top_n=top_n,
    )

    # ------------------------------------------------------------------ #
    # Steps 5 & 6 — run HiResCAM XAI on the selected slices
    # ------------------------------------------------------------------ #
    xai_results = run_xai_on_top_slices(
        report=report,
        model=predictor.model,
        alpha=0.4,
    )

    # ------------------------------------------------------------------ #
    # Step 7 — save JSON report, CSV, overlay PNGs, XAI grid
    # ------------------------------------------------------------------ #
    patient_out_dir = output_dir / patient_id
    save_report(
        report=report,
        xai_results=xai_results,
        output_dir=patient_out_dir,
    )

    logger.info("Outputs written to: %s", patient_out_dir)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Config and predictor setup (same as main_predict.py) ---
    config = Config(output_dir=Path("outputs"))
    if args.device is not None:
        config.device = args.device

    model_path: Path = args.model if args.model else config.best_model_path
    class_map_path: Path = args.class_map if args.class_map else config.class_map_path

    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)
    if not class_map_path.exists():
        logger.error("Class map not found: %s", class_map_path)
        sys.exit(1)

    predictor = Predictor(
        model_path=model_path,
        class_map_path=class_map_path,
        config=config,
    )

    output_dir: Path = args.output_dir
    skip_blank: bool = not args.keep_blank

    # --- Collect target files ---
    if args.volume is not None:
        nii_files = [args.volume]
    else:
        nii_files = _collect_nii_files(args.folder)
        if not nii_files:
            logger.error("No .nii / .nii.gz files found in %s", args.folder)
            sys.exit(1)
        logger.info("Found %d NIfTI files in %s", len(nii_files), args.folder)

    # --- Process each file ---
    for nii_path in nii_files:
        try:
            process_volume(
                nii_path=nii_path,
                predictor=predictor,
                output_dir=output_dir,
                axis=args.axis,
                top_n=args.top_n,
                skip_blank=skip_blank,
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", nii_path.name, exc, exc_info=True)

    logger.info("All done. Results in: %s", output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
