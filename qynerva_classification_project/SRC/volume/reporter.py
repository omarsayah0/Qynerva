"""
Report generator — writes the final patient-level outputs to disk.

Outputs:
  <output_dir>/
    <patient_id>_report.json          — full structured report
    <patient_id>_slice_predictions.csv — per-slice table
    <patient_id>_xai_grid.png         — matplotlib figure (original + overlay)
    xai/<patient_id>_slice_<N>_overlay.png — individual overlay PNGs
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from SRC.volume.aggregator import VolumeReport
from SRC.volume.xai_runner import SliceXAIResult
from SRC.xai.visualization import save_overlay

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Main public function
# --------------------------------------------------------------------------- #

def save_report(
    report: VolumeReport,
    xai_results: List[SliceXAIResult],
    output_dir: Path | str,
) -> None:
    """Persist all outputs for one patient scan.

    Args:
        report:      The aggregated :class:`~VolumeReport`.
        xai_results: XAI outputs from :func:`~run_xai_on_top_slices`.
        output_dir:  Root directory for this patient's outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pid = report.patient_id

    _save_json(report, xai_results, output_dir / f"{pid}_report.json")
    _save_csv(report, output_dir / f"{pid}_slice_predictions.csv")
    _save_individual_overlays(pid, xai_results, output_dir / "xai")
    _save_xai_grid(pid, report, xai_results, output_dir / f"{pid}_xai_grid.png")
    _print_summary(report)


# --------------------------------------------------------------------------- #
# JSON report
# --------------------------------------------------------------------------- #

def _save_json(
    report: VolumeReport,
    xai_results: List[SliceXAIResult],
    path: Path,
) -> None:
    data = {
        "patient_id": report.patient_id,
        "total_slices_analyzed": report.total_slices,
        "final_tumor_class": report.final_class,
        "class_vote_counts": report.class_counts,
        "class_vote_percentages": {
            cls: round(pct, 2) for cls, pct in report.class_percentages.items()
        },
        "top_slices_used_for_xai": [
            {
                "slice_index": sr.slice_index,
                "predicted_class": sr.predicted_class,
                "confidence": round(sr.confidence, 6),
            }
            for sr in report.top_slices
        ],
        "all_slice_predictions": [sr.as_dict() for sr in report.slice_results],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("JSON report saved: %s", path)


# --------------------------------------------------------------------------- #
# CSV per-slice table
# --------------------------------------------------------------------------- #

def _save_csv(report: VolumeReport, path: Path) -> None:
    rows = [sr.as_dict() for sr in report.slice_results]
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Slice predictions CSV saved: %s", path)


# --------------------------------------------------------------------------- #
# Individual overlay PNGs
# --------------------------------------------------------------------------- #

def _save_individual_overlays(
    patient_id: str,
    xai_results: List[SliceXAIResult],
    xai_dir: Path,
) -> None:
    for xr in xai_results:
        out_path = xai_dir / f"{patient_id}_slice_{xr.slice_index:04d}_overlay.png"
        save_overlay(overlay=xr.overlay_image, output_path=out_path)
        logger.info("Overlay saved: %s", out_path)


# --------------------------------------------------------------------------- #
# Matplotlib XAI grid
# --------------------------------------------------------------------------- #

def _save_xai_grid(
    patient_id: str,
    report: VolumeReport,
    xai_results: List[SliceXAIResult],
    path: Path,
) -> None:
    """Save a figure with one column per selected slice.

    Each column contains:
      row 0 — original slice image
      row 1 — HiResCAM overlay
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping XAI grid.")
        return

    if not xai_results:
        logger.warning("No XAI results — skipping grid figure.")
        return

    n_cols = len(xai_results)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_cols,
        figsize=(4 * n_cols, 8),
        squeeze=False,
    )

    # Header
    fig.suptitle(
        f"Patient: {patient_id}  |  Final diagnosis: {report.final_class}\n"
        f"Votes: "
        + "  ".join(
            f"{cls} {pct:.1f}%"
            for cls, pct in sorted(
                report.class_percentages.items(), key=lambda x: -x[1]
            )
        ),
        fontsize=11,
        y=1.01,
    )

    for col, xr in enumerate(xai_results):
        # Row 0: original slice
        ax_orig = axes[0][col]
        ax_orig.imshow(xr.original_image, cmap="gray")
        ax_orig.set_title(
            f"Slice {xr.slice_index}\n"
            f"{xr.predicted_class}\n"
            f"conf {xr.confidence * 100:.1f}%",
            fontsize=8,
        )
        ax_orig.axis("off")

        # Row 1: HiResCAM overlay
        ax_cam = axes[1][col]
        ax_cam.imshow(xr.overlay_image)
        ax_cam.set_title("HiResCAM", fontsize=8)
        ax_cam.axis("off")

    # Row labels on the leftmost column
    axes[0][0].set_ylabel("Original", fontsize=9)
    axes[1][0].set_ylabel("Explanation", fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("XAI grid saved: %s", path)


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #

def _print_summary(report: VolumeReport) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  PATIENT-LEVEL REPORT")
    print(f"  Patient ID : {report.patient_id}")
    print(f"  Total slices analyzed : {report.total_slices}")
    print("-" * width)
    print(f"  FINAL DIAGNOSIS : {report.final_class}")
    print("-" * width)
    print("  Class vote distribution:")
    # Sort by vote count descending
    for cls, pct in sorted(
        report.class_percentages.items(), key=lambda x: -x[1]
    ):
        count = report.class_counts[cls]
        bar = "█" * int(pct / 100 * 30)
        marker = " ← FINAL" if cls == report.final_class else ""
        print(f"    {cls:<28}  {count:>4} slices  {pct:>6.2f}%  {bar}{marker}")
    print("-" * width)
    print(f"  Top slices used for XAI (class={report.final_class}):")
    for sr in report.top_slices:
        print(
            f"    slice {sr.slice_index:>4}   conf {sr.confidence * 100:>6.2f}%"
        )
    print("=" * width)
    print()
