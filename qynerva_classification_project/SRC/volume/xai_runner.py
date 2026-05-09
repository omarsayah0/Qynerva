"""
XAI runner — applies the existing HiResCAM pipeline to the top-confidence
slices selected by the aggregator.

The existing functions from SRC.xai are called WITHOUT modification:
  - generate_hirescam(model, input_tensor, predicted_class)
  - generate_overlay(original_image, cam_map, alpha)

Only the top slices (those that voted for the final class and have the
highest confidence) are processed here, keeping XAI computation cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch.nn as nn

from SRC.volume.aggregator import VolumeReport
from SRC.xai.hirescam import generate_hirescam
from SRC.xai.visualization import generate_overlay

logger = logging.getLogger(__name__)


@dataclass
class SliceXAIResult:
    """XAI output for one selected slice."""

    slice_index: int
    predicted_class: str
    confidence: float
    original_image: np.ndarray    # float32 (H, W, 3) in [0, 1]
    cam_map: np.ndarray           # float32 (H, W)        in [0, 1]
    overlay_image: np.ndarray     # float32 (H, W, 3)     in [0, 1]


def run_xai_on_top_slices(
    report: VolumeReport,
    model: nn.Module,
    alpha: float = 0.4,
) -> list[SliceXAIResult]:
    """Run HiResCAM on each of the top slices stored in *report*.

    Args:
        report: A :class:`~SRC.volume.aggregator.VolumeReport` whose
                ``top_slices`` list was populated by :func:`~SRC.volume.aggregator.aggregate`.
        model:  The classifier ``nn.Module`` (``predictor.model``).  The same
                model used during inference — no re-loading is needed.
        alpha:  Blend weight for the CAM overlay (0 = only heatmap,
                1 = only original image).  Passed unchanged to
                :func:`~SRC.xai.visualization.generate_overlay`.

    Returns:
        List of :class:`SliceXAIResult`, one per top slice, in confidence
        descending order.
    """
    if not report.top_slices:
        logger.warning("No top slices available for XAI — skipping.")
        return []

    results: list[SliceXAIResult] = []

    for sr in report.top_slices:
        logger.info(
            "XAI  slice %d  class='%s'  conf=%.2f%%",
            sr.slice_index,
            sr.predicted_class,
            sr.confidence * 100,
        )

        # --- Existing XAI call (unchanged) ---
        cam_map = generate_hirescam(
            model=model,
            input_tensor=sr.input_tensor,
            predicted_class=sr.predicted_class_idx,
        )

        overlay = generate_overlay(
            original_image=sr.original_image,
            cam_map=cam_map,
            alpha=alpha,
        )

        results.append(SliceXAIResult(
            slice_index=sr.slice_index,
            predicted_class=sr.predicted_class,
            confidence=sr.confidence,
            original_image=sr.original_image,
            cam_map=cam_map,
            overlay_image=overlay,
        ))

    logger.info("XAI complete — %d slices processed.", len(results))
    return results
