from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch.nn as nn

from qynerva.classification.volume.aggregator import VolumeReport
from qynerva.classification.xai.hirescam import generate_hirescam
from qynerva.classification.xai.visualization import generate_overlay

logger = logging.getLogger(__name__)


@dataclass
class SliceXAIResult:
    slice_index: int
    predicted_class: str
    confidence: float
    original_image: np.ndarray
    cam_map: np.ndarray
    overlay_image: np.ndarray


def run_xai_on_top_slices(report: VolumeReport, model: nn.Module, alpha: float = 0.4) -> list[SliceXAIResult]:
    if not report.top_slices:
        logger.warning("No top slices available for XAI.")
        return []

    results = []
    for sr in report.top_slices:
        cam_map = generate_hirescam(model=model, input_tensor=sr.input_tensor, predicted_class=sr.predicted_class_idx)
        overlay = generate_overlay(original_image=sr.original_image, cam_map=cam_map, alpha=alpha)
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
