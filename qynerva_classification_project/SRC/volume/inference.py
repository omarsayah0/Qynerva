"""
Slice-level inference — runs the existing Predictor on each 2-D slice.

The Predictor instance is used without any modification:
  - predictor.transform  is applied directly to each PIL slice
  - predictor.model      is called for every slice
  - predictor.idx_to_class maps logit indices to class names

This module only adds the looping logic; the underlying classifier is
exactly the same as for single-image prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from SRC.prediction.predictor import Predictor

logger = logging.getLogger(__name__)


@dataclass
class SliceResult:
    """Per-slice classification output and everything needed for XAI later."""

    slice_index: int
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: Dict[str, float]

    # Retained for XAI — the preprocessed tensor that was fed to the model
    input_tensor: torch.Tensor = field(repr=False)
    # The original slice as float32 (H, W, 3) in [0, 1] — used for overlay
    original_image: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        """Serialisable representation (tensors/arrays are excluded)."""
        return {
            "slice_index": self.slice_index,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            **{f"prob_{k}": round(v, 6) for k, v in self.class_probabilities.items()},
        }


class VolumeInference:
    """Applies the existing 2-D Predictor to every slice of an MRI volume.

    Args:
        predictor: An already-initialised :class:`~SRC.prediction.predictor.Predictor`.
    """

    def __init__(self, predictor: Predictor) -> None:
        self._transform = predictor.transform
        self._model = predictor.model
        self._device = predictor.device
        self._idx_to_class = predictor.idx_to_class

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def classify_slices(
        self,
        pil_slices: List[Tuple[int, Image.Image]],
    ) -> List[SliceResult]:
        """Classify every slice in *pil_slices*.

        Args:
            pil_slices: List of ``(slice_index, PIL.Image)`` as returned by
                        :meth:`~SRC.volume.loader.MRIVolumeLoader.get_slices`.

        Returns:
            List of :class:`SliceResult`, one per input slice.
        """
        results: List[SliceResult] = []
        self._model.eval()

        for slice_idx, pil_img in pil_slices:
            # Apply exactly the same transform as single-image prediction
            input_tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

            # Keep gradients live so HiResCAM can back-propagate later
            with torch.enable_grad():
                logits = self._model(input_tensor)        # (1, num_classes)
                probs = F.softmax(logits, dim=1)[0]       # (num_classes,)

            top_idx = int(probs.argmax().item())
            confidence = float(probs[top_idx].item())
            predicted_class = self._idx_to_class[top_idx]

            class_probabilities: Dict[str, float] = {
                self._idx_to_class[i]: float(probs[i].item())
                for i in range(len(self._idx_to_class))
            }

            # Build the float32 (H, W, 3) image from the PIL slice
            original_image = np.asarray(pil_img, dtype=np.float32) / 255.0

            results.append(SliceResult(
                slice_index=slice_idx,
                predicted_class=predicted_class,
                predicted_class_idx=top_idx,
                confidence=confidence,
                class_probabilities=class_probabilities,
                input_tensor=input_tensor,
                original_image=original_image,
            ))

            logger.debug(
                "Slice %4d  →  %-25s  (conf %.2f%%)",
                slice_idx,
                predicted_class,
                confidence * 100,
            )

        _log_class_distribution(results)
        return results


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #

def _log_class_distribution(results: List[SliceResult]) -> None:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.predicted_class] = counts.get(r.predicted_class, 0) + 1
    total = len(results)
    summary = ", ".join(
        f"{cls}={cnt} ({cnt / total * 100:.1f}%)"
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1])
    )
    logger.info("Classified %d slices — %s", total, summary)
