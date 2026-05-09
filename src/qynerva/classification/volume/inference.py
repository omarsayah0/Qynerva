from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qynerva.classification.prediction.predictor import Predictor

logger = logging.getLogger(__name__)


@dataclass
class SliceResult:
    slice_index: int
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: Dict[str, float]
    input_tensor: torch.Tensor = field(repr=False)
    original_image: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "slice_index": self.slice_index,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            **{f"prob_{k}": round(v, 6) for k, v in self.class_probabilities.items()},
        }


class VolumeInference:
    def __init__(self, predictor: Predictor) -> None:
        self._transform = predictor.transform
        self._model = predictor.model
        self._device = predictor.device
        self._idx_to_class = predictor.idx_to_class

    def classify_slices(self, pil_slices: List[Tuple[int, Image.Image]]) -> List[SliceResult]:
        results: List[SliceResult] = []
        self._model.eval()

        for slice_idx, pil_img in pil_slices:
            input_tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

            with torch.enable_grad():
                logits = self._model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]

            top_idx = int(probs.argmax().item())
            results.append(SliceResult(
                slice_index=slice_idx,
                predicted_class=self._idx_to_class[top_idx],
                predicted_class_idx=top_idx,
                confidence=float(probs[top_idx].item()),
                class_probabilities={self._idx_to_class[i]: float(probs[i].item()) for i in range(len(self._idx_to_class))},
                input_tensor=input_tensor,
                original_image=np.asarray(pil_img, dtype=np.float32) / 255.0,
            ))

        _log_distribution(results)
        return results


def _log_distribution(results: List[SliceResult]) -> None:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.predicted_class] = counts.get(r.predicted_class, 0) + 1
    total = len(results)
    summary = ", ".join(f"{cls}={cnt} ({cnt/total*100:.1f}%)" for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]))
    logger.info("Classified %d slices — %s", total, summary)
