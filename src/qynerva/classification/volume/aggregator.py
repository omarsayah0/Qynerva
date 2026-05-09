from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from qynerva.classification.volume.inference import SliceResult

logger = logging.getLogger(__name__)


@dataclass
class VolumeReport:
    patient_id: str
    total_slices: int
    final_class: str
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    slice_results: List[SliceResult] = field(repr=False)
    top_slices: List[SliceResult] = field(default_factory=list, repr=False)


def aggregate(patient_id: str, slice_results: List[SliceResult], top_n: int = 5) -> VolumeReport:
    if not slice_results:
        raise ValueError("slice_results is empty.")

    total = len(slice_results)
    counts: Dict[str, int] = {}
    for r in slice_results:
        counts[r.predicted_class] = counts.get(r.predicted_class, 0) + 1

    final_class = max(counts, key=lambda c: (counts[c], c))
    percentages = {cls: round(cnt / total * 100.0, 2) for cls, cnt in counts.items()}

    candidates = sorted([r for r in slice_results if r.predicted_class == final_class], key=lambda r: r.confidence, reverse=True)
    top_slices = candidates[:min(top_n, len(candidates))]

    logger.info("Patient '%s': final=%s | votes=%d/%d (%.1f%%) | XAI slices=%d",
                patient_id, final_class, counts[final_class], total, percentages[final_class], len(top_slices))

    return VolumeReport(
        patient_id=patient_id,
        total_slices=total,
        final_class=final_class,
        class_counts=counts,
        class_percentages=percentages,
        slice_results=slice_results,
        top_slices=top_slices,
    )
