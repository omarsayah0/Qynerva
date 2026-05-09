"""
Aggregates per-slice predictions into a patient-level diagnosis.

Algorithm:
  1. Count how many slices were predicted as each class (majority voting).
  2. The class with the most votes is the patient-level prediction.
  3. From the winning-class slices, select the top-N by confidence — these
     are the slices that will be passed to the XAI module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from SRC.volume.inference import SliceResult

logger = logging.getLogger(__name__)


@dataclass
class VolumeReport:
    """Patient-level aggregation result."""

    patient_id: str

    # Volume stats
    total_slices: int

    # Majority-vote outcome
    final_class: str

    # Class vote distribution
    class_counts: Dict[str, int]        # {class_name: number_of_slices}
    class_percentages: Dict[str, float] # {class_name: percentage_of_slices}

    # All per-slice results (for the JSON report and CSV export)
    slice_results: List[SliceResult] = field(repr=False)

    # Top-N slices chosen for XAI (subset of winning-class slices)
    top_slices: List[SliceResult] = field(default_factory=list, repr=False)


def aggregate(
    patient_id: str,
    slice_results: List[SliceResult],
    top_n: int = 5,
) -> VolumeReport:
    """Aggregate slice-level results into a patient-level VolumeReport.

    Args:
        patient_id:    Identifier for the patient / scan (e.g. file stem).
        slice_results: List of per-slice results from
                       :class:`~SRC.volume.inference.VolumeInference`.
        top_n:         Number of top-confidence slices to select for XAI.
                       Capped at the number of winning-class slices available.

    Returns:
        A fully populated :class:`VolumeReport`.

    Raises:
        ValueError: If *slice_results* is empty.
    """
    if not slice_results:
        raise ValueError("slice_results is empty — nothing to aggregate.")

    total = len(slice_results)

    # --- Step 1: count votes per class ---
    counts: Dict[str, int] = {}
    for r in slice_results:
        counts[r.predicted_class] = counts.get(r.predicted_class, 0) + 1

    # --- Step 2: majority vote (ties broken alphabetically for reproducibility) ---
    final_class = max(counts, key=lambda c: (counts[c], c))

    # --- Step 3: compute percentages ---
    percentages: Dict[str, float] = {
        cls: round(cnt / total * 100.0, 2)
        for cls, cnt in counts.items()
    }

    # --- Step 4: select top-N slices of the winning class ---
    candidates = [r for r in slice_results if r.predicted_class == final_class]
    candidates.sort(key=lambda r: r.confidence, reverse=True)
    top_n_capped = min(top_n, len(candidates))
    top_slices = candidates[:top_n_capped]

    logger.info(
        "Patient '%s': final class = %s  |  votes: %d/%d (%.1f%%)  |  "
        "XAI slices selected: %d",
        patient_id,
        final_class,
        counts[final_class],
        total,
        percentages[final_class],
        len(top_slices),
    )

    return VolumeReport(
        patient_id=patient_id,
        total_slices=total,
        final_class=final_class,
        class_counts=counts,
        class_percentages=percentages,
        slice_results=slice_results,
        top_slices=top_slices,
    )
