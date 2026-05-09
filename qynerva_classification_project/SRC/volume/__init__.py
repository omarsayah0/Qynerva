"""
volume — 3-D MRI volume classification pipeline.

Extends the existing 2-D classifier to handle full NIfTI volumes by:
  1. Loading a .nii / .nii.gz file and extracting all 2-D slices.
  2. Running the existing Predictor on every slice unchanged.
  3. Aggregating slice-level predictions via majority voting.
  4. Running HiResCAM XAI only on the top-confident slices for the
     winning class.
  5. Saving a JSON report and a matplotlib visualisation grid.

Public API
----------
  MRIVolumeLoader   — load a NIfTI file and extract 2-D slices
  VolumeInference   — classify each slice with the existing Predictor
  aggregate         — majority voting + top-N slice selection
  run_xai_on_top_slices — HiResCAM on selected slices
  save_report       — JSON + matplotlib report
"""

from SRC.volume.loader import MRIVolumeLoader
from SRC.volume.inference import VolumeInference, SliceResult
from SRC.volume.aggregator import aggregate, VolumeReport
from SRC.volume.xai_runner import run_xai_on_top_slices, SliceXAIResult
from SRC.volume.reporter import save_report

__all__ = [
    "MRIVolumeLoader",
    "VolumeInference",
    "SliceResult",
    "aggregate",
    "VolumeReport",
    "run_xai_on_top_slices",
    "SliceXAIResult",
    "save_report",
]
