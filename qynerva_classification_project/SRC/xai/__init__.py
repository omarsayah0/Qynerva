"""
xai — Explainability module for brain MRI tumor classification.

Exposes the two primary entry points:
  - generate_hirescam  : produce a raw HiResCAM saliency map
  - generate_overlay   : blend the CAM map onto the original image
  - save_overlay       : persist the overlay to disk (optional helper)
"""

from SRC.xai.hirescam import generate_hirescam, get_target_layer
from SRC.xai.visualization import generate_overlay, save_overlay

__all__ = [
    "generate_hirescam",
    "get_target_layer",
    "generate_overlay",
    "save_overlay",
]
