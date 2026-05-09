"""
MRI volume loader — reads .nii / .nii.gz files and extracts 2-D slices.

No changes to any existing project file are required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Maps axis name → index in a (X, Y, Z) array
_AXIS_MAP = {"axial": 2, "coronal": 1, "sagittal": 0}

# Raw slice intensity range below which a slice is considered blank
_BLANK_THRESHOLD = 1e-3


class MRIVolumeLoader:
    """Loads a NIfTI MRI volume and exposes its 2-D slices as PIL Images.

    Args:
        nii_path:    Path to a ``.nii`` or ``.nii.gz`` file.
        axis:        Plane to slice along — "axial" (Z), "coronal" (Y),
                     or "sagittal" (X).  Defaults to ``"axial"``.
        skip_blank:  If ``True``, discard slices whose normalised range is
                     below ``_BLANK_THRESHOLD`` (nearly empty slices that
                     carry no diagnostic information).
    """

    def __init__(
        self,
        nii_path: Path | str,
        axis: str = "axial",
        skip_blank: bool = True,
    ) -> None:
        self.nii_path = Path(nii_path)
        self.axis = axis.lower()
        self.skip_blank = skip_blank

        if self.axis not in _AXIS_MAP:
            raise ValueError(
                f"axis must be one of {list(_AXIS_MAP)}; got {self.axis!r}"
            )

        self._volume: np.ndarray | None = None   # lazy load

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    @property
    def volume(self) -> np.ndarray:
        """Float32 numpy array of shape (X, Y, Z), loaded on first access."""
        if self._volume is None:
            self._volume = _load_nifti(self.nii_path)
        return self._volume

    def get_slices(self) -> List[Tuple[int, Image.Image]]:
        """Extract all 2-D slices along the chosen axis.

        Returns:
            List of ``(slice_index, PIL.Image)`` in original order.
            Blank slices are omitted when ``skip_blank=True``.
        """
        dim = _AXIS_MAP[self.axis]
        vol = self.volume
        n = vol.shape[dim]

        results: List[Tuple[int, Image.Image]] = []
        skipped = 0

        for i in range(n):
            raw = _get_slice(vol, dim, i)            # float32 (H, W)

            if self.skip_blank and _is_blank(raw):
                skipped += 1
                continue

            pil_img = _array_to_rgb_pil(raw)
            results.append((i, pil_img))

        logger.info(
            "Volume %s: %d %s slices extracted (%d blank skipped)",
            self.nii_path.name,
            len(results),
            self.axis,
            skipped,
        )
        return results

    def get_raw_slices(self) -> List[Tuple[int, np.ndarray]]:
        """Like :meth:`get_slices` but returns raw float32 arrays instead of PIL.

        Useful when you need the original intensity values (e.g. for logging).
        """
        dim = _AXIS_MAP[self.axis]
        vol = self.volume
        n = vol.shape[dim]

        results: List[Tuple[int, np.ndarray]] = []
        for i in range(n):
            raw = _get_slice(vol, dim, i)
            if self.skip_blank and _is_blank(raw):
                continue
            results.append((i, raw))
        return results


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #

def _load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return a float32 array."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI support.\n"
            "Install with:  pip install nibabel"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    # Handle 4-D volumes (e.g. fMRI): take the first time point
    if data.ndim == 4:
        logger.warning(
            "4-D volume detected (shape %s); using first time point.", data.shape
        )
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3-D volume; got shape {data.shape} from {path.name}"
        )

    logger.info("Loaded NIfTI volume: %s — shape %s", path.name, data.shape)
    return data


def _get_slice(volume: np.ndarray, dim: int, idx: int) -> np.ndarray:
    """Extract one 2-D slice from a 3-D volume."""
    if dim == 0:
        return volume[idx, :, :]
    if dim == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def _is_blank(arr: np.ndarray, threshold: float = _BLANK_THRESHOLD) -> bool:
    """Return True if the slice has negligible signal after normalisation."""
    lo, hi = float(arr.min()), float(arr.max())
    return (hi - lo) < threshold


def _array_to_rgb_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float32 2-D slice to a 3-channel PIL Image.

    Steps:
      1. Min-max normalise to [0, 1].
      2. Scale to uint8 [0, 255].
      3. Replicate across R/G/B (the existing model expects 3 channels).
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)

    uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.stack([uint8, uint8, uint8], axis=-1)   # (H, W, 3)
    return Image.fromarray(rgb, mode="RGB")
