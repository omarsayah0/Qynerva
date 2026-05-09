from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_AXIS_MAP = {"axial": 2, "coronal": 1, "sagittal": 0}
_BLANK_THRESHOLD = 1e-3


class MRIVolumeLoader:
    def __init__(self, nii_path: Path | str, axis: str = "axial", skip_blank: bool = True) -> None:
        self.nii_path = Path(nii_path)
        self.axis = axis.lower()
        self.skip_blank = skip_blank
        if self.axis not in _AXIS_MAP:
            raise ValueError(f"axis must be one of {list(_AXIS_MAP)}; got {self.axis!r}")
        self._volume: np.ndarray | None = None

    @property
    def volume(self) -> np.ndarray:
        if self._volume is None:
            self._volume = _load_nifti(self.nii_path)
        return self._volume

    def get_slices(self) -> List[Tuple[int, Image.Image]]:
        dim = _AXIS_MAP[self.axis]
        vol = self.volume
        results: List[Tuple[int, Image.Image]] = []
        skipped = 0

        for i in range(vol.shape[dim]):
            raw = _get_slice(vol, dim, i)
            if self.skip_blank and _is_blank(raw):
                skipped += 1
                continue
            results.append((i, _array_to_rgb_pil(raw)))

        logger.info("Volume %s: %d %s slices extracted (%d blank skipped)", self.nii_path.name, len(results), self.axis, skipped)
        return results

    def get_raw_slices(self) -> List[Tuple[int, np.ndarray]]:
        dim = _AXIS_MAP[self.axis]
        vol = self.volume
        results = []
        for i in range(vol.shape[dim]):
            raw = _get_slice(vol, dim, i)
            if self.skip_blank and _is_blank(raw):
                continue
            results.append((i, raw))
        return results


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D volume; got shape {data.shape}")
    return data


def _get_slice(volume: np.ndarray, dim: int, idx: int) -> np.ndarray:
    if dim == 0:
        return volume[idx, :, :]
    if dim == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def _is_blank(arr: np.ndarray) -> bool:
    return (float(arr.max()) - float(arr.min())) < _BLANK_THRESHOLD


def _array_to_rgb_pil(arr: np.ndarray) -> Image.Image:
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([uint8, uint8, uint8], axis=-1), mode="RGB")
