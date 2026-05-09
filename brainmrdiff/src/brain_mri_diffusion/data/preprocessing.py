"""BraTS preprocessing: load NIfTI → normalize → resize → extract masks → save .npy."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

logger = logging.getLogger(__name__)

# SynthSeg label map (FreeSurfer convention subset)
SYNTHSEG_LABELS = {
    "left_white_matter": 2,
    "left_gray_matter": 3,
    "left_ventricle": 4,
    "left_cerebellum_wm": 7,
    "left_cerebellum_cortex": 8,
    "left_thalamus": 10,
    "left_caudate": 11,
    "left_putamen": 12,
    "left_pallidum": 13,
    "third_ventricle": 14,
    "fourth_ventricle": 15,
    "brain_stem": 16,
    "left_hippocampus": 17,
    "left_amygdala": 18,
    "left_accumbens": 26,
    "left_ventral_dc": 28,
    "right_white_matter": 41,
    "right_gray_matter": 42,
    "right_ventricle": 43,
    "right_cerebellum_wm": 46,
    "right_cerebellum_cortex": 47,
    "right_thalamus": 49,
    "right_caudate": 50,
    "right_putamen": 51,
    "right_pallidum": 52,
    "right_hippocampus": 53,
    "right_amygdala": 54,
    "right_accumbens": 58,
    "right_ventral_dc": 60,
}

WMT_LABELS = [2, 41, 7, 46]
CGM_LABELS = [3, 42, 8, 47]
LV_LABELS = [4, 43, 14, 15]


class BraTSPreprocessor:
    """Preprocess BraTS patients into per-slice .npy arrays."""

    def __init__(
        self,
        data_dir: str,
        processed_dir: str,
        image_size: int = 128,
        modalities: Optional[List[str]] = None,
        use_synthseg: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.image_size = image_size
        self.modalities = modalities or ["t1n", "t1c", "t2w", "t2f"]
        self.use_synthseg = use_synthseg

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, overwrite: bool = False) -> None:
        patients = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        logger.info(f"Found {len(patients)} patients in {self.data_dir}")

        ok, failed = 0, 0
        for patient_dir in tqdm(patients, desc="Preprocessing patients"):
            out_dir = self.processed_dir / patient_dir.name
            if out_dir.exists() and not overwrite:
                logger.debug(f"Skipping {patient_dir.name} (already processed)")
                ok += 1
                continue
            try:
                self._process_patient(patient_dir, out_dir)
                ok += 1
            except Exception as exc:
                logger.warning(f"Failed {patient_dir.name}: {exc}")
                failed += 1

        logger.info(f"Preprocessing done. OK={ok}, Failed={failed}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_patient(self, patient_dir: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pid = patient_dir.name

        # Load all modalities + seg
        volumes: Dict[str, np.ndarray] = {}
        for mod in self.modalities:
            path = self._find_file(patient_dir, f"-{mod}.nii.gz")
            volumes[mod] = self._load_nifti(path)

        seg_path = self._find_file(patient_dir, "-seg.nii.gz")
        seg = self._load_nifti(seg_path)

        # Per-slice processing
        n_slices = volumes[self.modalities[0]].shape[2]
        slices: Dict[str, List[np.ndarray]] = {k: [] for k in self.modalities}
        tumor_slices: List[np.ndarray] = []
        brain_slices: List[np.ndarray] = []
        wmt_slices: List[np.ndarray] = []
        cgm_slices: List[np.ndarray] = []
        lv_slices: List[np.ndarray] = []

        t1n_vol = volumes["t1n"]

        for z in range(n_slices):
            seg_slice = seg[:, :, z]
            tumor_mask = (seg_slice > 0).astype(np.float32)

            # Basic anatomical masks derived from BraTS seg labels
            brain_mask = (seg_slice >= 0).astype(np.float32)  # all non-background
            wmt = self._label_mask(seg_slice, WMT_LABELS)
            cgm = self._label_mask(seg_slice, CGM_LABELS)
            lv = self._label_mask(seg_slice, LV_LABELS)

            tumor_slices.append(self._resize_slice(tumor_mask))
            brain_slices.append(self._resize_slice(brain_mask))
            wmt_slices.append(self._resize_slice(wmt))
            cgm_slices.append(self._resize_slice(cgm))
            lv_slices.append(self._resize_slice(lv))

            for mod in self.modalities:
                raw = volumes[mod][:, :, z].astype(np.float32)
                normalized = self._normalize(raw)
                slices[mod].append(self._resize_slice(normalized))

        # If SynthSeg is available, overwrite structural masks
        if self.use_synthseg:
            synthseg_result = self._run_synthseg(patient_dir, pid)
            if synthseg_result is not None:
                brain_slices, wmt_slices, cgm_slices, lv_slices = synthseg_result

        # Save stacked arrays
        for mod in self.modalities:
            np.save(out_dir / f"{mod}.npy", np.stack(slices[mod], axis=0))

        np.save(out_dir / "tumor_mask.npy", np.stack(tumor_slices, axis=0))
        np.save(out_dir / "brain_mask.npy", np.stack(brain_slices, axis=0))
        np.save(out_dir / "wmt.npy", np.stack(wmt_slices, axis=0))
        np.save(out_dir / "cgm.npy", np.stack(cgm_slices, axis=0))
        np.save(out_dir / "lv.npy", np.stack(lv_slices, axis=0))

        logger.debug(f"Saved {pid} → {out_dir} ({n_slices} slices)")

    def _run_synthseg(
        self, patient_dir: Path, pid: str
    ) -> Optional[Tuple[List, List, List, List]]:
        """
        Attempt to run SynthSeg segmentation on the T1n image.
        Falls back gracefully if SynthSeg is not installed.
        """
        try:
            import subprocess
            import tempfile

            t1n_path = self._find_file(patient_dir, "-t1n.nii.gz")
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                seg_out = Path(tmp.name)

            result = subprocess.run(
                ["mri_synthseg", "--i", str(t1n_path), "--o", str(seg_out), "--fast"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.debug(f"SynthSeg failed for {pid}: {result.stderr[:200]}")
                return None

            seg_vol = self._load_nifti(seg_out)
            seg_out.unlink(missing_ok=True)

            n_slices = seg_vol.shape[2]
            brain_slices, wmt_slices, cgm_slices, lv_slices = [], [], [], []
            for z in range(n_slices):
                sl = seg_vol[:, :, z]
                brain_slices.append(self._resize_slice((sl > 0).astype(np.float32)))
                wmt_slices.append(self._resize_slice(self._label_mask(sl, WMT_LABELS)))
                cgm_slices.append(self._resize_slice(self._label_mask(sl, CGM_LABELS)))
                lv_slices.append(self._resize_slice(self._label_mask(sl, LV_LABELS)))

            return brain_slices, wmt_slices, cgm_slices, lv_slices

        except (ImportError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug(f"SynthSeg not available ({exc}), using BraTS labels instead.")
            return None

    @staticmethod
    def _find_file(patient_dir: Path, suffix: str) -> Path:
        matches = list(patient_dir.glob(f"*{suffix}"))
        if not matches:
            raise FileNotFoundError(
                f"No file matching *{suffix} in {patient_dir}"
            )
        return matches[0]

    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32)

    @staticmethod
    def _normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Percentile-based intensity normalization to [0, 1]."""
        lo = np.percentile(volume, 1)
        hi = np.percentile(volume, 99)
        if hi - lo < eps:
            return np.zeros_like(volume)
        return np.clip((volume - lo) / (hi - lo + eps), 0.0, 1.0)

    def _resize_slice(self, sl: np.ndarray) -> np.ndarray:
        h, w = sl.shape
        if h == self.image_size and w == self.image_size:
            return sl
        zh = self.image_size / h
        zw = self.image_size / w
        return zoom(sl, (zh, zw), order=1).astype(np.float32)

    @staticmethod
    def _label_mask(seg: np.ndarray, labels: List[int]) -> np.ndarray:
        mask = np.zeros_like(seg, dtype=np.float32)
        for lbl in labels:
            mask[seg == lbl] = 1.0
        return mask
