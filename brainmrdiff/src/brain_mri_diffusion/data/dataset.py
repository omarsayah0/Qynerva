"""PyTorch Dataset and DataLoader factory for preprocessed BraTS data."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

MASK_KEYS = ["tumor_mask", "brain_mask", "wmt", "cgm", "lv"]
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


class BraTSDataset(Dataset):
    """
    Loads per-slice .npy files produced by BraTSPreprocessor.

    Each sample:
        image   : (1, H, W)  float32  [0, 1]  – target MRI slice
        cond    : (5, H, W)  float32  {0, 1}  – anatomical conditioning masks
        modality: int                           – index in MODALITIES list
        patient : str                           – patient ID
        slice_z : int                           – axial slice index
    """

    def __init__(
        self,
        processed_dir: str,
        modalities: Optional[List[str]] = None,
        image_size: int = 128,
        augment: bool = True,
        min_tissue_fraction: float = 0.02,
        patient_ids: Optional[List[str]] = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.modalities = modalities or MODALITIES
        self.image_size = image_size
        self.augment = augment
        self.min_tissue_fraction = min_tissue_fraction

        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed directory not found: {self.processed_dir}\n"
                "Run preprocessing first: bmd-preprocess"
            )

        self.samples: List[Dict] = []
        self._index(patient_ids)

        # Preload all patient arrays into RAM before workers fork.
        # Eliminates repeated disk reads across epochs and workers.
        # Each worker inherits the parent's copy-on-write memory pages.
        self._cache: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        if cache_in_memory:
            self._cache = self._preload_cache()

        logger.info(
            f"Dataset ready: {len(self.samples)} slices "
            f"from {self.processed_dir}"
            + (" (cached in RAM)" if cache_in_memory else "")
        )

    # ------------------------------------------------------------------

    def _preload_cache(self) -> Dict[str, Dict[str, np.ndarray]]:
        required_keys = MASK_KEYS + self.modalities
        seen_patients = {s["patient_dir"] for s in self.samples}
        cache: Dict[str, Dict[str, np.ndarray]] = {}
        logger.info(f"Preloading {len(seen_patients)} patients into RAM …")
        for pdir in seen_patients:
            p = Path(pdir)
            cache[pdir] = {k: np.load(p / f"{k}.npy") for k in required_keys}
        logger.info("Preload complete.")
        return cache

    # ------------------------------------------------------------------

    def _index(self, patient_ids: Optional[List[str]]) -> None:
        patients = sorted([p for p in self.processed_dir.iterdir() if p.is_dir()])
        if patient_ids is not None:
            patients = [p for p in patients if p.name in patient_ids]

        for patient_dir in patients:
            # Check all required files exist
            required = MASK_KEYS + self.modalities
            if not all((patient_dir / f"{k}.npy").exists() for k in required):
                logger.debug(f"Skipping incomplete patient: {patient_dir.name}")
                continue

            # Load brain_mask to count valid slices
            brain_mask = np.load(patient_dir / "brain_mask.npy")  # (Z, H, W)
            n_slices = brain_mask.shape[0]

            for z in range(n_slices):
                # Only include slices with enough brain tissue
                if brain_mask[z].mean() < self.min_tissue_fraction:
                    continue
                for mod_idx, mod in enumerate(self.modalities):
                    self.samples.append(
                        {
                            "patient": patient_dir.name,
                            "patient_dir": str(patient_dir),
                            "slice_z": z,
                            "n_slices": n_slices,
                            "modality": mod_idx,
                        }
                    )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        info = self.samples[idx]
        patient_dir = Path(info["patient_dir"])
        pdir_str = info["patient_dir"]
        z = info["slice_z"]

        mod_name = self.modalities[info["modality"]]

        if self._cache is not None:
            arrays = self._cache[pdir_str]
            image = arrays[mod_name][z]
            masks = [arrays[k][z] for k in MASK_KEYS]
        else:
            image = np.load(patient_dir / f"{mod_name}.npy", mmap_mode="r")[z]
            masks = [np.load(patient_dir / f"{k}.npy", mmap_mode="r")[z] for k in MASK_KEYS]

        cond = np.stack(masks, axis=0)  # (5, H, W)

        image_t = torch.from_numpy(image).unsqueeze(0).float()    # (1, H, W)
        cond_t = torch.from_numpy(cond).float()                    # (5, H, W)

        # Augmentation: horizontal flip
        if self.augment and random.random() > 0.5:
            image_t = torch.flip(image_t, dims=[-1])
            cond_t = torch.flip(cond_t, dims=[-1])

        return {
            "image": image_t,
            "cond": cond_t,
            "modality": torch.tensor(info["modality"], dtype=torch.long),
            "patient": info["patient"],
            "slice_z": info["slice_z"],
        }


# ---------------------------------------------------------------------------


def get_dataloaders(
    processed_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    modalities: Optional[List[str]] = None,
    image_size: int = 128,
    seed: int = 42,
    cache_in_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Split patients into train / val and return DataLoaders."""
    processed_path = Path(processed_dir)
    patients = sorted([p.name for p in processed_path.iterdir() if p.is_dir()])

    if not patients:
        raise ValueError(f"No patient directories found in {processed_dir}")

    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_fraction))
    val_patients = patients[:n_val]
    train_patients = patients[n_val:]

    logger.info(
        f"Split: {len(train_patients)} train patients, {len(val_patients)} val patients"
    )

    train_ds = BraTSDataset(
        processed_dir=processed_dir,
        modalities=modalities,
        image_size=image_size,
        augment=True,
        patient_ids=train_patients,
        cache_in_memory=cache_in_memory,
    )
    val_ds = BraTSDataset(
        processed_dir=processed_dir,
        modalities=modalities,
        image_size=image_size,
        augment=False,
        patient_ids=val_patients,
        cache_in_memory=cache_in_memory,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
