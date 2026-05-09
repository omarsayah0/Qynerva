from __future__ import annotations

from pathlib import Path

MODALITIES = ("t2f", "t1n", "t1c", "t2w")


def discover_patients(root_dir: str | Path) -> list[dict]:
    root = Path(root_dir)
    patient_dirs = sorted({file.parent for file in root.rglob("*-t2f.nii.gz")})
    samples = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        sample = {
            "patient_id": patient_id,
            "image": [str(patient_dir / f"{patient_id}-{mod}.nii.gz") for mod in MODALITIES],
            "mask": str(patient_dir / f"{patient_id}-seg.nii.gz"),
        }
        if all(Path(p).exists() for p in sample["image"]) and Path(sample["mask"]).exists():
            samples.append(sample)

    if not samples:
        raise FileNotFoundError("No valid BraTS cases found. Check root_dir and folder structure.")
    return samples
