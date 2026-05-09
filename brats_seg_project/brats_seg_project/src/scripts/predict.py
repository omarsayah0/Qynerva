from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureType, Invertd, SaveImaged
from monai.data import Dataset, DataLoader, decollate_batch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.transforms import get_predict_transforms
from models.unet3d import build_model
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--patient-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "predictions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    patient_dir = Path(args.patient_dir)
    patient_id = patient_dir.name
    checkpoint_path = args.checkpoint or str(Path(config["training"]["save_dir"]) / config["training"]["checkpoint_name"])

    item = {
        "patient_id": patient_id,
        "image": [
            str(patient_dir / f"{patient_id}-t2f.nii.gz"),
            str(patient_dir / f"{patient_id}-t1n.nii.gz"),
            str(patient_dir / f"{patient_id}-t1c.nii.gz"),
            str(patient_dir / f"{patient_id}-t2w.nii.gz"),
        ],
    }

    transforms = get_predict_transforms()
    loader = DataLoader(Dataset([item], transform=transforms), batch_size=1)

    device = torch.device("cuda" if config["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = sliding_window_inference(images, roi_size=tuple(config["data"]["patch_size"]), sw_batch_size=1, predictor=model)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0].astype(np.uint8)

            reference = nib.load(item["image"][0])
            nib.save(nib.Nifti1Image(pred, reference.affine, reference.header), output_dir / f"{patient_id}_pred.nii.gz")
            print(output_dir / f"{patient_id}_pred.nii.gz")


if __name__ == "__main__":
    main()
