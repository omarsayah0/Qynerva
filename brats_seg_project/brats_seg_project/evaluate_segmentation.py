"""
Evaluate the trained 3D U-Net segmentation model on the held-out test set.

Metrics reported (voxel-level, per class and aggregate):
  - Overall Accuracy
  - Precision, Recall (Sensitivity), F1 / Dice  — per class
  - 4 x 4 Confusion Matrix

How it works
------------
1. Loads the best checkpoint saved by the training script.
2. Reconstructs the identical test split (same seed + ratios as training).
3. Runs sliding-window inference on every test volume.
4. Accumulates a single 4x4 confusion matrix across all voxels of all patients
   (memory-efficient — raw voxel arrays are never held in memory).
5. Derives all metrics from that matrix and prints a formatted report.

Run from the brats_seg_project/brats_seg_project/ directory:
    python evaluate_segmentation.py

Resource-saving flags (use these if your PC overheats / shuts down):
    --patch_size 64 64 64   Use smaller inference patches (less GPU memory)
    --overlap 0.125         Lower overlap = less redundant computation
    --device cpu            Force CPU (slower but no GPU heat)
    --max_patients 5        Only evaluate on N patients (quick sanity check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data.splits import make_splits
from data.transforms import get_eval_transforms
from models.unet3d import build_model
from utils.config import load_config
from utils.io import discover_patients
from utils.seed import set_seed

CLASS_NAMES = ["Background", "NCR (Necrotic Core)", "ED (Edema)", "ET (Enhancing Tumor)"]
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        saved_dice = ckpt.get("best_val_dice", None)
        epoch = ckpt.get("epoch", "?")
        if saved_dice is not None:
            print(f"Checkpoint : epoch {epoch}  |  best_val_dice = {saved_dice:.4f}")
        else:
            print(f"Checkpoint : epoch {epoch}")
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference + confusion-matrix accumulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(model, test_loader, patch_size: tuple, overlap: float, device: torch.device) -> np.ndarray:
    """
    Slide a window over every test volume, argmax the predictions,
    and accumulate a global 4x4 confusion matrix (rows=true, cols=pred).

    Masks come out of the dataloader as one-hot (4, H, W, D).
    Logits come out of the model as (1, 4, H, W, D).
    Both are collapsed to integer class indices (H, W, D) before comparison.
    """
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    post_label = AsDiscrete(argmax=True)

    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for batch in tqdm(test_loader, desc="Evaluating test set"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)          # (1, 4, H, W, D)  one-hot

        logits = sliding_window_inference(
            inputs=images,
            roi_size=patch_size,
            sw_batch_size=1,
            overlap=overlap,
            predictor=model,
        )                                          # (1, 4, H, W, D)

        # Collapse channel dim -> class index, flatten to 1-D
        y_pred = post_pred(logits[0]).squeeze(0).cpu().numpy().ravel().astype(np.int32)
        y_true = post_label(masks[0]).squeeze(0).cpu().numpy().ravel().astype(np.int32)

        cm_total += confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

        # Free GPU memory between patients to avoid thermal build-up
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return cm_total


# ---------------------------------------------------------------------------
# Metric derivation
# ---------------------------------------------------------------------------

def derive_metrics(cm: np.ndarray):
    """Derive accuracy, per-class precision, recall, and F1 from the confusion matrix."""
    accuracy = np.diag(cm).sum() / cm.sum()
    precision = np.zeros(NUM_CLASSES)
    recall    = np.zeros(NUM_CLASSES)
    f1        = np.zeros(NUM_CLASSES)

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp       # row sum minus diagonal
        fp = cm[:, i].sum() - tp    # col sum minus diagonal
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom = precision[i] + recall[i]
        f1[i] = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0

    return accuracy, precision, recall, f1


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(cm: np.ndarray, accuracy: float, precision: np.ndarray, recall: np.ndarray, f1: np.ndarray):
    sep = "=" * 72

    print(f"\n{sep}")
    print("  SEGMENTATION MODEL EVALUATION  —  TEST SET")
    print(f"{sep}\n")

    print(f"  Overall Voxel Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)\n")

    print(f"  {'Class':<30} {'Precision':>10} {'Recall':>10} {'F1/Dice':>10}")
    print("  " + "-" * 62)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<30} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")

    mean_prec_tumor   = precision[1:].mean()
    mean_recall_tumor = recall[1:].mean()
    mean_f1_tumor     = f1[1:].mean()
    print("  " + "-" * 62)
    print(
        f"  {'Mean Tumor (NCR + ED + ET)':<30} "
        f"{mean_prec_tumor:>10.4f} {mean_recall_tumor:>10.4f} {mean_f1_tumor:>10.4f}"
    )

    print(f"\n{sep}")
    print("  CONFUSION MATRIX  (rows = True class,  cols = Predicted class)")
    print(f"{sep}")
    col_w       = 14
    short_names = ["BG", "NCR", "ED", "ET"]
    print(f"  {'':>8}" + "".join(f"{s:>{col_w}}" for s in short_names))
    print("  " + "-" * (8 + col_w * NUM_CLASSES))
    for i, row_label in enumerate(short_names):
        row = f"  {row_label:>8}" + "".join(f"{cm[i, j]:>{col_w},}" for j in range(NUM_CLASSES))
        print(row)

    print(f"\n{sep}")
    print("  Note: F1/Dice per class = voxel-wise Dice (2TP / (2TP + FP + FN)).")
    print(f"  Tumor classes: NCR = Necrotic Core, ED = Edema, ET = Enhancing Tumor.")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the BraTS segmentation model.")
    parser.add_argument(
        "--patch_size", nargs=3, type=int, default=None, metavar=("H", "W", "D"),
        help="Override inference patch size, e.g. --patch_size 64 64 64  (default: from config)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.25,
        help="Sliding-window overlap fraction (default: 0.25). Lower = faster, less accurate.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device, e.g. --device cpu  (default: from config, falls back to cpu if no CUDA)",
    )
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Evaluate on at most N patients (quick sanity check). Default: all test patients.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    here        = Path(__file__).parent
    config_path = here / "configs" / "train.yaml"
    config      = load_config(str(config_path))

    set_seed(config["seed"])

    if args.device:
        device = torch.device(args.device)
    else:
        device_str = config["training"]["device"]
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    save_dir        = here / config["training"]["save_dir"]
    checkpoint_path = save_dir / config["training"]["checkpoint_name"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run the training script first, or check save_dir / checkpoint_name in configs/train.yaml."
        )
    print(f"Checkpoint : {checkpoint_path}")

    root_dir = here / config["data"]["root_dir"]
    print(f"Data root  : {root_dir.resolve()}")
    samples = discover_patients(root_dir)

    _, _, test_items = make_splits(
        samples,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["seed"],
    )

    if args.max_patients is not None:
        test_items = test_items[: args.max_patients]
        print(f"Test set   : {len(test_items)} patients  (limited by --max_patients)\n")
    else:
        print(f"Test set   : {len(test_items)} patients\n")

    if not test_items:
        raise RuntimeError("No test samples found. Lower test_ratio or provide more data.")

    test_ds = Dataset(data=test_items, transform=get_eval_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,          # 0 avoids multiprocessing issues on Windows/WSL
        collate_fn=list_data_collate,
        pin_memory=False,
    )

    model = load_model(checkpoint_path, config, device)

    patch_size = tuple(args.patch_size) if args.patch_size else tuple(config["data"]["patch_size"])
    overlap    = args.overlap
    print(f"Patch size : {patch_size}")
    print(f"Overlap    : {overlap}")
    print("Running sliding-window inference — may take several minutes per case on CPU...\n")

    cm = run_evaluation(model, test_loader, patch_size, overlap, device)
    accuracy, precision, recall, f1 = derive_metrics(cm)
    print_results(cm, accuracy, precision, recall, f1)


if __name__ == "__main__":
    main()
