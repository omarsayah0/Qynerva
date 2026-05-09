"""
Evaluate the trained classification model on the held-out test set.

Run from the qynerva_classification_project/ directory:
    python evaluate_classification.py

Outputs
-------
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix (printed + saved as PNG to outputs/plots/)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure SRC is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from SRC.config.config import Config
from SRC.data.dataset import get_eval_transform
from SRC.data.splitter import split_dataset
from SRC.models.efficientnet import BrainTumorClassifier


def load_model(model_path: Path, config: Config, device: torch.device) -> BrainTumorClassifier:
    model = BrainTumorClassifier(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        hidden_units=config.hidden_units,
        pretrained=False,
        backbone=config.backbone,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_evaluation(model, test_data, transform, device, idx_to_class):
    """Run inference over all test samples and return (all_preds, all_labels)."""
    from PIL import Image

    all_preds = []
    all_labels = []

    for img_path, label in test_data:
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        logits = model(tensor)
        pred = logits.argmax(dim=1).item()

        all_preds.append(pred)
        all_labels.append(label)

    return all_preds, all_labels


def plot_confusion_matrix(cm, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close(fig)


def main():
    config = Config()
    device = torch.device(config.device)
    print(f"Device: {device}")

    # Locate model — prefer best_model.pth, fall back to final_model.pth
    best = config.model_dir / "best_model.pth"
    final = config.model_dir / "final_model.pth"
    if best.exists():
        model_path = best
    elif final.exists():
        model_path = final
    else:
        raise FileNotFoundError(
            f"No model found. Expected one of:\n  {best}\n  {final}"
        )
    print(f"Loading model: {model_path}")

    # Load class mapping
    with open(config.class_map_path) as fh:
        class_to_idx: dict = json.load(fh)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Classes: {class_names}")

    # Reconstruct the SAME test split used during training (same seed)
    print(f"\nReconstructing test split from: {config.data_dir.resolve()}")
    _, _, test_data, _ = split_dataset(
        data_dir=config.data_dir,
        class_names=config.class_names,
        val_split=config.val_split,
        test_split=config.test_split,
        random_seed=config.random_seed,
    )
    print(f"Test set size: {len(test_data)} images")

    if not test_data:
        print("No test data found (test_split=0?). Aborting.")
        return

    # Load model and transform
    model = load_model(model_path, config, device)
    transform = get_eval_transform(config)

    # Run evaluation
    print("\nRunning inference on test set...")
    all_preds, all_labels = run_evaluation(model, test_data, transform, device, idx_to_class)

    # Metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print("\n" + "=" * 60)
    print(f"  TEST SET ACCURACY: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("=" * 60)

    print("\nPer-class report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    header = "        " + "  ".join(f"{n[:8]:>8}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"{class_names[i][:8]:>8}  {row_str}")

    # Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, config.plots_dir / "confusion_matrix_test.png")


if __name__ == "__main__":
    main()
