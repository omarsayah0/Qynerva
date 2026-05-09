"""
Visualisation utilities — training history plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict[str, List], output_dir: Path) -> None:
    """Save loss and accuracy curves to *output_dir*.

    Two PNG files are generated:
        - ``loss_curve.png``   — train vs validation loss per epoch.
        - ``accuracy_curve.png`` — train vs validation accuracy per epoch.

    Stage boundaries are indicated by a vertical dashed line.

    Args:
        history:    Dictionary as produced by the training pipeline.
        output_dir: Directory where plots will be saved.
    """
    # matplotlib is an optional runtime dependency at prediction time so
    # we import it lazily here to avoid hard failures when it is absent.
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (safe on headless servers)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    stages = history.get("stage", [])

    # Find stage-boundary indices (where stage changes)
    boundaries: List[int] = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            boundaries.append(i)  # 0-based index

    def _add_stage_lines(ax, label: bool = True) -> None:
        for idx in boundaries:
            ax.axvline(
                x=epochs[idx],
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label="Stage 2 start" if label else "",
            )

    # ------------------------------------------------------------------ #
    # Loss curve
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)
    _add_stage_lines(ax, label=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved: %s", loss_path)

    # ------------------------------------------------------------------ #
    # Accuracy curve
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_acc"], label="Train Acc", linewidth=1.5)
    ax.plot(epochs, history["val_acc"], label="Val Acc", linewidth=1.5)
    _add_stage_lines(ax, label=False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    acc_path = output_dir / "accuracy_curve.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    logger.info("Accuracy curve saved: %s", acc_path)


def load_and_plot(history_path: Path, output_dir: Path) -> None:
    """Convenience wrapper: read a JSON history file and plot it.

    Args:
        history_path: Path to ``training_history.json``.
        output_dir:   Directory where plots will be saved.
    """
    with open(history_path) as fh:
        history = json.load(fh)
    plot_training_history(history, output_dir)
