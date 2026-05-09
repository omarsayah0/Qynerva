"""
Two-stage training pipeline for BrainTumorClassifier.

Stage 1: Backbone frozen — trains only the classification head.
Stage 2: Top backbone blocks unfrozen — fine-tunes with a much smaller LR.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from SRC.config.config import Config
from SRC.models.efficientnet import BrainTumorClassifier
from SRC.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Epoch helpers
# --------------------------------------------------------------------------- #

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Run a single training or evaluation epoch.

    Pass *optimizer=None* to run in evaluation (validation) mode.

    Returns:
        ``(mean_loss, accuracy)`` for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp and is_train:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                if is_train:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #

def _train_stage(
    stage: int,
    model: BrainTumorClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    early_stopping: EarlyStopping,
    checkpoint: ModelCheckpoint,
    epochs: int,
    device: torch.device,
    use_amp: bool,
    history: Dict[str, List],
) -> None:
    """Execute *epochs* of training for one stage.

    Modifies *history* in-place, adding per-epoch entries for
    ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``.
    """
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None
    effective_amp = use_amp and device.type == "cuda"

    logger.info("=" * 60)
    logger.info("Stage %d — starting (%d epochs max)", stage, epochs)
    logger.info("=" * 60)

    early_stopping.reset()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, use_amp=effective_amp,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device,
        )

        scheduler.step(val_loss)
        checkpoint(model, val_loss)

        history["stage"].append(stage)
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        elapsed = time.perf_counter() - t0
        logger.info(
            "Stage %d | Epoch %3d/%d | "
            "train_loss: %.4f  train_acc: %.4f | "
            "val_loss: %.4f  val_acc: %.4f | "
            "lr: %.2e | %.1fs",
            stage, epoch, epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        if early_stopping(val_loss):
            logger.info("Early stopping at epoch %d (stage %d).", epoch, stage)
            break


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_training(config: Config) -> None:
    """Orchestrate the full two-stage training pipeline.

    1. Prepare the dataset (split + DataLoaders).
    2. Build the model.
    3. Stage 1 — frozen backbone, train head.
    4. Stage 2 — unfreeze top blocks, fine-tune.
    5. Save final model and training history.

    Args:
        config: Project configuration instance.
    """
    from SRC.data.splitter import split_dataset
    from SRC.data.dataset import create_dataloaders

    config.create_output_dirs()
    device = torch.device(config.device)

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    logger.info("Scanning dataset: %s", config.data_dir.resolve())
    train_data, val_data, test_data, class_to_idx = split_dataset(
        data_dir=config.data_dir,
        class_names=config.class_names,
        val_split=config.val_split,
        test_split=config.test_split,
        random_seed=config.random_seed,
    )

    # Persist class mapping
    with open(config.class_map_path, "w") as fh:
        json.dump(class_to_idx, fh, indent=2)
    logger.info("Class mapping saved: %s", config.class_map_path)

    loaders = create_dataloaders(train_data, val_data, test_data, config)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    logger.info("Building model: %s", config.backbone)
    model = BrainTumorClassifier(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        hidden_units=config.hidden_units,
        pretrained=config.pretrained,
        backbone=config.backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Shared history dict
    history: Dict[str, List] = {
        "stage": [], "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    checkpoint = ModelCheckpoint(config.best_model_path, mode="min")
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min")

    # ================================================================== #
    # Stage 1 — freeze backbone, train head only
    # ================================================================== #
    model.freeze_backbone()
    params_s1 = model.count_parameters()
    logger.info(
        "Stage 1 trainable params: %d / %d",
        params_s1["trainable"], params_s1["total"],
    )

    optimizer_s1 = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.stage1_lr,
    )
    scheduler_s1 = ReduceLROnPlateau(
        optimizer_s1,
        mode="min",
        patience=config.lr_scheduler_patience,
        factor=config.lr_scheduler_factor,
        min_lr=config.lr_scheduler_min_lr,
    )

    _train_stage(
        stage=1,
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer_s1,
        criterion=criterion,
        scheduler=scheduler_s1,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        epochs=config.stage1_epochs,
        device=device,
        use_amp=config.use_amp,
        history=history,
    )

    # ================================================================== #
    # Stage 2 — unfreeze top blocks, fine-tune
    # ================================================================== #
    model.unfreeze_top_blocks(n=config.unfreeze_last_n_blocks)
    params_s2 = model.count_parameters()
    logger.info(
        "Stage 2 trainable params: %d / %d",
        params_s2["trainable"], params_s2["total"],
    )

    optimizer_s2 = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.stage2_lr,
    )
    scheduler_s2 = ReduceLROnPlateau(
        optimizer_s2,
        mode="min",
        patience=config.lr_scheduler_patience,
        factor=config.lr_scheduler_factor,
        min_lr=config.lr_scheduler_min_lr,
    )

    _train_stage(
        stage=2,
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer_s2,
        criterion=criterion,
        scheduler=scheduler_s2,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        epochs=config.stage2_epochs,
        device=device,
        use_amp=config.use_amp,
        history=history,
    )

    # ================================================================== #
    # Save final model and history
    # ================================================================== #
    torch.save(model.state_dict(), config.final_model_path)
    logger.info("Final model saved: %s", config.final_model_path)

    with open(config.history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training history saved: %s", config.history_path)

    # Optional: evaluate on test set if it exists
    if "test" in loaders:
        test_loss, test_acc = _run_epoch(
            model, loaders["test"], criterion, device,
        )
        logger.info(
            "Test set — loss: %.4f | accuracy: %.4f", test_loss, test_acc
        )
