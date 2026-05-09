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

from qynerva.classification.config import Config
from qynerva.classification.models.efficientnet import BrainTumorClassifier
from qynerva.classification.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
) -> Tuple[float, float]:
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


def _train_stage(
    stage: int,
    model: BrainTumorClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    early_stopping: EarlyStopping,
    checkpoint: ModelCheckpoint,
    epochs: int,
    device: torch.device,
    use_amp: bool,
    history: Dict[str, List],
) -> None:
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None
    effective_amp = use_amp and device.type == "cuda"
    early_stopping.reset()

    logger.info("=" * 60)
    logger.info("Stage %d — starting (%d epochs max)", stage, epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, use_amp=effective_amp)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        checkpoint(model, val_loss)

        history["stage"].append(stage)
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            "Stage %d | Epoch %3d/%d | train_loss: %.4f  train_acc: %.4f | val_loss: %.4f  val_acc: %.4f | lr: %.2e | %.1fs",
            stage, epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"], time.perf_counter() - t0,
        )

        if early_stopping(val_loss):
            logger.info("Early stopping at epoch %d.", epoch)
            break


def run_training(config: Config) -> None:
    from qynerva.classification.data.splitter import split_dataset
    from qynerva.classification.data.dataset import create_dataloaders

    config.create_output_dirs()
    device = torch.device(config.device)

    logger.info("Scanning dataset: %s", config.data_dir.resolve())
    train_data, val_data, test_data, class_to_idx = split_dataset(
        data_dir=config.data_dir,
        class_names=config.class_names,
        val_split=config.val_split,
        test_split=config.test_split,
        random_seed=config.random_seed,
    )

    with open(config.class_map_path, "w") as fh:
        json.dump(class_to_idx, fh, indent=2)

    loaders = create_dataloaders(train_data, val_data, test_data, config)

    model = BrainTumorClassifier(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        hidden_units=config.hidden_units,
        pretrained=config.pretrained,
        backbone=config.backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List] = {"stage": [], "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    checkpoint = ModelCheckpoint(config.best_model_path, mode="min")
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min")

    # Stage 1 — frozen backbone
    model.freeze_backbone()
    optimizer_s1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.stage1_lr)
    scheduler_s1 = ReduceLROnPlateau(optimizer_s1, mode="min", patience=config.lr_scheduler_patience, factor=config.lr_scheduler_factor, min_lr=config.lr_scheduler_min_lr)
    _train_stage(1, model, loaders["train"], loaders["val"], optimizer_s1, criterion, scheduler_s1, early_stopping, checkpoint, config.stage1_epochs, device, config.use_amp, history)

    # Stage 2 — partial fine-tune
    model.unfreeze_top_blocks(n=config.unfreeze_last_n_blocks)
    optimizer_s2 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.stage2_lr)
    scheduler_s2 = ReduceLROnPlateau(optimizer_s2, mode="min", patience=config.lr_scheduler_patience, factor=config.lr_scheduler_factor, min_lr=config.lr_scheduler_min_lr)
    _train_stage(2, model, loaders["train"], loaders["val"], optimizer_s2, criterion, scheduler_s2, early_stopping, checkpoint, config.stage2_epochs, device, config.use_amp, history)

    torch.save(model.state_dict(), config.final_model_path)
    with open(config.history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training complete. Outputs: %s", config.output_dir.resolve())

    if "test" in loaders:
        test_loss, test_acc = _run_epoch(model, loaders["test"], criterion, device)
        logger.info("Test set — loss: %.4f | accuracy: %.4f", test_loss, test_acc)


def main_cli() -> None:
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Train EfficientNetB3 brain-tumor classifier.")
    parser.add_argument("--data-dir", type=str, default="Data")
    parser.add_argument("--output-dir", type=str, default="outputs/classification")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        random_seed=args.seed,
    )
    if args.device:
        config.device = args.device

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    run_training(config)


if __name__ == "__main__":
    main_cli()
