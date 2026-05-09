from __future__ import annotations

from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from utils.metrics import SegmentationMetrics


class Trainer:
    def __init__(self, model, criterion, optimizer, device: torch.device, amp: bool, save_dir: str | Path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.amp = amp
        self.metrics = SegmentationMetrics()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dice = -1.0

    def train_epoch(self, loader) -> float:
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(loader, desc="train", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item()

        return running_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader, patch_size: tuple[int, int, int]) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0

        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = sliding_window_inference(images, roi_size=patch_size, sw_batch_size=1, predictor=self.model)
                loss = self.criterion(logits, masks)

            dice = self.metrics(logits, masks)
            total_loss += loss.item()
            total_dice += dice

        mean_loss = total_loss / max(len(loader), 1)
        mean_dice = total_dice / max(len(loader), 1)
        return mean_loss, mean_dice

    def save_best(self, epoch: int, val_dice: float, checkpoint_name: str) -> None:
        if val_dice <= self.best_dice:
            return
        self.best_dice = val_dice
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_dice": val_dice,
            },
            self.save_dir / checkpoint_name,
        )
