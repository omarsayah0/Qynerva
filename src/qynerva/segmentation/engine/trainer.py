from __future__ import annotations

import logging
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from qynerva.segmentation.utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


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
            total_loss += loss.item()
            total_dice += self.metrics(logits, masks)
        return total_loss / max(len(loader), 1), total_dice / max(len(loader), 1)

    def save_best(self, epoch: int, val_dice: float, checkpoint_name: str) -> None:
        if val_dice <= self.best_dice:
            return
        self.best_dice = val_dice
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "best_val_dice": val_dice}, self.save_dir / checkpoint_name)


def main_cli() -> None:
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train 3D U-Net brain tumor segmentation.")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml")
    args = parser.parse_args()

    from qynerva.segmentation.utils.config import load_config
    from qynerva.segmentation.utils.io import discover_patients
    from qynerva.segmentation.utils.seed import set_seed
    from qynerva.segmentation.data.splits import make_splits
    from qynerva.segmentation.data.dataset import create_dataloaders
    from qynerva.segmentation.models.unet3d import build_model
    from qynerva.segmentation.losses.segmentation import DiceTverskyLoss

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    config = load_config(args.config)
    set_seed(config["seed"])

    samples = discover_patients(config["data"]["root_dir"])
    train_items, val_items, test_items = make_splits(samples, config["data"]["val_ratio"], config["data"]["test_ratio"], config["seed"])
    print(f"cases: total={len(samples)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    train_loader, val_loader, test_loader = create_dataloaders(config, train_items, val_items, test_items)

    device = torch.device("cuda" if config["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    criterion = DiceTverskyLoss(**config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, amp=config["training"]["amp"] and device.type == "cuda", save_dir=config["training"]["save_dir"])

    patch_size = tuple(config["data"]["patch_size"])
    ckpt_path = trainer.save_dir / "checkpoint.pt"
    start_epoch = 1

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_dice = trainer.evaluate(val_loader, patch_size)
        trainer.save_best(epoch, val_dice, config["training"]["checkpoint_name"])
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

    ckpt = torch.load(Path(config["training"]["save_dir"]) / config["training"]["checkpoint_name"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_dice = trainer.evaluate(test_loader, patch_size)
    print(f"best_val_dice={ckpt['best_val_dice']:.4f} test_loss={test_loss:.4f} test_dice={test_dice:.4f}")


if __name__ == "__main__":
    main_cli()
