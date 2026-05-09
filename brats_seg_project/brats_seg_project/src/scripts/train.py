from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.dataset import create_dataloaders
from data.splits import make_splits
from losses.segmentation import DiceTverskyLoss
from models.unet3d import build_model
from engine.trainer import Trainer
from utils.config import load_config
from utils.io import discover_patients
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    samples = discover_patients(config["data"]["root_dir"])
    train_items, val_items, test_items = make_splits(
        samples=samples,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["seed"],
    )

    print(f"cases: total={len(samples)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    train_loader, val_loader, test_loader = create_dataloaders(config, train_items, val_items, test_items)

    requested_device = config["training"]["device"]
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    criterion = DiceTverskyLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        amp=config["training"]["amp"] and device.type == "cuda",
        save_dir=config["training"]["save_dir"],
    )

    ckpt_path = trainer.save_dir / "checkpoint.pt"
    start_epoch = 1
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")

    patch_size = tuple(config["data"]["patch_size"])
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_dice = trainer.evaluate(val_loader, patch_size)
        trainer.save_best(epoch, val_dice, config["training"]["checkpoint_name"])
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f}"
        )

    checkpoint_path = Path(config["training"]["save_dir"]) / config["training"]["checkpoint_name"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_dice = trainer.evaluate(test_loader, patch_size)
    print(f"best_val_dice={checkpoint['best_val_dice']:.4f} test_loss={test_loss:.4f} test_dice={test_dice:.4f}")


if __name__ == "__main__":
    main()
