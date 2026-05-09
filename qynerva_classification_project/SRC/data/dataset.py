"""
PyTorch Dataset and DataLoader factories for the brain-tumor classification task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from SRC.config.config import Config
from SRC.data.splitter import SplitData


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class BrainTumorDataset(Dataset):
    """Loads MRI images from a pre-split list of ``(path, label)`` pairs.

    Args:
        samples:   List of ``(image_path, label_index)`` tuples.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(self, samples: SplitData, transform: Optional[transforms.Compose] = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_train_transform(config: Config) -> transforms.Compose:
    """Light augmentation pipeline for training (conservative for medical data)."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def get_eval_transform(config: Config) -> transforms.Compose:
    """Deterministic pipeline for validation, test, and prediction."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def create_dataloaders(
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    config: Config,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for each split.

    Args:
        train_data: List of ``(path, label)`` for training.
        val_data:   List of ``(path, label)`` for validation.
        test_data:  List of ``(path, label)`` for test (may be empty).
        config:     Project configuration.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, and optionally ``"test"``.
    """
    train_transform = get_train_transform(config)
    eval_transform = get_eval_transform(config)

    loaders: Dict[str, DataLoader] = {}

    loaders["train"] = DataLoader(
        BrainTumorDataset(train_data, transform=train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    loaders["val"] = DataLoader(
        BrainTumorDataset(val_data, transform=eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    if test_data:
        loaders["test"] = DataLoader(
            BrainTumorDataset(test_data, transform=eval_transform),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return loaders
