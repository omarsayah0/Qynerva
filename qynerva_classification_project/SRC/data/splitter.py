"""
Dataset splitter.

Scans the raw `Data/` folder (which contains one sub-folder per class) and
returns stratified train / val / (optional) test splits as plain Python lists
of ``(image_path, label_index)`` tuples.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

SplitData = List[Tuple[Path, int]]


def _collect_images(data_dir: Path, class_names: list[str]) -> Tuple[List[Path], List[int], Dict[str, int]]:
    """Walk *data_dir* and collect all image paths with integer labels.

    Args:
        data_dir:     Path to the root data directory (e.g. ``Data/``).
        class_names:  Ordered list of expected class folder names.

    Returns:
        Tuple of (image_paths, labels, class_to_idx).

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError:        If no images are found.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            logger.warning("Expected class folder not found: %s", class_dir)
            continue

        found = 0
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
                found += 1

        logger.info("  %-25s  %d images", class_name, found)

    if not image_paths:
        raise ValueError(f"No images found in {data_dir.resolve()}")

    logger.info("Total images collected: %d", len(image_paths))
    return image_paths, labels, class_to_idx


def split_dataset(
    data_dir: Path,
    class_names: list[str],
    val_split: float = 0.15,
    test_split: float = 0.10,
    random_seed: int = 42,
) -> Tuple[SplitData, SplitData, SplitData, Dict[str, int]]:
    """Stratified split of the raw dataset into train / val / test.

    Args:
        data_dir:     Root data directory containing one folder per class.
        class_names:  Ordered list of class folder names.
        val_split:    Fraction of total data reserved for validation.
        test_split:   Fraction of total data reserved for test.
                      Set to ``0`` to skip the test split.
        random_seed:  Random seed for reproducibility.

    Returns:
        ``(train_data, val_data, test_data, class_to_idx)``

        Each *data* item is a list of ``(Path, label_index)`` tuples.
        *test_data* will be an empty list when *test_split* is ``0``.
    """
    image_paths, labels, class_to_idx = _collect_images(data_dir, class_names)

    # ------------------------------------------------------------------ #
    # First split off the test set (if requested)
    # ------------------------------------------------------------------ #
    if test_split > 0.0:
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths,
            labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels,
        )
    else:
        train_val_paths, train_val_labels = image_paths, labels
        test_paths, test_labels = [], []

    # ------------------------------------------------------------------ #
    # Split the remaining data into train / val
    # ------------------------------------------------------------------ #
    # val_split is expressed as a fraction of the *original* dataset, so we
    # must adjust it relative to the train+val subset size.
    effective_val_fraction = val_split / (1.0 - test_split) if test_split > 0.0 else val_split

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=effective_val_fraction,
        random_state=random_seed,
        stratify=train_val_labels,
    )

    train_data: SplitData = list(zip(train_paths, train_labels))
    val_data: SplitData = list(zip(val_paths, val_labels))
    test_data: SplitData = list(zip(test_paths, test_labels))

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_data),
        len(val_data),
        len(test_data),
    )

    return train_data, val_data, test_data, class_to_idx
