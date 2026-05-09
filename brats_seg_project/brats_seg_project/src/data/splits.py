from __future__ import annotations

from sklearn.model_selection import train_test_split


def make_splits(samples: list[dict], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    indices = list(range(len(samples)))
    train_idx, temp_idx = train_test_split(indices, test_size=val_ratio + test_ratio, random_state=seed, shuffle=True)

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=relative_test_ratio, random_state=seed, shuffle=True)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples
