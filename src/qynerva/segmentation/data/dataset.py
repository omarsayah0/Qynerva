from __future__ import annotations

from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

from qynerva.segmentation.data.transforms import get_eval_transforms, get_train_transforms


def _build_dataset(items, transform, cache_rate: float):
    if cache_rate > 0:
        return CacheDataset(data=items, transform=transform, cache_rate=cache_rate, num_workers=0)
    return Dataset(data=items, transform=transform)


def create_dataloaders(config: dict, train_items, val_items, test_items):
    patch_size = tuple(config["data"]["patch_size"])
    samples_per_volume = config["data"]["samples_per_volume"]
    cache_rate = config["data"]["cache_rate"]
    num_workers = config["data"]["num_workers"]
    batch_size = config["data"]["batch_size"]

    train_ds = _build_dataset(train_items, get_train_transforms(patch_size, samples_per_volume), cache_rate)
    val_ds = _build_dataset(val_items, get_eval_transforms(), cache_rate)
    test_ds = _build_dataset(test_items, get_eval_transforms(), cache_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=list_data_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
