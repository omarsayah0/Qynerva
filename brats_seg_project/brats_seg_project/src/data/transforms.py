from __future__ import annotations

from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    SpatialPadd,
)


def get_train_transforms(patch_size: tuple[int, int, int], samples_per_volume: int):
    return Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # BraTS labels are 0,1,2,4 — remap 4→3 so one-hot size 4 works
            MapLabelValued(keys="mask", orig_labels=[4], target_labels=[3]),
            AsDiscreted(keys="mask", to_onehot=4),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            SpatialPadd(keys=["image", "mask"], spatial_size=patch_size),
            RandSpatialCropSamplesd(
                keys=["image", "mask"],
                roi_size=patch_size,
                num_samples=samples_per_volume,
                random_center=True,
                random_size=False,
            ),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )


def get_eval_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # BraTS labels are 0,1,2,4 — remap 4→3 so one-hot size 4 works
            MapLabelValued(keys="mask", orig_labels=[4], target_labels=[3]),
            AsDiscreted(keys="mask", to_onehot=4),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )


def get_predict_transforms():
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
