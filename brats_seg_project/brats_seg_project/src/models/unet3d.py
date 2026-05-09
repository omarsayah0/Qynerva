from __future__ import annotations

from monai.networks.nets import UNet


def build_model(config: dict):
    model_cfg = config["model"]
    return UNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        num_res_units=model_cfg["num_res_units"],
        dropout=model_cfg["dropout"],
    )
