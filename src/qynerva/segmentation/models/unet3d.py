from __future__ import annotations

from monai.networks.nets import UNet


def build_model(config: dict):
    m = config["model"]
    return UNet(
        spatial_dims=3,
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m["num_res_units"],
        dropout=m["dropout"],
    )
