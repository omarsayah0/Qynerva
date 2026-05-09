from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model: nn.Module) -> nn.Module:
    for _, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError("No nn.Conv2d layer found in the model.")


def generate_hirescam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    predicted_class: int,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    if target_layer is None:
        target_layer = get_target_layer(model)

    with HiResCAM(model=model, target_layers=[target_layer]) as cam:
        cam_map: np.ndarray = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(predicted_class)],
        )[0]

    return cam_map
