
import torch
import torch.nn as nn
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import numpy as np


# ---------------------------------------------------------------------------
# Target-layer selection
# ---------------------------------------------------------------------------

def get_target_layer(model: nn.Module) -> nn.Module:
   
    for _, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.Conv2d):
            return layer

    raise ValueError(
        "No nn.Conv2d layer found in the model. "
        "Please supply the target layer explicitly via the "
        "`target_layer` parameter of `generate_hirescam`."
    )


# ---------------------------------------------------------------------------
# CAM generation
# ---------------------------------------------------------------------------

def generate_hirescam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    predicted_class: int,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    
    if target_layer is None:
        target_layer = get_target_layer(model)

    targets = [ClassifierOutputTarget(predicted_class)]

    with HiResCAM(model=model, target_layers=[target_layer]) as cam:
        # cam() returns shape (batch, H, W); we take the first (and only) item.
        cam_map: np.ndarray = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]  # shape: (H, W), float32 in [0, 1]

    return cam_map
