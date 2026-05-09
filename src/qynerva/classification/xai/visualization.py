from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_overlay(
    original_image: np.ndarray,
    cam_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    original_image = _validate_image(original_image)
    cam_map = _resize_cam_if_needed(cam_map, target_hw=original_image.shape[:2])
    overlay = show_cam_on_image(img=original_image, mask=cam_map, use_rgb=True, colormap=colormap, image_weight=1.0 - alpha)
    return overlay.astype(np.float32) / 255.0


def save_overlay(overlay: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_uint8 = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))


def _validate_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"original_image must have shape (H, W, 3), got {image.shape}.")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def _resize_cam_if_needed(cam_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h_target, w_target = target_hw
    if cam_map.shape == (h_target, w_target):
        return cam_map
    return cv2.resize(cam_map, (w_target, h_target), interpolation=cv2.INTER_LINEAR).astype(np.float32)
