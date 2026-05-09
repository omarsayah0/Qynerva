"""Evaluation metrics: PSNR, SSIM, Dice score."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor

logger = logging.getLogger(__name__)


def _to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def compute_psnr(pred: Tensor, target: Tensor, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    psnr_vals = []
    for p, t in zip(pred_np, target_np):
        p = p.squeeze()
        t = t.squeeze()
        # Normalize to [0, 1]
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        psnr_vals.append(peak_signal_noise_ratio(t, p, data_range=data_range))
    return float(np.mean(psnr_vals))


def compute_ssim(pred: Tensor, target: Tensor, data_range: float = 1.0) -> float:
    """Structural Similarity Index (higher is better)."""
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    ssim_vals = []
    for p, t in zip(pred_np, target_np):
        p = p.squeeze()
        t = t.squeeze()
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        ssim_vals.append(structural_similarity(t, p, data_range=data_range))
    return float(np.mean(ssim_vals))


def compute_dice(
    pred_mask: Tensor,
    target_mask: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Dice coefficient for binary segmentation masks (higher is better)."""
    pred_bin = (pred_mask > threshold).float()
    target_bin = (target_mask > threshold).float()
    intersection = (pred_bin * target_bin).sum()
    dice = (2.0 * intersection + smooth) / (pred_bin.sum() + target_bin.sum() + smooth)
    return float(dice.item())


def evaluate_batch(
    pred: Tensor,
    target: Tensor,
    pred_mask: Optional[Tensor] = None,
    target_mask: Optional[Tensor] = None,
) -> Dict[str, float]:
    """Compute all metrics for a batch, return mean values."""
    results: Dict[str, float] = {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
    }
    if pred_mask is not None and target_mask is not None:
        results["dice"] = compute_dice(pred_mask, target_mask)
    return results


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of per-batch metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
