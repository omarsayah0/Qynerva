from __future__ import annotations

import torch
import torch.nn as nn


class DiceTverskyLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, tversky_weight: float = 0.3, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * targets, dim=dims)
        pred_sum = torch.sum(probs, dim=dims)
        target_sum = torch.sum(targets, dim=dims)
        false_neg = torch.sum((1.0 - probs) * targets, dim=dims)
        false_pos = torch.sum(probs * (1.0 - targets), dim=dims)

        dice = 1.0 - ((2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)).mean()
        tversky = 1.0 - ((intersection + self.eps) / (intersection + self.alpha * false_neg + self.beta * false_pos + self.eps)).mean()
        return self.dice_weight * dice + self.tversky_weight * tversky
