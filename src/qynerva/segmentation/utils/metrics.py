from __future__ import annotations

import torch
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose


class SegmentationMetrics:
    def __init__(self) -> None:
        self.metric = DiceMetric(include_background=True, reduction="mean")
        self.post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([AsDiscrete(argmax=True, to_onehot=4)])

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = [self.post_pred(x) for x in logits]
        gts = [self.post_label(x) for x in labels]
        self.metric.reset()
        self.metric(preds, gts)
        value = self.metric.aggregate().item()
        self.metric.reset()
        return float(value)
