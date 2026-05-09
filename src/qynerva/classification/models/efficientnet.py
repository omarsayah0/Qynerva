from __future__ import annotations

import logging
from typing import List

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BrainTumorClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        hidden_units: int = 256,
        pretrained: bool = True,
        backbone: str = "efficientnet_b3",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg",
        )
        in_features: int = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, n: int = 3) -> None:
        all_blocks: List[nn.Module] = list(self.backbone.blocks.children())
        for block in (all_blocks[-n:] if n <= len(all_blocks) else all_blocks):
            for param in block.parameters():
                param.requires_grad = True
        for attr in ("conv_head", "bn2"):
            layer = getattr(self.backbone, attr, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}
