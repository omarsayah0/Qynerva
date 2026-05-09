"""
EfficientNetB3-based classifier for brain-tumor MRI classification.

Architecture:
    Backbone  : EfficientNetB3 (pretrained via timm, global-average-pool output)
    Head      : Dropout → Linear(hidden) → ReLU → Dropout → Linear(num_classes)
"""

from __future__ import annotations

import logging
from typing import List

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BrainTumorClassifier(nn.Module):
    """EfficientNetB3 with a custom classification head.

    Args:
        num_classes:  Number of output classes.
        dropout_rate: Dropout probability applied before each linear layer.
        hidden_units: Width of the intermediate fully-connected layer.
        pretrained:   Whether to load ImageNet pre-trained weights.
        backbone:     timm model name (default: ``"efficientnet_b3"``).
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        hidden_units: int = 256,
        pretrained: bool = True,
        backbone: str = "efficientnet_b3",
    ) -> None:
        super().__init__()

        # Load backbone with the pooling layer but without the original head.
        # num_classes=0 tells timm to return pooled features directly.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,       # removes the original classifier
            global_pool="avg",   # keep global average pooling
        )

        in_features: int = self.backbone.num_features  # 1536 for EfficientNetB3

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_units, num_classes),
        )

        logger.info(
            "BrainTumorClassifier — backbone: %s | in_features: %d | num_classes: %d",
            backbone,
            in_features,
            num_classes,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, in_features)
        return self.classifier(features)  # (B, num_classes)

    # ------------------------------------------------------------------ #
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------ #

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Stage 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only classifier head will be trained.")

    def unfreeze_top_blocks(self, n: int = 3) -> None:
        """Unfreeze the last *n* EfficientNet blocks plus the head conv layers.

        Lower-level blocks remain frozen to preserve low-level feature
        representations that generalise well across medical images.

        Args:
            n: Number of top-most block groups to unfreeze.
        """
        # Unfreeze top-n blocks
        all_blocks: List[nn.Module] = list(self.backbone.blocks.children())
        blocks_to_unfreeze = all_blocks[-n:] if n <= len(all_blocks) else all_blocks

        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze conv_head and its batch-norm
        for attr in ("conv_head", "bn2"):
            layer = getattr(self.backbone, attr, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Unfroze top-%d blocks — trainable params: %d / %d (%.1f%%)",
            n,
            trainable,
            total,
            100.0 * trainable / total,
        )

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter in the model."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen.")

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> dict:
        """Return a dict with trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}
