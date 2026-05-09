from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from qynerva.classification.config import Config
from qynerva.classification.data.dataset import get_eval_transform
from qynerva.classification.models.efficientnet import BrainTumorClassifier

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: Path, class_map_path: Path, config: Config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.transform = get_eval_transform(config)

        with open(class_map_path) as fh:
            class_to_idx: Dict[str, int] = json.load(fh)

        self.idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}
        self.class_names: List[str] = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.model = BrainTumorClassifier(
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            hidden_units=config.hidden_units,
            pretrained=False,
            backbone=config.backbone,
        )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Predictor ready — model: %s | device: %s", model_path.name, self.device)

    def predict_pil(self, pil_image: Image.Image) -> dict:
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        original_image = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0

        self.model.eval()
        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())

        return {
            "model": self.model,
            "input_tensor": input_tensor,
            "predicted_class": top_idx,
            "predicted_class_name": self.idx_to_class[top_idx],
            "original_image": original_image,
            "confidence": confidence,
            "class_probabilities": {self.idx_to_class[i]: float(probs[i].item()) for i in range(len(self.class_names))},
        }
