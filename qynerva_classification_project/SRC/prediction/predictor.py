"""
Prediction pipeline for single images and folders.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from SRC.config.config import Config
from SRC.data.dataset import get_eval_transform
from SRC.models.efficientnet import BrainTumorClassifier

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class PredictionResult:
    """Container for a single prediction."""

    __slots__ = ("image_name", "predicted_class", "confidence", "class_probabilities")

    def __init__(
        self,
        image_name: str,
        predicted_class: str,
        confidence: float,
        class_probabilities: Dict[str, float],
    ) -> None:
        self.image_name = image_name
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.class_probabilities = class_probabilities

    def __repr__(self) -> str:
        return (
            f"PredictionResult(image={self.image_name!r}, "
            f"class={self.predicted_class!r}, "
            f"confidence={self.confidence:.4f})"
        )

    def as_dict(self) -> dict:
        return {
            "image_name": self.image_name,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            **{f"prob_{k}": round(v, 6) for k, v in self.class_probabilities.items()},
        }


class Predictor:
    """Load a trained model and run inference on images.

    Args:
        model_path:     Path to a ``.pth`` model state-dict file.
        class_map_path: Path to the ``class_to_idx.json`` produced during training.
        config:         Project configuration (used for image transforms and device).
    """

    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        config: Config,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.transform = get_eval_transform(config)

        # Load class mapping  {class_name: index}
        with open(class_map_path) as fh:
            class_to_idx: Dict[str, int] = json.load(fh)

        # Invert to {index: class_name}
        self.idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}
        self.class_names: List[str] = [
            self.idx_to_class[i] for i in range(len(self.idx_to_class))
        ]

        # Build and load model
        self.model = BrainTumorClassifier(
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            hidden_units=config.hidden_units,
            pretrained=False,   # weights come from the checkpoint
            backbone=config.backbone,
        )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Predictor ready — model: %s | classes: %s | device: %s",
            model_path.name,
            self.class_names,
            self.device,
        )

    # ------------------------------------------------------------------ #
    # Core inference
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a single image into a (1, C, H, W) tensor."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)          # (C, H, W)
        return tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

    def predict_image(self, image_path: Path | str) -> PredictionResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            A :class:`PredictionResult` with the predicted class and confidence.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        tensor = self._preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor)           # (1, num_classes)
            probs = F.softmax(logits, dim=1)[0]   # (num_classes,)

        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()
        predicted_class = self.idx_to_class[top_idx]

        class_probabilities = {
            self.idx_to_class[i]: probs[i].item()
            for i in range(len(self.class_names))
        }

        return PredictionResult(
            image_name=image_path.name,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
        )

    # ------------------------------------------------------------------ #
    # Folder prediction
    # ------------------------------------------------------------------ #

    def predict_folder(
        self,
        folder_path: Path | str,
        save_csv: Optional[Path] = None,
        save_json: Optional[Path] = None,
    ) -> List[PredictionResult]:
        """Run inference on every image in *folder_path*.

        Args:
            folder_path: Directory containing image files.
            save_csv:    Optional path to write results as CSV.
            save_json:   Optional path to write results as JSON.

        Returns:
            List of :class:`PredictionResult` instances (one per image).
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        image_files = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("No images found in %s", folder_path)
            return []

        logger.info("Running prediction on %d images in %s", len(image_files), folder_path)

        results: List[PredictionResult] = []
        for img_path in image_files:
            try:
                result = self.predict_image(img_path)
                results.append(result)
                logger.info(
                    "  %-40s  ->  %-25s  (%.2f%%)",
                    result.image_name,
                    result.predicted_class,
                    result.confidence * 100,
                )
            except Exception as exc:
                logger.warning("Skipping %s — %s", img_path.name, exc)

        if save_csv and results:
            _save_csv(results, save_csv)
            logger.info("Results saved to CSV: %s", save_csv)

        if save_json and results:
            _save_json(results, save_json)
            logger.info("Results saved to JSON: %s", save_json)

        return results
    
    def predict_image_with_xai_payload(self, image_path: Path | str) -> dict:
        """
        Run inference on a single image and return everything needed for XAI.

        Returns
        -------
        dict
         {
                "model": nn.Module,
                "input_tensor": torch.Tensor,        # (1, C, H, W)
                "predicted_class": int,              # class index
                "predicted_class_name": str,         # class label
                "original_image": np.ndarray,        # (H, W, 3) float32 in [0,1]
                "confidence": float,
                "image_name": str,
                "class_probabilities": dict[str, float],
            }   
        """ 
        import numpy as np
    
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    
        # Original image for overlay
        image = Image.open(image_path).convert("RGB")
        original_image = np.asarray(image, dtype=np.float32) / 255.0
    
        # Preprocessed tensor for the model
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
    
        # IMPORTANT: keep gradients enabled for CAM methods
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(input_tensor)         # (1, num_classes)
            probs = F.softmax(logits, dim=1)[0]       # (num_classes,)
    
        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        predicted_class_name = self.idx_to_class[top_idx]
    
        class_probabilities = {
            self.idx_to_class[i]: float(probs[i].item())
            for i in range(len(self.class_names))
        }
    
        return {
            "model": self.model,
            "input_tensor": input_tensor,
            "predicted_class": top_idx,                  # index required by XAI
            "predicted_class_name": predicted_class_name,
            "original_image": original_image,
            "confidence": confidence,
            "image_name": image_path.name,
            "class_probabilities": class_probabilities,
        }


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def _save_csv(results: List[PredictionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].as_dict().keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(r.as_dict() for r in results)


def _save_json(results: List[PredictionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump([r.as_dict() for r in results], fh, indent=2)
