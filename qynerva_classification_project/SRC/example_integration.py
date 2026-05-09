"""
example_integration.py
-----------------------
Shows how your existing classifier feeds into the XAI module.

This file is for illustration only — it is NOT part of the XAI module.
Connected here to the real Predictor inside SRC.
"""

from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn

from SRC.config.config import Config
from SRC.prediction.predictor import Predictor
from SRC.utils.logger import setup_logging

from SRC.xai.hirescam import generate_hirescam
from SRC.xai.visualization import generate_overlay, save_overlay
# ---------------------------------------------------------------------------
# Real classifier call
# ---------------------------------------------------------------------------

def run_classifier(image_path: str) -> dict:
    """
    Runs your real classifier through Predictor and returns the exact
    dictionary shape required by the XAI module.

    Expected returned keys:
      - model
      - input_tensor
      - predicted_class
      - original_image
      - confidence
    """
    setup_logging(log_level=logging.INFO)

    config = Config(output_dir=Path("outputs"))

    predictor = Predictor(
        model_path=config.best_model_path,
        class_map_path=config.class_map_path,
        config=config,
    )

    return predictor.predict_image_with_xai_payload(Path(image_path))


# ---------------------------------------------------------------------------
# XAI integration
# ---------------------------------------------------------------------------

def explain_prediction(classifier_output: dict) -> dict:
    """
    Generate a HiResCAM explanation for a classifier prediction.

    Parameters
    ----------
    classifier_output : dict
        The dictionary returned by your classifier. Must contain:
          "model"          - nn.Module, already loaded and in eval mode
          "input_tensor"   - torch.Tensor, shape (1, C, H, W)
          "predicted_class"- int, class index from the classifier
          "original_image" - np.ndarray, float32 (H, W, 3) in [0, 1]

    Returns
    -------
    dict with keys:
      "cam_map"      - raw HiResCAM map, np.ndarray (H, W) float32 [0,1]
      "overlay_image"- blended overlay, np.ndarray (H, W, 3) float32 [0,1]
    """
    model: nn.Module = classifier_output["model"]
    input_tensor: torch.Tensor = classifier_output["input_tensor"]
    predicted_class: int = classifier_output["predicted_class"]
    original_image: np.ndarray = classifier_output["original_image"]

    cam_map = generate_hirescam(
        model=model,
        input_tensor=input_tensor,
        predicted_class=predicted_class,
        # target_layer=model.layer4[-1]  # Uncomment if you want to force a layer
    )

    overlay_image = generate_overlay(
        original_image=original_image,
        cam_map=cam_map,
        alpha=0.4,
    )

    return {
        "cam_map": cam_map,
        "overlay_image": overlay_image,
    }


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    IMAGE_PATH = "path/to/brain_mri.jpg"   # set your real image path

    # Step 1: run your existing classifier
    classifier_output = run_classifier(IMAGE_PATH)

    print(f"Predicted class : {classifier_output['predicted_class_name']}")
    print(f"Confidence      : {classifier_output['confidence']:.2%}")

    # Step 2: generate XAI explanation
    xai_output = explain_prediction(classifier_output)

    # Step 3: save the overlay
    output_path = Path("outputs/hirescam_overlay.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_overlay(
        overlay=xai_output["overlay_image"],
        output_path=str(output_path),
    )

    print(f"Overlay saved to {output_path}")

    # xai_output["cam_map"]       -> raw float32 (H, W) map
    # xai_output["overlay_image"] -> float32 (H, W, 3) image ready for display