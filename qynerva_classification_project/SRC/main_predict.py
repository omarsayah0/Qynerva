"""
CLI entry point for prediction.

Usage
-----
Single image:
    qynerva_classification_predict --image path/to/image.jpg

Folder (batch):
    qynerva_classification_predict --folder path/to/folder [--save-csv out.csv] [--save-json out.json]

Options
-------
    --image      PATH   Single image to classify.
    --folder     PATH   Folder of images to classify.
    --model      PATH   Path to .pth checkpoint         (default: outputs/models/best_model.pth)
    --class-map  PATH   Path to class_to_idx.json       (default: outputs/models/class_to_idx.json)
    --output-dir PATH   Base output dir for defaults    (default: outputs)
    --save-csv   PATH   Save folder-mode results to CSV.
    --save-json  PATH   Save folder-mode results to JSON.
    --device     STR    Force device: "cpu" | "cuda".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from SRC.config.config import Config
from SRC.prediction.predictor import Predictor
from SRC.utils.logger import setup_logging

from SRC.xai.hirescam import generate_hirescam
from SRC.xai.visualization import generate_overlay, save_overlay


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qynerva_classification_predict",
        description="Run brain-tumor MRI classification inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — exactly one of --image or --folder is required
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, metavar="PATH",
                      help="Path to a single image file.")
    mode.add_argument("--folder", type=Path, metavar="PATH",
                      help="Path to a folder of images.")

    # Model / class-map
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Base output directory (used to derive default model/class-map paths).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to .pth model checkpoint. "
                             "Defaults to <output-dir>/models/best_model.pth")
    parser.add_argument("--class-map", type=Path, default=None,
                        help="Path to class_to_idx.json. "
                             "Defaults to <output-dir>/models/class_to_idx.json")

    # Output files (folder mode only)
    parser.add_argument("--save-csv", type=Path, default=None, metavar="PATH",
                        help="Save results to a CSV file (folder mode).")
    parser.add_argument("--save-json", type=Path, default=None, metavar="PATH",
                        help="Save results to a JSON file (folder mode).")

    # Misc
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'. Auto-detects if omitted.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = Config(output_dir=args.output_dir)
    if args.device is not None:
        config.device = args.device

    # Resolve model and class-map paths
    model_path: Path = args.model if args.model else config.best_model_path
    class_map_path: Path = args.class_map if args.class_map else config.class_map_path

    # Validate paths before loading
    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)
    if not class_map_path.exists():
        logger.error("Class map not found: %s", class_map_path)
        sys.exit(1)

    predictor = Predictor(
        model_path=model_path,
        class_map_path=class_map_path,
        config=config,
    )

    # ------------------------------------------------------------------ #
    # Single image mode
    # ------------------------------------------------------------------ #
    if args.image is not None:
        classifier_output = predictor.predict_image_with_xai_payload(args.image)

        print("\n" + "=" * 55)
        print(f"  Image          : {classifier_output['image_name']}")
        print(f"  Predicted class: {classifier_output['predicted_class_name']}")
        print(f"  Confidence     : {classifier_output['confidence'] * 100:.2f}%")
        print("-" * 55)
        print("  Class probabilities:")
        for cls_name, prob in sorted(
            classifier_output["class_probabilities"].items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(prob * 30)
            print(f"    {cls_name:<28}  {prob * 100:6.2f}%  {bar}")
        print("=" * 55 + "\n")

        # Generate HiResCAM explanation
        cam_map = generate_hirescam(
            model=classifier_output["model"],
            input_tensor=classifier_output["input_tensor"],
            predicted_class=classifier_output["predicted_class"],
        )

        overlay_image = generate_overlay(
            original_image=classifier_output["original_image"],
            cam_map=cam_map,
            alpha=0.4,
        )

        # Save overlay
        xai_output_dir = config.output_dir / "xai"
        xai_output_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = xai_output_dir / f"{Path(classifier_output['image_name']).stem}_hirescam_overlay.png"

        save_overlay(
            overlay=overlay_image,
            output_path=str(overlay_path),
        )

        print(f"Overlay saved to: {overlay_path}")

    # ------------------------------------------------------------------ #
    # Folder mode
    # ------------------------------------------------------------------ #
    else:
        results = predictor.predict_folder(
            folder_path=args.folder,
            save_csv=args.save_csv,
            save_json=args.save_json,
        )

        if not results:
            logger.warning("No results produced.")
            return

        print("\n" + "=" * 75)
        print(f"{'Image':<40}  {'Predicted Class':<25}  {'Confidence':>10}")
        print("-" * 75)
        for r in results:
            print(f"{r.image_name:<40}  {r.predicted_class:<25}  {r.confidence * 100:>9.2f}%")
        print("=" * 75)
        print(f"Total: {len(results)} images processed.\n")

        if args.save_csv:
            print(f"Results saved to CSV : {args.save_csv}")
        if args.save_json:
            print(f"Results saved to JSON: {args.save_json}")


if __name__ == "__main__":
    main(sys.argv[1:])