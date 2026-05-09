"""
Train the EfficientNetB3 classification model.

Usage
-----
    python train_classification.py --data-dir Data --output-dir outputs/classification
    python train_classification.py --help
"""

from qynerva.classification.training.trainer import main_cli

if __name__ == "__main__":
    main_cli()
