"""
Train the 3D U-Net segmentation model.

Usage
-----
    python train_segmentation.py --config configs/segmentation.yaml
    python train_segmentation.py --help

Edit configs/segmentation.yaml first — set data.root_dir to your BraTS folder.
"""

from qynerva.segmentation.engine.trainer import main_cli

if __name__ == "__main__":
    main_cli()
