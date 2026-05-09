"""
Qynerva — single entry point.

Usage
-----
    python run.py --input scan.nii.gz
    python run.py --input scan.nii.gz --seg-model outputs/segmentation/best_model.pt
    python run.py --input scan.nii.gz --no-display

See python run.py --help for all options.
"""

from qynerva.pipeline.runner import main

if __name__ == "__main__":
    main()
