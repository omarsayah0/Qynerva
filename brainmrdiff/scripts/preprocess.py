#!/usr/bin/env python3
"""Convenience wrapper – run from project root: python scripts/preprocess.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_mri_diffusion.scripts.preprocess import main

if __name__ == "__main__":
    main()
