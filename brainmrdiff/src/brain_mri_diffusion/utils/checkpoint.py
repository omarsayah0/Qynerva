"""Checkpoint save / load utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        step: int,
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        filename = f"step_{step:07d}.pt" if tag is None else f"{tag}.pt"
        path = self.checkpoint_dir / filename
        torch.save(state, path)
        logger.info(f"Checkpoint saved → {path}")

        latest = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest)

        if is_best:
            best = self.checkpoint_dir / "best.pt"
            torch.save(state, best)
            logger.info(f"Best checkpoint updated → {best}")

        return path

    def load(self, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if path is None:
            path = self.checkpoint_dir / "latest.pt"
        else:
            path = Path(path)

        if not path.exists():
            logger.info("No checkpoint found, starting from scratch.")
            return None

        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_best(self) -> Optional[Dict[str, Any]]:
        return self.load(str(self.checkpoint_dir / "best.pt"))
