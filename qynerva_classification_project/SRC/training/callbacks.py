"""
Training callbacks: early stopping and model checkpointing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Early Stopping
# --------------------------------------------------------------------------- #

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:  Number of epochs with no improvement before stopping.
        min_delta: Minimum change in the monitored metric to count as improvement.
        mode:      ``"min"`` (lower is better, e.g. loss) or
                   ``"max"`` (higher is better, e.g. accuracy).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._counter: int = 0
        self._best: Optional[float] = None
        self.triggered: bool = False

    def __call__(self, value: float) -> bool:
        """Update state with the latest metric value.

        Args:
            value: Current epoch metric (loss or accuracy).

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self._best is None:
            self._best = value
            return False

        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )

        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d / %d epochs.",
                self._counter,
                self.patience,
            )
            if self._counter >= self.patience:
                self.triggered = True
                logger.info("Early stopping triggered after %d epochs.", self.patience)
                return True

        return False

    def reset(self) -> None:
        """Reset state (useful between stages)."""
        self._counter = 0
        self._best = None
        self.triggered = False


# --------------------------------------------------------------------------- #
# Model Checkpoint
# --------------------------------------------------------------------------- #

class ModelCheckpoint:
    """Save the model whenever a monitored metric improves.

    Args:
        save_path:  File path for the saved checkpoint (``*.pth``).
        mode:       ``"min"`` or ``"max"``.
        verbose:    Whether to log when a new best is saved.
    """

    def __init__(
        self,
        save_path: Path,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.save_path = Path(save_path)
        self.mode = mode
        self.verbose = verbose

        self._best: Optional[float] = None

    def __call__(self, model: nn.Module, value: float) -> bool:
        """Check *value* and save *model* if it is a new best.

        Args:
            model: The model to save.
            value: Current epoch metric.

        Returns:
            ``True`` if the model was saved (new best), ``False`` otherwise.
        """
        improved = (
            self._best is None
            or (self.mode == "min" and value < self._best)
            or (self.mode == "max" and value > self._best)
        )

        if improved:
            self._best = value
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                logger.info(
                    "ModelCheckpoint: new best %.6f — saved to %s",
                    value,
                    self.save_path,
                )
            return True

        return False

    @property
    def best_value(self) -> Optional[float]:
        return self._best

    def reset(self) -> None:
        """Reset best-value tracker (useful between stages)."""
        self._best = None
