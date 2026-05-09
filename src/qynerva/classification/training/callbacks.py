from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._counter: int = 0
        self._best: Optional[float] = None
        self.triggered: bool = False

    def __call__(self, value: float) -> bool:
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
            if self._counter >= self.patience:
                self.triggered = True
                return True
        return False

    def reset(self) -> None:
        self._counter = 0
        self._best = None
        self.triggered = False


class ModelCheckpoint:
    def __init__(self, save_path: Path, mode: str = "min", verbose: bool = True) -> None:
        self.save_path = Path(save_path)
        self.mode = mode
        self.verbose = verbose
        self._best: Optional[float] = None

    def __call__(self, model: nn.Module, value: float) -> bool:
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
                logger.info("New best %.6f — saved to %s", value, self.save_path)
            return True
        return False

    @property
    def best_value(self) -> Optional[float]:
        return self._best

    def reset(self) -> None:
        self._best = None
