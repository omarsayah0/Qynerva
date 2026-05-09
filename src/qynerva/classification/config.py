from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    data_dir: Path = Path("Data")
    output_dir: Path = Path("outputs/classification")

    image_size: int = 300

    num_classes: int = 4
    class_names: list = field(default_factory=lambda: [
        "glioma_tumor",
        "meningioma_tumor",
        "normal",
        "pituitary_tumor",
    ])

    val_split: float = 0.15
    test_split: float = 0.10
    random_seed: int = 42

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    stage1_epochs: int = 10
    stage1_lr: float = 1e-3

    stage2_epochs: int = 20
    stage2_lr: float = 1e-5
    unfreeze_last_n_blocks: int = 3

    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-7

    early_stopping_patience: int = 7

    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    dropout_rate: float = 0.3
    hidden_units: int = 256

    normalize_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def history_path(self) -> Path:
        return self.model_dir / "training_history.json"

    @property
    def class_map_path(self) -> Path:
        return self.model_dir / "class_to_idx.json"

    @property
    def best_model_path(self) -> Path:
        return self.model_dir / "best_model.pth"

    @property
    def final_model_path(self) -> Path:
        return self.model_dir / "final_model.pth"

    def create_output_dirs(self) -> None:
        for d in (self.model_dir, self.logs_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
