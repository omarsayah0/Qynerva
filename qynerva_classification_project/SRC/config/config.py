"""Central configuration for the qynerva_classification project."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    data_dir: Path = Path("Data")
    output_dir: Path = Path("outputs")

    # ------------------------------------------------------------------ #
    # Image settings
    # ------------------------------------------------------------------ #
    image_size: int = 300

    # ------------------------------------------------------------------ #
    # Classes
    # ------------------------------------------------------------------ #
    num_classes: int = 4
    class_names: list = field(default_factory=lambda: [
        "glioma_tumor",
        "meningioma_tumor",
        "normal",
        "pituitary_tumor",
    ])

    # ------------------------------------------------------------------ #
    # Dataset split
    # ------------------------------------------------------------------ #
    val_split: float = 0.15    # fraction of total data used for validation
    test_split: float = 0.10   # fraction of total data used for test (0 = no test set)
    random_seed: int = 42

    # ------------------------------------------------------------------ #
    # DataLoader
    # ------------------------------------------------------------------ #
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # ------------------------------------------------------------------ #
    # Stage 1 — backbone frozen, only head trains
    # ------------------------------------------------------------------ #
    stage1_epochs: int = 10
    stage1_lr: float = 1e-3

    # ------------------------------------------------------------------ #
    # Stage 2 — partial fine-tuning of backbone
    # ------------------------------------------------------------------ #
    stage2_epochs: int = 20
    stage2_lr: float = 1e-5
    unfreeze_last_n_blocks: int = 3   # how many EfficientNet blocks to unfreeze

    # ------------------------------------------------------------------ #
    # Learning-rate scheduler  (ReduceLROnPlateau)
    # ------------------------------------------------------------------ #
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-7

    # ------------------------------------------------------------------ #
    # Early stopping
    # ------------------------------------------------------------------ #
    early_stopping_patience: int = 7

    # ------------------------------------------------------------------ #
    # Model architecture
    # ------------------------------------------------------------------ #
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    dropout_rate: float = 0.3
    hidden_units: int = 256

    # ------------------------------------------------------------------ #
    # ImageNet normalisation statistics
    # ------------------------------------------------------------------ #
    normalize_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ------------------------------------------------------------------ #
    # Compute device
    # ------------------------------------------------------------------ #
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------------ #
    # Mixed-precision training (only effective when device == "cuda")
    # ------------------------------------------------------------------ #
    use_amp: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    # ------------------------------------------------------------------
    # Derived paths (computed from output_dir)
    # ------------------------------------------------------------------
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
        """Create all required output directories."""
        for directory in (self.model_dir, self.logs_dir, self.plots_dir):
            directory.mkdir(parents=True, exist_ok=True)
