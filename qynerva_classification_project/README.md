# qynerva_classification

2D brain tumor MRI classification using **EfficientNetB3** (PyTorch + timm).

Classifies MRI images into four categories:

| Label | Folder |
|---|---|
| 0 | `glioma_tumor` |
| 1 | `meningioma_tumor` |
| 2 | `normal` |
| 3 | `pituitary_tumor` |

---

## Project structure

```
qynerva_classification_project/
├── Data/                        ← raw dataset (you place this here)
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── normal/
│   └── pituitary_tumor/
│
├── SRC/
│   ├── config/
│   │   └── config.py            ← central Config dataclass
│   ├── data/
│   │   ├── splitter.py          ← stratified train/val/test split
│   │   └── dataset.py           ← Dataset, transforms, DataLoader factory
│   ├── models/
│   │   └── efficientnet.py      ← BrainTumorClassifier (EfficientNetB3 + custom head)
│   ├── training/
│   │   ├── callbacks.py         ← EarlyStopping, ModelCheckpoint
│   │   └── trainer.py           ← two-stage training pipeline
│   ├── prediction/
│   │   └── predictor.py         ← Predictor (single image / folder)
│   ├── utils/
│   │   ├── logger.py            ← logging setup
│   │   └── visualization.py     ← training-history plots
│   ├── main_train.py            ← CLI entry: qynerva_classification_train
│   └── main_predict.py          ← CLI entry: qynerva_classification_predict
│
├── outputs/                     ← created automatically during training
│   ├── models/
│   │   ├── best_model.pth
│   │   ├── final_model.pth
│   │   ├── class_to_idx.json
│   │   └── training_history.json
│   ├── logs/
│   │   └── train.log
│   └── plots/
│       ├── loss_curve.png
│       └── accuracy_curve.png
│
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python ≥ 3.10
- PyTorch (CPU or CUDA)
- torchvision
- timm
- scikit-learn
- Pillow
- matplotlib
- pandas
- numpy

---

## Installation

```bash
# 1 — (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2 — install the package in editable mode
pip install -e .
```

> For GPU training install the CUDA-enabled PyTorch build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Dataset setup

Place the dataset so that the folder structure looks like:

```
Data/
    glioma_tumor/      ← JPG images
    meningioma_tumor/  ← JPG images
    normal/            ← JPG images
    pituitary_tumor/   ← JPG images
```

The project automatically performs a **stratified split** into train / val / test
sets at runtime. No manual splitting is required.

---

## Training

```bash
# Default settings (auto-detects GPU/CPU)
qynerva_classification_train

# Custom settings
qynerva_classification_train \
    --data-dir  Data \
    --output-dir outputs \
    --batch-size 32 \
    --stage1-epochs 10 \
    --stage2-epochs 20 \
    --stage1-lr 1e-3 \
    --stage2-lr 1e-5 \
    --seed 42
```

### Training stages

| Stage | Backbone | Epochs | LR | Purpose |
|---|---|---|---|---|
| 1 | Frozen | 10 | 1e-3 | Train head only |
| 2 | Top 3 blocks unfrozen | 20 | 1e-5 | Fine-tune |

Both stages use:
- `CrossEntropyLoss`
- `Adam` optimiser
- `ReduceLROnPlateau` scheduler
- Early stopping (patience = 7)
- Best-model checkpoint

After training, the following files are created under `outputs/`:

```
outputs/models/best_model.pth        ← best validation-loss checkpoint
outputs/models/final_model.pth       ← model after all training completes
outputs/models/class_to_idx.json     ← {"glioma_tumor": 0, ...}
outputs/models/training_history.json ← per-epoch loss / acc / lr
outputs/logs/train.log
outputs/plots/loss_curve.png
outputs/plots/accuracy_curve.png
```

---

## Prediction

### Single image

```bash
qynerva_classification_predict --image path/to/brain_scan.jpg
```

Example output:

```
=======================================================
  Image          : brain_scan.jpg
  Predicted class: glioma_tumor
  Confidence     : 97.43%
-------------------------------------------------------
  Class probabilities:
    glioma_tumor                   97.43%  ██████████████████████████████
    meningioma_tumor                1.82%
    pituitary_tumor                 0.53%
    normal                          0.22%
=======================================================
```

### Folder (batch)

```bash
# Print results to console
qynerva_classification_predict --folder path/to/images/

# Save results to CSV and JSON
qynerva_classification_predict \
    --folder path/to/images/ \
    --save-csv results/predictions.csv \
    --save-json results/predictions.json
```

### Pointing to a different model

```bash
qynerva_classification_predict \
    --image scan.jpg \
    --model   outputs/models/final_model.pth \
    --class-map outputs/models/class_to_idx.json
```

---

## Configuration

All defaults live in `SRC/config/config.py`.  The `Config` dataclass can be
imported and instantiated directly when using the library programmatically:

```python
from SRC.config.config import Config
from SRC.training.trainer import run_training

cfg = Config(batch_size=64, stage1_epochs=15)
run_training(cfg)
```

---

## Model architecture

```
EfficientNetB3 (pretrained, global avg pool)
    └─ Dropout(0.3)
    └─ Linear(1536 → 256)
    └─ ReLU
    └─ Dropout(0.15)
    └─ Linear(256 → 4)
```

Total parameters: ~12.3 M  |  Trainable (stage 1): ~0.4 M
