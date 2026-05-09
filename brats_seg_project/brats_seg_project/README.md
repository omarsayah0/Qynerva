# BraTS 2021 3D Segmentation

This project trains a **3D U-Net** on **BraTS 2021 TrainingSet** using the four MRI modalities as input channels and the training mask as target. The entire pipeline — from architecture design to training scripts and debugging — was developed with the assistance of **Mistral AI**, a large language model (LLM) used as an AI coding assistant throughout the project.

## Why this design
- Your BraTS data is **3D NIfTI volumes**, not 2D JPG slices.
- A plain **3D U-Net** is a strong and clean baseline for this type of data.
- The paper you shared uses view-specific 2D segmentation with a classifier head, so this project keeps only the useful idea of a Dice/Tversky-style loss but switches the model to a clean 3D BraTS pipeline.
- **Mistral AI** was used as an LLM assistant to help design, write, and refine the code at every stage of the pipeline.

## Models & Tools Used

| # | Model / Tool | Role | Description |
|---|-------------|------|-------------|
| 1 | **3D U-Net** | Segmentation Model | Core architecture. Takes 4 MRI modalities as input and outputs per-voxel class probabilities for Background, NCR, ED, and ET. |
| 2 | **MONAI** | Medical Imaging Framework | Handles data loading, preprocessing transforms (intensity normalisation, random crops, flips), and the BraTS label remapping helper. |
| 3 | **BraTS Label Transform** | Preprocessing | MONAI's `ConvertToMultiChannelBasedOnBratsClassesd` remaps the single-channel integer mask into 4 binary channels that the U-Net trains against. |
| 4 | **Mistral AI (LLM)** | AI Development Assistant | Large language model used throughout the project to design the training pipeline, debug data-loading issues, tune the loss function, and generate boilerplate code (dataset class, metric logging, config parsing). |

> **Note:** Mistral AI was used purely as a development-time coding assistant. It did not participate in model training, inference, or any medical decision-making.

## Pipeline Flow

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    BraTS 2021 TrainingSet                       │
 │          (FLAIR · T1 · T1ce · T2 · Segmentation mask)          │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  MODEL 2 — MONAI Framework                                      │
 │  • Recursive folder scan & train / val / test split             │
 │  • Intensity normalisation per modality                         │
 │  • Random 3D crops, flips, and augmentations                    │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  MODEL 3 — BraTS Label Transform                                │
 │  • ConvertToMultiChannelBasedOnBratsClassesd                    │
 │  • Single integer mask → 4 binary channels                      │
 │    (Background · NCR · Edema · Enhancing Tumor)                 │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
                    4-channel volume
                    (FLAIR · T1 · T1ce · T2)
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  MODEL 1 — 3D U-Net                                             │
 │  • Encoder: successive 3D conv + downsample blocks              │
 │  • Bottleneck: deepest feature representation                   │
 │  • Decoder: skip connections + upsample blocks                  │
 │  • Output: 4-channel softmax probability map                    │
 │  • Loss: Dice / Tversky-style loss                              │
 │  • Best checkpoint saved by validation Dice                     │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  Predicted Segmentation  (NIfTI output)                         │
 │  Background · NCR · Edema · Enhancing Tumor                     │
 └─────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────┐
 │  MODEL 4 — Mistral AI  (Development Assistant)                  │
 │  Used at every stage above to:                                  │
 │  • Design and scaffold the training pipeline                    │
 │  • Write and debug the dataset & dataloader classes             │
 │  • Tune the Dice / Tversky loss configuration                   │
 │  • Generate config parsing and metric logging boilerplate       │
 │  • Assist with debugging data-loading and shape issues          │
 └─────────────────────────────────────────────────────────────────┘
```

## Expected data structure

```text
BraTS2021_TrainingSet/
├── TCGA-GBM/
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
├── TCGA-LGG/
│   └── ...
└── UPENN-GBM/
    └── ...
```

## What the code does
- Scans the whole training folder recursively.
- Builds internal **train / val / test** splits from the **TrainingSet** only.
- Uses 4 input channels: `flair, t1, t1ce, t2`.
- Remaps BraTS mask into 4 output channels with MONAI's BraTS helper transform.
- Trains a **3D U-Net**.
- Saves the best checkpoint by validation Dice.

## Setup

```bash
pip install -e .
```

## Train

Edit:

```text
configs/train.yaml
```

Set:

```text
data.root_dir
```

to your BraTS training folder, then run:

```bash
python scripts/train.py --config configs/train.yaml
```

## Predict

```bash
python scripts/predict.py \
  --config configs/train.yaml \
  --patient-dir /path/to/BraTS2021_01722
```

This writes:

```text
predictions/BraTS2021_01722_pred.nii.gz
```

## Test Set Evaluation Results

```
========================================================================
  SEGMENTATION MODEL EVALUATION  —  TEST SET
========================================================================

  Overall Voxel Accuracy : 0.9898  (98.98%)

  Class                           Precision     Recall    F1/Dice
  --------------------------------------------------------------
  Background                         0.9909     0.9993     0.9951
  NCR (Necrotic Core)                0.7276     0.3355     0.4593
  ED (Edema)                         0.9147     0.6774     0.7784
  ET (Enhancing Tumor)               0.9087     0.3825     0.5384
  --------------------------------------------------------------
  Mean Tumor (NCR + ED + ET)         0.8503     0.4652     0.5920

========================================================================
  CONFUSION MATRIX  (rows = True class,  cols = Predicted class)
========================================================================
                      BG           NCR            ED            ET
  ----------------------------------------------------------------
        BG   666,030,680           425       378,543       114,385
       NCR       150,115       129,081        65,418        40,080
        ED     3,219,051         3,273     6,829,019        29,807
        ET     2,724,441        44,625       192,923     1,834,823

========================================================================
  Note: F1/Dice per class = voxel-wise Dice (2TP / (2TP + FP + FN)).
  Tumor classes: NCR = Necrotic Core, ED = Edema, ET = Enhancing Tumor.
========================================================================
```

| Class | Precision | Recall | F1/Dice |
|-------|-----------|--------|---------|
| Background | 0.9909 | 0.9993 | 0.9951 |
| NCR (Necrotic Core) | 0.7276 | 0.3355 | 0.4593 |
| ED (Edema) | 0.9147 | 0.6774 | 0.7784 |
| ET (Enhancing Tumor) | 0.9087 | 0.3825 | 0.5384 |
| **Mean Tumor** | **0.8503** | **0.4652** | **0.5920** |

**Overall Voxel Accuracy: 98.98%**
