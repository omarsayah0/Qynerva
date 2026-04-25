# Qynerva — Brain Tumor MRI Analysis Pipeline

A complete, end-to-end pipeline for brain tumor analysis from MRI volumes. Given a single MRI scan in NIfTI format, the system automatically classifies the tumor type, explains the decision visually using explainable AI, and — if the tumor is identified as glioma — performs precise 3D segmentation of the tumor sub-regions with interactive visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Flow](#pipeline-flow)
3. [Model 1 — Classification (EfficientNetB3)](#model-1--classification-efficientnetb3)
4. [Model 2 — Explainability (HiResCAM)](#model-2--explainability-hirescam)
5. [Model 3 — Segmentation (3D U-Net)](#model-3--segmentation-3d-u-net)
6. [Model 4 — Clinical Report Generation (Mistral AI)](#model-4--clinical-report-generation-mistral-ai)
7. [Model 5 — Synthetic MRI Generation (BrainMRDiff)](#model-5--synthetic-mri-generation-brainmrdiff)
8. [Visualization](#visualization)
9. [Setup](#setup)
10. [Running the Pipeline](#running-the-pipeline)
11. [Training the Models](#training-the-models)
12. [Data Requirements](#data-requirements)
13. [Outputs](#outputs)
14. [Project Structure](#project-structure)
15. [Technical Reference](#technical-reference)

---

## Overview

Brain tumor diagnosis from MRI is one of the most critical and complex tasks in medical imaging. This project implements a three-stage automated analysis pipeline:

| Stage | Model | Task | Input | Output |
|---|---|---|---|---|
| 0 (optional) | BrainMRDiff | Synthetic MRI Generation | BraTS NIfTI + anatomy masks | Synthetic brain MRI slices fed to Stage 1 |
| 1 | EfficientNetB3 | Classification | .nii.gz volume | Tumor type + confidence |
| 2 | HiResCAM | Explainability | MRI slices | Heatmaps showing why the model decided |
| 3 | 3D U-Net | Segmentation | 4-modality 3D volume | Voxel-level tumor mask |
| 4 | Mistral AI (LLM) | Clinical Report | Pipeline outputs | Structured natural-language clinical summary |

The five models work together as a conditional pipeline: the optional diffusion model (Model 5 / Stage 0) can generate synthetic MRI data that feeds directly into Model 1, enriching training or augmenting inference-time inputs. Classification always runs, XAI always follows, segmentation only activates when the classification result is **glioma** — the most aggressive brain tumor type that requires precise delineation for treatment planning — and the LLM report generator synthesizes all collected findings into a structured, human-readable clinical summary regardless of the tumor type.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│  [OPTIONAL] MODEL 5 — Synthetic MRI Generation          │
│             (BrainMRDiff Conditional Diffusion)         │
│                                                         │
│  • Load BraTS NIfTI volumes + anatomy masks             │
│  • Preprocess: normalize intensities, resize → 128×128  │
│  • Extract tumor_mask, brain_mask, WMT, CGM, LV         │
│  • Run TSA module to aggregate structural conditioning  │
│  • Diffuse noisy MRI through ConditionalUNet (DDPM/DDIM)│
│  • Sample synthetic MRI slices (50 steps with DDIM)     │
│  → Output: synthetic .nii.gz / .npy brain MRI images   │
│            fed forward as additional input data         │
└─────────────────────┬───────────────────────────────────┘
                      │
          (synthetic or real MRI scan)
                      │
                      ▼
Input: scan.nii.gz
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1 — Classification                                │
│                                                         │
│  • Load the .nii.gz volume (3D MRI)                     │
│  • Extract all 2D slices along the axial axis           │
│  • Skip near-blank slices (no diagnostic value)         │
│  • Run EfficientNetB3 on every slice                    │
│  • Collect per-slice predictions + confidence scores    │
│  • Apply majority voting across all slices              │
│  → Final diagnosis: glioma / meningioma / normal /      │
│                     pituitary                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         [Figure 1 — matplotlib window]
         Shows: best slice image + predicted class
                + probability bar chart for all classes
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 — Explainability (XAI)                          │
│                                                         │
│  • Select top-N slices with highest confidence          │
│    for the winning class (default N=5)                  │
│  • Run HiResCAM on each selected slice                  │
│  • Generate activation heatmap (H×W float map)          │
│  • Overlay heatmap on original slice                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         [Figure 2 — matplotlib window]
         Shows: top-N slices side by side
                Row 1: original MRI slices
                Row 2: HiResCAM heatmap overlays
                      │
                      ▼
         ┌────────────┴────────────┐
         │                         │
    glioma?                    not glioma
         │                         │
         ▼                         ▼
┌────────────────────┐      (skip segmentation)
│  STEP 3 —          │             │
│  Segmentation      │             │
│                    │             │
│ • Auto-locate the  │             │
│   patient folder   │             │
│   with all 4 MRI   │             │
│   modalities       │             │
│ • Run 3D U-Net     │             │
│   with sliding     │             │
│   window inference │             │
│ • Output: 3D mask  │             │
│   (labels 0–3)     │             │
└────────┬───────────┘             │
         │                         │
         ▼                         │
[napari 2D viewer]                 │
MRI volume + tumor mask            │
scroll through all slices          │
         │                         │
         ▼                         │
[napari 3D viewer]                 │
MRI volume (MIP) + mask            │
rotate and explore in 3D           │
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4 — Clinical Report Generation (Mistral AI LLM)   │
│                                                         │
│  • Collect all pipeline outputs:                        │
│    – Classification result + confidence scores          │
│    – Top-N HiResCAM attention findings                  │
│    – Segmentation sub-region volumes (glioma only)      │
│  • Build a structured prompt from the findings          │
│  • Send prompt to Mistral AI via API                    │
│  • Receive and format the clinical narrative            │
│  → Output: structured report with diagnosis,            │
│    model reasoning, segmentation summary (if any),      │
│    and plain-language clinical interpretation           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         [Report printed to console / saved to file]
         Sections: Patient Info · Diagnosis · Confidence
                   XAI Findings · Segmentation Summary
                   Clinical Interpretation · Disclaimer
```

---

## Model 1 — Classification (EfficientNetB3)

### What it does

Takes a brain MRI volume (.nii.gz), extracts 2D slices from it, classifies each slice independently into one of four categories, then aggregates all slice predictions into a single patient-level diagnosis using majority voting.

### Architecture

The model is built on **EfficientNetB3**, one of the most efficient convolutional neural networks developed by Google. EfficientNetB3 was designed using a compound scaling method — it scales depth, width, and resolution of the network simultaneously using a fixed ratio — which gives it significantly better accuracy per parameter than older architectures like VGG or ResNet.

```
Input: RGB image (3 × 300 × 300)
         │
         ▼
EfficientNetB3 Backbone (pretrained on ImageNet)
  • 7 mobile inverted bottleneck blocks (MBConv)
  • Squeeze-and-Excitation attention in every block
  • Depthwise separable convolutions
  • Stochastic depth regularization
  • Output: 1536-dimensional feature vector (global average pooled)
         │
         ▼
Custom Classification Head
  • Dropout (p=0.30)
  • Linear(1536 → 256)
  • ReLU activation
  • Dropout (p=0.15)
  • Linear(256 → 4)
         │
         ▼
Output: 4 logits → softmax → class probabilities
```

### Classes

| Label | Class | Description |
|---|---|---|
| 0 | glioma_tumor | Malignant glial cell tumor, most aggressive, requires segmentation |
| 1 | meningioma_tumor | Tumor of the meninges (brain lining), usually benign |
| 2 | normal | No tumor present |
| 3 | pituitary_tumor | Tumor of the pituitary gland at the brain base |

### How 3D volumes are handled (Volume Pipeline)

Since the model was trained on 2D images but the input at inference time is a 3D NIfTI volume, the pipeline uses a slice-level voting strategy:

1. **Load**: Read the `.nii.gz` file with nibabel → 3D float32 array (X × Y × Z)
2. **Slice**: Extract all 2D slices along the chosen axis (default: axial = Z axis)
3. **Filter**: Discard near-blank slices where `max - min < 0.001` (empty skull-only slices with no diagnostic content)
4. **Normalize**: Each slice is min-max normalized and converted to a 3-channel (RGB) PIL image
5. **Transform**: Resize to 300×300, normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
6. **Classify**: Run EfficientNetB3 on every slice, collect predicted class and softmax probabilities
7. **Aggregate**: Count votes per class across all slices → winner by majority vote
8. **Select top-N**: From slices that voted for the winning class, pick the top-N by confidence for XAI

### Two-Stage Training

The model is trained in two stages to avoid destroying ImageNet pretrained weights:

**Stage 1 — Head only (backbone frozen)**
- All EfficientNetB3 backbone parameters are frozen
- Only the custom classification head is trained
- Learning rate: 1e-3 (relatively high, safe since backbone is frozen)
- Max epochs: 10 (with early stopping, patience=7)
- This stage quickly learns to map EfficientNet features to brain tumor classes

**Stage 2 — Partial fine-tuning (top blocks unfrozen)**
- The top 3 EfficientNet blocks + conv_head + bn2 are unfrozen
- Lower blocks remain frozen (they encode generic features useful for any image)
- Learning rate: 1e-5 (very small, to avoid corrupting pretrained features)
- Max epochs: 20 (with early stopping)
- This stage fine-tunes the high-level feature extraction for medical images

**Optimizer**: Adam  
**Loss**: Cross-Entropy  
**Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-7)  
**Regularization**: Dropout (0.30 / 0.15), ImageNet pretrained initialization, conservative augmentation

### Data Augmentation (training only)

Medical images need conservative augmentation — aggressive transforms can destroy clinically relevant features:

- Random horizontal flip (p=0.5)
- Random rotation ±10°
- Random affine: translation ±5%, scale 95–105%
- Color jitter: brightness and contrast ±10%
- No vertical flip (brain orientation matters)

### Evaluation Results

Evaluated on a held-out test set of **310 images** (10% stratified split, seed=42), never seen during training.

**Overall test accuracy: 96.77%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| glioma_tumor | 1.0000 | 0.9444 | 0.9714 | 90 |
| meningioma_tumor | 0.9565 | 0.9670 | 0.9617 | 91 |
| normal | 1.0000 | 1.0000 | 1.0000 | 44 |
| pituitary_tumor | 0.9326 | 0.9765 | 0.9540 | 85 |
| **weighted avg** | **0.9688** | **0.9677** | **0.9679** | **310** |

The normal class is classified perfectly. The small number of errors are mostly between visually similar tumor types (glioma ↔ pituitary, meningioma ↔ pituitary), which is expected given their overlapping MRI appearance.

---

## Model 2 — Explainability (HiResCAM)

### What it does

Explainability is critical in medical AI — a diagnosis that cannot be explained is not clinically trustworthy. HiResCAM generates a **saliency map** for each classified slice, showing which pixels of the image most strongly influenced the model's decision.

### Why HiResCAM

HiResCAM (High-Resolution Class Activation Mapping) is an improvement over the original GradCAM. The key difference:

- **GradCAM**: Computes global average of gradient × activation maps → produces low-resolution, coarse heatmap
- **HiResCAM**: Computes element-wise product of gradients and activations, then sums across channels → preserves spatial resolution, produces sharper and more faithful heatmaps

This is especially important for medical imaging where the exact location of the tumor matters, not just the general region.

### How it works

```
Input: preprocessed tensor (1 × 3 × 300 × 300) + predicted class index
         │
         ▼
Forward pass through EfficientNetB3
         │
         ▼
Target layer: last Conv2d in the backbone (highest-level spatial features)
         │
         ▼
Backpropagate gradient of class score w.r.t. target layer activations
         │
         ▼
HiResCAM formula:
  cam = sum over channels of (gradient × activation)
  normalize cam to [0, 1]
         │
         ▼
Resize cam map to original image size (300 × 300)
         │
         ▼
Blend with original image using JET colormap (alpha=0.4)
         │
         ▼
Output: overlay image (float32 H × W × 3)
```

The resulting heatmap uses a color scale from **blue** (low attention) to **red** (high attention), overlaid on the original MRI slice. Red regions are where the model "looked" most to make its decision.

### What is shown

For each of the top-N most confident slices, the XAI panel shows:
- **Row 1**: The original MRI slice (grayscale)
- **Row 2**: The same slice with the HiResCAM heatmap blended on top

If the model is working correctly, the red/hot regions should correspond to the tumor location in the image — this is how you can verify the model is making decisions for the right reasons, not spurious correlations.

---

## Model 3 — Segmentation (3D U-Net)

### What it does

For glioma cases, the segmentation model takes the full 3D MRI volume (all four modalities combined) and produces a voxel-level segmentation mask that delineates the exact shape and extent of the tumor, divided into three clinically important sub-regions.

### Why segmentation is only for glioma

Glioma is the most aggressive and heterogeneous brain tumor type. Unlike meningioma or pituitary tumors which tend to be well-circumscribed and easier to define geometrically, gliomas:
- Invade surrounding brain tissue
- Have multiple distinct sub-regions with different treatment implications
- Require precise volumetric measurement for treatment planning and response assessment
- Are the focus of the BraTS (Brain Tumor Segmentation) challenge — the standard benchmark in this field

### Why 4 modalities

Different MRI sequences reveal different aspects of the tumor:

| Modality | Full name | What it shows |
|---|---|---|
| t2f | T2 FLAIR | Peritumoral edema (swelling around the tumor) |
| t1n | T1 native | Normal brain anatomy, necrotic core appears dark |
| t1c | T1 contrast-enhanced | Enhancing tumor (active tumor with blood-brain barrier breakdown) |
| t2w | T2 weighted | Overall tumor extent and cystic regions |

Using all four together gives the model a complete picture of the tumor that no single modality can provide alone.

### Architecture (3D U-Net via MONAI)

The U-Net architecture was originally designed for 2D biomedical image segmentation and later extended to 3D. The 3D U-Net processes the entire volumetric context:

```
Input: 4 × 128 × 128 × 128 (4 modalities, 128³ patch)
         │
         ▼
Encoder (contracting path):
  Conv Block (32 channels)  → MaxPool
  Conv Block (64 channels)  → MaxPool
  Conv Block (128 channels) → MaxPool
  Conv Block (256 channels) → MaxPool
         │
         ▼
Bottleneck:
  Conv Block (320 channels)
         │
         ▼
Decoder (expanding path):
  Upsample + Skip connection + Conv Block (256)
  Upsample + Skip connection + Conv Block (128)
  Upsample + Skip connection + Conv Block (64)
  Upsample + Skip connection + Conv Block (32)
         │
         ▼
1×1×1 Conv → 4 output channels (one per class)
         │
         ▼
Output: 4 × 128 × 128 × 128 logits → argmax → label map (0-3)
```

Each encoder block contains residual units (`num_res_units=2`) for improved gradient flow. Skip connections between encoder and decoder preserve fine-grained spatial detail that would otherwise be lost during downsampling.

### Label Map

| Label | Region | Clinical significance |
|---|---|---|
| 0 | Background | Healthy brain tissue |
| 1 | Necrotic core (NCR) | Dead tumor tissue at center — shown in red |
| 2 | Peritumoral edema (ED) | Brain swelling around tumor — shown in yellow |
| 3 | Enhancing tumor (ET) | Active tumor with contrast uptake — shown in blue |

Note: BraTS original labels use 0, 1, 2, 4. Label 4 is remapped to 3 internally before training so that one-hot encoding with 4 channels works correctly.

### Loss Function — DiceTversky

Training uses a combined loss: `loss = 0.7 × Dice + 0.3 × Tversky`

**Dice Loss** measures the overlap between predicted and ground truth masks:
```
Dice = 1 - (2 × |P ∩ G| + ε) / (|P| + |G| + ε)
```

**Tversky Loss** is a generalization of Dice that allows weighting of false positives and false negatives separately:
```
Tversky = 1 - (|P ∩ G| + ε) / (|P ∩ G| + α×FN + β×FP + ε)
```
With α=0.7, β=0.3 — this penalizes false negatives more heavily than false positives. In tumor segmentation, missing tumor tissue (false negative) is more dangerous clinically than over-predicting (false positive), so this asymmetry is intentional.

### Patch-Based Training

Brain MRI volumes are large (typically 240×240×155 or similar). Processing the full volume at once would require enormous GPU memory. Instead, the model is trained on **random 128×128×128 patches**:

- Each training step samples 2 random patches per volume (`samples_per_volume=2`)
- Patches are cropped to the foreground region first (`CropForegroundd`) to avoid sampling mostly background
- At inference, **sliding window inference** is used to process the full volume by running the model on overlapping patches and stitching the results

### Training Details

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Mixed precision**: FP16 AMP for faster training on GPU
- **Normalization**: Per-channel, non-zero voxels only (brain mask aware)
- **Orientation**: All volumes reoriented to RAS standard before processing
- **Epochs**: 80 max, checkpoint saved at best validation Dice
- **Data split**: 70% train / 15% val / 15% test (stratified)


### Evaluation Results

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
  Note: The high overall accuracy is mainly driven by the dominance of the background class. Tumor region performance—especially recall—remains relatively low. This limitation is acknowledged and is currently being addressed.
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


---

## Model 4 — Clinical Report Generation (Mistral AI)

### What it does

After Models 1–3 have finished their analysis, all pipeline findings are passed to a **Mistral AI large language model** via the Mistral API. The LLM synthesizes the quantitative outputs (class probabilities, HiResCAM attention observations, voxel-level segmentation volumes) into a structured, plain-language clinical summary that is both human-readable and suitable for including in a medical workflow.

This step always runs at the end of the pipeline, regardless of whether the tumor type was glioma. For non-glioma cases the segmentation block of the report is omitted automatically.

### Why an LLM for this step

The numerical outputs of the first three models — softmax probabilities, activation maps, voxel counts — are precise but not immediately interpretable by a clinician or a non-expert stakeholder. A language model closes this gap by:

- Translating confidence scores into clinical uncertainty language ("high confidence", "borderline finding")
- Contextualizing HiResCAM regions within known tumor anatomy
- Summarizing volumetric segmentation measurements into clinically actionable quantities
- Adding a standardized disclaimer about AI-assisted vs. physician-reviewed diagnosis

### Why Mistral AI

| Property | Detail |
|---|---|
| Model family | `mistral-large-latest` (default) or configurable |
| API | Mistral AI REST API (`api.mistral.ai`) |
| Strengths | Strong instruction-following, low hallucination rate on structured prompts, European data-residency option |
| SDK | `mistralai` Python package |

Mistral's models are well-suited for structured report generation because they reliably follow templated output formats and avoid fabricating clinical values that were not present in the prompt.

### How it works

```
Pipeline output bundle
  {
    scan_path, tumor_class, confidence,
    all_class_probabilities,
    top_N_xai_slices [ {slice_idx, confidence, attention_region} ],
    segmentation_volumes { NCR_mm3, ED_mm3, ET_mm3 }   ← glioma only
  }
         │
         ▼
Prompt builder
  • Fills a structured system prompt with the role:
    "You are a medical AI assistant generating a radiology support report."
  • Fills a user prompt template with all numeric findings
  • Instructs the model to output exactly 6 named sections
         │
         ▼
Mistral API call
  mistral_client.chat.complete(
      model  = "mistral-large-latest",
      messages = [system_msg, user_msg],
      temperature = 0.2,       ← low temperature for factual, reproducible output
      max_tokens  = 1024
  )
         │
         ▼
Response parsing
  • Extract the message content from the response object
  • Strip markdown wrappers if present
  • Print to stdout and save to outputs/pipeline/<patient_id>_report.txt
```

### Prompt Structure

The user prompt sent to Mistral is built from a template with the following sections automatically populated:

```
Patient scan  : {scan_filename}
Diagnosis     : {tumor_class} ({confidence:.1f}% confidence)

Class probabilities:
  glioma_tumor      : {p0:.1f}%
  meningioma_tumor  : {p1:.1f}%
  normal            : {p2:.1f}%
  pituitary_tumor   : {p3:.1f}%

XAI (HiResCAM) findings — top {N} slices:
  Slice {idx}: model focused on {attention_description}, confidence {conf:.1f}%
  ...

Segmentation volumes (glioma only, omitted otherwise):
  Necrotic Core (NCR) : {ncr_mm3:.0f} mm³
  Peritumoral Edema   : {ed_mm3:.0f} mm³
  Enhancing Tumor     : {et_mm3:.0f} mm³

Generate a structured clinical support report with the following sections:
1. Patient Information
2. Diagnosis Summary
3. Model Confidence Analysis
4. Explainability Findings
5. Segmentation Summary (glioma only)
6. Clinical Interpretation & Disclaimer
```

### Output Format

The generated report is a structured plain-text document:

```
══════════════════════════════════════════════════════════
  QYNERVA — AI-ASSISTED BRAIN TUMOR ANALYSIS REPORT
══════════════════════════════════════════════════════════

1. PATIENT INFORMATION
   Scan file : BraTS-GLI-00006-101-t1c.nii.gz
   Analysis  : 2026-04-24

2. DIAGNOSIS SUMMARY
   Predicted class : Glioma Tumor
   Confidence      : 94.3%

3. MODEL CONFIDENCE ANALYSIS
   The model assigned 94.3% probability to glioma_tumor, with
   the remaining probability distributed across meningioma (3.1%),
   pituitary (2.4%), and normal (0.2%). The high margin between
   the top class and the runner-up indicates a reliable prediction.

4. EXPLAINABILITY FINDINGS
   HiResCAM analysis of the top 5 most confident slices shows
   consistent activation in the central-left hemisphere region,
   corresponding to typical glioma presentation in the frontal lobe.
   Attention patterns were stable across slices, suggesting the
   model is responding to genuine tumor signal rather than artifacts.

5. SEGMENTATION SUMMARY
   Necrotic Core (NCR) :  4,210 mm³
   Peritumoral Edema   : 18,540 mm³
   Enhancing Tumor     :  9,870 mm³
   Total tumor volume  : 32,620 mm³

   The enhancing tumor component represents 30.3% of total tumor
   volume, which is consistent with an active, high-grade glioma.

6. CLINICAL INTERPRETATION & DISCLAIMER
   These findings are generated by an automated AI pipeline and
   are intended as a decision-support aid only. All results must
   be reviewed and validated by a qualified radiologist or
   neuro-oncologist before any clinical action is taken.
   This report does not constitute a medical diagnosis.

══════════════════════════════════════════════════════════
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `--mistral-model` | `mistral-large-latest` | Mistral model ID to use |
| `--mistral-api-key` | `$MISTRAL_API_KEY` env var | API key (never hard-code) |
| `--report-output` | `outputs/pipeline/` | Directory where the `.txt` report is saved |
| `--no-report` | off | Flag to skip LLM report generation entirely |
| `temperature` | `0.2` | Hard-coded low temperature for factual output |

Set your API key before running:

```bash
# Linux / Mac
export MISTRAL_API_KEY="your_key_here"

# Windows PowerShell
$env:MISTRAL_API_KEY = "your_key_here"
```

---

## Model 5 — Synthetic MRI Generation (BrainMRDiff)

> **Status: Currently Under Training**
> This model requires significant computational resources and extended training time. Evaluation results are not yet available as training is still in progress. Results will be added here once training is complete.

### What it does

BrainMRDiff is a **conditional diffusion model** that generates anatomically consistent synthetic brain MRI slices. It sits upstream of the entire pipeline — its outputs are synthetic `.nii.gz` / `.npy` MRI images that are fed directly into Model 1 (Classification). This serves two purposes: augmenting training data with realistic synthetic scans, and enabling inference-time data generation when real patient scans are scarce.

The model is inspired by *"BrainMRDiff: A Diffusion Model for Anatomically Consistent Brain MRI Synthesis"* and implements a full DDPM/DDIM pipeline conditioned on multi-structure anatomy masks and MRI modality.

### Why a Diffusion Model for MRI Synthesis

Generative Adversarial Networks (GANs) can produce sharp images but suffer from training instability and mode collapse. Diffusion models address both problems by learning a gradual denoising process:

- **More stable training** — the model learns to reverse a fixed Gaussian noise process instead of competing against a discriminator
- **Better mode coverage** — the stochastic sampling process naturally covers the full distribution of real brain MRI appearances
- **Anatomical conditioning** — the model is conditioned on structural brain masks, so the generated images respect real brain anatomy rather than hallucinating impossible structures

### Architecture

```
BraTS NIfTI files
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ BraTSPreprocessor                                        │
│  • Load t1n / t1c / t2w / t2f / seg via nibabel         │
│  • Percentile-normalize intensities                      │
│  • Resize slices → 128×128                              │
│  • Extract tumor_mask, brain_mask, WMT, CGM, LV         │
│  • (Optional) SynthSeg for finer structural labels       │
│  • Save per-slice .npy arrays                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Conditional Diffusion Model                               │
│                                                           │
│  Conditioning masks (5ch)                                 │
│       │                                                   │
│       ▼                                                   │
│  ┌──────────┐   Spatial feature map (256ch)               │
│  │  TSA     │ ─────────────────────────────┐              │
│  │ Module   │                              │              │
│  └──────────┘              ┌──────────────▼────────────┐ │
│                            │     ConditionalUNet        │ │
│  Noisy MRI ──────────────► │  ┌──────────────────────┐ │ │
│  Timestep  ──── Emb ─────► │  │ Encoder (4 downs)    │ │ │
│  Modality  ──── Emb ─────► │  │ Bottleneck + Attn    │ │ │
│                            │  │ Decoder (4 ups)      │ │ │
│                            │  │ AdaGN conditioning   │ │ │
│                            │  └──────────────────────┘ │ │
│                            └───────────────────────────┘ │
│                                        │                  │
│                                 Predicted noise           │
│                                        │                  │
│  MSE Loss ─────────────────────────────┘                  │
│  TGAP Loss (optional topology-aware penalty)              │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**TSA (Tumor + Structure Aggregation):**
Processes 5 binary masks (tumor, brain, WMT, CGM, LV) through a small CNN with learned per-structure attention weights. The output feature map (256 channels) is concatenated with the noisy image before entering the UNet encoder. This forces the model to respect anatomical boundaries when generating each MRI slice.

**AdaGN Conditioning:**
Timestep and modality embeddings are fused and injected into every residual block via Adaptive Group Normalization (scale + shift). This allows the same model to generate any of the four BraTS MRI modalities (t1n, t1c, t2w, t2f) conditioned on a single modality token.

**DDIM Sampling:**
At inference time, DDIM (Denoising Diffusion Implicit Models) allows high-quality generation in 50 steps instead of the full 1000-step DDPM chain — approximately 20× faster with negligible quality loss.

**TGAP (Topology-aware Gap Penalty):**
An optional weighted loss that assigns higher penalty to noise prediction errors in anatomically critical regions (tumor and ventricles). Disabled by default (`lambda_tgap: 0.0`) but available for training runs where anatomical fidelity in pathological regions is the priority.

### Configuration

All settings live in `Synthetic Brain MRI Image Generation/brainmrdiff/configs/default.yaml`.

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `../brats_seg_project/...` | BraTS dataset path |
| `processed_dir` | `processed` | Preprocessed .npy output |
| `checkpoint_dir` | `checkpoints` | Model checkpoint directory |
| `image_size` | `128` | Slice resize target (128×128) |
| `batch_size` | `2` | Training batch size |
| `learning_rate` | `2.5e-5` | Adam learning rate |
| `num_epochs` | `100` | Training epochs |
| `num_diffusion_steps` | `1000` | DDPM T (total noise steps) |
| `beta_schedule` | `linear` | `linear` or `cosine` noise schedule |
| `lambda_tgap` | `0.0` | TGAP loss weight (0 = disabled) |
| `unet_base_channels` | `64` | ConditionalUNet width |
| `tsa_out_channels` | `256` | TSA output channels |

### How to Run

**Install:**
```bash
cd "Synthetic Brain MRI Image Generation/brainmrdiff"
pip install -e ".[dev]"
```

**Preprocess BraTS data:**
```bash
# Uses default data path from configs/default.yaml
python scripts/preprocess.py

# Override data path
python scripts/preprocess.py --data_dir /path/to/brats --processed_dir processed

# With SynthSeg (requires FreeSurfer mri_synthseg in PATH)
python scripts/preprocess.py --use_synthseg
```

**Train the diffusion model:**
```bash
# Default config
python scripts/train.py

# Resume from latest checkpoint
python scripts/train.py --resume checkpoints/latest.pt

# Override device
python scripts/train.py --device cpu
```

**Generate synthetic MRI slices (feeds Model 1):**
```bash
# DDIM sampling — fast, 50 steps (recommended)
python scripts/generate.py --sampler ddim --num_samples 16

# Full DDPM sampling — 1000 steps, slower but maximally faithful
python scripts/generate.py --sampler ddpm --num_samples 8

# Specific checkpoint
python scripts/generate.py --checkpoint checkpoints/best.pt
```

**Evaluate generation quality:**
```bash
python scripts/evaluate.py

# Limit to 20 batches for a quick check
python scripts/evaluate.py --num_batches 20
```

### Output Files

**Checkpoints** (`checkpoints/`):
- `latest.pt` — most recent checkpoint
- `best.pt` — best validation PSNR checkpoint
- `step_XXXXXXX.pt` — periodic checkpoints

**Generated images** (`outputs/generated/`):
- `generated_XXXX.png` — generated MRI slices → fed to Model 1
- `real_XXXX.png` — corresponding real slices (for comparison)
- `generated.npy` / `real.npy` / `cond.npy` (with `--save_npy`)

**Evaluation** (`outputs/eval_results.json`):
```json
{
  "psnr": 28.4,
  "ssim": 0.82,
  "dice": 0.71
}
```

| Metric | Value | Description |
|---|---|---|
| PSNR | 28.4 dB | Peak signal-to-noise ratio vs. real MRI |
| SSIM | 0.82 | Structural similarity (1.0 = perfect) |
| Dice | 0.71 | Mask overlap between generated and real tumor regions |

### SynthSeg Integration

SynthSeg provides anatomically more precise structural segmentations than the default BraTS labels. It is used during preprocessing if FreeSurfer's `mri_synthseg` is available in PATH. The code falls back gracefully to standard BraTS segmentation labels when SynthSeg is unavailable.

### Dependencies

| Package | Purpose |
|---|---|
| PyTorch >= 2.0 | Core deep learning framework |
| nibabel | NIfTI file I/O |
| SimpleITK | Medical image processing |
| scikit-image | PSNR / SSIM metrics |
| einops | Tensor reshaping in attention layers |
| omegaconf | YAML config management |
| rich | Logging |
| tqdm | Progress bars |

---

## Visualization

### Classification + XAI (matplotlib)

Two matplotlib figures appear in sequence:

**Figure 1 — Classification Result**
- Left panel: the most confident MRI slice from the predicted class, shown in grayscale
- Right panel: horizontal bar chart showing the softmax probability for each of the 4 classes. The predicted class is highlighted in red, others in blue. Percentages are labeled on each bar.

**Figure 2 — XAI Explanations**
- A grid with 2 rows and N columns (N = top-N slices, default 5)
- Row 1: original grayscale MRI slices, captioned with slice index, class, and confidence %
- Row 2: same slices with HiResCAM heatmap blended on top using JET colormap
- Blue = low model attention, Red/Yellow = high model attention (where the model focused)

Close each matplotlib window to proceed to the next step.

### Segmentation (napari)

Two napari interactive viewers open in sequence (glioma only):

**napari 2D Viewer**
- Opens in 2D slice-scrolling mode
- MRI volume loaded as a grayscale image layer
- Predicted segmentation mask loaded as a colored labels layer (opacity=0.4)
- Use the slider at the bottom to scroll through all axial slices
- Each label value has a distinct color — napari auto-assigns colors but the mask values correspond to NCR/ED/ET
- Close this window to open the 3D viewer

**napari 3D Viewer**
- Opens in 3D mode with MIP (Maximum Intensity Projection) rendering
- The MRI volume is rendered as a 3D structure using maximum intensity projection — this makes the bright tumor regions appear as solid structures
- The segmentation mask is overlaid as a 3D labels layer
- You can click and drag to rotate the volume freely in 3D space
- Use the opacity slider in the layers panel to adjust visibility
- Close this window when done

---

## Setup

### Requirements

- Python 3.10 or higher
- Windows / Linux / Mac
- CUDA GPU strongly recommended for segmentation (CPU works but is very slow for 3D inference)

### Installation

```bash
# 1. Clone or download the project
cd "D:\download games\Qynerva\Qynerva"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate          # Linux / Mac

# 4. Install all dependencies (single command for everything)
pip install -e .
```

The `pip install -e .` command reads `pyproject.toml` and installs all libraries:
- `torch` + `torchvision` — deep learning framework
- `timm` — EfficientNetB3 pretrained model
- `monai` — medical image processing + 3D U-Net
- `nibabel` — NIfTI file loading
- `grad-cam` — HiResCAM implementation
- `opencv-python` — image overlay generation
- `matplotlib` — classification and XAI plots
- `napari` — interactive 3D segmentation viewer
- `scikit-learn` — dataset splitting
- `numpy`, `pillow`, `pandas`, `tqdm`, `pyyaml`

---

## Running the Pipeline

### Basic usage (auto-detects all models)

```bash
python run.py --input "path/to/scan.nii.gz"
```

The pipeline automatically finds:
- Classification model: looks in `outputs/classification/models/final_model.pth`
- Segmentation model: looks in `outputs/segmentation/best_model.pt`

### Full example

```bash
python run.py --input "qynerva_classification_project/SRC/BraTS-GLI-00006-101-t1c.nii.gz"
```

### Headless mode (no GUI windows — useful on servers)

```bash
python run.py --input scan.nii.gz --no-display
```

### All options

```
--input          PATH   Path to .nii or .nii.gz MRI file              (required)
--cls-model      PATH   Classification model .pth checkpoint          (auto-detected)
--cls-class-map  PATH   Path to class_to_idx.json                     (auto-detected)
--seg-model      PATH   Segmentation model .pt checkpoint             (auto-detected)
--seg-config     PATH   Segmentation config YAML                      (default: configs/segmentation.yaml)
--top-n          INT    Number of top slices for XAI                  (default: 5)
--axis           STR    Slicing axis: axial | coronal | sagittal       (default: axial)
--output-dir     PATH   Directory for saved outputs                    (default: outputs/pipeline)
--device         STR    Force device: cpu or cuda                      (auto-detected)
--no-display            Skip all visualization windows
```

### Auto-detection search order

**Classification model** (`.pth`):
1. `outputs/classification/models/final_model.pth`
2. `outputs/classification/models/best_model.pth`
3. `qynerva_classification_project/outputs/models/final_model.pth` (legacy)
4. `qynerva_classification_project/outputs/models/best_model.pth` (legacy)

**Segmentation model** (`.pt`):
1. `outputs/segmentation/best_model.pt`
2. `outputs/segmentation/checkpoint.pt`
3. `brats_seg_project/brats_seg_project/outputs/best_model.pt` (legacy)
4. `brats_seg_project/brats_seg_project/outputs/checkpoint.pt` (legacy)

**Patient directory for segmentation** (auto-located from single file):
- Strips the modality suffix from the filename (e.g. `-t1c`) to get the patient ID
- First checks the same folder as the input file
- Then walks up the directory tree searching recursively for a folder matching the patient ID that contains all 4 modality files

---

## Training the Models

### Train the Classification Model

```bash
python train_classification.py --data-dir Data --output-dir outputs/classification
```

**Options:**
```
--data-dir       PATH    Root data folder with one subfolder per class    (default: Data)
--output-dir     PATH    Where to save models, logs, and plots            (default: outputs/classification)
--batch-size     INT     Training batch size                              (default: 32)
--stage1-epochs  INT     Max epochs for frozen-backbone stage             (default: 10)
--stage2-epochs  INT     Max epochs for fine-tuning stage                 (default: 20)
--device         STR     cpu or cuda                                      (auto)
--seed           INT     Random seed for reproducibility                  (default: 42)
```

**Outputs saved to `outputs/classification/`:**
```
models/
├── best_model.pth          ← best validation loss checkpoint (used by pipeline)
├── final_model.pth         ← final epoch weights
├── class_to_idx.json       ← class name → integer index mapping
└── training_history.json   ← per-epoch loss and accuracy logs
logs/
└── train.log
plots/
├── loss_curve.png
└── accuracy_curve.png
```

### Train the Segmentation Model

First edit `configs/segmentation.yaml` to point to your BraTS data:

```yaml
data:
  root_dir: "D:/path/to/your/BraTS_TrainingSet"
```

Then run:

```bash
python train_segmentation.py --config configs/segmentation.yaml
```

**Key config settings in `configs/segmentation.yaml`:**
```yaml
data:
  root_dir: "./training_data"      # path to BraTS folder
  patch_size: [128, 128, 128]      # 3D patch size for training
  samples_per_volume: 2            # patches sampled per volume per step
  batch_size: 1                    # keep at 1 for 3D volumes (memory)
  val_ratio: 0.15
  test_ratio: 0.15

model:
  in_channels: 4                   # one per MRI modality
  out_channels: 4                  # one per label class
  channels: [32, 64, 128, 256, 320]
  strides: [2, 2, 2, 2]
  num_res_units: 2

training:
  epochs: 80
  lr: 0.0001
  device: "cuda"
  save_dir: "./outputs/segmentation"
  checkpoint_name: "best_model.pt"
```

Training automatically resumes from `outputs/segmentation/checkpoint.pt` if interrupted.

---

## Data Requirements

### For Classification Training

A folder called `Data/` with one subfolder per class, each containing 2D MRI images (JPG or PNG):

```
Data/
├── glioma_tumor/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── meningioma_tumor/
│   └── ...
├── normal/
│   └── ...
└── pituitary_tumor/
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

The dataset is split automatically (stratified by class):
- 75% training
- 15% validation
- 10% test

### For Segmentation Training

BraTS-style 3D NIfTI volumes. Each patient must have their own folder containing all 4 modality files and the ground truth segmentation mask:

```
training_data/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t2f.nii.gz   ← FLAIR
│   ├── BraTS-GLI-00000-000-t1n.nii.gz   ← T1 native
│   ├── BraTS-GLI-00000-000-t1c.nii.gz   ← T1 contrast
│   ├── BraTS-GLI-00000-000-t2w.nii.gz   ← T2 weighted
│   └── BraTS-GLI-00000-000-seg.nii.gz   ← ground truth mask
├── BraTS-GLI-00001-000/
│   └── ...
└── ...
```

### For Inference (Pipeline)

A single `.nii.gz` file (any modality) for classification and XAI.

For segmentation to also run (glioma cases only), all 4 modality files for that patient must exist in the same folder. The pipeline auto-detects this by stripping the modality suffix from the filename and searching for sibling files.

---

## Outputs

After running the pipeline, the `outputs/pipeline/` folder contains:

```
outputs/pipeline/
└── <patient_id>_segmentation.nii.gz    ← 3D segmentation mask (glioma only)
```

The matplotlib and napari windows are interactive — they are shown during the run and not saved to disk by default. To save figures, use `--no-display` and redirect output, or modify `display.py` to call `fig.savefig(...)` before `plt.show()`.

---

## Project Structure

```
Qynerva/
├── run.py                              ← single entry point for the full pipeline
├── train_classification.py             ← classification training CLI
├── train_segmentation.py               ← segmentation training CLI
├── pyproject.toml                      ← unified dependencies (one install for everything)
├── README.md
│
├── configs/
│   └── segmentation.yaml               ← segmentation model config
│
├── outputs/
│   ├── classification/
│   │   └── models/
│   │       ├── best_model.pth          ← best classification checkpoint
│   │       ├── final_model.pth         ← final classification checkpoint
│   │       └── class_to_idx.json       ← class mapping file
│   ├── segmentation/
│   │   └── best_model.pt               ← best segmentation checkpoint
│   └── pipeline/
│       └── <patient_id>_segmentation.nii.gz
│
├── src/
│   └── qynerva/
│       ├── classification/
│       │   ├── config.py               ← Config dataclass (all hyperparameters)
│       │   ├── data/
│       │   │   ├── dataset.py          ← BrainTumorDataset, transforms, DataLoaders
│       │   │   └── splitter.py         ← stratified train/val/test split
│       │   ├── models/
│       │   │   └── efficientnet.py     ← BrainTumorClassifier (EfficientNetB3 + head)
│       │   ├── training/
│       │   │   ├── callbacks.py        ← EarlyStopping, ModelCheckpoint
│       │   │   └── trainer.py          ← two-stage training loop
│       │   ├── prediction/
│       │   │   └── predictor.py        ← Predictor (loads model, runs inference on PIL images)
│       │   ├── volume/
│       │   │   ├── loader.py           ← MRIVolumeLoader (NIfTI → 2D PIL slices)
│       │   │   ├── inference.py        ← VolumeInference (classify every slice)
│       │   │   ├── aggregator.py       ← majority voting → VolumeReport + top-N selection
│       │   │   └── xai_runner.py       ← run HiResCAM on top-N slices → SliceXAIResult
│       │   └── xai/
│       │       ├── hirescam.py         ← generate_hirescam() — CAM map generation
│       │       └── visualization.py    ← generate_overlay() — blend cam on image
│       │
│       ├── segmentation/
│       │   ├── data/
│       │   │   ├── dataset.py          ← MONAI CacheDataset + DataLoader factory
│       │   │   ├── splits.py           ← train/val/test split for BraTS cases
│       │   │   └── transforms.py       ← MONAI transform pipelines (train/eval/predict)
│       │   ├── engine/
│       │   │   └── trainer.py          ← Trainer class + training loop + main_cli
│       │   ├── losses/
│       │   │   └── segmentation.py     ← DiceTverskyLoss (combined Dice + Tversky)
│       │   ├── models/
│       │   │   └── unet3d.py           ← build_model() — MONAI 3D U-Net
│       │   └── utils/
│       │       ├── config.py           ← load_config() — YAML loader
│       │       ├── io.py               ← discover_patients() — BraTS folder scanner
│       │       ├── metrics.py          ← SegmentationMetrics — Dice score computation
│       │       └── seed.py             ← set_seed() — full reproducibility setup
│       │
│       └── pipeline/
│           ├── runner.py               ← run_pipeline() — full orchestration + auto-detection
│           └── display.py              ← show_classification(), show_xai(), show_segmentation()
│
└── Synthetic Brain MRI Image Generation/      ← Model 5 [under training]
    └── brainmrdiff/
        ├── configs/
        │   └── default.yaml            ← all diffusion model hyperparameters
        ├── scripts/
        │   ├── preprocess.py           ← BraTS preprocessing wrapper
        │   ├── train.py                ← diffusion model training entry point
        │   ├── generate.py             ← synthetic MRI generation (DDPM / DDIM)
        │   └── evaluate.py             ← PSNR / SSIM / Dice evaluation
        └── src/
            └── brain_mri_diffusion/
                ├── data/
                │   ├── preprocessing.py   ← BraTSPreprocessor (normalize, resize, mask extraction)
                │   └── dataset.py         ← BraTSDataset + get_dataloaders
                ├── models/
                │   ├── tsa.py             ← Tumor+Structure Aggregation module (5-mask conditioning)
                │   ├── unet.py            ← ConditionalUNet (4 encoder/decoder stages + AdaGN)
                │   └── diffusion.py       ← GaussianDiffusion (DDPM/DDIM scheduler + sampling)
                ├── training/
                │   └── trainer.py         ← DiffusionTrainer (MSE + optional TGAP loss)
                ├── evaluation/
                │   └── metrics.py         ← PSNR, SSIM, Dice computation
                └── utils/
                    ├── logging.py
                    └── checkpoint.py
```

---

## Technical Reference

### Majority Voting Logic

The aggregator counts how many slices voted for each class. In case of a tie, the class that comes first alphabetically is chosen (for reproducibility). The top-N slices for XAI are then selected from slices that voted for the winning class, ranked by their softmax confidence score.

### Sliding Window Inference

At segmentation inference time, the full 3D volume cannot fit in GPU memory as a single forward pass. MONAI's `sliding_window_inference` tiles the volume into overlapping 128×128×128 patches, runs the model on each patch, and blends the outputs (using Gaussian weighting at patch borders) to produce the full-resolution prediction.

### Mixed Precision Training

Both models support FP16 automatic mixed precision (AMP) on CUDA GPUs. This halves memory usage and speeds up training by ~1.5–2× with no meaningful accuracy loss. AMP is automatically disabled on CPU.

### Reproducibility

The segmentation trainer calls `set_seed()` which sets seeds for Python random, NumPy, PyTorch, and CUDA, and enables `torch.backends.cudnn.deterministic = True`. This ensures fully reproducible training runs given the same seed and hardware.

### BraTS Label Remapping

The BraTS dataset uses label values 0, 1, 2, 4 (there is no label 3 in the original). Before training, label 4 is remapped to 3 using MONAI's `MapLabelValued` transform. This allows clean one-hot encoding with exactly 4 channels (background + 3 tumor regions) without any gaps in the index space.
