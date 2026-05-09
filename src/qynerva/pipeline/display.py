"""
Matplotlib visualization for the unified pipeline.

Three separate figures are shown in sequence:
  1. Classification  — representative slice + class label + confidence bar
  2. XAI             — top-N slices with HiResCAM overlays
  3. Segmentation    — axial slices with tumor-region overlay (glioma only)
"""

from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Figure 1 — Classification
# --------------------------------------------------------------------------- #

def show_classification(
    original_image: np.ndarray,
    predicted_class: str,
    confidence: float,
    class_probabilities: dict[str, float],
    patient_id: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left — representative MRI slice
    ax_img = axes[0]
    ax_img.imshow(original_image, cmap="gray")
    ax_img.axis("off")
    ax_img.set_title(
        f"{'Patient: ' + patient_id + '  |  ' if patient_id else ''}Predicted class: {predicted_class}\nConfidence: {confidence * 100:.1f}%",
        fontsize=13, fontweight="bold",
    )

    # Right — probability bar chart
    ax_bar = axes[1]
    classes = list(class_probabilities.keys())
    probs = [class_probabilities[c] * 100 for c in classes]
    colours = ["#e74c3c" if c == predicted_class else "#3498db" for c in classes]

    bars = ax_bar.barh(classes, probs, color=colours)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Probability (%)", fontsize=10)
    ax_bar.set_title("Class probabilities", fontsize=11)
    for bar, prob in zip(bars, probs):
        ax_bar.text(min(prob + 1, 95), bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1f}%", va="center", fontsize=9)

    fig.suptitle("STEP 1 — Classification Result", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Figure 2 — XAI
# --------------------------------------------------------------------------- #

def show_xai(xai_results, patient_id: str = "") -> None:
    if not xai_results:
        print("[XAI] No results to display.")
        return

    n = len(xai_results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)

    title = f"STEP 2 — XAI: HiResCAM Explanations"
    if patient_id:
        title += f"  |  Patient: {patient_id}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for col, xr in enumerate(xai_results):
        # Row 0 — original slice
        ax_orig = axes[0][col]
        ax_orig.imshow(xr.original_image, cmap="gray")
        ax_orig.set_title(
            f"Slice {xr.slice_index}\n{xr.predicted_class}\nconf {xr.confidence * 100:.1f}%",
            fontsize=9,
        )
        ax_orig.axis("off")

        # Row 1 — HiResCAM overlay
        ax_cam = axes[1][col]
        ax_cam.imshow(xr.overlay_image)
        ax_cam.set_title("HiResCAM", fontsize=9)
        ax_cam.axis("off")

    axes[0][0].set_ylabel("Original", fontsize=10)
    axes[1][0].set_ylabel("Explanation", fontsize=10)

    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Figure 3 — Segmentation (napari interactive viewer)
# --------------------------------------------------------------------------- #

def show_segmentation(
    volume: np.ndarray,
    seg_mask: np.ndarray,
    patient_id: str = "",
) -> None:
    """
    Open a napari interactive viewer showing the MRI volume and the
    predicted segmentation mask as a labels layer.

    The viewer opens in 2D mode so the user can scroll through slices.
    A second viewer opens in 3D mode with MIP rendering.

    Args:
        volume:     3D float32 array (X, Y, Z) — one MRI modality.
        seg_mask:   3D uint8 array (X, Y, Z)   — label values 0-3.
        patient_id: Patient identifier shown in the window title.
    """
    import napari

    title_2d = f"STEP 3 — Segmentation (2D)  |  {patient_id}" if patient_id else "STEP 3 — Segmentation (2D)"
    title_3d = f"STEP 3 — Segmentation (3D)  |  {patient_id}" if patient_id else "STEP 3 — Segmentation (3D)"

    print("      Opening napari 2D viewer — scroll through slices, then close to continue...")
    viewer_2d = napari.Viewer(ndisplay=2, title=title_2d)
    viewer_2d.add_image(volume, name="MRI", colormap="gray")
    viewer_2d.add_labels(seg_mask.astype(int), name="Tumor Mask", opacity=0.4)
    napari.run()

    print("      Opening napari 3D viewer — close when done...")
    viewer_3d = napari.Viewer(ndisplay=3, title=title_3d)
    viewer_3d.add_image(volume, name="MRI", colormap="gray", rendering="mip")
    viewer_3d.add_labels(seg_mask.astype(int), name="Tumor Mask", opacity=0.5)
    napari.run()
