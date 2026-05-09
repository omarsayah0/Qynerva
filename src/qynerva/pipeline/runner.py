"""
Unified Qynerva pipeline.

Flow:
  1. Load .nii.gz → extract 2D slices
  2. Classify every slice (EfficientNetB3) → majority vote → final class + confidence
  3. Show Figure 1: Classification (image + class + probability bar)
  4. Run HiResCAM XAI on top-N most confident slices
  5. Show Figure 2: XAI grid
  6. If final class == glioma_tumor AND a patient directory with 4 modalities exists:
       Run 3D U-Net segmentation → Show Figure 3: segmentation slices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

logger = logging.getLogger(__name__)

GLIOMA_CLASS = "glioma_tumor"
SEG_MODALITIES = ("t2f", "t1n", "t1c", "t2w")


# --------------------------------------------------------------------------- #
# Segmentation helpers
# --------------------------------------------------------------------------- #

def _find_patient_dir(nii_path: Path) -> tuple[Path, str] | tuple[None, None]:
    """Given a single .nii.gz file, find a patient directory with all 4 modalities.

    Returns (patient_dir, patient_id) or (None, None).
    Searches: same folder first, then scans the whole project tree.
    """
    stem = nii_path.name.replace(".nii.gz", "").replace(".nii", "")

    # Derive patient_id by stripping modality suffix
    patient_id = stem
    for mod in SEG_MODALITIES:
        if stem.endswith(f"-{mod}"):
            patient_id = stem[: -len(f"-{mod}")]
            break

    # 1. Check the same folder as the input file
    parent = nii_path.parent
    if all((parent / f"{patient_id}-{m}.nii.gz").exists() for m in SEG_MODALITIES):
        return parent, patient_id

    # 2. Search the whole project tree for a directory named patient_id
    #    Start from the grandparent of the input file and walk upward up to 5 levels
    search_root = nii_path.resolve().parent
    for _ in range(5):
        search_root = search_root.parent
        candidate = search_root / patient_id
        if candidate.is_dir() and all((candidate / f"{patient_id}-{m}.nii.gz").exists() for m in SEG_MODALITIES):
            return candidate, patient_id
        # Also search one level deeper (e.g. training_data1_v2/patient_id)
        for sub in search_root.rglob(patient_id):
            if sub.is_dir() and all((sub / f"{patient_id}-{m}.nii.gz").exists() for m in SEG_MODALITIES):
                return sub, patient_id

    return None, None


def _run_segmentation(patient_dir: Path, patient_id: str, seg_model_path: Path, seg_config_path: Path, device: str) -> np.ndarray | None:
    """Run 3D U-Net and return the predicted segmentation mask as a numpy array."""
    try:
        from qynerva.segmentation.utils.config import load_config
        from qynerva.segmentation.models.unet3d import build_model
        from qynerva.segmentation.data.transforms import get_predict_transforms
        from monai.inferers import sliding_window_inference
        from monai.data import Dataset, DataLoader, decollate_batch

        config = load_config(seg_config_path)
        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

        model = build_model(config).to(dev)
        ckpt = torch.load(seg_model_path, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        item = {
            "patient_id": patient_id,
            "image": [str(patient_dir / f"{patient_id}-{mod}.nii.gz") for mod in SEG_MODALITIES],
        }

        transforms = get_predict_transforms()
        loader = DataLoader(Dataset([item], transform=transforms), batch_size=1)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(dev)
                logits = sliding_window_inference(
                    images,
                    roi_size=tuple(config["data"]["patch_size"]),
                    sw_batch_size=1,
                    predictor=model,
                )
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0].astype(np.uint8)
                return pred

    except Exception as exc:
        logger.error("Segmentation failed: %s", exc, exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(
    nii_path: Path,
    cls_model_path: Path,
    cls_class_map_path: Path,
    seg_model_path: Path | None,
    seg_config_path: Path | None,
    top_n: int = 5,
    axis: str = "axial",
    output_dir: Path = Path("outputs/pipeline"),
    device: str | None = None,
    no_display: bool = False,
    no_chat: bool = False,
    gemini_key: str | None = None,
) -> None:
    from qynerva.classification.config import Config
    from qynerva.classification.prediction.predictor import Predictor
    from qynerva.classification.volume.loader import MRIVolumeLoader
    from qynerva.classification.volume.inference import VolumeInference
    from qynerva.classification.volume.aggregator import aggregate
    from qynerva.classification.volume.xai_runner import run_xai_on_top_slices
    from qynerva.pipeline.display import show_classification, show_xai, show_segmentation

    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = nii_path.name.replace(".nii.gz", "").replace(".nii", "")

    # Collect all results for the chatbot
    pipeline_results: dict = {
        "patient_id": patient_id,
        "axis": axis,
        "seg_performed": False,
        "seg_path": None,
    }

    # ── Config & predictor ──────────────────────────────────────────────────
    config = Config(output_dir=output_dir)
    if device:
        config.device = device

    predictor = Predictor(
        model_path=cls_model_path,
        class_map_path=cls_class_map_path,
        config=config,
    )

    # ── Step 1: extract slices ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Qynerva Pipeline — {nii_path.name}")
    print(f"{'='*60}")
    print(f"[1/4] Loading volume and extracting {axis} slices...")

    loader = MRIVolumeLoader(nii_path, axis=axis, skip_blank=True)
    pil_slices = loader.get_slices()

    if not pil_slices:
        logger.error("No usable slices found. Aborting.")
        return

    print(f"      {len(pil_slices)} slices extracted.")

    # ── Step 2: classify all slices ──────────────────────────────────────────
    print("[2/4] Classifying slices...")
    volume_inf = VolumeInference(predictor)
    slice_results = volume_inf.classify_slices(pil_slices)

    report = aggregate(patient_id=patient_id, slice_results=slice_results, top_n=top_n)

    # Store in results dict for chatbot
    pipeline_results.update({
        "final_class": report.final_class,
        "total_slices": report.total_slices,
        "class_percentages": report.class_percentages,
        "class_counts": report.class_counts,
        "top_slices": [
            {"slice_index": s.slice_index, "predicted_class": s.predicted_class, "confidence": s.confidence}
            for s in report.top_slices
        ],
    })

    print(f"\n  CLASSIFICATION RESULT: {report.final_class.upper()}")
    print(f"  Votes: ", end="")
    for cls, pct in sorted(report.class_percentages.items(), key=lambda x: -x[1]):
        marker = " <--" if cls == report.final_class else ""
        print(f"{cls}={pct:.1f}%{marker}  ", end="")
    print()

    # ── Step 3: XAI ─────────────────────────────────────────────────────────
    print("[3/4] Running HiResCAM XAI on top slices...")
    xai_results = run_xai_on_top_slices(report=report, model=predictor.model, alpha=0.4)

    # ── Display Figure 1: Classification ────────────────────────────────────
    if not no_display:
        best_slice = report.top_slices[0] if report.top_slices else slice_results[len(slice_results) // 2]
        show_classification(
            original_image=best_slice.original_image,
            predicted_class=report.final_class,
            confidence=max(s.confidence for s in report.top_slices) if report.top_slices else best_slice.confidence,
            class_probabilities=best_slice.class_probabilities,
            patient_id=patient_id,
        )

    # ── Display Figure 2: XAI ───────────────────────────────────────────────
    if not no_display:
        show_xai(xai_results, patient_id=patient_id)

    # ── Step 4: Segmentation (glioma only) ───────────────────────────────────
    if report.final_class != GLIOMA_CLASS:
        print(f"[4/4] Skipping segmentation — class is '{report.final_class}' (not glioma).")
    else:
        print(f"[4/4] Glioma detected — running 3D segmentation...")

        if seg_model_path is None or not seg_model_path.exists():
            print("      Segmentation model not found. Skipping.")
            print(f"      Pass --seg-model <path> to enable segmentation.")
        elif seg_config_path is None or not seg_config_path.exists():
            print("      Segmentation config not found. Skipping.")
        else:
            patient_dir, seg_patient_id = _find_patient_dir(nii_path)
            if patient_dir is None:
                print("      Could not find a patient directory with all 4 modalities (t2f, t1n, t1c, t2w).")
                print(f"      Make sure all 4 files exist in the same folder as your input.")
            else:
                print(f"      Patient directory: {patient_dir}")
                seg_mask = _run_segmentation(
                    patient_dir=patient_dir,
                    patient_id=seg_patient_id,
                    seg_model_path=seg_model_path,
                    seg_config_path=seg_config_path,
                    device=config.device,
                )

                if seg_mask is not None:
                    ref = nib.load(str(nii_path))
                    out_seg = output_dir / f"{seg_patient_id}_segmentation.nii.gz"
                    nib.save(nib.Nifti1Image(seg_mask, ref.affine, ref.header), out_seg)
                    print(f"      Segmentation saved: {out_seg}")

                    pipeline_results["seg_performed"] = True
                    pipeline_results["seg_path"] = str(out_seg)

                    if not no_display:
                        show_segmentation(
                            volume=loader.volume,
                            seg_mask=seg_mask,
                            patient_id=patient_id,
                        )
                else:
                    print("      Segmentation could not be completed.")

    print(f"\n  Pipeline complete. Outputs saved to: {output_dir.resolve()}")
    print(f"{'='*60}\n")

    # ── Step 5: Medical report + chatbot ────────────────────────────────────
    if not no_chat:
        _run_chatbot(pipeline_results, gemini_key)


def _run_chatbot(pipeline_results: dict, gemini_key: str | None) -> None:
    from qynerva.pipeline.chatbot import generate_report, save_pdf

    print(f"{'='*60}")
    print("  STEP 5 — Generating Medical Report (PDF)")
    print(f"{'='*60}")

    try:
        report = generate_report(pipeline_results, api_key=gemini_key)

        print()
        print(f"{'='*60}")
        print("  MEDICAL REPORT")
        print(f"{'='*60}")
        print()

        for line in report.split("\n"):
            print(f"  {line}")

        print()

        patient_id = pipeline_results["patient_id"]
        out_dir = Path("outputs/pipeline")

        # Save plain-text version
        txt_path = out_dir / f"{patient_id}_medical_report.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Text report saved : {txt_path}")

        # Save PDF version
        pdf_path = out_dir / f"{patient_id}_medical_report.pdf"
        save_pdf(report, pdf_path)
        print(f"  PDF  report saved : {pdf_path}")
        print()

    except Exception as exc:
        print(f"  [Report error: {exc}]")
        print("  Skipping report generation. Check your Gemini API key.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _find_default_cls_model() -> Path | None:
    candidates = [
        Path("outputs/classification/models/final_model.pth"),
        Path("outputs/classification/models/best_model.pth"),
        Path("qynerva_classification_project/outputs/models/final_model.pth"),
        Path("qynerva_classification_project/outputs/models/best_model.pth"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_default_cls_class_map() -> Path | None:
    candidates = [
        Path("outputs/classification/models/class_to_idx.json"),
        Path("qynerva_classification_project/outputs/models/class_to_idx.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_default_seg_model() -> Path | None:
    candidates = [
        Path("outputs/segmentation/best_model.pt"),
        Path("outputs/segmentation/checkpoint.pt"),
        Path("brats_seg_project/brats_seg_project/outputs/best_model.pt"),
        Path("brats_seg_project/brats_seg_project/outputs/checkpoint.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_default_seg_config() -> Path | None:
    candidates = [
        Path("configs/segmentation.yaml"),
        Path("brats_seg_project/brats_seg_project/configs/train.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="qynerva-run",
        description="Qynerva unified pipeline: classification → XAI → segmentation (glioma).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, metavar="PATH",
                        help="Path to a .nii or .nii.gz MRI file.")
    parser.add_argument("--cls-model", type=Path, default=None,
                        help="Path to classification model .pth checkpoint.")
    parser.add_argument("--cls-class-map", type=Path, default=None,
                        help="Path to class_to_idx.json.")
    parser.add_argument("--seg-model", type=Path, default=None,
                        help="Path to segmentation model checkpoint .pt (optional — only used for glioma).")
    parser.add_argument("--seg-config", type=Path, default=Path("configs/segmentation.yaml"),
                        help="Path to segmentation config YAML.")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top-confidence slices for XAI.")
    parser.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Axis for 2D slice extraction.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline"),
                        help="Directory for saved outputs.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu or cuda.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip matplotlib windows (useful in headless environments).")
    parser.add_argument("--no-chat", action="store_true",
                        help="Skip the medical report PDF generation step.")
    parser.add_argument("--mistral-key", type=str, default=None,
                        help="Mistral API key. Falls back to MISTRAL_API_KEY env var if not set.")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Auto-detect all model paths
    cls_model = args.cls_model or _find_default_cls_model()
    cls_map = args.cls_class_map or _find_default_cls_class_map()
    seg_model = args.seg_model or _find_default_seg_model()
    seg_config = args.seg_config if args.seg_config.exists() else _find_default_seg_config()

    if cls_model is None or not cls_model.exists():
        print("ERROR: Classification model not found.")
        print("  Place it at: outputs/classification/models/final_model.pth")
        sys.exit(1)

    if cls_map is None or not cls_map.exists():
        print("ERROR: class_to_idx.json not found.")
        print("  Place it at: outputs/classification/models/class_to_idx.json")
        sys.exit(1)

    if seg_model:
        logger.info("Segmentation model: %s", seg_model)
    else:
        logger.info("No segmentation model found — segmentation will be skipped even if glioma is detected.")

    run_pipeline(
        nii_path=args.input,
        cls_model_path=cls_model,
        cls_class_map_path=cls_map,
        seg_model_path=seg_model,
        seg_config_path=seg_config,
        top_n=args.top_n,
        axis=args.axis,
        output_dir=args.output_dir,
        device=args.device,
        no_display=args.no_display,
        no_chat=args.no_chat,
        gemini_key=args.mistral_key,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
