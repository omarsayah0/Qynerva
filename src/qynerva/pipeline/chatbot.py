"""
Mistral-powered medical PDF report generator.

Takes the pipeline results dict, calls the Mistral API to generate a
structured medical report, prints it, saves it as .txt and .pdf.
No interactive chat.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_API_KEY = "xFhT0TxBGEhDYDzSnEVpf1IlngOY5nYl"

_MODELS = [
    "mistral-large-latest",
    "mistral-small-latest",
    "open-mistral-7b",
]

_SYSTEM_PROMPT = """You are an expert medical AI assistant specializing in neuroradiology and brain tumor analysis.
You are given the results of an automated MRI analysis pipeline that includes:
- 2D slice-level classification using EfficientNetB3
- Explainability maps using HiResCAM
- 3D tumor segmentation using a U-Net (only for glioma cases)

Your role is to generate a clear, professional, and detailed medical report based on the analysis results.

Important:
- Always be clear that this is an AI-assisted analysis and not a final clinical diagnosis
- Use proper medical terminology while also providing plain-language explanations
- Be specific about confidence levels and uncertainty
- For glioma cases, discuss the significance of each tumor sub-region (NCR, ED, ET)
"""


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #

def _build_context(results: dict) -> str:
    patient_id        = results.get("patient_id", "Unknown")
    final_class       = results.get("final_class", "Unknown")
    total_slices      = results.get("total_slices", 0)
    class_percentages = results.get("class_percentages", {})
    class_counts      = results.get("class_counts", {})
    top_slices        = results.get("top_slices", [])
    seg_performed     = results.get("seg_performed", False)
    seg_path          = results.get("seg_path", None)
    axis              = results.get("axis", "axial")

    lines = [
        "=== PIPELINE ANALYSIS RESULTS ===",
        "",
        f"Patient ID            : {patient_id}",
        f"Slicing axis          : {axis}",
        f"Total slices analyzed : {total_slices}",
        "",
        "--- CLASSIFICATION RESULT ---",
        f"Final Diagnosis : {final_class.replace('_', ' ').upper()}",
        "",
        "Vote distribution across all slices:",
    ]

    for cls, pct in sorted(class_percentages.items(), key=lambda x: -x[1]):
        count = class_counts.get(cls, 0)
        marker = "  <-- FINAL DIAGNOSIS" if cls == final_class else ""
        lines.append(f"  {cls:<28}  {count:>4} slices  ({pct:.1f}%){marker}")

    lines += [
        "",
        "--- XAI (HiResCAM) ANALYSIS ---",
        f"Top {len(top_slices)} most confident slices selected for explanation:",
    ]
    for i, s in enumerate(top_slices):
        lines.append(
            f"  Slice {i+1}: index={s.get('slice_index','?')}, "
            f"class={s.get('predicted_class','?')}, "
            f"confidence={s.get('confidence',0)*100:.1f}%"
        )

    lines.append("")
    lines.append("HiResCAM heatmaps highlight which regions of the MRI most influenced the classification decision.")

    if seg_performed and seg_path:
        lines += [
            "",
            "--- 3D SEGMENTATION RESULT ---",
            "Segmentation was performed because the diagnosis is glioma.",
            f"Segmentation mask saved to: {seg_path}",
            "",
            "The 3D U-Net model segmented the tumor into the following sub-regions:",
            "  Label 0 -- Background (healthy brain tissue)",
            "  Label 1 -- Necrotic tumor core (NCR): dead/necrotic tissue at the center",
            "  Label 2 -- Peritumoral edema (ED): brain swelling and invasion zone",
            "  Label 3 -- Enhancing tumor (ET): active tumor cells with contrast uptake",
            "",
            "These regions follow the BraTS (Brain Tumor Segmentation) challenge convention.",
        ]
    elif final_class == "glioma_tumor":
        lines += ["", "--- SEGMENTATION ---", "Diagnosis is glioma but segmentation was not performed."]
    else:
        lines += ["", "--- SEGMENTATION ---", f"Segmentation was NOT performed -- diagnosis is '{final_class}' (not glioma)."]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Report generation (Mistral)
# --------------------------------------------------------------------------- #

def _make_client(key: str):
    """Return a Mistral client. Tries multiple import paths across SDK versions."""
    # Speakeasy-generated SDK (mistralai >= 1.x, class lives in mistralai.client)
    try:
        from mistralai.client import Mistral
        return Mistral(api_key=key)
    except (ImportError, Exception):
        pass
    # Flat import (some builds re-export at top level)
    try:
        from mistralai import Mistral
        return Mistral(api_key=key)
    except (ImportError, Exception):
        pass
    raise RuntimeError("Could not import Mistral client. Run: pip install --upgrade mistralai")


def _chat(client, model_name: str, system: str, user: str) -> str:
    response = client.chat.complete(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_report(results: dict, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("MISTRAL_API_KEY", _DEFAULT_API_KEY)
    client = _make_client(key)
    context = _build_context(results)

    prompt = f"""Based on the following automated MRI brain tumor analysis results, generate a comprehensive medical report.

{context}

Please structure the report with the following sections:
1. PATIENT INFORMATION
2. EXAMINATION METHOD
3. FINDINGS
   a. Classification Analysis
   b. Explainability Analysis (XAI)
   c. Segmentation Analysis (if applicable)
4. IMPRESSION
5. RECOMMENDATIONS
6. DISCLAIMER

Use professional medical language appropriate for a radiology report. Be specific about confidence levels and what each finding means clinically."""

    print("\n  Generating medical report...", end="", flush=True)

    last_err = None
    for model_name in _MODELS:
        try:
            text = _chat(client, model_name, _SYSTEM_PROMPT, prompt)
            print(f" done. (model: {model_name})")
            return text
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All Mistral models failed. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# PDF export
# --------------------------------------------------------------------------- #

def save_pdf(report_text: str, output_path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "QYNERVA -- AI-Assisted MRI Brain Tumor Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Body
    pdf.set_font("Helvetica", size=10)
    for line in report_text.split("\n"):
        stripped = line.strip()
        if stripped and (stripped.isupper() or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in (".", " "))):
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.ln(3)
            pdf.multi_cell(0, 6, line)
            pdf.set_font("Helvetica", size=10)
        else:
            pdf.multi_cell(0, 5, line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
