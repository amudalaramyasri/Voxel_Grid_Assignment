"""
assistant/knowledge.py — Compiles the facts about this project into a single
context block that the assistant (rule-based or LLM-based) grounds its
answers in. This is the "retrieval" half of the lightweight assistant:
everything it can say is traceable back to this file + results/metrics.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PROJECT_FACTS = {
    "problem": (
        "Binary classification of chest CT-scan images as COVID-19 positive "
        "or non-COVID, to help triage/flag scans for radiologist review."
    ),
    "clinical_relevance": (
        "CT-based COVID screening can support rapid triage when RT-PCR turnaround "
        "is slow or capacity-constrained, and CT is sensitive to characteristic "
        "ground-glass opacities even early in disease progression."
    ),
    "dataset": (
        "SARS-COV-2 Ct-Scan Dataset (Kaggle, Soares et al.): ~2,482 CT images "
        "from 120 patients, labeled COVID / non-COVID, collected from hospitals "
        "in Sao Paulo, Brazil. Roughly balanced between the two classes. "
        "Known weakness: single-source data (one set of hospitals/scanners), "
        "and the public release doesn't cleanly expose patient IDs, so the "
        "train/val/test split is stratified by class but not guaranteed to be "
        "patient-disjoint."
    ),
    "model": (
        f"{config.BACKBONE} pretrained on ImageNet, used as a frozen feature "
        f"extractor with a small classification head (GlobalAveragePooling -> "
        f"Dense({config.DENSE_UNITS}) -> Dropout({config.DROPOUT}) -> Dense(1, sigmoid)), "
        f"followed by light fine-tuning of the top backbone layers. Chosen over "
        f"larger backbones (ResNet50, DenseNet201, Xception used in earlier "
        f"experiments) because it trains and infers fast enough on CPU only, "
        f"while earlier results on this dataset showed only a small accuracy gap "
        f"versus the larger models (ResNet50 ~96.0% vs MobileNetV2 ~92.7% in the "
        f"earlier GPU experiments)."
    ),
    "training": (
        f"Images resized to {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}, pixel "
        f"values scaled to [0,1]. Stratified split: "
        f"{int((1 - config.TEST_SPLIT) * (1 - config.VAL_SPLIT) * 100)}% train / "
        f"{int((1 - config.TEST_SPLIT) * config.VAL_SPLIT * 100)}% val / "
        f"{int(config.TEST_SPLIT * 100)}% test. Light augmentation (flip, small "
        f"rotation/zoom, contrast jitter). Adam optimizer, binary cross-entropy "
        f"loss, batch size {config.BATCH_SIZE}, up to {config.EPOCHS} epochs with "
        f"early stopping and LR reduction on plateau. Phase 1 trains only the new "
        f"head (backbone frozen); phase 2 (from epoch {config.FINE_TUNE_AT_EPOCH}) "
        f"unfreezes the top ~30 backbone layers at a 100x lower learning rate."
    ),
    "limitations": (
        "Single-center-style dataset (limited hospital/scanner diversity) means "
        "the model may not generalize to other scanners, protocols, or "
        "populations. It classifies whole CT slices, not 3D volumes, and gives "
        "no localization of abnormal regions. It has not been validated against "
        "RT-PCR ground truth in a clinical pipeline and is not a diagnostic "
        "device -- it is a research/triage-support prototype only. The "
        "train/val/test split is also not guaranteed patient-disjoint, which "
        "could optimistically bias the reported test metrics."
    ),
    "interpretation": (
        "The model outputs a COVID probability in [0,1]; values near 0.5 should "
        "be treated as low-confidence and routed to a radiologist rather than "
        "trusted directly. Metrics (precision/recall/F1/ROC-AUC) describe "
        "expected behavior on data similar to the test split -- not a guarantee "
        "on new hospitals' data. See results/metrics.json and "
        "results/figures/confusion_matrix.png for the actual numbers and where "
        "the model tends to fail."
    ),
    "improvements": (
        "If this became a real product, priorities in order: (1) patient-aware, "
        "multi-site evaluation so no patient appears in both train and test, plus "
        "an external test set from a different hospital; (2) domain-specific "
        "pretrained weights (e.g. RadImageNet-style) instead of generic ImageNet; "
        "(3) hyperparameter search over learning rate, fine-tune depth, "
        "augmentation strength; (4) calibration (temperature scaling) so "
        "predicted probabilities are trustworthy; (5) ensembling a couple of "
        "lightweight backbones; (6) Grad-CAM-style saliency maps for "
        "explainability/trust; (7) active learning -- routing low-confidence "
        "predictions to radiologists and feeding corrections back into retraining."
    ),
    "deployment": (
        "This should only ever be deployed as a triage/second-signal tool "
        "alongside a radiologist, never as a standalone diagnostic. Predictions "
        "near 0.5 probability should be flagged for mandatory human review. "
        "Before any real deployment it would need: clinical validation against "
        "RT-PCR ground truth, patient-disjoint multi-site evaluation, "
        "regulatory/clinical-safety review, drift monitoring once live, and "
        "explainability tooling so radiologists can see why the model flagged "
        "a given scan."
    ),
    "assistant_design": (
        "This assistant itself is intentionally lightweight (per the assignment "
        "brief, not a production agent). It tries a local open-source LLM via "
        "Ollama first (free, offline, no API key), falls back to the Anthropic "
        "API if ANTHROPIC_API_KEY is set, and falls back further to a "
        "keyword-matched rule-based answer if neither LLM backend is available -- "
        "so it always works. In LLM mode it's grounded in this same project-facts "
        "file plus the live contents of results/metrics.json, and is instructed "
        "to say when it doesn't know something rather than guess."
    ),
}


def load_metrics():
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    return None


def build_context_block() -> str:
    """A single string with everything the assistant is allowed to draw on."""
    lines = ["# Project facts"]
    for key, value in PROJECT_FACTS.items():
        lines.append(f"\n## {key}\n{value}")

    metrics = load_metrics()
    lines.append("\n## evaluation_results")
    if metrics:
        lines.append(json.dumps(metrics, indent=2))
    else:
        lines.append(
            "Not available yet -- run train.py then evaluate.py to populate "
            "results/metrics.json."
        )
    return "\n".join(lines)
