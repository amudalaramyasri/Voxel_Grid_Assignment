"""
generate_report.py — Builds report.pdf from static content + (optionally)
real results/metrics.json + results/figures/*.png if they exist.

Run this AFTER train.py + evaluate.py on the real dataset to get a report
populated with your actual numbers and plots. If results aren't present yet,
it builds the report with clearly marked placeholders so the structure and
narrative are still complete.
"""

import json
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, ListFlowable, ListItem
)

import config

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=14, spaceBefore=6, textColor=colors.HexColor("#1a1a2e")))
styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceAfter=8, spaceBefore=16, textColor=colors.HexColor("#16213e")))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, spaceAfter=8))
styles.add(ParagraphStyle(name="Note", fontSize=9.5, leading=13, spaceAfter=8, textColor=colors.HexColor("#7a1f1f")))
styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=11, textColor=colors.HexColor("#555555")))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def note(text):
    story.append(Paragraph(f"<b>Note:</b> {text}", styles["Note"]))


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, styles["Body"])) for i in items],
                               bulletType="bullet", leftIndent=18))


def image_if_exists(path, width=4.5 * inch, caption=None):
    if os.path.exists(path):
        story.append(Image(path, width=width, height=width * 0.65))
        if caption:
            story.append(Paragraph(caption, styles["Small"]))
        story.append(Spacer(1, 10))
        return True
    else:
        story.append(Paragraph(f"[Figure not yet generated: {os.path.basename(path)} — "
                                f"run train.py and evaluate.py to produce it]", styles["Note"]))
        return False


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
h1("COVID-19 CT-Scan Classification — Project Report")
body(
    "AI/ML Engineer Hiring Assignment submission. Problem: binary classification "
    "of chest CT-scan images into COVID / non-COVID. All training and evaluation "
    "are designed to run on CPU only."
    "Submitted by Amudala Ramyasri"
)
story.append(Spacer(1, 6))

# ---------------------------------------------------------------------------
# Part 1 — Problem Definition
# ---------------------------------------------------------------------------
h2("Part 1 — Problem Definition")

body("<b>Problem chosen:</b> Disease classification — binary classification of chest "
     "CT-scan images as <i>COVID-19 positive</i> or <i>non-COVID</i>.")

body("<b>Why it is clinically relevant:</b> During surges in respiratory illness, "
     "RT-PCR testing capacity and turnaround time can become a bottleneck. Chest CT "
     "imaging is widely available in hospitals and is sensitive to characteristic "
     "findings (e.g. ground-glass opacities) that can appear even when PCR results "
     "are pending or borderline. An automated triage classifier does not replace "
     "PCR or radiologist judgment, but can help flag scans for prioritized review "
     "and provide a fast, consistent second signal in resource-constrained settings.")

body("<b>Why this problem was selected:</b> It is a well-scoped binary classification "
     "task with a clean, publicly available, moderately sized dataset — large enough "
     "to be meaningful, small enough to train and iterate on entirely on CPU within "
     "the assignment's time and compute constraints, while still being a realistic, "
     "clinically motivated imaging problem rather than a toy task.")

# ---------------------------------------------------------------------------
# Part 2 — Dataset
# ---------------------------------------------------------------------------
h2("Part 2 — Dataset")

dataset_table_data = [
    ["Property", "Value"],
    ["Source", "Kaggle — \"SARS-COV-2 Ct-Scan Dataset\" (plameneduardo/sarscov2-ctscan-dataset)"],
    ["Original authors", "Soares, E. et al. — SARS-CoV-2 CT-scan dataset"],
    ["Total images", "~2,482 CT-scan images"],
    ["Patients", "120 patients (hospitals in Sao Paulo, Brazil)"],
    ["Classes", "2 — COVID, non-COVID"],
    ["Class balance", "Roughly balanced (~1,252 COVID / ~1,230 non-COVID per the original dataset card)"],
    ["Format", "PNG images, variable resolution, RGB-encoded grayscale CT slices"],
]
t = Table(dataset_table_data, colWidths=[1.6 * inch, 4.6 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f8")]),
]))
story.append(t)
story.append(Spacer(1, 10))

body("<b>Challenges within the dataset:</b>")
bullets([
    "Single-source data: all images come from one set of hospitals/scanners, so the model "
    "may not generalize to other CT equipment, acquisition protocols, or populations.",
    "Patient-level leakage risk: multiple slices can come from the same patient. Without "
    "patient-aware splitting, the same patient's anatomy could appear in both train and "
    "test sets, inflating apparent performance. (See Part 3 for how this is handled / its "
    "limitation in the current split.)",
    "Variable image quality and framing — some slices include more of the chest cavity "
    "than others, and pre-processing (e.g. windowing) is not standardized across sources.",
    "No pixel-level (segmentation) labels — only image-level COVID/non-COVID labels, so "
    "the model cannot be supervised to localize abnormal regions directly.",
])

body("<b>Why this dataset was chosen:</b> it is publicly available, ethically sourced and "
     "de-identified, and a reasonable size for CPU-only training — large enough to train "
     "a meaningful classifier, small enough to iterate on quickly without a GPU.")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# Part 3 — Model Development
# ---------------------------------------------------------------------------
h2("Part 3 — Model Development")

body("<b>Data preprocessing:</b> Images are read with OpenCV, converted BGR→RGB, resized "
     f"to {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} (a deliberate choice to keep "
     "per-image compute low enough for CPU training while retaining enough detail for "
     "the classifier), and scaled to [0,1]. MobileNetV2's expected preprocessing "
     "(channel-wise scaling/centering) is applied inside the model graph, so the saved "
     "model is self-contained for inference.")

body("<b>Data augmentation:</b> Random horizontal flip, small rotation (±5%), small zoom "
     "(±10%), and mild contrast jitter — implemented as Keras preprocessing layers so they "
     "run as part of the training graph (only during training, not at eval/inference). "
     "Augmentation is intentionally light: CT images are not natural photos, so aggressive "
     "color/hue jitter or large rotations could distort clinically meaningful structure.")

body("<b>Train/Validation/Test split:</b> Stratified, file-level split — "
     f"{int(config.TEST_SPLIT * 100)}% held out as test, then "
     f"{int(config.VAL_SPLIT * 100)}% of the remainder held out as validation "
     f"(net ≈{int((1 - config.TEST_SPLIT) * (1 - config.VAL_SPLIT) * 100)}% train / "
     f"{int((1 - config.TEST_SPLIT) * config.VAL_SPLIT * 100)}% val / "
     f"{int(config.TEST_SPLIT * 100)}% test), seeded for reproducibility "
     f"(seed={config.RANDOM_SEED}). The exact split is saved to "
     "results/split_indices.json so evaluate.py always scores the same held-out set.")

note("Known limitation: the split is stratified by class but not by patient. Because "
     "this dataset's metadata doesn't cleanly expose a patient ID per file in the public "
     "release, some patient overlap between splits is possible, which can optimistically "
     "bias test metrics. This is flagged explicitly here and addressed as a priority "
     "item in Part 5 (Improving the Model).")

body("<b>Model architecture:</b> MobileNetV2 (ImageNet-pretrained) as a feature extractor, "
     f"with a custom head: GlobalAveragePooling2D → Dense({config.DENSE_UNITS}, ReLU) → "
     f"Dropout({config.DROPOUT}) → Dense(1, sigmoid).")

body("<b>Why MobileNetV2:</b> Several ImageNet-pretrained CNN backbones were considered "
     "for this task — ResNet50, Xception, DenseNet121/201, MobileNet, and MobileNetV2.")
bullets([
    "CPU feasibility: MobileNetV2 was designed for mobile/edge inference — it has far "
    "fewer FLOPs and parameters (~3.4M) than deeper backbones like ResNet50 (~25M) or "
    "DenseNet201 (~20M), which directly translates to faster training and inference on "
    "CPU-only hardware, the hard constraint for this assignment.",
    "Acceptable accuracy trade-off: lightweight backbones like MobileNetV2 are well "
    "documented to trade a few points of top-line accuracy for a large reduction in "
    "compute cost compared to deeper networks. Under a CPU-only constraint, that "
    "trade-off favors a model that actually finishes training in a reasonable time "
    "over one that's marginally more accurate but impractical to iterate on locally.",
    "Transfer learning is well-justified here: a few thousand images is small for "
    "training a CNN from scratch, so ImageNet features (edges, textures, shapes) give "
    "a strong starting point that a randomly-initialized network of any size could not "
    "match with this little data.",
])

body("<b>Loss function:</b> Binary cross-entropy — natural choice for a single-sigmoid-"
     "output binary classifier with roughly balanced classes.")

body(f"<b>Optimizer:</b> Adam. <b>Learning rate:</b> {config.LEARNING_RATE} for the head-only "
     f"phase, dropped to {config.FINE_TUNE_LEARNING_RATE} for fine-tuning (with "
     "ReduceLROnPlateau on validation loss) — small fine-tuning LR avoids destroying "
     "the pretrained features when unfreezing the top of the backbone.")

body(f"<b>Batch size:</b> {config.BATCH_SIZE} — kept small deliberately, since smaller "
     "batches keep peak memory and per-step latency manageable on CPU-only hardware.")

body(f"<b>Epochs:</b> up to {config.EPOCHS}, split into two phases — "
     f"phase 1 (epochs 1–{config.FINE_TUNE_AT_EPOCH}): backbone frozen, only the new "
     f"head trains; phase 2 (remaining epochs): top ~30 layers of the backbone are "
     "unfrozen and fine-tuned. EarlyStopping (patience 5, restores best weights) and "
     "ModelCheckpoint (best val_loss) prevent overfitting and wasted CPU time.")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# Part 4 — Evaluation
# ---------------------------------------------------------------------------
h2("Part 4 — Evaluation")

metrics = None
if os.path.exists(config.METRICS_PATH):
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)

if metrics:
    body("Results below are from this project's own trained model "
         "(results/metrics.json), evaluated on the held-out test split.")
    metrics_table = [["Metric", "Value"]] + [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
                                              for k, v in metrics.items()]
else:
    note("No results/metrics.json was found yet — run train.py then evaluate.py, then "
         "re-run this script (python generate_report.py) to populate this section with "
         "this model's actual evaluation numbers and figures.")
    metrics_table = [
        ["Metric", "Value"],
        ["accuracy", "—"],
        ["precision", "—"],
        ["recall", "—"],
        ["f1", "—"],
        ["roc_auc", "—"],
    ]

t2 = Table(metrics_table, colWidths=[2.2 * inch] + [0.95 * inch] * (len(metrics_table[0]) - 1))
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f8")]),
]))
story.append(t2)
story.append(Spacer(1, 12))

h2("Training & validation curves")
image_if_exists(os.path.join(config.FIGURES_DIR, "training_curves.png"),
                 caption="Training/validation accuracy and loss across both training phases.")

h2("Confusion matrix")
image_if_exists(os.path.join(config.FIGURES_DIR, "confusion_matrix.png"), width=3.2 * inch)

h2("ROC curve")
image_if_exists(os.path.join(config.FIGURES_DIR, "roc_curve.png"), width=3.2 * inch)

h2("Sample predictions")
image_if_exists(os.path.join(config.SAMPLES_DIR, "correct_predictions.png"),
                 caption="Correctly classified examples.")
image_if_exists(os.path.join(config.SAMPLES_DIR, "failure_cases.png"),
                 caption="Misclassified examples (failure cases).")

h2("Discussion")
body("<b>Where the model performs well (expected):</b> Cases with clear, classic "
     "ground-glass or consolidation patterns typical of the training distribution, and "
     "images framed/cropped similarly to the bulk of the training data.")
body("<b>Where the model is expected to fail:</b> Borderline/early-stage cases with subtle "
     "findings; images with atypical framing, noise, or artifacts not well represented in "
     "training; non-COVID cases with other lung pathology that visually resembles COVID "
     "findings (a known confusion in this literature).")
body("<b>Possible reasons for failure:</b> Limited dataset diversity (single source), "
     "image-level labels without abnormality localization (the model can't \"point at\" "
     "the relevant region, making subtle cases harder to fit), and possible patient-level "
     "split leakage masking some generalization weaknesses until tested on a truly external "
     "patient/hospital.")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# Part 5 — Improving the Model
# ---------------------------------------------------------------------------
h2("Part 5 — Improving the Model")
body("If this were to become part of a real-world medical imaging product, priorities "
     "in roughly the order I'd tackle them:")

improvements = [
    ("Patient-aware, multi-site evaluation", 
     "Re-split (and ideally re-collect) data so no patient appears in both train and "
     "test, and add an external test set from a different hospital/scanner. "
     "<b>Benefit:</b> honest estimate of real-world generalization. "
     "<b>Trade-off:</b> requires more data-sourcing effort and may reveal a real "
     "accuracy drop versus the single-source numbers reported here."),
    ("Better preprocessing",
     "CT-specific windowing/normalization (e.g. lung-window leveling) instead of "
     "generic RGB scaling. <b>Benefit:</b> emphasizes clinically relevant contrast. "
     "<b>Trade-off:</b> needs DICOM-level metadata, which this PNG-only public dataset "
     "doesn't provide — would require sourcing a DICOM dataset."),
    ("Transfer learning from a domain-specific foundation model",
     "Use a chest-CT or medical-imaging pretrained backbone (e.g. CT-CLIP-style or "
     "RadImageNet weights) instead of generic ImageNet weights. "
     "<b>Benefit:</b> features already tuned to medical imaging statistics, likely "
     "better small-data performance. <b>Trade-off:</b> larger/heavier checkpoints, "
     "license/availability constraints, possibly GPU-only inference, eroding the "
     "CPU-only property of this build."),
    ("Self-supervised / semi-supervised pretraining on unlabeled CT scans",
     "Pretrain on a larger pool of unlabeled chest CTs before fine-tuning on labeled "
     "COVID data. <b>Benefit:</b> reduces dependence on scarce labels. "
     "<b>Trade-off:</b> meaningfully more compute and engineering complexity than "
     "transfer learning from existing weights."),
    ("Better / more diverse data collection",
     "Actively source data across more hospitals, scanners, and patient demographics. "
     "<b>Benefit:</b> the single highest-leverage way to improve real-world robustness. "
     "<b>Trade-off:</b> slow, costly, and subject to data-sharing/privacy agreements."),
    ("Calibration",
     "Apply temperature scaling or Platt scaling on a held-out calibration set so "
     "predicted probabilities reflect true likelihoods. <b>Benefit:</b> a 0.51 vs 0.93 "
     "COVID probability becomes meaningfully different to a clinician. "
     "<b>Trade-off:</b> needs a dedicated calibration split, and calibration can drift "
     "if the deployment population shifts from the calibration population."),
    ("Hyperparameter optimization",
     "Systematic search (e.g. small Bayesian or grid search) over learning rate, "
     "fine-tune depth, dropout, and augmentation strength. <b>Benefit:</b> likely a "
     "few more points of accuracy/F1 for free. <b>Trade-off:</b> multiplies compute "
     "cost, which competes directly with the CPU-only constraint."),
    ("Ensembling",
     "Combine 2–3 lightweight backbones (e.g. MobileNetV2 + a small EfficientNet) by "
     "averaging predictions. <b>Benefit:</b> typically a reliable, low-risk accuracy "
     "boost. <b>Trade-off:</b> multiplies inference cost and model maintenance burden "
     "for a relatively modest gain."),
    ("Localization / explainability (e.g. Grad-CAM)",
     "Even without segmentation labels, Grad-CAM-style saliency maps can show *where* "
     "the model is focusing. <b>Benefit:</b> critical for radiologist trust and for "
     "catching the model focusing on spurious artifacts instead of anatomy. "
     "<b>Trade-off:</b> visualization only — doesn't directly improve accuracy, and "
     "saliency maps can be unreliable/misleading if over-interpreted."),
    ("Active learning",
     "Once deployed (in a research/triage-support capacity), route low-confidence "
     "predictions to radiologists and feed their corrections back into retraining. "
     "<b>Benefit:</b> targets labeling effort where it matters most. "
     "<b>Trade-off:</b> needs a feedback/labeling pipeline and careful governance "
     "around using clinician corrections as training data."),
]
for title, txt in improvements:
    body(f"<b>{title}.</b> {txt}")

story.append(PageBreak())

# ---------------------------------------------------------------------------
# Part 6 — AI Assistant
# ---------------------------------------------------------------------------
h2("Part 6 — AI Assistant")

body("A lightweight project assistant lives in <font face='Courier'>assistant/</font> and "
     "is exposed via CLI (<font face='Courier'>python assistant/assistant.py</font>).")

body("<b>Design:</b> <font face='Courier'>assistant/knowledge.py</font> compiles a single "
     "\"project facts\" context block — problem statement, dataset summary, model "
     "rationale, training configuration, limitations, improvement plan, deployment "
     "guidance, and (live) the contents of results/metrics.json. "
     "<font face='Courier'>assistant/assistant.py</font> answers each question through "
     "a layered fallback chain, tried in order:")

bullets([
    "<b>Guaranteed-match layer:</b> the assignment's six required questions (what "
    "problem it solves, which dataset was used, why this model was selected, how it "
    "was trained, what the limitations are, how to interpret the results) — and close "
    "rewordings of them — are matched directly against the project-facts dictionary, "
    "so these always return a correct, complete answer regardless of which backend "
    "below is available.",
    "<b>Local open-source LLM via Ollama</b> (free, offline, no API key): for anything "
    "outside the six required questions, the same context block is passed as a system "
    "prompt to a small local model (e.g. llama3.2:1b), letting it reason and respond "
    "more naturally instead of relying on fixed keyword matches.",
    "<b>Rule-based keyword fallback:</b> if Ollama isn't running, a broader keyword "
    "router still answers from the same project-facts dictionary, so the assistant "
    "works out-of-the-box with zero setup, zero cost, and no internet dependency.",
])

body("It can answer the questions the assignment specifies: what problem the model "
     "solves, which dataset was used, why this model was selected, how it was trained, "
     "what its limitations are, and how to interpret its results — all grounded in the "
     "same facts documented in this report, so the assistant's answers stay consistent "
     "with the written report and with results/metrics.json as it gets updated.")

note("This is intentionally lightweight, per the assignment brief — a context-grounded "
     "Q&A layer over the project's own facts, not a production agent with tool use, "
     "memory, or multi-turn planning.")

doc = SimpleDocTemplate(
    "report.pdf", pagesize=letter,
    leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    topMargin=0.85 * inch, bottomMargin=0.85 * inch,
    title="COVID-19 CT-Scan Classification — Project Report",
)
doc.build(story)
print("Wrote report.pdf")
