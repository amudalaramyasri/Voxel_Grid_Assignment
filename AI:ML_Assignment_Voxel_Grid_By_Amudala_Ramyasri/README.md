# COVID-19 CT-Scan Classification (CPU-only)

Binary classification of chest CT-scan images as **COVID** / **non-COVID**,
built as a CPU-only, laptop-reproducible pipeline using transfer learning.

See `report.pdf` for the full writeup — problem framing, dataset, design
decisions, evaluation, failure analysis, and improvement plan.

## Repository structure

```
project/
├── README.md
├── requirements.txt
├── report.pdf
├── config.py                # all paths / hyperparameters in one place
├── data_utils.py              # dataset loading + splitting (shared)
├── train.py
├── evaluate.py
├── inference.py
├── generate_report.py          # rebuilds report.pdf from results/
├── download_dataset.py          # fetches the dataset via kagglehub
├── dataset/                       # put the downloaded dataset here (see below)
├── models/                         # trained model weights land here
├── results/                         # metrics.json, figures/, sample_predictions/
└── assistant/                        # Part 6 — lightweight AI assistant
    ├── knowledge.py
    └── assistant.py
```

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.9–3.12. No GPU required — `tensorflow-cpu` is used
intentionally, and every script in this repo runs on CPU only.

## 2. Get the dataset

This project uses the **SARS-COV-2 Ct-Scan Dataset**
(Kaggle: `plameneduardo/sarscov2-ctscan-dataset`, Soares et al., ~2,482 CT
images from 120 patients). It isn't bundled in this repo (size + license).

**Option A — kagglehub (simplest):**

```bash
pip install kagglehub
python download_dataset.py
```

This downloads the dataset and copies it into `dataset/COVID` and
`dataset/non-COVID` automatically. First run may prompt for Kaggle auth —
follow the prompt, or set up `~/.kaggle/kaggle.json` beforehand.

**Option B — Kaggle CLI:**

```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p dataset --unzip
```

Either way, `dataset/` should end up looking like:

```
dataset/
├── COVID/
│   ├── img1.png
│   └── ...
└── non-COVID/
    ├── img1.png
    └── ...
```

(If your unzip produces nested folders, move the two class folders directly
under `dataset/`.)

## 3. Train

```bash
python train.py
```

Builds a stratified train/val/test split, then trains a MobileNetV2-based
classifier in two phases — first with the backbone frozen (only the new
classification head learns), then with the top layers of the backbone
unfrozen and fine-tuned at a lower learning rate. Saves:

- `models/mobilenetv2_covid_ct.keras`
- `results/training_history.json`
- `results/split_indices.json` (so evaluate.py always scores the same test set)

On a standard laptop CPU, expect roughly 15–40 minutes depending on core
count — see `config.py` to shorten `EPOCHS` / `BATCH_SIZE` for a quick run.

## 4. Evaluate

```bash
python evaluate.py
```

Produces, under `results/`:
- `metrics.json` — accuracy, precision, recall, F1, ROC-AUC
- `figures/training_curves.png`, `figures/confusion_matrix.png`, `figures/roc_curve.png`
- `sample_predictions/correct_predictions.png`, `sample_predictions/failure_cases.png`

## 5. Generate the report

```bash
python generate_report.py
```

Rebuilds `report.pdf` from whatever is currently in `results/` — run this
again any time after retraining to refresh the report with new numbers and
figures.

## 6. Run inference on a single image

```bash
python inference.py path/to/scan.png
```

## 7. Ask the project assistant (Part 6)

```bash
python assistant/assistant.py
```

Answers questions about the problem, dataset, model choice, training setup,
limitations, and how to interpret results. It works in layers, tried in
this order:

1. **The assignment's required questions** (what problem it solves, which
   dataset was used, why this model was selected, how it was trained, what
   the limitations are, how to interpret the results) are matched directly
   against the project's own facts, so these always answer correctly.
2. **Local LLM via Ollama** (free, fully offline) for anything else — gives
   more natural, free-form answers:
   ```bash
   ollama pull llama3.2:1b
   ollama serve                 # if not already running
   ```
   To use a different local model:
   ```bash
   export OLLAMA_MODEL=qwen2.5:1.5b
   ```
3. **Rule-based fallback** if Ollama isn't running — still answers from the
   same project facts, so the assistant works with zero setup.

## Notes on reproducibility

- All hyperparameters and paths live in `config.py`.
- `RANDOM_SEED = 42` is used for the split and for TF/numpy seeding.
- `results/split_indices.json` pins the exact train/val/test file lists so
  `evaluate.py` always scores the same held-out set `train.py` produced.

## Limitations

See `report.pdf` Parts 4–5, and the assistant's answer to "what are the
limitations" — summarized: single-source dataset, no 3D/volumetric context,
no localization of abnormal regions, not clinically validated. This is a
research/demo prototype, not a diagnostic device.
