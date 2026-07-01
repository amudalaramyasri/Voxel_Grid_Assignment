# How to Run This Project — Step by Step

Run these commands in order, from inside the project folder.

## 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download the dataset

```bash
pip install kagglehub
python download_dataset.py
```

This downloads the SARS-COV-2 Ct-Scan Dataset and places it into
`dataset/COVID/` and `dataset/non-COVID/` automatically. You may be
prompted to authenticate with Kaggle the first time — follow the on-screen
instructions.

(Alternative: if you'd rather use the Kaggle CLI instead, see README.md
section 2, Option B.)

## 4. Train the model

```bash
python train.py
```

Trains a MobileNetV2-based classifier in two phases (frozen backbone, then
fine-tuning). Takes roughly 15–40 minutes on a laptop CPU. Saves the
trained model to `models/mobilenetv2_covid_ct.keras`.

## 5. Evaluate the model

```bash
python evaluate.py
```

Runs the trained model on the held-out test set and saves metrics
(`results/metrics.json`) and figures (training curves, confusion matrix,
ROC curve, sample predictions) under `results/`.

## 6. Generate the report

```bash
python generate_report.py
```

Builds `report.pdf` using the real metrics and figures from step 5.

## 7. (Optional) Run inference on a single image

```bash
python inference.py path/to/some_image.png
```

## 8. (Optional) Talk to the project assistant

```bash
python assistant/assistant.py
```

Works immediately with no setup (rule-based answers). For more natural,
free-form answers, run a local model via Ollama first:

```bash
ollama pull llama3.2:1b
ollama serve
python assistant/assistant.py
```

---

That's the full pipeline: **venv → install → dataset → train → evaluate →
report**. Steps 7 and 8 are optional extras you can run any time after
step 4.

See `README.md` for more detail on each step, and `report.pdf` for the
full written report.
