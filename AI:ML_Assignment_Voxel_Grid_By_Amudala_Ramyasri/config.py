"""
Central configuration for the COVID-19 CT-Scan Classification project.

Keeping every tunable in one place makes the design decisions easy to find,
explain, and change without hunting through train.py / evaluate.py / inference.py.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")          # expects COVID/ and non-COVID/ subfolders
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "sample_predictions")

MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_covid_ct.keras")
HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
SPLIT_INDEX_PATH = os.path.join(RESULTS_DIR, "split_indices.json")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CLASS_NAMES = ["non-COVID", "COVID"]   # label 0, label 1 (kept consistent with original repo)
IMAGE_SIZE = (160, 160)                # smaller than the original 200x200 -> ~30% fewer pixels, friendlier for CPU
CHANNELS = 3

# Split ratios (applied as: first carve out TEST, then carve VAL out of remaining TRAIN)
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15           # fraction of the remaining (non-test) data
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
BACKBONE = "MobileNetV2"   # chosen for CPU feasibility, see report.pdf Part 3 for justification
BATCH_SIZE = 16             # smaller than the original 64 -> realistic for CPU memory/throughput
EPOCHS = 15
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT_EPOCH = 8       # epoch at which we unfreeze the top backbone layers for fine-tuning
DROPOUT = 0.3
DENSE_UNITS = 128

# Reproducibility
import random
import numpy as np


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
