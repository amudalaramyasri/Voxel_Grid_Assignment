"""
evaluate.py — Evaluate the trained model on the held-out test split.

Produces (all saved under results/):
  - metrics.json            : accuracy, precision, recall, F1, ROC-AUC
  - figures/training_curves.png
  - figures/confusion_matrix.png
  - figures/roc_curve.png
  - sample_predictions/*.png : a grid of correct + incorrect predictions

Usage:
    python evaluate.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

import config
from data_utils import load_split_arrays


def load_test_split():
    with open(config.SPLIT_INDEX_PATH) as f:
        splits = json.load(f)
    test = splits["test"]
    return test["files"], np.array(test["labels"])


def plot_training_curves(history_path, out_path):
    with open(history_path) as f:
        hist = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return cm


def plot_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return auc


def save_sample_predictions(X, y_true, y_pred, y_prob, test_fp, out_dir, n_correct=4, n_wrong=4):
    os.makedirs(out_dir, exist_ok=True)
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]

    def _grid(indices, fname, title):
        if len(indices) == 0:
            return
        n = min(len(indices), 8)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        if n == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices[:n]):
            ax.imshow(X[idx])
            ax.set_title(
                f"true={config.CLASS_NAMES[y_true[idx]]}\n"
                f"pred={config.CLASS_NAMES[y_pred[idx]]} ({y_prob[idx]:.2f})",
                fontsize=8,
            )
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

    _grid(correct_idx, "correct_predictions.png", "Correct predictions")
    _grid(wrong_idx, "failure_cases.png", "Failure cases")


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)

    print("Loading test split...")
    test_fp, test_y = load_test_split()
    X_test, y_test = load_split_arrays(test_fp, test_y)
    print(f"  test set: {X_test.shape}")

    print(f"Loading model from {config.MODEL_PATH} ...")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    y_prob = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    except ValueError:
        metrics["roc_auc"] = None

    print("Test metrics:", json.dumps(metrics, indent=2))
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if os.path.exists(config.HISTORY_PATH):
        plot_training_curves(config.HISTORY_PATH, os.path.join(config.FIGURES_DIR, "training_curves.png"))

    plot_confusion_matrix(y_test, y_pred, os.path.join(config.FIGURES_DIR, "confusion_matrix.png"))
    if metrics["roc_auc"] is not None:
        plot_roc_curve(y_test, y_prob, os.path.join(config.FIGURES_DIR, "roc_curve.png"))

    save_sample_predictions(X_test, y_test, y_pred, y_prob, test_fp, config.SAMPLES_DIR)

    print(f"\nSaved metrics + figures to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
