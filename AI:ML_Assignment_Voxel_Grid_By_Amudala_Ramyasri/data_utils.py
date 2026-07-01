"""
Dataset loading and preprocessing utilities shared by train.py and evaluate.py.

Expected layout (matches the Kaggle "SARS-COV-2 Ct-Scan Dataset"):

    dataset/
        COVID/
            img1.png
            img2.png
            ...
        non-COVID/
            img1.png
            ...

Download instructions are in README.md (the dataset is not bundled in this
repo because of size / license, and this sandbox could not reach Kaggle to
fetch it directly).
"""

import json
import os

import numpy as np

import config


def list_image_files():
    """Return (filepaths, labels) for every image under config.DATASET_DIR."""
    filepaths, labels = [], []
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)
    return filepaths, np.array(labels)


def load_image(path):
    """Load a single image as a float32 array in [0, 1], resized to config.IMAGE_SIZE."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def build_splits():
    """
    Stratified train/val/test split at the *filepath* level (images are loaded
    lazily later) so this stays cheap even for large datasets.

    Returns a dict of {split_name: (filepaths, labels)}.
    """
    from sklearn.model_selection import train_test_split

    filepaths, labels = list_image_files()
    if len(filepaths) == 0:
        raise FileNotFoundError(
            f"No images found under {config.DATASET_DIR}. "
            "Did you download the dataset? See README.md."
        )

    train_val_fp, test_fp, train_val_y, test_y = train_test_split(
        filepaths, labels,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    train_fp, val_fp, train_y, val_y = train_test_split(
        train_val_fp, train_val_y,
        test_size=config.VAL_SPLIT,
        stratify=train_val_y,
        random_state=config.RANDOM_SEED,
    )

    splits = {
        "train": (train_fp, train_y),
        "val": (val_fp, val_y),
        "test": (test_fp, test_y),
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.SPLIT_INDEX_PATH, "w") as f:
        json.dump(
            {name: {"files": fp, "labels": y.tolist()} for name, (fp, y) in splits.items()},
            f,
        )
    return splits


def load_split_arrays(filepaths, labels):
    """Materialize a list of filepaths into an (N, H, W, C) float32 array + labels."""
    images = np.zeros((len(filepaths), config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.CHANNELS),
                       dtype=np.float32)
    keep = []
    for i, fp in enumerate(filepaths):
        img = load_image(fp)
        if img is not None:
            images[i] = img
            keep.append(i)
    if len(keep) != len(filepaths):
        images = images[keep]
        labels = labels[keep]
    return images, labels
