"""
download_dataset.py — Downloads the SARS-COV-2 Ct-Scan Dataset via kagglehub
and links it into dataset/COVID and dataset/non-COVID.

Usage:
    pip install kagglehub
    python download_dataset.py
"""

import os
import shutil

import kagglehub

import config


def find_class_dirs(root):
    """kagglehub may nest the dataset under an extra folder; search for COVID/non-COVID."""
    for dirpath, dirnames, _ in os.walk(root):
        if "COVID" in dirnames and "non-COVID" in dirnames:
            return os.path.join(dirpath, "COVID"), os.path.join(dirpath, "non-COVID")
    raise FileNotFoundError(f"Could not find COVID/ and non-COVID/ folders under {root}")


def main():
    print("Downloading dataset via kagglehub (requires Kaggle auth - see "
          "https://github.com/Kagglehub/kagglehub#authenticate if prompted)...")
    path = kagglehub.dataset_download("plameneduardo/sarscov2-ctscan-dataset")
    print("Downloaded to:", path)

    covid_src, noncovid_src = find_class_dirs(path)

    os.makedirs(config.DATASET_DIR, exist_ok=True)
    for src, name in [(covid_src, "COVID"), (noncovid_src, "non-COVID")]:
        dst = os.path.join(config.DATASET_DIR, name)
        if os.path.exists(dst):
            print(f"  {dst} already exists, skipping")
            continue
        shutil.copytree(src, dst)
        print(f"  copied {src} -> {dst}")

    print(f"\nDone. {config.DATASET_DIR} is ready for train.py")


if __name__ == "__main__":
    main()
