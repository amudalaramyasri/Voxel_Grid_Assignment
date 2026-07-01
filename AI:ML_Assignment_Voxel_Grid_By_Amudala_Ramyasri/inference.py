"""
inference.py — Run the trained model on a single CT-scan image.

Usage:
    python inference.py path/to/image.png
"""

import sys
import json

import numpy as np
import tensorflow as tf

import config
from data_utils import load_image


def predict(image_path: str) -> dict:
    img = load_image(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = tf.keras.models.load_model(config.MODEL_PATH)
    prob = float(model.predict(img[np.newaxis, ...], verbose=0)[0, 0])
    label_idx = int(prob >= 0.5)

    return {
        "image": image_path,
        "predicted_class": config.CLASS_NAMES[label_idx],
        "covid_probability": prob,
        "confidence": prob if label_idx == 1 else 1 - prob,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/image.png")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
