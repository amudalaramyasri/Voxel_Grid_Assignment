"""
train.py — Train a MobileNetV2-based COVID/non-COVID CT-scan classifier.

Usage:
    python train.py

Design decisions are documented in report.pdf (Part 3). Summary:
  - Backbone: MobileNetV2 (ImageNet weights), chosen for CPU feasibility.
  - Two-phase training: (1) frozen backbone + new head, (2) light fine-tuning
    of the top backbone layers at a low learning rate.
  - Augmentation: light geometric/intensity jitter appropriate for CT scans
    (no heavy color jitter, since CT images are grayscale-like and color
    artifacts are not clinically meaningful).
"""

import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

import config
from data_utils import build_splits, load_split_arrays


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMAGE_SIZE + (config.CHANNELS,),
    )
    base_model.trainable = False  # phase 1: frozen backbone

    inputs = layers.Input(shape=config.IMAGE_SIZE + (config.CHANNELS,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def make_augmenter():
    """Light, clinically-sensible augmentation for CT images."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


def unfreeze_top_layers(base_model, n_trainable=30):
    """Unfreeze the top n_trainable layers of the backbone for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_trainable]:
        layer.trainable = False


def main():
    config.set_global_seed()
    print(f"TensorFlow {tf.__version__}, using CPU only.")

    print("Building train/val/test splits...")
    splits = build_splits()
    train_fp, train_y = splits["train"]
    val_fp, val_y = splits["val"]
    print(f"  train={len(train_fp)}  val={len(val_fp)}  test={len(splits['test'][0])}")

    print("Loading images into memory (this is the slow step on CPU)...")
    t0 = time.time()
    X_train, y_train = load_split_arrays(train_fp, train_y)
    X_val, y_val = load_split_arrays(val_fp, val_y)
    print(f"  loaded in {time.time() - t0:.1f}s  X_train={X_train.shape} X_val={X_val.shape}")

    augmenter = make_augmenter()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train), seed=config.RANDOM_SEED)
    train_ds = train_ds.batch(config.BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                             num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(config.BATCH_SIZE)

    model, base_model = build_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    import os
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    base_callbacks = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    # --- Phase 1: train the new head only, backbone frozen ---
    print(f"\n[phase 1] Training head only (backbone frozen) for up to {config.FINE_TUNE_AT_EPOCH} epochs")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FINE_TUNE_AT_EPOCH,
        callbacks=base_callbacks,
        verbose=1,
    )

    # --- Phase 2: unfreeze top of backbone, fine-tune at a low LR ---
    print(f"\n[phase 2] Fine-tuning top of {config.BACKBONE} for remaining epochs")
    unfreeze_top_layers(base_model, n_trainable=30)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    remaining_epochs = max(config.EPOCHS - config.FINE_TUNE_AT_EPOCH, 1)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=remaining_epochs,
        callbacks=base_callbacks + [
            callbacks.ModelCheckpoint(config.MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        ],
        verbose=1,
    )

    model.save(config.MODEL_PATH)

    # Merge both phases into one continuous history for plotting.
    merged_history = {}
    for key in history1.history:
        merged_history[key] = list(history1.history[key]) + list(history2.history.get(key, []))
    with open(config.HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in merged_history.items()}, f)

    print(f"\nSaved model to {config.MODEL_PATH}")
    print(f"Saved training history to {config.HISTORY_PATH}")


if __name__ == "__main__":
    main()
