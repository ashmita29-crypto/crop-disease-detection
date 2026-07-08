"""
train_transfer.py

Trains the MobileNetV2 transfer learning model on PlantVillage.

Two-phase training:
  Phase 1: Train only the custom head (base frozen) - 10 epochs
  Phase 2: Fine-tune top layers of MobileNetV2 - 10 more epochs

This two-phase approach gives better accuracy than training
from scratch or fine-tuning immediately.
"""

import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our model builder
import sys
sys.path.append('src')
from mobilenet_model import build_mobilenet_model, unfreeze_model

# ============================================================
# SETTINGS
# ============================================================
SPLITS_DIR       = "data/splits"
MODEL_SAVE_PATH  = "models/mobilenet_crop_disease.h5"
RESULTS_DIR      = "results"
NUM_CLASSES      = 15
IMAGE_SIZE       = (224, 224)
BATCH_SIZE       = 32
PHASE1_EPOCHS    = 10
PHASE2_EPOCHS    = 10

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
print("Loading data...")

# MobileNetV2 expects inputs preprocessed with its own function
# preprocess_input scales pixels to [-1, 1] (not [0,1])
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Train samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")

# ============================================================
# BUILD MODEL
# ============================================================
print("\nBuilding MobileNetV2 model...")
model, base_model = build_mobilenet_model(num_classes=NUM_CLASSES)

# ============================================================
# PHASE 1: Train with frozen base
# ============================================================
print("\n--- PHASE 1: Training classification head (base frozen) ---")

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/mobilenet_phase1_best.h5",
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

history1 = model.fit(
    train_gen,
    epochs=PHASE1_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_phase1
)

print(f"\nPhase 1 best val accuracy: "
      f"{max(history1.history['val_accuracy']):.4f}")

# ============================================================
# PHASE 2: Fine-tune top layers
# ============================================================
print("\n--- PHASE 2: Fine-tuning top MobileNetV2 layers ---")

model = unfreeze_model(model, base_model, fine_tune_from=100)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=PHASE2_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_phase2
)

print(f"\nPhase 2 best val accuracy: "
      f"{max(history2.history['val_accuracy']):.4f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")

# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================
def save_history_plot(h1, h2, metric, save_path):
    """Combines phase 1 and phase 2 history and plots."""
    combined = h1.history[metric] + h2.history[metric]
    combined_val = (h1.history[f'val_{metric}']
                    + h2.history[f'val_{metric}'])
    epochs = range(1, len(combined) + 1)
    phase2_start = len(h1.history[metric])
    
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, combined, label=f'Train {metric}',
             color='steelblue')
    plt.plot(epochs, combined_val, label=f'Val {metric}',
             color='darkorange')
    plt.axvline(x=phase2_start, color='gray', linestyle='--',
                label='Fine-tuning starts')
    plt.title(f"MobileNetV2 — {metric.capitalize()} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

save_history_plot(history1, history2, 'accuracy',
    f"{RESULTS_DIR}/mobilenet_accuracy.png")
save_history_plot(history1, history2, 'loss',
    f"{RESULTS_DIR}/mobilenet_loss.png")

print("\nTraining complete!")