"""
train.py

Trains the CNN model on the PlantVillage dataset.
Uses EarlyStopping to stop automatically when the model
stops improving.
"""

import os
import tensorflow as tf
from model import build_model

# =====================================================================
# SETTINGS
# =====================================================================
SPLITS_DIR   = "data/splits"
MODEL_SAVE_PATH = "models/crop_disease_cnn.h5"
NUM_CLASSES  = 38
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32       # How many images to process at once
EPOCHS       = 30       # Maximum number of training passes
LEARNING_RATE = 0.001

# =====================================================================
# DATA LOADING (using ImageDataGenerator)
# =====================================================================
# ImageDataGenerator loads images from folders, applies normalisation,
# and feeds batches to the model during training.

print("Loading training data...")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,         # Normalise: divide all pixels by 255
    rotation_range=15,            # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,        # Randomly shift horizontally
    height_shift_range=0.1,       # Randomly shift vertically
    horizontal_flip=True,         # Randomly flip left-right
)

# Validation data should only be normalised, not augmented
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',     # One-hot encoded labels for 38 classes
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Training images  : {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")
print(f"Classes detected : {train_generator.num_classes}")

# =====================================================================
# BUILD THE MODEL
# =====================================================================
print("\nBuilding model...")
model = build_model(num_classes=NUM_CLASSES)

# =====================================================================
# CALLBACKS
# =====================================================================
# EarlyStopping: stops training if validation accuracy stops improving
# for 5 epochs in a row. Saves the best version of the model.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',   # Watch validation accuracy
    patience=5,               # Stop if no improvement for 5 epochs
    restore_best_weights=True # Revert to the best weights found
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,      # Only save when accuracy improves
    verbose=1
)

# =====================================================================
# TRAIN
# =====================================================================
print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nTraining complete!")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")