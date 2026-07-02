"""
hyperparameter_tuning.py

Systematically tests different hyperparameter combinations
and saves all results to a CSV file for comparison.

Hyperparameters we test:
- Learning Rate : controls how big each weight update step is
- Batch Size    : how many images per training step
- Dropout Rate  : fraction of neurons switched off during training

We train a smaller version for each config (fewer epochs)
to save time, then pick the best config for the final model.
"""

import os
import csv
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SPLITS_DIR  = "data/splits"
RESULTS_DIR = "results"
IMAGE_SIZE  = (224, 224)
NUM_CLASSES = 38
TUNE_EPOCHS = 5       # Short runs to compare configs

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# HYPERPARAMETER GRID
# ============================================================
# We test every combination of these values
hyperparams = {
    "learning_rate" : [0.001, 0.0001, 0.00001],
    "batch_size"    : [16, 32],
    "dropout_rate"  : [0.2, 0.4],
}

def build_tuning_model(learning_rate, dropout_rate, num_classes=38):
    """Builds a quick MobileNetV2 model for tuning experiments."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax'
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_data_generators(batch_size):
    """Creates data generators for a given batch size."""
    dg = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    train_gen = dg.flow_from_directory(
        os.path.join(SPLITS_DIR, "train"),
        target_size=IMAGE_SIZE, batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )
    val_gen = dg.flow_from_directory(
        os.path.join(SPLITS_DIR, "val"),
        target_size=IMAGE_SIZE, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================
results = []
experiment_num = 1

for lr in hyperparams["learning_rate"]:
    for bs in hyperparams["batch_size"]:
        for dr in hyperparams["dropout_rate"]:
            print(f"\n{'='*55}")
            print(f"Experiment {experiment_num}: "
                  f"LR={lr}, Batch={bs}, Dropout={dr}")
            print(f"{'='*55}")
            
            train_gen, val_gen = get_data_generators(bs)
            model = build_tuning_model(lr, dr)
            
            history = model.fit(
                train_gen, epochs=TUNE_EPOCHS,
                validation_data=val_gen, verbose=1
            )
            
            best_val_acc  = max(history.history['val_accuracy'])
            best_val_loss = min(history.history['val_loss'])
            
            results.append({
                "experiment"    : experiment_num,
                "learning_rate" : lr,
                "batch_size"    : bs,
                "dropout_rate"  : dr,
                "best_val_acc"  : round(best_val_acc, 4),
                "best_val_loss" : round(best_val_loss, 4)
            })
            
            print(f"Result: val_acc={best_val_acc:.4f}, "
                  f"val_loss={best_val_loss:.4f}")
            experiment_num += 1

# ============================================================
# SAVE RESULTS
# ============================================================
csv_path = os.path.join(RESULTS_DIR, "tuning_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nAll results saved to {csv_path}")

# Print best config
best = max(results, key=lambda x: x["best_val_acc"])
print(f"\nBEST CONFIGURATION:")
print(f"  Learning Rate : {best['learning_rate']}")
print(f"  Batch Size    : {best['batch_size']}")
print(f"  Dropout Rate  : {best['dropout_rate']}")
print(f"  Val Accuracy  : {best['best_val_acc']}")