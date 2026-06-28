"""
callbacks.py

Defines training callbacks for the CNN.

What is EarlyStopping?
------------------------
Imagine you're studying for an exam. You study for 10 hours and your
practice test score keeps improving. But after hour 12, it stops
improving and even gets slightly worse. EarlyStopping is like a
timer that says: "Stop studying — you've reached your peak."

This prevents OVERFITTING: where the model memorises training
images instead of learning general patterns.

What is ModelCheckpoint?
------------------------
It automatically saves the model every time it gets better.
Like pressing Save in a game right after you level up.

What is ReduceLROnPlateau?
--------------------------
If the model gets stuck and stops improving, this callback
automatically reduces the learning rate to take smaller,
more careful steps.
"""

import tensorflow as tf
import os

def get_callbacks(model_save_path, monitor='val_accuracy'):
    """
    Returns a list of training callbacks.
    
    Args:
        model_save_path: where to save the best model
        monitor: which metric to watch ('val_accuracy' or 'val_loss')
    
    Returns:
        List of Keras callbacks
    """
    
    # Make sure the models folder exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor=monitor,
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # Reduce learning rate by half
        patience=3,           # If no improvement for 3 epochs
        min_lr=1e-6,          # Don't go below this learning rate
        verbose=1
    )
    
    return [early_stopping, model_checkpoint, reduce_lr]