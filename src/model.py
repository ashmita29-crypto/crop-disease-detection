"""
model.py

The complete CNN model for crop disease detection.
Combines feature extraction (Conv2D + MaxPooling) with
classification (Flatten + Dense + Dropout).

This file is the central model definition used by train.py.
"""

import tensorflow as tf

def build_model(num_classes=15, input_shape=(224, 224, 3)):
    """
    Builds the full CNN for crop disease classification.
    
    Args:
        num_classes  : number of disease classes (default 15 for PlantVillage)
        input_shape  : (height, width, channels) — default (224, 224, 3)
    
    Returns:
        A compiled Keras model ready for training
    """
    
    # INPUT LAYER
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # =====================================================================
    # FEATURE EXTRACTION (Conv2D + MaxPooling blocks)
    # =====================================================================
    
    # Block 1: Learn basic edges and colours
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    # Image: 224x224 -> 112x112
    
    # Block 2: Learn shapes and textures
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    # Image: 112x112 -> 56x56
    
    # Block 3: Learn complex disease patterns
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    # Image: 56x56 -> 28x28
    
    # =====================================================================
    # CLASSIFICATION HEAD (Flatten + Dense + Dropout)
    # =====================================================================
    
    x = tf.keras.layers.Flatten(name='flatten')(x)
    # Shape: (28, 28, 128) -> (100352,)
    
    x = tf.keras.layers.Dense(256, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout1')(x)
    
    x = tf.keras.layers.Dense(128, activation='relu', name='dense2')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout2')(x)
    
    # OUTPUT LAYER
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # BUILD THE MODEL
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CropDiseaseCNN")
    
    # =====================================================================
    # COMPILE THE MODEL
    # =====================================================================
    # optimizer='adam'       — controls how the model updates its weights during training
    # loss='categorical_crossentropy' — measures how wrong predictions are (for multi-class)
    # metrics=['accuracy']   — what we track during training
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    model = build_model(num_classes=38)
    print("Full CNN Model Summary:")
    model.summary()
    print("\nModel successfully built and compiled!")