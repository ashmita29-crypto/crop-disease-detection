"""
cnn_layers.py

Explains and demonstrates Conv2D and MaxPooling layers.
These are the feature-extraction layers of our CNN.
"""

import tensorflow as tf

def build_feature_extractor():
    """
    Builds the feature extraction part of the CNN.
    
    Conv2D: scans the image with a filter to find patterns
    MaxPooling2D: shrinks the image to keep only the strongest signals
    """
    
    # Input: each image is 224x224 pixels with 3 colour channels (R, G, B)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # --- Block 1 ---
    # 32 filters, each 3x3 pixels in size
    # 'relu' activation: turns negative values to 0 (keeps only positive signals)
    # 'same' padding: keeps image size the same after convolution
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # MaxPooling: reduces 224x224 to 112x112
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # --- Block 2 ---
    # 64 filters — more filters = learns more complex patterns
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # MaxPooling: reduces 112x112 to 56x56
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # --- Block 3 ---
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # MaxPooling: reduces 56x56 to 28x28
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x, name="feature_extractor")
    return model


if __name__ == "__main__":
    feature_extractor = build_feature_extractor()
    print("Feature extractor summary:")
    feature_extractor.summary()