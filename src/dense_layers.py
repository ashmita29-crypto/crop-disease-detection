"""
dense_layers.py

Explains and demonstrates the Dense (classification) part of the CNN.

After Conv2D and MaxPooling extract features from the image,
we Flatten the result and pass it through Dense layers to make
the final disease classification.
"""

import tensorflow as tf

def build_classifier(num_classes=38):
    """
    Builds the classification head of the CNN.
    
    Args:
        num_classes: how many disease classes to predict (PlantVillage has 38)
    """
    
    # The input here is the output of our feature extractor
    # After 3 MaxPooling operations: 224 -> 112 -> 56 -> 28
    # With 128 filters: shape is (28, 28, 128)
    inputs = tf.keras.Input(shape=(28, 28, 128))
    
    # FLATTEN: convert 3D feature maps (28, 28, 128) into 1D vector (28*28*128 = 100352)
    # Think: unrolling a 3D cube into a single long strip
    x = tf.keras.layers.Flatten()(inputs)
    
    # DENSE layer: 256 neurons, each connected to every number in the flattened vector
    # This is where the network "thinks" about what features mean
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    
    # DROPOUT: randomly turn off 50% of neurons during training
    # Rate=0.5 means 50% dropout. This prevents memorisation (overfitting)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    # DENSE layer: 128 neurons — another layer of thinking
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Dropout again — 30% this time
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    # OUTPUT layer: one neuron per class
    # 'softmax' converts raw scores into probabilities that sum to 1.0
    # e.g. [0.85, 0.10, 0.05] means 85% chance it's class 0
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier_head")
    return model


if __name__ == "__main__":
    classifier = build_classifier(num_classes=38)
    print("Classifier head summary:")
    classifier.summary()