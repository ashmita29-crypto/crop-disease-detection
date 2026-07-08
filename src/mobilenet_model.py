"""
mobilenet_model.py

Builds a transfer learning model using MobileNetV2.

What is MobileNetV2?
- A lightweight neural network designed by Google
- Pre-trained on ImageNet (1.4 million images, 1000 categories)
- Already knows how to detect edges, textures, shapes
- We take this knowledge and add our own classification layer on top

Transfer Learning has two phases:
Phase 1 - FROZEN: MobileNetV2 weights are locked. Only our new
          classification head trains. Fast and stable.
Phase 2 - FINE-TUNING: We unfreeze the top layers of MobileNetV2
          and train everything together at a very small learning rate.
"""

import tensorflow as tf

def build_mobilenet_model(num_classes=15, input_shape=(224, 224, 3)):
    """
    Builds MobileNetV2 transfer learning model.
    
    Args:
        num_classes : number of disease classes (15 for PlantVillage)
        input_shape : (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    
    # ============================================================
    # STEP 1: Load MobileNetV2 base (without its top layer)
    # ============================================================
    # include_top=False means we remove MobileNetV2's original
    # 1000-class output layer. We'll add our own 38-class layer.
    # weights='imagenet' loads the pre-trained weights from Google.
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # ============================================================
    # STEP 2: Freeze the base model
    # ============================================================
    # Freezing means: do NOT update these weights during training.
    # We want to keep Google's learned knowledge intact.
    # Only our new layers on top will learn from our crop data.
    
    base_model.trainable = False
    
    print(f"MobileNetV2 base layers    : {len(base_model.layers)}")
    print(f"Trainable layers (frozen)  : 0")
    
    # ============================================================
    # STEP 3: Add our custom classification head
    # ============================================================
    
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Pass through MobileNetV2 (frozen)
    # training=False ensures BatchNorm layers behave correctly
    x = base_model(inputs, training=False)
    
    # GlobalAveragePooling: converts (7,7,1280) feature maps
    # into a single vector of 1280 numbers
    # Better than Flatten for transfer learning
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    
    # Dropout to prevent overfitting
    x = tf.keras.layers.Dropout(0.3, name="dropout1")(x)
    
    # Dense layer for learning crop-specific features
    x = tf.keras.layers.Dense(256, activation='relu', name="dense1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout2")(x)
    
    # Final output: 38 classes with softmax probabilities
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name="output"
    )(x)
    
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name="MobileNetV2_CropDisease")
    
    # ============================================================
    # STEP 4: Compile
    # ============================================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def unfreeze_model(model, base_model, fine_tune_from=100):
    """
    Phase 2: Unfreeze top layers of MobileNetV2 for fine-tuning.
    
    We unfreeze only the LAST few layers, not all 155.
    Fine-tuning with a very small learning rate (0.00001) prevents
    destroying the pre-trained weights.
    
    Args:
        model       : the full model
        base_model  : the MobileNetV2 base
        fine_tune_from : unfreeze layers from this index onward
    """
    base_model.trainable = True
    
    # Freeze all layers BEFORE fine_tune_from
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    
    trainable_count = sum(
        1 for l in base_model.layers if l.trainable
    )
    print(f"Layers unfrozen for fine-tuning: {trainable_count}")
    
    # Recompile with a much smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    model, base_model = build_mobilenet_model(num_classes=15)
    model.summary()
    print("\nMobileNetV2 model built successfully!")