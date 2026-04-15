import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np

def create_simplified_model(input_shape=(224, 224, 3)):
    """Simplified hybrid model with CNN + Frequency features"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN branch - MobileNetV2
    mobilenet = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    mobilenet.trainable = False
    cnn_features = mobilenet(inputs)
    cnn_features = layers.Dense(256, activation='relu')(cnn_features)
    
    # Frequency domain branch - simplified feature extraction
    gray_images = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(inputs)
    freq_features = layers.Flatten()(gray_images)
    freq_features = layers.Dense(256, activation='relu')(freq_features)
    freq_features = layers.Dense(256, activation='relu')(freq_features)
    
    # Attention-based fusion
    concatenated = layers.Concatenate()([cnn_features, freq_features])
    attention_weights = layers.Dense(2, activation='softmax')(concatenated)
    
    weighted_cnn = layers.Multiply()([cnn_features, layers.Lambda(lambda x: x[:, 0:1])(attention_weights)])
    weighted_freq = layers.Multiply()([freq_features, layers.Lambda(lambda x: x[:, 1:2])(attention_weights)])
    
    fused = layers.Concatenate()([weighted_cnn, weighted_freq])
    
    # Classification head
    x = layers.Dense(128, activation='relu')(fused)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    print("Building model...")
    model = create_simplified_model()
    model.summary()