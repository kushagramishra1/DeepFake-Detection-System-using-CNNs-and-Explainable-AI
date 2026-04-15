import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
import numpy as np

# Vision Transformer implementation (simplified)
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(embed_dim, patch_size, patch_size, strides=patch_size)

    def call(self, x):
        x = self.projection(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, self.embed_dim])
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_vit_branch(input_shape=(224, 224, 3), patch_size=16, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=6):
    inputs = layers.Input(shape=input_shape)
    patches = PatchEmbedding(patch_size, embed_dim)(inputs)
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    x = patches + pos_embed
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x, training=True)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    return models.Model(inputs, x)

def extract_fft_features(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = tf.image.rgb_to_grayscale(image)
    image = tf.squeeze(image, axis=-1)
    fft = tf.signal.fft2d(tf.cast(image, tf.complex64))
    magnitude = tf.abs(fft)
    # Take log for better visualization
    magnitude = tf.math.log(magnitude + 1e-8)
    # Flatten or pool - for 2D image, reduce over spatial dimensions
    features = tf.reduce_mean(magnitude, axis=[0, 1])
    return features

def create_fft_branch(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    fft_features = layers.Lambda(lambda x: tf.map_fn(extract_fft_features, x))(inputs)
    x = layers.Dense(512, activation='relu')(fft_features)
    x = layers.Dense(256, activation='relu')(x)
    return models.Model(inputs, x)

def attention_fusion(features_list, embed_dim=256):
    # features_list: [cnn_features, fft_features, vit_features]
    concatenated = layers.Concatenate()(features_list)
    attention_weights = layers.Dense(len(features_list), activation='softmax')(concatenated)
    weighted_features = []
    for i, feat in enumerate(features_list):
        weight = layers.Lambda(lambda x: x[:, i:i+1])(attention_weights)
        weighted = layers.Multiply()([feat, weight])
        weighted_features.append(weighted)
    fused = layers.Concatenate()(weighted_features)
    return fused

def create_hybrid_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # CNN branch
    efficientnet = EfficientNetB4(include_top=False, input_shape=input_shape, pooling='avg')
    cnn_features = efficientnet(inputs)
    cnn_features = layers.Dense(512, activation='relu')(cnn_features)

    # FFT branch
    fft_model = create_fft_branch(input_shape)
    fft_features = fft_model(inputs)

    # ViT branch
    vit_model = create_vit_branch(input_shape)
    vit_features = vit_model(inputs)
    vit_features = layers.Dense(512, activation='relu')(vit_features)

    # Fusion
    fused = attention_fusion([cnn_features, fft_features, vit_features])

    # Classification head
    x = layers.Dense(256, activation='relu')(fused)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = create_hybrid_model()
    model.summary()