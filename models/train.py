import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import create_hybrid_model

def create_synthetic_dataset(num_samples=100, batch_size=32, image_size=(224, 224)):
    """Create synthetic dataset for demo/testing"""
    # Generate synthetic real images
    real_images = np.random.rand(num_samples // 2, *image_size, 3).astype(np.float32)
    real_labels = np.ones((num_samples // 2, 1))
    
    # Generate synthetic fake images (with artifacts)
    fake_images = np.random.rand(num_samples // 2, *image_size, 3).astype(np.float32)
    # Add noise pattern to fake images
    fake_images += np.random.normal(0.1, 0.05, fake_images.shape)
    fake_images = np.clip(fake_images, 0, 1)
    fake_labels = np.zeros((num_samples // 2, 1))
    
    # Combine and shuffle
    all_images = np.vstack([real_images, fake_images])
    all_labels = np.vstack([real_labels, fake_labels])
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_real_dataset(data_dir, batch_size=32, image_size=(224, 224)):
    """Load real dataset from directory"""
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Dataset directories not found. Using synthetic data.")
        return None
    
    real_files = tf.data.Dataset.list_files(os.path.join(real_dir, '*.jpg'))
    fake_files = tf.data.Dataset.list_files(os.path.join(fake_dir, '*.jpg'))
    
    real_count = len(list(real_files))
    fake_count = len(list(fake_files))
    
    real_labels = tf.data.Dataset.from_tensor_slices(np.ones(real_count))
    fake_labels = tf.data.Dataset.from_tensor_slices(np.zeros(fake_count))
    
    real_dataset = tf.data.Dataset.zip((real_files, real_labels))
    fake_dataset = tf.data.Dataset.zip((fake_files, fake_labels))
    
    dataset = real_dataset.concatenate(fake_dataset)
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda x, y: (parse_image(x), y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    # Paths
    train_dir = os.path.join(Path(__file__).parent.parent, 'data', 'train')
    val_dir = os.path.join(Path(__file__).parent.parent, 'data', 'val')
    model_save_path = os.path.join(Path(__file__).parent, 'deepfake_detector.h5')
    
    print("[INFO] Creating hybrid model...")
    model = create_hybrid_model()
    
    print("[INFO] Compiling model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("[INFO] Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        model_save_path, 
        save_best_only=True, 
        monitor='val_accuracy',
        verbose=1
    )
    
    # Load datasets
    print("[INFO] Loading datasets...")
    train_dataset = load_real_dataset(train_dir)
    if train_dataset is None:
        print("[INFO] Using synthetic training data...")
        train_dataset = create_synthetic_dataset(num_samples=200, batch_size=16)
    
    val_dataset = load_real_dataset(val_dir)
    if val_dataset is None:
        print("[INFO] Using synthetic validation data...")
        val_dataset = create_synthetic_dataset(num_samples=50, batch_size=16)
    
    # Train
    print("[INFO] Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=1
    )
    
    # Save final model
    print(f"[INFO] Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("[SUCCESS] Model trained and saved!")
    
    return model, history

if __name__ == "__main__":
    model, history = main()