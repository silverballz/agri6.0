#!/usr/bin/env python3
"""
Train Simplified CNN Model for Crop Health Classification

This script uses a simpler CNN architecture to avoid TensorFlow compatibility issues.

Usage:
    python scripts/train_cnn_simple.py [--epochs 10]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import os

# Disable TensorFlow warnings and force CPU usage to avoid Metal issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cnn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_simple_cnn_model():
    """
    Create a simplified CNN model that's more stable.
    
    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Force CPU execution to avoid Metal/GPU issues on macOS
    tf.config.set_visible_devices([], 'GPU')
    
    logger.info("Creating simplified CNN model (CPU mode)...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(64, 64, 4)),
        
        # Conv block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Conv block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Conv block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Global pooling instead of flatten
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer (4 classes)
        layers.Dense(4, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model created with {model.count_params():,} parameters")
    
    return model


def load_training_data(data_dir: Path):
    """Load CNN training data."""
    logger.info(f"Loading training data from {data_dir}...")
    
    X_train = np.load(data_dir / 'cnn_X_train.npy')
    y_train = np.load(data_dir / 'cnn_y_train.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    
    return X_train, y_train, metadata


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """Train the model."""
    from tensorflow import keras
    
    logger.info("="*70)
    logger.info("Training CNN Model")
    logger.info("="*70)
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training complete!")
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logger.info("="*70)
    logger.info("Evaluating Model")
    logger.info("="*70)
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    pred_confidences = np.max(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_classes)
    
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Mean confidence: {pred_confidences.mean():.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, pred_classes)
    logger.info("\n  Confusion Matrix:")
    logger.info(f"  {cm}")
    
    # Classification report
    class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    report = classification_report(y_test, pred_classes, target_names=class_names, zero_division=0)
    logger.info("\n  Classification Report:")
    logger.info(f"\n{report}")
    
    return {
        'accuracy': float(accuracy),
        'mean_confidence': float(pred_confidences.mean()),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def save_model_and_metrics(model, history, metrics, model_dir: Path):
    """Save trained model and metrics."""
    logger.info("="*70)
    logger.info("Saving Model and Metrics")
    logger.info("="*70)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'crop_health_cnn.h5'
    model.save(str(model_path))
    logger.info(f"  Saved model to: {model_path}")
    
    # Save training history
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    logger.info(f"  Saved training history to: {history_path}")
    
    # Save metrics
    metrics_path = model_dir / 'model_metrics.json'
    model_metadata = {
        'model_type': 'CNN',
        'model_name': 'SimplifiedCropHealthCNN',
        'training_date': datetime.now().isoformat(),
        'version': '1.0',
        'metrics': metrics,
        'final_training_accuracy': float(history.history['accuracy'][-1]),
        'final_validation_accuracy': float(history.history['val_accuracy'][-1]),
        'num_epochs': len(history.history['accuracy']),
        'classes': ['Healthy', 'Moderate', 'Stressed', 'Critical']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"  Saved metrics to: {metrics_path}")


def update_env_file():
    """Update .env file to enable AI models."""
    logger.info("="*70)
    logger.info("Updating .env File")
    logger.info("="*70)
    
    env_path = Path('.env')
    
    if not env_path.exists():
        logger.warning("  .env file not found, skipping update")
        return
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update USE_AI_MODELS setting
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('USE_AI_MODELS='):
            lines[i] = 'USE_AI_MODELS=true\n'
            updated = True
            break
    
    if not updated:
        lines.append('\n# AI Models\nUSE_AI_MODELS=true\n')
    
    # Write updated .env
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    logger.info("  Updated USE_AI_MODELS=true in .env")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train simplified CNN model for crop health classification'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/training'),
        help='Directory containing training data'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('models'),
        help='Directory to save trained model'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("Simplified CNN Model Training Pipeline")
    logger.info("="*70)
    
    try:
        # Load training data
        X_train, y_train, metadata = load_training_data(args.data_dir)
        
        # Split into train/validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Training: {len(X_train_split)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        
        # Create model
        model = create_simple_cnn_model()
        
        # Train model
        history = train_model(
            model,
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        
        # Save model and metrics
        save_model_and_metrics(model, history, metrics, args.model_dir)
        
        # Update .env file
        update_env_file()
        
        # Print summary
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print("\n" + "="*70)
        print("CNN TRAINING SUMMARY")
        print("="*70)
        print(f"Training samples: {len(X_train_split):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Epochs trained: {len(history.history['accuracy'])}")
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nModel saved to: {args.model_dir}/crop_health_cnn.h5")
        print("="*70)
        
        if metrics['accuracy'] >= 0.85:
            logger.info("\n✅ Model achieved >85% accuracy target!")
        else:
            logger.warning(f"\n⚠️  Model accuracy ({metrics['accuracy']:.2%}) below 85% target")
        
        logger.info("\n✅ CNN training complete!")
        logger.info("Model is ready for inference.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
