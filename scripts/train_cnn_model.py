#!/usr/bin/env python3
"""
Train CNN Model for Crop Health Classification

This script trains the CNN model on the generated training data:
1. Loads training data from data/training/
2. Trains CNN on synthetic labeled patches (4 classes)
3. Achieves >85% validation accuracy
4. Saves model weights to models/crop_health_cnn.h5
5. Updates .env to set USE_AI_MODELS=true

Usage:
    python scripts/train_cnn_model.py [--epochs 20] [--batch-size 32]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_models.crop_health_cnn import CropHealthCNN

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


def load_training_data(data_dir: Path):
    """
    Load CNN training data.
    
    Args:
        data_dir: Directory containing training data
        
    Returns:
        Tuple of (X_train, y_train, metadata)
    """
    logger.info(f"Loading training data from {data_dir}...")
    
    X_train = np.load(data_dir / 'cnn_X_train.npy')
    y_train = np.load(data_dir / 'cnn_y_train.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    logger.info(f"  Number of classes: {metadata['cnn']['num_classes']}")
    
    return X_train, y_train, metadata


def prepare_labels_for_training(y_train: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """
    Prepare labels for CNN training (per-pixel classification).
    
    Args:
        y_train: Array of class labels (one per patch)
        patch_size: Size of patches
        
    Returns:
        One-hot encoded labels with shape (N, patch_size, patch_size, num_classes)
    """
    logger.info("Preparing labels for training...")
    
    num_samples = len(y_train)
    num_classes = 4
    
    # Create one-hot encoded labels for each pixel in the patch
    y_train_onehot = np.zeros((num_samples, patch_size, patch_size, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        # Set all pixels in the patch to the same class
        y_train_onehot[i, :, :, y_train[i]] = 1.0
    
    logger.info(f"  Label shape: {y_train_onehot.shape}")
    
    return y_train_onehot


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32
):
    """
    Train the CNN model.
    
    Args:
        X_train: Training patches
        y_train: Training labels (one-hot encoded)
        X_val: Validation patches
        y_val: Validation labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Tuple of (model, history)
    """
    logger.info("="*70)
    logger.info("Training CNN Model")
    logger.info("="*70)
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Create model
    cnn = CropHealthCNN()
    
    # Train model
    history = cnn.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    logger.info("Training complete!")
    
    return cnn, history


def evaluate_model(cnn: CropHealthCNN, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate the trained model.
    
    Args:
        cnn: Trained CNN model
        X_test: Test patches
        y_test: Test labels (class indices)
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("="*70)
    logger.info("Evaluating Model")
    logger.info("="*70)
    
    # Get predictions
    predictions, confidences = cnn.predict_with_confidence(X_test)
    
    # Extract center pixel predictions
    center = predictions.shape[1] // 2
    pred_classes = predictions[:, center, center]
    pred_confidences = confidences[:, center, center]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_classes, average='weighted', zero_division=0
    )
    
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
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
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'mean_confidence': float(pred_confidences.mean()),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def save_model_and_metrics(
    cnn: CropHealthCNN,
    history: dict,
    metrics: dict,
    model_dir: Path
):
    """
    Save trained model and metrics.
    
    Args:
        cnn: Trained CNN model
        history: Training history
        metrics: Evaluation metrics
        model_dir: Directory to save model
    """
    logger.info("="*70)
    logger.info("Saving Model and Metrics")
    logger.info("="*70)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = model_dir / 'crop_health_cnn.h5'
    cnn.save_model(str(model_path))
    logger.info(f"  Saved model to: {model_path}")
    
    # Save training history
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    logger.info(f"  Saved training history to: {history_path}")
    
    # Save metrics
    metrics_path = model_dir / 'model_metrics.json'
    model_metadata = {
        'model_type': 'CNN',
        'model_name': 'CropHealthCNN',
        'training_date': datetime.now().isoformat(),
        'version': '1.0',
        'metrics': metrics,
        'final_training_accuracy': float(history['accuracy'][-1]),
        'final_validation_accuracy': float(history['val_accuracy'][-1]),
        'num_epochs': len(history['accuracy']),
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
        # Add the setting if it doesn't exist
        lines.append('\n# AI Models\nUSE_AI_MODELS=true\n')
    
    # Write updated .env
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    logger.info("  Updated USE_AI_MODELS=true in .env")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train CNN model for crop health classification'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split fraction (default: 0.2)'
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
    logger.info("CNN Model Training Pipeline")
    logger.info("="*70)
    
    try:
        # Load training data
        X_train, y_train, metadata = load_training_data(args.data_dir)
        
        # Prepare labels
        patch_size = metadata['cnn']['patch_size']
        y_train_onehot = prepare_labels_for_training(y_train, patch_size)
        
        # Split into train/validation
        X_train_split, X_val, y_train_split, y_val_onehot = train_test_split(
            X_train, y_train_onehot, test_size=args.validation_split, random_state=42
        )
        
        # Also split the original labels for evaluation
        _, _, y_train_labels, y_val_labels = train_test_split(
            X_train, y_train, test_size=args.validation_split, random_state=42
        )
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Training: {len(X_train_split)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        
        # Train model
        cnn, history = train_model(
            X_train_split,
            y_train_split,
            X_val,
            y_val_onehot,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        metrics = evaluate_model(cnn, X_val, y_val_labels)
        
        # Save model and metrics
        save_model_and_metrics(cnn, history, metrics, args.model_dir)
        
        # Update .env file
        update_env_file()
        
        # Print summary
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        
        print("\n" + "="*70)
        print("CNN TRAINING SUMMARY")
        print("="*70)
        print(f"Training samples: {len(X_train_split):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Epochs trained: {len(history['accuracy'])}")
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1-score: {metrics['f1_score']:.4f}")
        print(f"\nModel saved to: {args.model_dir}/crop_health_cnn.h5")
        print("="*70)
        
        if metrics['accuracy'] >= 0.85:
            logger.info("\n✅ Model achieved >85% accuracy target!")
        else:
            logger.warning(f"\n⚠️  Model accuracy ({metrics['accuracy']:.2%}) below 85% target")
            logger.warning("Consider training for more epochs or adjusting hyperparameters")
        
        logger.info("\n✅ CNN training complete!")
        logger.info("Model is ready for inference.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
