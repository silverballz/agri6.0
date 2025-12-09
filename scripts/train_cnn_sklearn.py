#!/usr/bin/env python3
"""
Train CNN Model using sklearn-compatible wrapper

This script uses a simpler approach with sklearn's MLPClassifier as a workaround
for TensorFlow compatibility issues.

Usage:
    python scripts/train_cnn_sklearn.py
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

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


def prepare_data_for_mlp(X):
    """Flatten patches for MLP input."""
    # Flatten from (N, 64, 64, 4) to (N, 16384)
    N = X.shape[0]
    X_flat = X.reshape(N, -1)
    logger.info(f"  Flattened shape: {X_flat.shape}")
    return X_flat


def create_and_train_model(X_train, y_train, X_val, y_val):
    """Create and train MLP model."""
    logger.info("="*70)
    logger.info("Training Neural Network Model")
    logger.info("="*70)
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    
    # Standardize features
    logger.info("  Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create MLP model
    logger.info("  Creating MLP model...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=50,
        shuffle=True,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    )
    
    # Train model
    logger.info("  Training model...")
    model.fit(X_train_scaled, y_train)
    
    logger.info("  Training complete!")
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate the model."""
    logger.info("="*70)
    logger.info("Evaluating Model")
    logger.info("="*70)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    pred_classes = model.predict(X_test_scaled)
    pred_proba = model.predict_proba(X_test_scaled)
    pred_confidences = np.max(pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_classes)
    
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Mean confidence: {pred_confidences.mean():.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, pred_classes)
    logger.info("\n  Confusion Matrix:")
    logger.info(f"  {cm}")
    
    # Classification report
    # Get unique classes in the data
    unique_classes = np.unique(np.concatenate([y_test, pred_classes]))
    class_names_all = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    class_names = [class_names_all[i] for i in unique_classes]
    report = classification_report(y_test, pred_classes, target_names=class_names, zero_division=0)
    logger.info("\n  Classification Report:")
    logger.info(f"\n{report}")
    
    return {
        'accuracy': float(accuracy),
        'mean_confidence': float(pred_confidences.mean()),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'training_loss': float(model.loss_),
        'n_iterations': int(model.n_iter_)
    }


def save_model_and_metrics(model, scaler, metrics, model_dir: Path):
    """Save trained model and metrics."""
    logger.info("="*70)
    logger.info("Saving Model and Metrics")
    logger.info("="*70)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'crop_health_mlp.pkl'
    joblib.dump(model, str(model_path))
    logger.info(f"  Saved model to: {model_path}")
    
    # Save scaler
    scaler_path = model_dir / 'feature_scaler.pkl'
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"  Saved scaler to: {scaler_path}")
    
    # Save metrics
    metrics_path = model_dir / 'model_metrics.json'
    model_metadata = {
        'model_type': 'MLP',
        'model_name': 'CropHealthMLP',
        'training_date': datetime.now().isoformat(),
        'version': '1.0',
        'metrics': metrics,
        'architecture': {
            'hidden_layers': [512, 256, 128],
            'activation': 'relu',
            'solver': 'adam'
        },
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
        description='Train MLP model for crop health classification'
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
    logger.info("MLP Model Training Pipeline")
    logger.info("="*70)
    
    try:
        # Load training data
        X_train, y_train, metadata = load_training_data(args.data_dir)
        
        # Prepare data
        logger.info("\nPreparing data...")
        X_train_flat = prepare_data_for_mlp(X_train)
        
        # Split into train/validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_flat, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Training: {len(X_train_split)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        
        # Train model
        model, scaler = create_and_train_model(
            X_train_split,
            y_train_split,
            X_val,
            y_val
        )
        
        # Evaluate model
        metrics = evaluate_model(model, scaler, X_val, y_val)
        
        # Save model and metrics
        save_model_and_metrics(model, scaler, metrics, args.model_dir)
        
        # Update .env file
        update_env_file()
        
        # Print summary
        print("\n" + "="*70)
        print("MLP TRAINING SUMMARY")
        print("="*70)
        print(f"Training samples: {len(X_train_split):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Iterations: {metrics['n_iterations']}")
        print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
        print(f"\nModel saved to: {args.model_dir}/crop_health_mlp.pkl")
        print("="*70)
        
        if metrics['accuracy'] >= 0.85:
            logger.info("\n✅ Model achieved >85% accuracy target!")
        else:
            logger.info(f"\n✅ Model trained successfully with {metrics['accuracy']:.2%} accuracy")
        
        logger.info("\n✅ Model training complete!")
        logger.info("Model is ready for inference.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
