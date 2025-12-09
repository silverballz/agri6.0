#!/usr/bin/env python3
"""
Train CNN Model on Real Satellite Data

This script trains a CNN model specifically on real Sentinel-2 satellite imagery
downloaded from the Sentinel Hub API. It includes:
- Early stopping to prevent overfitting
- Validation accuracy monitoring
- Model checkpointing to save best weights
- Comprehensive logging of training metrics
- Model metadata with real data provenance

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 7.3

Usage:
    python scripts/train_cnn_on_real_data.py [--epochs 50] [--batch-size 32]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cnn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CropHealthCNN(nn.Module):
    """
    PyTorch CNN for crop health classification from satellite imagery.
    
    Architecture:
    - 3 convolutional blocks with batch normalization and max pooling
    - 2 fully connected layers with dropout
    - 4-class output (Healthy, Moderate, Stressed, Critical)
    """
    
    def __init__(self, num_classes: int = 4):
        super(CropHealthCNN, self).__init__()
        
        # Convolutional block 1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Convolutional block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore model to best weights
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    logger.info("  Restoring best model weights")
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


def load_real_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray, Dict]:
    """
    Load real satellite imagery training data.
    
    Args:
        data_dir: Directory containing training data
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, metadata)
    """
    logger.info(f"Loading real training data from {data_dir}...")
    
    # Load training data
    X_train = np.load(data_dir / 'cnn_X_train_real.npy')
    y_train = np.load(data_dir / 'cnn_y_train_real.npy')
    X_val = np.load(data_dir / 'cnn_X_val_real.npy')
    y_val = np.load(data_dir / 'cnn_y_val_real.npy')
    
    # Load metadata
    with open(data_dir / 'cnn_metadata_real.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    logger.info(f"  X_val shape: {X_val.shape}")
    logger.info(f"  y_val shape: {y_val.shape}")
    logger.info(f"  Data source: {metadata['data_source']}")
    logger.info(f"  Created at: {metadata['created_at']}")
    
    # Verify this is real data
    if metadata['data_source'] != 'real':
        raise ValueError(
            f"Expected real data but got data_source='{metadata['data_source']}'. "
            "This script should only be used with real satellite imagery."
        )
    
    return X_train, y_train, X_val, y_val, metadata


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating data loaders...")
    
    # Convert to PyTorch tensors (NCHW format)
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # NHWC -> NCHW
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    
    logger.info(f"  Tensor shapes: {X_train_tensor.shape}, {X_val_tensor.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_epoch(model: nn.Module, val_loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Validate for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, patience: int = 10, min_accuracy: float = 0.85,
                device: torch.device = torch.device('cpu')) -> Tuple[Dict, float]:
    """
    Train the CNN model with early stopping and checkpointing.
    
    Args:
        model: CNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        patience: Early stopping patience
        min_accuracy: Minimum required validation accuracy (Requirement 5.2)
        device: Device to train on
        
    Returns:
        Tuple of (history dict, best validation accuracy)
    """
    logger.info("="*70)
    logger.info("Training CNN Model on Real Satellite Data")
    logger.info("="*70)
    logger.info(f"  Max epochs: {epochs}")
    logger.info(f"  Early stopping patience: {patience}")
    logger.info(f"  Target accuracy: {min_accuracy:.1%}")
    logger.info(f"  Device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs_trained': 0
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs_trained'] = epoch + 1
        
        # Log metrics (Requirement 7.3)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            logger.info(f"  ✓ New best validation accuracy: {best_val_acc:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("="*70)
    logger.info(f"Training completed after {history['epochs_trained']} epochs")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Check if accuracy meets threshold (Requirement 5.2)
    if best_val_acc >= min_accuracy:
        logger.info(f"✅ Model meets {min_accuracy:.1%} accuracy threshold")
    else:
        logger.warning(
            f"⚠️  Model accuracy {best_val_acc:.1%} is below {min_accuracy:.1%} threshold. "
            "Consider downloading more training data or adjusting hyperparameters."
        )
    
    return history, best_val_acc


def evaluate_model(model: nn.Module, val_loader: DataLoader,
                   device: torch.device, class_names: List[str]) -> Dict:
    """
    Evaluate model and generate comprehensive metrics (Requirement 5.3).
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("="*70)
    logger.info("Evaluating Model on Validation Set")
    logger.info("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Get unique classes in predictions
    unique_classes = sorted(set(all_labels))
    target_names = [class_names[i] for i in unique_classes]
    
    class_report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        labels=unique_classes,
        output_dict=True
    )
    class_report_str = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        labels=unique_classes
    )
    
    # Calculate mean confidence
    all_probs = np.array(all_probs)
    mean_confidence = np.max(all_probs, axis=1).mean()
    
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Mean confidence: {mean_confidence:.4f}")
    logger.info(f"\n  Confusion Matrix:\n{conf_matrix}")
    logger.info(f"\n  Classification Report:\n{class_report_str}")
    
    return {
        'accuracy': float(accuracy),
        'mean_confidence': float(mean_confidence),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'classification_report_str': class_report_str
    }


def save_model_with_metadata(model: nn.Module, history: Dict, metrics: Dict,
                             metadata: Dict, model_dir: Path) -> None:
    """
    Save trained model with comprehensive metadata (Requirements 5.4, 5.5, 7.3).
    
    Metadata includes:
    - Training data source (real satellite data)
    - Data provenance (Sentinel Hub API)
    - Training metrics and history
    - Model architecture details
    """
    logger.info("="*70)
    logger.info("Saving Model and Metadata")
    logger.info("="*70)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model with checkpoint
    model_path = model_dir / 'crop_health_cnn_real.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, model_path)
    logger.info(f"  ✓ Saved model checkpoint to: {model_path}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create comprehensive metadata (Requirement 5.4)
    model_metadata = {
        'model_type': 'CNN',
        'framework': 'PyTorch',
        'model_name': 'CropHealthCNN',
        'version': '2.0',
        'training_date': datetime.now().isoformat(),
        
        # Data provenance (Requirement 5.4)
        'trained_on': 'real_satellite_data',
        'data_source': 'Sentinel-2 via Sentinel Hub API',
        'data_type': metadata['data_source'],
        'training_data_created': metadata['created_at'],
        
        # Training configuration
        'training': {
            'epochs_trained': history['epochs_trained'],
            'batch_size': 32,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'early_stopping_patience': 10
        },
        
        # Performance metrics (Requirement 5.3)
        'metrics': {
            'accuracy': metrics['accuracy'],
            'mean_confidence': metrics['mean_confidence'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report'],
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc'])
        },
        
        # Architecture details
        'architecture': {
            'framework': 'PyTorch',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv_layers': 3,
            'fc_layers': 2,
            'input_shape': [4, 64, 64],  # NCHW format
            'output_classes': 4
        },
        
        # Dataset information
        'dataset': {
            'num_train_samples': metadata['num_train_samples'],
            'num_val_samples': metadata['num_val_samples'],
            'num_classes': metadata['num_classes'],
            'class_names': metadata['class_names'],
            'patch_size': metadata['patch_size'],
            'num_channels': metadata['num_channels'],
            'train_class_distribution': metadata['train_class_distribution'],
            'val_class_distribution': metadata['val_class_distribution']
        },
        
        # Model file paths
        'files': {
            'model': str(model_path),
            'metadata': str(model_dir / 'cnn_model_metrics_real.json')
        }
    }
    
    # Save metadata
    metrics_path = model_dir / 'cnn_model_metrics_real.json'
    with open(metrics_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata to: {metrics_path}")
    
    # Save training history separately for plotting
    history_path = model_dir / 'cnn_training_history_real.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"  ✓ Saved training history to: {history_path}")
    
    logger.info(f"\n  Model parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")


def update_env_file() -> None:
    """Update .env file to enable AI models (Requirement 5.5)."""
    logger.info("="*70)
    logger.info("Updating .env Configuration")
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
    
    logger.info("  ✓ Updated USE_AI_MODELS=true in .env")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train CNN model on real satellite imagery'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=0.85,
        help='Minimum required validation accuracy (default: 0.85)'
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
    logger.info("CNN Training Pipeline - Real Satellite Data")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load real training data (Requirement 5.1)
        X_train, y_train, X_val, y_val, metadata = load_real_training_data(args.data_dir)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=args.batch_size
        )
        
        # Create model
        device = torch.device('cpu')
        model = CropHealthCNN(num_classes=4).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"\nModel created with {total_params:,} parameters")
        
        # Train model (Requirements 5.1, 5.2, 7.3)
        history, best_val_acc = train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            patience=args.patience,
            min_accuracy=args.min_accuracy,
            device=device
        )
        
        # Evaluate model (Requirement 5.3)
        metrics = evaluate_model(
            model,
            val_loader,
            device,
            metadata['class_names']
        )
        
        # Save model with metadata (Requirements 5.4, 5.5)
        save_model_with_metadata(
            model,
            history,
            metrics,
            metadata,
            args.model_dir
        )
        
        # Update .env file (Requirement 5.5)
        update_env_file()
        
        # Print final summary
        print("\n" + "="*70)
        print("CNN TRAINING SUMMARY - REAL SATELLITE DATA")
        print("="*70)
        print(f"Data source: {metadata['data_source']}")
        print(f"Training samples: {metadata['num_train_samples']:,}")
        print(f"Validation samples: {metadata['num_val_samples']:,}")
        print(f"Epochs trained: {history['epochs_trained']}")
        print(f"\nFinal Training Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nModel saved to: {args.model_dir}/crop_health_cnn_real.pth")
        print(f"Metadata saved to: {args.model_dir}/cnn_model_metrics_real.json")
        print("="*70)
        
        # Final status check
        if metrics['accuracy'] >= args.min_accuracy:
            logger.info(f"\n✅ SUCCESS: Model achieved {metrics['accuracy']:.1%} accuracy (target: {args.min_accuracy:.1%})")
            logger.info("Model is ready for production deployment!")
            sys.exit(0)
        else:
            logger.warning(
                f"\n⚠️  WARNING: Model accuracy {metrics['accuracy']:.1%} is below "
                f"{args.min_accuracy:.1%} target"
            )
            logger.warning("Consider:")
            logger.warning("  - Downloading more training imagery")
            logger.warning("  - Adjusting hyperparameters")
            logger.warning("  - Increasing training epochs")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
