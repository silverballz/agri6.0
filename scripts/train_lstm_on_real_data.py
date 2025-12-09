#!/usr/bin/env python3
"""
Train LSTM Model on Real Temporal Satellite Data

This script trains an LSTM model specifically on real Sentinel-2 temporal sequences
downloaded from the Sentinel Hub API. It includes:
- LSTM training loop with early stopping
- Temporal validation metrics (MAE, MSE, RMSE, R²)
- Model checkpointing to save best weights
- Comprehensive logging of training metrics
- Model metadata with real data provenance

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.3

Usage:
    python scripts/train_lstm_on_real_data.py [--epochs 100] [--batch-size 32]
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure comprehensive logging (Requirement 7.3)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CropHealthLSTM(nn.Module):
    """
    PyTorch LSTM for temporal crop health prediction from satellite time series.
    
    Architecture:
    - 2 bidirectional LSTM layers with dropout
    - 2 fully connected layers with dropout
    - Single output for next time step prediction
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 1):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features (default: 1 for NDVI)
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            output_size: Number of output values (default: 1)
        """
        super(CropHealthLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # FC layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore model to best weights
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, 
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
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        return False


def load_real_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray, Dict]:
    """
    Load real temporal satellite imagery training data (Requirement 6.1).
    
    Args:
        data_dir: Directory containing training data
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, metadata)
    """
    logger.info(f"Loading real temporal training data from {data_dir}...")
    
    # Load training data
    X_train = np.load(data_dir / 'lstm_X_train_real.npy')
    y_train = np.load(data_dir / 'lstm_y_train_real.npy')
    X_val = np.load(data_dir / 'lstm_X_val_real.npy')
    y_val = np.load(data_dir / 'lstm_y_val_real.npy')
    
    # Load metadata
    with open(data_dir / 'lstm_metadata_real.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    logger.info(f"  X_val shape: {X_val.shape}")
    logger.info(f"  y_val shape: {y_val.shape}")
    logger.info(f"  Data source: {metadata['data_source']}")
    logger.info(f"  Created at: {metadata['created_at']}")
    logger.info(f"  Sequence length: {metadata['sequence_length']}")
    
    # Verify this is real data (Requirement 6.1)
    if metadata['data_source'] != 'real':
        raise ValueError(
            f"Expected real data but got data_source='{metadata['data_source']}'. "
            "This script should only be used with real temporal satellite imagery."
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
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Add dimension for output
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
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
        Tuple of (average_loss, mae)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    avg_loss = running_loss / len(train_loader)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return avg_loss, mae


def validate_epoch(model: nn.Module, val_loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Validate for one epoch.
    
    Returns:
        Tuple of (average_loss, mae)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    avg_loss = running_loss / len(val_loader)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return avg_loss, mae


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 100, patience: int = 15, min_accuracy: float = 0.80,
                device: torch.device = torch.device('cpu')) -> Tuple[Dict, float]:
    """
    Train the LSTM model with early stopping and checkpointing (Requirement 6.2).
    
    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        patience: Early stopping patience
        min_accuracy: Minimum required validation accuracy (Requirement 6.3)
        device: Device to train on
        
    Returns:
        Tuple of (history dict, best validation loss)
    """
    logger.info("="*70)
    logger.info("Training LSTM Model on Real Temporal Satellite Data")
    logger.info("="*70)
    logger.info(f"  Max epochs: {epochs}")
    logger.info(f"  Early stopping patience: {patience}")
    logger.info(f"  Target accuracy threshold: {min_accuracy:.1%}")
    logger.info(f"  Device: {device}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Training history (Requirement 7.3)
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'epochs_trained': 0
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_mae = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['epochs_trained'] = epoch + 1
        
        # Log metrics (Requirement 7.3)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} - "
            f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}"
        )
        
        # Track best model (model checkpointing)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            logger.info(f"  ✓ New best validation loss: {best_val_loss:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("="*70)
    logger.info(f"Training completed after {history['epochs_trained']} epochs")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    
    return history, best_val_loss


def evaluate_model(model: nn.Module, val_loader: DataLoader,
                   device: torch.device) -> Dict:
    """
    Evaluate model with temporal validation metrics (Requirement 6.4).
    
    Calculates:
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - R² Score (Coefficient of Determination)
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("="*70)
    logger.info("Evaluating Model on Validation Set")
    logger.info("="*70)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate temporal validation metrics (Requirement 6.4)
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    # Calculate accuracy as 1 - normalized MAE
    # For NDVI values typically in range [-1, 1], we normalize by range
    ndvi_range = 2.0  # NDVI range from -1 to 1
    normalized_mae = mae / ndvi_range
    accuracy = max(0.0, 1.0 - normalized_mae)
    
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R² Score: {r2:.6f}")
    logger.info(f"  Accuracy (1 - normalized MAE): {accuracy:.4f}")
    
    # Calculate prediction statistics
    pred_mean = np.mean(all_preds)
    pred_std = np.std(all_preds)
    target_mean = np.mean(all_targets)
    target_std = np.std(all_targets)
    
    logger.info(f"\n  Prediction statistics:")
    logger.info(f"    Mean: {pred_mean:.6f} ± {pred_std:.6f}")
    logger.info(f"    Range: [{np.min(all_preds):.6f}, {np.max(all_preds):.6f}]")
    logger.info(f"  Target statistics:")
    logger.info(f"    Mean: {target_mean:.6f} ± {target_std:.6f}")
    logger.info(f"    Range: [{np.min(all_targets):.6f}, {np.max(all_targets):.6f}]")
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'accuracy': float(accuracy),
        'prediction_mean': float(pred_mean),
        'prediction_std': float(pred_std),
        'target_mean': float(target_mean),
        'target_std': float(target_std)
    }


def save_model_with_metadata(model: nn.Module, history: Dict, metrics: Dict,
                             metadata: Dict, model_dir: Path) -> None:
    """
    Save trained model with comprehensive metadata (Requirements 6.4, 6.5, 7.3).
    
    Metadata includes:
    - Training data source (real temporal satellite data)
    - Data provenance (Sentinel Hub API)
    - Training metrics and history
    - Model architecture details
    """
    logger.info("="*70)
    logger.info("Saving Model and Metadata")
    logger.info("="*70)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model with checkpoint
    model_path = model_dir / 'crop_health_lstm_real.pth'
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
    
    # Create comprehensive metadata (Requirement 6.5)
    model_metadata = {
        'model_type': 'LSTM',
        'framework': 'PyTorch',
        'model_name': 'CropHealthLSTM',
        'version': '2.0',
        'training_date': datetime.now().isoformat(),
        
        # Data provenance (Requirement 6.5)
        'trained_on': 'real_temporal_sequences',
        'data_source': 'Sentinel-2 time-series via Sentinel Hub API',
        'data_type': metadata['data_source'],
        'training_data_created': metadata['created_at'],
        
        # Training configuration
        'training': {
            'epochs_trained': history['epochs_trained'],
            'batch_size': 32,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'early_stopping_patience': 15
        },
        
        # Performance metrics (Requirement 6.4)
        'metrics': {
            'mae': metrics['mae'],
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'r2_score': metrics['r2_score'],
            'accuracy': metrics['accuracy'],
            'final_train_loss': history['train_loss'][-1],
            'final_train_mae': history['train_mae'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_mae': history['val_mae'][-1],
            'best_val_loss': min(history['val_loss'])
        },
        
        # Architecture details
        'architecture': {
            'framework': 'PyTorch',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lstm_layers': 2,
            'fc_layers': 2,
            'hidden_size': 128,
            'bidirectional': True,
            'sequence_length': metadata['sequence_length'],
            'input_features': metadata['input_features'],
            'output_size': 1
        },
        
        # Dataset information
        'dataset': {
            'num_train_sequences': metadata['num_train_sequences'],
            'num_val_sequences': metadata['num_val_sequences'],
            'sequence_length': metadata['sequence_length'],
            'input_features': metadata['input_features'],
            'train_target_stats': metadata['train_target_stats'],
            'val_target_stats': metadata['val_target_stats']
        },
        
        # Model file paths
        'files': {
            'model': str(model_path),
            'metadata': str(model_dir / 'lstm_model_metrics_real.json')
        }
    }
    
    # Save metadata
    metrics_path = model_dir / 'lstm_model_metrics_real.json'
    with open(metrics_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata to: {metrics_path}")
    
    # Save training history separately for plotting
    history_path = model_dir / 'lstm_training_history_real.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"  ✓ Saved training history to: {history_path}")
    
    logger.info(f"\n  Model parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")


def update_env_file() -> None:
    """Update .env file to enable AI models (Requirement 6.5)."""
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
        description='Train LSTM model on real temporal satellite imagery'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs (default: 100)'
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
        default=15,
        help='Early stopping patience (default: 15)'
    )
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=0.80,
        help='Minimum required validation accuracy (default: 0.80)'
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
    logger.info("LSTM Training Pipeline - Real Temporal Satellite Data")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load real temporal training data (Requirement 6.1)
        X_train, y_train, X_val, y_val, metadata = load_real_training_data(args.data_dir)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=args.batch_size
        )
        
        # Create model
        device = torch.device('cpu')
        input_size = X_train.shape[2]  # Number of features
        model = CropHealthLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=1
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"\nModel created with {total_params:,} parameters")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Sequence length: {X_train.shape[1]}")
        
        # Train model (Requirements 6.2, 6.3, 7.3)
        history, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            patience=args.patience,
            min_accuracy=args.min_accuracy,
            device=device
        )
        
        # Evaluate model (Requirement 6.4)
        metrics = evaluate_model(
            model,
            val_loader,
            device
        )
        
        # Save model with metadata (Requirements 6.4, 6.5)
        save_model_with_metadata(
            model,
            history,
            metrics,
            metadata,
            args.model_dir
        )
        
        # Update .env file (Requirement 6.5)
        update_env_file()
        
        # Print final summary
        print("\n" + "="*70)
        print("LSTM TRAINING SUMMARY - REAL TEMPORAL SATELLITE DATA")
        print("="*70)
        print(f"Data source: {metadata['data_source']}")
        print(f"Training sequences: {metadata['num_train_sequences']:,}")
        print(f"Validation sequences: {metadata['num_val_sequences']:,}")
        print(f"Sequence length: {metadata['sequence_length']} time steps")
        print(f"Epochs trained: {history['epochs_trained']}")
        print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
        print(f"Final Training MAE: {history['train_mae'][-1]:.6f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        print(f"Final Validation MAE: {history['val_mae'][-1]:.6f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"\nTemporal Validation Metrics:")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R² Score: {metrics['r2_score']:.6f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nModel saved to: {args.model_dir}/crop_health_lstm_real.pth")
        print(f"Metadata saved to: {args.model_dir}/lstm_model_metrics_real.json")
        print("="*70)
        
        # Final status check (Requirement 6.3)
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
            logger.warning("  - Downloading more temporal imagery")
            logger.warning("  - Adjusting hyperparameters")
            logger.warning("  - Increasing training epochs")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
