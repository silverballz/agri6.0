#!/usr/bin/env python3
"""
Train LSTM Model using PyTorch for Temporal Trend Analysis

This script trains an LSTM model for vegetation trend forecasting.
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VegetationTrendLSTM(nn.Module):
    """PyTorch LSTM for vegetation trend forecasting"""
    
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(VegetationTrendLSTM, self).__init__()
        
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
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # FC layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def train_model(model, train_loader, val_loader, epochs=20, device='cpu'):
    """Train the LSTM model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss = train_loss / len(train_loader)
        train_mae = mean_absolute_error(train_targets, train_predictions)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return history, best_val_loss


def main():
    logger.info("="*70)
    logger.info("PyTorch LSTM Model Training Pipeline")
    logger.info("="*70)
    
    # Load training data
    data_dir = Path('data/training')
    logger.info(f"Loading training data from {data_dir}...")
    
    X_train = np.load(data_dir / 'lstm_X_train.npy')
    y_train = np.load(data_dir / 'lstm_y_train.npy')
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Training: {len(X_train)} sequences")
    logger.info(f"  Validation: {len(X_val)} sequences")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)  # Add dimension for output
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    device = torch.device('cpu')
    sequence_length = X_train.shape[1]
    input_size = X_train.shape[2]
    
    model = VegetationTrendLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel created with {total_params:,} parameters")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Input features: {input_size}")
    
    # Train model
    logger.info("="*70)
    logger.info("Training LSTM Model")
    logger.info("="*70)
    
    history, best_val_loss = train_model(model, train_loader, val_loader, epochs=20, device=device)
    
    # Evaluate on validation set
    logger.info("="*70)
    logger.info("Evaluating Model")
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
    
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    
    # Save model
    logger.info("="*70)
    logger.info("Saving Model and Metrics")
    logger.info("="*70)
    
    model_dir = Path('models/lstm_temporal')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    model_path = model_dir / 'vegetation_trend_lstm.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'best_val_loss': best_val_loss,
        'history': history,
        'sequence_length': sequence_length,
        'input_size': input_size
    }, model_path)
    logger.info(f"  Saved model to: {model_path}")
    
    # Save metrics
    metrics = {
        'model_type': 'LSTM',
        'framework': 'PyTorch',
        'model_name': 'VegetationTrendLSTM',
        'training_date': datetime.now().isoformat(),
        'version': '1.0',
        'metrics': {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1])
        },
        'architecture': {
            'framework': 'PyTorch',
            'total_parameters': total_params,
            'hidden_size': 64,
            'num_layers': 2,
            'bidirectional': True,
            'sequence_length': int(sequence_length),
            'input_features': int(input_size)
        }
    }
    
    metrics_path = model_dir / 'lstm_model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved metrics to: {metrics_path}")
    
    logger.info("="*70)
    logger.info("LSTM TRAINING SUMMARY")
    logger.info("="*70)
    logger.info(f"Training sequences: {len(X_train):,}")
    logger.info(f"Validation sequences: {len(X_val):,}")
    logger.info(f"Epochs: 20")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("="*70)
    
    logger.info("\n✅ LSTM training complete!")
    logger.info("Model is ready for temporal trend forecasting.")


if __name__ == '__main__':
    main()
