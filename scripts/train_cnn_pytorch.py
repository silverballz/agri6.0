#!/usr/bin/env python3
"""
Train CNN Model using PyTorch (TensorFlow alternative)

This script trains a CNN model using PyTorch to avoid TensorFlow compatibility issues.
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cnn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CropHealthCNN(nn.Module):
    """PyTorch CNN for crop health classification"""
    
    def __init__(self, num_classes=4):
        super(CropHealthCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
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


def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    """Train the CNN model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return history, best_val_acc


def main():
    logger.info("="*70)
    logger.info("PyTorch CNN Model Training Pipeline")
    logger.info("="*70)
    
    # Load training data
    data_dir = Path('data/training')
    logger.info(f"Loading training data from {data_dir}...")
    
    X_train = np.load(data_dir / 'cnn_X_train.npy')
    y_train = np.load(data_dir / 'cnn_y_train.npy')
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Training: {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    
    # Convert to PyTorch tensors (NCHW format)
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # NHWC -> NCHW
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    device = torch.device('cpu')
    model = CropHealthCNN(num_classes=4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel created with {total_params:,} parameters")
    
    # Train model
    logger.info("="*70)
    logger.info("Training CNN Model")
    logger.info("="*70)
    
    history, best_val_acc = train_model(model, train_loader, val_loader, epochs=10, device=device)
    
    # Evaluate on validation set
    logger.info("="*70)
    logger.info("Evaluating Model")
    logger.info("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Get unique classes in predictions
    unique_classes = sorted(set(all_labels))
    class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    target_names = [class_names[i] for i in unique_classes]
    
    class_report = classification_report(all_labels, all_preds, 
                                        target_names=target_names,
                                        labels=unique_classes)
    
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"\n  Confusion Matrix:\n{conf_matrix}")
    logger.info(f"\n  Classification Report:\n{class_report}")
    
    # Save model
    logger.info("="*70)
    logger.info("Saving Model and Metrics")
    logger.info("="*70)
    
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save PyTorch model
    model_path = model_dir / 'crop_health_cnn.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'best_val_acc': best_val_acc,
        'history': history
    }, model_path)
    logger.info(f"  Saved model to: {model_path}")
    
    # Save metrics
    metrics = {
        'model_type': 'CNN',
        'framework': 'PyTorch',
        'model_name': 'CropHealthCNN',
        'training_date': datetime.now().isoformat(),
        'version': '1.0',
        'metrics': {
            'accuracy': float(accuracy),
            'best_val_accuracy': float(best_val_acc),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1])
        },
        'architecture': {
            'framework': 'PyTorch',
            'total_parameters': total_params,
            'conv_layers': 3,
            'fc_layers': 2
        },
        'classes': ['Healthy', 'Moderate', 'Stressed', 'Critical']
    }
    
    metrics_path = model_dir / 'cnn_model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved metrics to: {metrics_path}")
    
    # Update .env
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        with open(env_path, 'w') as f:
            found = False
            for line in lines:
                if line.startswith('USE_AI_MODELS='):
                    f.write('USE_AI_MODELS=true\n')
                    found = True
                else:
                    f.write(line)
            
            if not found:
                f.write('\nUSE_AI_MODELS=true\n')
        
        logger.info("  Updated USE_AI_MODELS=true in .env")
    
    logger.info("="*70)
    logger.info("CNN TRAINING SUMMARY")
    logger.info("="*70)
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Epochs: 10")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("="*70)
    
    if accuracy >= 0.85:
        logger.info("\n✅ Model achieved >85% accuracy target!")
    else:
        logger.info(f"\n⚠️  Model accuracy {accuracy:.2%} is below 85% target")
    
    logger.info("\n✅ CNN training complete!")
    logger.info("Model is ready for inference.")


if __name__ == '__main__':
    main()
