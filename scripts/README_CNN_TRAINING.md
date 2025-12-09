# CNN Training on Real Satellite Data

## Overview

The `train_cnn_on_real_data.py` script trains a Convolutional Neural Network (CNN) model specifically on real Sentinel-2 satellite imagery downloaded from the Sentinel Hub API. This script is part of the real satellite data integration pipeline.

## Features

### Core Functionality
- **Real Data Training**: Uses only real satellite imagery (not synthetic data)
- **Early Stopping**: Prevents overfitting with configurable patience
- **Validation Monitoring**: Tracks validation accuracy throughout training
- **Model Checkpointing**: Saves best model weights automatically
- **Comprehensive Logging**: Detailed logs of all training metrics
- **Data Provenance**: Model metadata includes complete data source information

### Requirements Satisfied
- **5.1**: Training uses real satellite imagery patches
- **5.2**: Validation accuracy monitoring (target ≥ 85%)
- **5.3**: Comprehensive evaluation metrics (confusion matrix, classification report)
- **5.4**: Model metadata with real data provenance
- **5.5**: Model saved and .env updated to enable AI predictions
- **7.3**: Comprehensive logging of training metrics

## Usage

### Basic Usage

```bash
python scripts/train_cnn_on_real_data.py
```

### With Custom Parameters

```bash
python scripts/train_cnn_on_real_data.py \
    --epochs 50 \
    --batch-size 32 \
    --patience 10 \
    --min-accuracy 0.85
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Maximum number of training epochs |
| `--batch-size` | 32 | Batch size for training |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--min-accuracy` | 0.85 | Minimum required validation accuracy |
| `--data-dir` | `data/training` | Directory containing training data |
| `--model-dir` | `models` | Directory to save trained model |

## Prerequisites

### Required Files

The script expects the following files to exist:

```
data/training/
├── cnn_X_train_real.npy      # Training images (N, 64, 64, 4)
├── cnn_y_train_real.npy      # Training labels (N,)
├── cnn_X_val_real.npy        # Validation images (N, 64, 64, 4)
├── cnn_y_val_real.npy        # Validation labels (N,)
└── cnn_metadata_real.json    # Dataset metadata
```

These files are created by running:
```bash
python scripts/prepare_real_training_data.py
```

### Python Dependencies

- PyTorch
- NumPy
- scikit-learn

## Output Files

### Model Files

```
models/
├── crop_health_cnn_real.pth           # PyTorch model checkpoint
├── cnn_model_metrics_real.json        # Model metadata and metrics
└── cnn_training_history_real.json     # Training history
```

### Log Files

```
logs/
└── cnn_training.log                   # Comprehensive training logs
```

### Environment Configuration

The script automatically updates `.env`:
```
USE_AI_MODELS=true
```

## Model Architecture

### CropHealthCNN

```
Input: (batch_size, 4, 64, 64)  # 4 bands: B02, B03, B04, B08

Conv Block 1:
  - Conv2d(4 → 32, kernel=3, padding=1)
  - BatchNorm2d(32)
  - ReLU
  - MaxPool2d(2, 2)
  → Output: (batch_size, 32, 32, 32)

Conv Block 2:
  - Conv2d(32 → 64, kernel=3, padding=1)
  - BatchNorm2d(64)
  - ReLU
  - MaxPool2d(2, 2)
  → Output: (batch_size, 64, 16, 16)

Conv Block 3:
  - Conv2d(64 → 128, kernel=3, padding=1)
  - BatchNorm2d(128)
  - ReLU
  - MaxPool2d(2, 2)
  → Output: (batch_size, 128, 8, 8)

Fully Connected:
  - Flatten → (batch_size, 8192)
  - Linear(8192 → 128)
  - ReLU
  - Dropout(0.3)
  - Linear(128 → 4)
  → Output: (batch_size, 4)  # 4 classes

Total Parameters: 1,143,204
```

### Output Classes

0. **Healthy**: NDVI > 0.6
1. **Moderate**: 0.4 < NDVI ≤ 0.6
2. **Stressed**: 0.2 < NDVI ≤ 0.4
3. **Critical**: NDVI ≤ 0.2

## Training Process

### 1. Data Loading
- Loads real satellite imagery from `data/training/`
- Verifies data source is 'real' (not synthetic)
- Converts to PyTorch tensors in NCHW format

### 2. Training Loop
- Trains for specified number of epochs
- Monitors validation loss for early stopping
- Saves best model weights automatically
- Logs metrics every epoch

### 3. Early Stopping
- Monitors validation loss
- Stops if no improvement for `patience` epochs
- Restores best model weights

### 4. Evaluation
- Generates confusion matrix
- Creates classification report
- Calculates accuracy and confidence metrics

### 5. Model Saving
- Saves PyTorch checkpoint with:
  - Model state dict
  - Training history
  - Evaluation metrics
  - Timestamp
- Saves comprehensive metadata JSON
- Updates .env configuration

## Model Metadata

The saved metadata includes:

```json
{
  "model_type": "CNN",
  "framework": "PyTorch",
  "trained_on": "real_satellite_data",
  "data_source": "Sentinel-2 via Sentinel Hub API",
  "data_type": "real",
  "training_date": "2025-12-09T07:00:13.802568",
  "metrics": {
    "accuracy": 0.725,
    "confusion_matrix": [...],
    "classification_report": {...}
  },
  "architecture": {
    "total_parameters": 1143204,
    "conv_layers": 3,
    "fc_layers": 2
  },
  "dataset": {
    "num_train_samples": 6400,
    "num_val_samples": 1600,
    "class_names": ["Healthy", "Moderate", "Stressed", "Critical"]
  }
}
```

## Logging

The script provides comprehensive logging:

### Console Output
- Training progress with epoch metrics
- Validation accuracy updates
- Best model checkpoints
- Final evaluation results
- Summary statistics

### Log File
- Detailed training metrics per epoch
- Loss and accuracy values
- Confusion matrix
- Classification report
- Model saving confirmation
- Warnings and errors

## Performance Expectations

### Training Time
- ~20-25 seconds per epoch (CPU)
- ~2-3 minutes for 5 epochs
- ~15-20 minutes for 50 epochs

### Accuracy Targets
- **Minimum**: 85% validation accuracy (Requirement 5.2)
- **Expected**: 85-92% with sufficient training data
- **Best**: 90%+ with optimal hyperparameters

### Factors Affecting Accuracy
1. **Training data quantity**: More imagery dates = better accuracy
2. **Training epochs**: More epochs = better convergence
3. **Data quality**: Lower cloud coverage = better features
4. **Class balance**: Equal samples per class = better performance

## Troubleshooting

### Low Accuracy (< 85%)

**Possible causes:**
- Insufficient training data (< 15 imagery dates)
- Too few training epochs
- Poor data quality (high cloud coverage)

**Solutions:**
```bash
# Download more imagery
python scripts/download_real_satellite_data.py --target-count 30

# Prepare new training data
python scripts/prepare_real_training_data.py

# Train with more epochs
python scripts/train_cnn_on_real_data.py --epochs 100
```

### Out of Memory

**Solution:**
```bash
# Reduce batch size
python scripts/train_cnn_on_real_data.py --batch-size 16
```

### Training Too Slow

**Solution:**
```bash
# Reduce epochs or use early stopping
python scripts/train_cnn_on_real_data.py --epochs 30 --patience 5
```

## Verification

To verify the trained model:

```bash
# Test model loading and inference
python test_cnn_real_model_loading.py

# Verify all requirements
python test_cnn_training_requirements.py
```

## Next Steps

After training the CNN model:

1. **Train LSTM Model**:
   ```bash
   python scripts/train_lstm_on_real_data.py
   ```

2. **Compare Models**:
   ```bash
   python scripts/compare_model_performance.py
   ```

3. **Deploy to Production**:
   ```bash
   python scripts/deploy_real_models.py
   ```

## References

- Design Document: `.kiro/specs/real-satellite-data-integration/design.md`
- Requirements: `.kiro/specs/real-satellite-data-integration/requirements.md`
- Task List: `.kiro/specs/real-satellite-data-integration/tasks.md`
