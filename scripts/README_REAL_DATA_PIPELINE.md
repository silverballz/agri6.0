# Real Data Pipeline Scripts

This directory contains scripts for downloading real Sentinel-2 satellite imagery and training AI models on actual agricultural data.

## Scripts Overview

### Data Download and Processing

#### `download_real_satellite_data.py`
Downloads real Sentinel-2 imagery from Sentinel Hub API.

**Usage:**
```bash
python scripts/download_real_satellite_data.py \
    --region ludhiana \
    --days-back 365 \
    --target-count 20 \
    --cloud-threshold 20.0
```

**Options:**
- `--region`: Region name (default: ludhiana)
- `--days-back`: Days to look back from today (default: 365)
- `--target-count`: Number of imagery dates to download (default: 20)
- `--cloud-threshold`: Maximum cloud coverage % (default: 20.0)
- `--output-dir`: Output directory (default: data/processed)
- `--db-path`: Database path (default: data/agriflux.db)

**Output:**
- Processed imagery in `data/processed/`
- Database records with `synthetic=false`
- Download log in `logs/real_data_download.log`

**Requirements:**
- Valid Sentinel Hub API credentials in `.env`
- Internet connection
- ~1-2 GB free disk space

---

#### `validate_data_quality.py`
Validates downloaded imagery meets quality requirements.

**Usage:**
```bash
python scripts/validate_data_quality.py
```

**Checks:**
- All required bands present (B02, B03, B04, B08)
- Vegetation indices within valid ranges
- Minimum 15 imagery dates available
- Metadata synthetic flag is false
- No corrupted or incomplete downloads

**Output:**
- Validation report in `logs/validation_report_YYYYMMDD_HHMMSS.json`
- Console summary of checks passed/failed

---

### Training Data Preparation

#### `prepare_real_training_data.py`
Prepares CNN training dataset from real imagery.

**Usage:**
```bash
python scripts/prepare_real_training_data.py \
    --patch-size 64 \
    --stride 32 \
    --samples-per-class 2000 \
    --output-dir data/training
```

**Options:**
- `--patch-size`: Size of extracted patches (default: 64)
- `--stride`: Stride for patch extraction (default: 32)
- `--samples-per-class`: Samples per health class (default: 2000)
- `--output-dir`: Output directory (default: data/training)

**Output:**
- `cnn_X_train_real.npy`: Training patches (N, 64, 64, 4)
- `cnn_y_train_real.npy`: Training labels (N,)
- `cnn_X_val_real.npy`: Validation patches
- `cnn_y_val_real.npy`: Validation labels
- `cnn_metadata_real.json`: Dataset metadata

**Process:**
1. Scans for real imagery (synthetic=false)
2. Extracts 64x64 patches with stride
3. Generates labels using rule-based classifier
4. Balances dataset across 4 health classes
5. Splits 80/20 train/validation

---

#### `prepare_lstm_training_data.py`
Prepares LSTM training dataset from temporal sequences.

**Usage:**
```bash
python scripts/prepare_lstm_training_data.py \
    --sequence-length 10 \
    --samples 1000 \
    --output-dir data/training
```

**Options:**
- `--sequence-length`: Length of temporal sequences (default: 10)
- `--samples`: Number of sequences to generate (default: 1000)
- `--output-dir`: Output directory (default: data/training)

**Output:**
- `lstm_X_sequences_real.npy`: Input sequences (N, 10, H, W)
- `lstm_y_targets_real.npy`: Target values (N, H, W)
- `lstm_X_val_real.npy`: Validation sequences
- `lstm_y_val_real.npy`: Validation targets
- `lstm_metadata_real.json`: Dataset metadata

**Process:**
1. Loads real imagery sorted by date
2. Creates sliding window sequences
3. Generates prediction targets
4. Splits 80/20 train/validation

---

### Model Training

#### `train_cnn_on_real_data.py`
Trains CNN model on real satellite imagery.

**Usage:**
```bash
python scripts/train_cnn_on_real_data.py \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --patience 10 \
    --min-accuracy 0.85
```

**Options:**
- `--epochs`: Maximum training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--min-accuracy`: Minimum validation accuracy (default: 0.85)
- `--device`: Device to use (default: cuda if available)

**Output:**
- `models/crop_health_cnn_real.pth`: Trained model weights
- `models/cnn_model_metrics_real.json`: Performance metrics
- `models/cnn_training_history_real.json`: Training history
- `logs/cnn_training.log`: Training log

**Training Process:**
1. Loads real training data
2. Creates CNN model architecture
3. Trains with early stopping
4. Validates on held-out set
5. Saves best model with metadata

**Expected Results:**
- Validation accuracy ≥85%
- Training time: ~15-30 min (GPU), ~2-4 hours (CPU)

---

#### `train_lstm_on_real_data.py`
Trains LSTM model on real temporal sequences.

**Usage:**
```bash
python scripts/train_lstm_on_real_data.py \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --patience 15 \
    --min-accuracy 0.80
```

**Options:**
- `--epochs`: Maximum training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 15)
- `--min-accuracy`: Minimum validation accuracy (default: 0.80)
- `--device`: Device to use (default: cuda if available)

**Output:**
- `models/crop_health_lstm_real.pth`: Trained model weights
- `models/lstm_model_metrics_real.json`: Performance metrics
- `models/lstm_training_history_real.json`: Training history
- `logs/lstm_training.log`: Training log

**Training Process:**
1. Loads real temporal sequences
2. Creates LSTM model architecture
3. Trains with early stopping
4. Validates on held-out sequences
5. Saves best model with metadata

**Expected Results:**
- Validation accuracy ≥80%
- Training time: ~30-60 min (GPU), ~4-8 hours (CPU)

---

### Model Evaluation and Deployment

#### `compare_model_performance.py`
Compares synthetic-trained vs real-trained models.

**Usage:**
```bash
python scripts/compare_model_performance.py
```

**Output:**
- `reports/model_comparison_report.json`: Detailed comparison
- `reports/confusion_matrix_comparison.png`: Visual comparison
- `reports/metrics_comparison.png`: Metrics bar chart

**Metrics Compared:**
- Accuracy
- Precision
- Recall
- F1 Score
- Per-class performance
- Confusion matrices

---

#### `deploy_real_trained_models.py`
Deploys real-trained models to production.

**Usage:**
```bash
python scripts/deploy_real_trained_models.py
```

**Process:**
1. Backs up existing models to `models/backups/`
2. Copies real-trained models to production location
3. Updates model registry with metadata
4. Verifies models load correctly
5. Updates `.env` to enable AI predictions

**Output:**
- `models/deployment_report.json`: Deployment summary
- Backup in `models/backups/YYYYMMDD_HHMMSS/`

---

### Utility Scripts

#### `fix_existing_data.py`
Fixes metadata for existing processed imagery.

**Usage:**
```bash
python scripts/fix_existing_data.py
```

**Purpose:**
- Updates old imagery records to include synthetic flag
- Fixes missing metadata fields
- Validates data integrity

---

#### `delete_synthetic_record.py`
Removes synthetic data records from database.

**Usage:**
```bash
python scripts/delete_synthetic_record.py --imagery-id <id>
```

**Purpose:**
- Cleans up synthetic data records
- Useful when transitioning to real data only

---

## Complete Pipeline Example

Here's how to run the complete pipeline from scratch:

```bash
# Step 1: Download real satellite data
python scripts/download_real_satellite_data.py \
    --target-count 20 \
    --cloud-threshold 20.0

# Step 2: Validate data quality
python scripts/validate_data_quality.py

# Step 3: Prepare CNN training data
python scripts/prepare_real_training_data.py \
    --samples-per-class 2000

# Step 4: Prepare LSTM training data
python scripts/prepare_lstm_training_data.py \
    --sequence-length 10

# Step 5: Train CNN model
python scripts/train_cnn_on_real_data.py \
    --epochs 50 \
    --min-accuracy 0.85

# Step 6: Train LSTM model
python scripts/train_lstm_on_real_data.py \
    --epochs 100 \
    --min-accuracy 0.80

# Step 7: Compare model performance
python scripts/compare_model_performance.py

# Step 8: Deploy models
python scripts/deploy_real_trained_models.py

# Step 9: Verify pipeline
python verify_complete_pipeline.py
```

## Prerequisites

### Environment Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Sentinel Hub credentials:**
```bash
# Add to .env file
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret
```

3. **Create required directories:**
```bash
mkdir -p data/processed data/training models logs reports
```

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: 5 GB minimum for complete pipeline
- **GPU**: Optional but recommended for training (CUDA-compatible)
- **Internet**: Required for downloading satellite data

## Troubleshooting

### Common Issues

1. **API Authentication Errors**
   - Verify credentials in `.env`
   - Check Sentinel Hub account status
   - See: `docs/API_TROUBLESHOOTING_GUIDE.md`

2. **Insufficient Training Data**
   - Download more imagery dates
   - Reduce `--samples-per-class`
   - Increase `--cloud-threshold`

3. **Model Accuracy Below Threshold**
   - Download more imagery
   - Adjust hyperparameters
   - Check data quality

4. **Out of Memory**
   - Reduce `--batch-size`
   - Use CPU instead of GPU
   - Close other applications

### Getting Help

- **Full Documentation**: `docs/REAL_DATA_PIPELINE_GUIDE.md`
- **Quick Reference**: `docs/REAL_DATA_QUICK_REFERENCE.md`
- **API Troubleshooting**: `docs/API_TROUBLESHOOTING_GUIDE.md`
- **Logging System**: `docs/LOGGING_SYSTEM.md`

## Script Dependencies

### Data Flow

```
download_real_satellite_data.py
    ↓
validate_data_quality.py
    ↓
prepare_real_training_data.py
prepare_lstm_training_data.py
    ↓
train_cnn_on_real_data.py
train_lstm_on_real_data.py
    ↓
compare_model_performance.py
    ↓
deploy_real_trained_models.py
```

### Module Dependencies

- `src.data_processing.sentinel_hub_client`: API client
- `src.data_processing.vegetation_indices`: Index calculations
- `src.database.db_manager`: Database operations
- `src.ai_models.crop_health_predictor`: Model inference
- `src.utils.logging_config`: Logging setup

## Performance Benchmarks

### Download Performance
- **1 imagery date**: ~30-60 seconds
- **20 imagery dates**: ~10-20 minutes
- **Bottleneck**: API rate limits, network speed

### Training Performance (GPU)
- **CNN (50 epochs)**: ~15-30 minutes
- **LSTM (100 epochs)**: ~30-60 minutes
- **Bottleneck**: Dataset size, batch size

### Training Performance (CPU)
- **CNN (50 epochs)**: ~2-4 hours
- **LSTM (100 epochs)**: ~4-8 hours
- **Bottleneck**: CPU speed, RAM

### Expected Accuracy
- **CNN**: 85-92% validation accuracy
- **LSTM**: 80-88% validation accuracy
- **Improvement over synthetic**: +5-15%

## Logging

All scripts log to both console and files:

- **Download**: `logs/real_data_download.log`
- **Validation**: `logs/data_quality_validation.log`
- **CNN Training**: `logs/cnn_training.log`
- **LSTM Training**: `logs/lstm_training.log`
- **Pipeline**: `logs/pipeline_orchestration.log`

Log format:
```
2024-12-09 10:00:00 - INFO - [ScriptName] Message
```

## Testing

### Unit Tests
```bash
# Test data processing
pytest tests/test_sentinel_hub_integration.py

# Test training data preparation
pytest tests/test_training_data_source_properties.py
pytest tests/test_dataset_balancing_properties.py

# Test model training
pytest tests/test_lstm_accuracy_threshold_properties.py
```

### Integration Tests
```bash
# Test complete pipeline
python verify_complete_pipeline.py

# Test database queries
python test_real_db_queries.py

# Test model deployment
python test_model_deployment.py
```

## Version History

- **v1.0** (2024-12-09): Initial release with complete pipeline
  - Fixed Sentinel Hub API client
  - Added real data download
  - Implemented model retraining
  - Added comprehensive logging

## License

See main project LICENSE file.

## Support

For issues or questions:
1. Check documentation in `docs/`
2. Review log files in `logs/`
3. Run validation scripts
4. Contact project maintainers
