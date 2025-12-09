# Model Deployment Guide

## Overview

This guide explains how to deploy real-trained models to production using the automated deployment script. The deployment process ensures that models trained on real Sentinel-2 satellite data replace any synthetic-trained models safely and reliably.

## Prerequisites

Before deploying models, ensure:

1. **Real-trained models exist**: Both CNN and LSTM models must be trained on real data
   - `models/crop_health_cnn_real.pth`
   - `models/crop_health_lstm_real.pth`
   - Corresponding metadata files (`*_real.json`)

2. **PyTorch is installed**: Required for model verification
   ```bash
   pip install torch
   ```

3. **Backup space available**: The script creates backups of existing models

## Deployment Script

The deployment script is located at `scripts/deploy_real_trained_models.py`.

### What It Does

The deployment script performs five key operations:

1. **Backup Existing Models**
   - Creates timestamped backup directory
   - Copies current production models to backup
   - Preserves model registry if it exists

2. **Deploy Real Models**
   - Copies real-trained models to production names
   - Updates model files: `crop_health_cnn.pth`, `crop_health_lstm.pth`
   - Updates metrics files: `cnn_model_metrics.json`, `lstm_model_metrics.json`

3. **Update Model Registry**
   - Creates/updates `models/model_registry.json`
   - Records deployment metadata
   - Tracks model provenance (real vs synthetic)

4. **Verify Models Load**
   - Tests that models can be loaded with PyTorch
   - Validates model structure and parameters
   - Confirms metadata files are correct

5. **Update Environment Configuration**
   - Sets `USE_AI_MODELS=true` in `.env`
   - Enables AI predictions in the application

## Usage

### Basic Deployment

Deploy models with default settings:

```bash
python scripts/deploy_real_trained_models.py
```

### Custom Models Directory

Deploy from a custom directory:

```bash
python scripts/deploy_real_trained_models.py --models-dir /path/to/models
```

### Dry Run

Test the deployment without making changes:

```bash
python scripts/deploy_real_trained_models.py --dry-run
```

This performs backup and verification only, without deploying models.

## Output

### Console Output

The script provides detailed progress information:

```
================================================================================
REAL-TRAINED MODELS DEPLOYMENT
================================================================================
Started at: 2025-12-09 11:28:17
Models directory: /path/to/models
================================================================================

STEP 1: Backing up existing models
  ✓ Backed up model: crop_health_cnn.pth
  ✓ Backed up metrics: cnn_model_metrics.json

STEP 2: Deploying real-trained models
  ✓ Deployed model: crop_health_cnn_real.pth -> crop_health_cnn.pth
  ✓ Deployed metrics: cnn_model_metrics_real.json -> cnn_model_metrics.json

STEP 3: Updating model registry
  ✓ Added CNN to registry
    - Trained on: real_satellite_data
    - Data source: Sentinel-2 via Sentinel Hub API
    - Accuracy: 0.83625

STEP 4: Verifying models load correctly
  ✓ CNN model verification passed
  ✓ Model has 1,143,655 parameters

STEP 5: Updating .env configuration
  ✓ .env file updated

DEPLOYMENT SUMMARY
  ✓ ALL TESTS PASSED
  All models deployed and verified successfully!
  AI predictions are now enabled with real-trained models.
```

### Generated Files

The deployment creates several files:

1. **Backup Directory**: `models/backups/YYYYMMDD_HHMMSS/`
   - Contains copies of previous production models
   - Timestamped for easy identification

2. **Model Registry**: `models/model_registry.json`
   - Central registry of deployed models
   - Tracks model metadata and provenance

3. **Deployment Report**: `models/deployment_report.json`
   - Detailed report of deployment operations
   - Status of each step
   - Success/failure indicators

## Model Registry Format

The model registry (`models/model_registry.json`) contains:

```json
{
  "last_updated": "2025-12-09T11:28:17.213078",
  "deployment_type": "real_trained_models",
  "models": {
    "cnn": {
      "model_file": "crop_health_cnn.pth",
      "metrics_file": "cnn_model_metrics.json",
      "model_type": "CNN",
      "framework": "PyTorch",
      "version": "2.0",
      "trained_on": "real_satellite_data",
      "data_source": "Sentinel-2 via Sentinel Hub API",
      "training_date": "2025-12-09T07:36:45.322425",
      "deployed_date": "2025-12-09T11:28:17.213295",
      "accuracy": 0.83625,
      "status": "active",
      "backup_location": "models/backups/20251209_112817"
    },
    "lstm": {
      "model_file": "crop_health_lstm.pth",
      "metrics_file": "lstm_model_metrics.json",
      "model_type": "LSTM",
      "framework": "PyTorch",
      "version": "2.0",
      "trained_on": "real_temporal_sequences",
      "data_source": "Sentinel-2 time-series via Sentinel Hub API",
      "training_date": "2025-12-09T08:05:09.431343",
      "deployed_date": "2025-12-09T11:28:17.213482",
      "accuracy": 0.9787034895271063,
      "status": "active",
      "backup_location": "models/backups/20251209_112817"
    }
  }
}
```

## Verification

After deployment, verify the models are working:

```bash
python test_model_deployment.py
```

This runs comprehensive tests to ensure:
- Model files exist at correct locations
- Models load successfully with PyTorch
- Metadata indicates real data training
- Model registry is correct
- Backups were created

## Rollback

If you need to rollback to previous models:

1. **Identify backup directory**:
   ```bash
   ls -la models/backups/
   ```

2. **Restore from backup**:
   ```bash
   # Replace YYYYMMDD_HHMMSS with your backup timestamp
   cp models/backups/YYYYMMDD_HHMMSS/crop_health_cnn.pth models/
   cp models/backups/YYYYMMDD_HHMMSS/cnn_model_metrics.json models/
   ```

3. **Update .env if needed**:
   ```bash
   # Edit .env and set USE_AI_MODELS=false if reverting to rule-based
   ```

## Troubleshooting

### Model Not Found Error

**Problem**: `Real model not found: models/crop_health_cnn_real.pth`

**Solution**: Train the models first using:
```bash
python scripts/train_cnn_on_real_data.py
python scripts/train_lstm_on_real_data.py
```

### Model Load Failure

**Problem**: `Failed to load CNN model: ...`

**Solution**: 
1. Check PyTorch is installed: `pip install torch`
2. Verify model file is not corrupted
3. Check model was saved correctly during training

### Permission Denied

**Problem**: Cannot write to models directory

**Solution**: 
1. Check directory permissions
2. Run with appropriate user permissions
3. Ensure models directory is writable

### Backup Space Issues

**Problem**: Not enough disk space for backup

**Solution**:
1. Clean old backups: `rm -rf models/backups/old_timestamp/`
2. Free up disk space
3. Use external storage for backups

## Best Practices

1. **Always backup before deployment**
   - The script does this automatically
   - Keep at least 2-3 recent backups

2. **Verify after deployment**
   - Run `test_model_deployment.py`
   - Check model predictions in dashboard

3. **Monitor model performance**
   - Track accuracy metrics
   - Compare with previous models
   - Watch for degradation

4. **Document deployments**
   - Keep deployment reports
   - Note any issues or observations
   - Track model versions

5. **Test in staging first**
   - Deploy to staging environment
   - Verify functionality
   - Then deploy to production

## Integration with Application

After deployment, the models are automatically used by:

1. **Dashboard**: `production_dashboard.py`
   - Loads models from `models/crop_health_cnn.pth`
   - Uses real-trained models for predictions

2. **AI Predictor**: `src/ai_models/crop_health_predictor.py`
   - Checks `USE_AI_MODELS` environment variable
   - Falls back to rule-based if models unavailable

3. **Model Monitoring**: `src/ai_models/model_monitoring.py`
   - Tracks model performance
   - Uses model registry for metadata

## Requirements Validation

This deployment script satisfies the following requirements:

- **Requirement 5.4**: Model metadata indicates training on real data
- **Requirement 5.5**: Models are saved and deployed correctly
- **Requirement 6.4**: LSTM model metadata indicates real data
- **Requirement 6.5**: LSTM model is saved and deployed
- **Requirement 9.5**: .env is updated to enable AI models

## Related Documentation

- [CNN Training Guide](../scripts/README_CNN_TRAINING.md)
- [LSTM Training Guide](../scripts/train_lstm_on_real_data.py)
- [Model Comparison Report](../TASK_15_MODEL_COMPARISON_REPORT.md)
- [Pipeline Orchestration](../scripts/README_PIPELINE.md)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review deployment logs in console output
3. Examine `models/deployment_report.json`
4. Check model registry: `models/model_registry.json`
