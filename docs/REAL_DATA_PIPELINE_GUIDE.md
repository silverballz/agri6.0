# Real Satellite Data Pipeline Guide

## Overview

This guide documents the complete pipeline for downloading real Sentinel-2 satellite imagery, processing it, and training AI models on actual agricultural data. This pipeline replaces the synthetic data workflow and ensures production-ready model accuracy.

## Table of Contents

1. [API Client Fixes and Usage](#api-client-fixes-and-usage)
2. [Downloading Additional Data](#downloading-additional-data)
3. [Training Data Preparation](#training-data-preparation)
4. [Model Retraining Instructions](#model-retraining-instructions)
5. [Troubleshooting Guide](#troubleshooting-guide)

---

## API Client Fixes and Usage

### Overview of Fixes

The Sentinel Hub API client had several critical bugs that prevented real data download. The following fixes were implemented:

#### 1. Date Validation Fix

**Problem**: System was querying future dates, causing API rejections.

**Solution**: Added date validation to prevent future date queries.

```python
def _validate_date_range(self, date_range: Tuple[str, str]) -> Tuple[str, str]:
    """Ensure dates are valid and not in future."""
    start_str, end_str = date_range
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d')
    now = datetime.now()
    
    if start > now or end > now:
        raise ValueError(
            f"Date range cannot be in future. "
            f"Requested: {start_str} to {end_str}, "
            f"Current date: {now.strftime('%Y-%m-%d')}"
        )
    
    return start_str, end_str
```

#### 2. STAC API Request Format Fix

**Problem**: Incorrect API endpoint and request format causing 406 errors.

**Solution**: Updated to use correct STAC API v1 endpoint with proper payload structure.

```python
# Correct endpoint
catalog_url = f"{self.config.base_url}/api/v1/catalog/1.0.0/search"

# Correct payload format
payload = {
    "bbox": self._geometry_to_bbox(geometry),
    "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
    "collections": ["sentinel-2-l2a"],
    "limit": min(max_results, 100),
    "query": {
        "eo:cloud_cover": {"lte": cloud_threshold}
    },
    "fields": {
        "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
        "exclude": []
    }
}

# Correct headers
headers = {
    'Authorization': f'Bearer {self.access_token}',
    'Content-Type': 'application/json',
    'Accept': 'application/geo+json'  # STAC format
}
```

#### 3. Error Handling and Retry Logic

**Problem**: No retry logic for rate limits and transient failures.

**Solution**: Implemented exponential backoff with proper error handling.

```python
def _execute_query_with_retry(
    self,
    url: str,
    payload: Dict,
    headers: Dict,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """Execute query with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 429:
                # Rate limit - respect Retry-After header
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            logger.warning(f"Request failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
```

### Using the API Client

#### Basic Usage

```python
from src.data_processing.sentinel_hub_client import create_client_from_env

# Create client from environment variables
client = create_client_from_env()

# Define region of interest (Ludhiana example)
geometry = {
    "type": "Polygon",
    "coordinates": [[
        [75.8, 30.9],
        [75.9, 30.9],
        [75.9, 31.0],
        [75.8, 31.0],
        [75.8, 30.9]
    ]]
}

# Query available imagery
imagery_list = client.query_sentinel_imagery(
    geometry=geometry,
    date_range=("2024-01-01", "2024-12-31"),
    cloud_threshold=20.0,
    max_results=20
)

# Download bands for a specific date
bands = client.download_multispectral_bands(
    geometry=geometry,
    acquisition_date="2024-09-23",
    bands=['B02', 'B03', 'B04', 'B08'],
    resolution=10
)
```

#### Configuration

Set the following environment variables in your `.env` file:

```bash
# Sentinel Hub API credentials
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret
SENTINEL_HUB_BASE_URL=https://services.sentinel-hub.com
```

#### API Rate Limits

- **Free tier**: 30,000 processing units/month
- **Rate limit**: Varies by plan, typically 10-20 requests/second
- **Retry-After**: System automatically respects rate limit headers

---

## Downloading Additional Data

### Using the Download Script

The `download_real_satellite_data.py` script orchestrates the complete download pipeline.

#### Basic Download

```bash
# Download 20 imagery dates for Ludhiana region
python scripts/download_real_satellite_data.py \
    --region ludhiana \
    --days-back 365 \
    --target-count 20 \
    --cloud-threshold 20.0
```

#### Command-Line Options

```
--region TEXT           Region name (default: ludhiana)
--days-back INTEGER     Days to look back from today (default: 365)
--target-count INTEGER  Target number of imagery dates (default: 20)
--cloud-threshold FLOAT Maximum cloud coverage % (default: 20.0)
--output-dir PATH       Output directory (default: data/processed)
--db-path PATH          Database path (default: data/agriflux.db)
```

### Downloading for Custom Regions

#### 1. Define Custom Geometry

Create a GeoJSON file with your region boundary:

```json
{
  "type": "Polygon",
  "coordinates": [[
    [longitude1, latitude1],
    [longitude2, latitude2],
    [longitude3, latitude3],
    [longitude4, latitude4],
    [longitude1, latitude1]
  ]]
}
```

#### 2. Modify Download Script

```python
from pathlib import Path
import json

# Load custom geometry
with open('my_region.geojson') as f:
    geometry = json.load(f)

# Create downloader
downloader = RealDataDownloader(
    output_dir=Path('data/processed'),
    db_path=Path('data/agriflux.db')
)

# Download for custom region
results = downloader.download_timeseries(
    geometry=geometry,
    days_back=365,
    target_count=20,
    cloud_threshold=20.0
)
```

### Download Output Structure

```
data/processed/
├── S2A_43REQ_20240923/
│   ├── bands/
│   │   ├── B02.tif
│   │   ├── B03.tif
│   │   ├── B04.tif
│   │   └── B08.tif
│   ├── indices/
│   │   ├── NDVI.tif
│   │   ├── SAVI.tif
│   │   ├── EVI.tif
│   │   └── NDWI.tif
│   ├── arrays/
│   │   ├── bands.npy
│   │   └── indices.npy
│   └── metadata.json
├── S2A_43REQ_20240915/
│   └── ...
└── ...
```

### Metadata Format

Each imagery directory contains a `metadata.json` file:

```json
{
  "tile_id": "43REQ",
  "acquisition_date": "2024-09-23T05:36:41Z",
  "cloud_coverage": 5.2,
  "synthetic": false,
  "data_source": "Sentinel Hub API",
  "bands": ["B02", "B03", "B04", "B08"],
  "indices": ["NDVI", "SAVI", "EVI", "NDWI"],
  "resolution": 10,
  "processed_at": "2024-12-09T10:30:00Z",
  "statistics": {
    "NDVI": {"min": -0.2, "max": 0.9, "mean": 0.65, "std": 0.15},
    "SAVI": {"min": -0.15, "max": 0.85, "mean": 0.55, "std": 0.12}
  }
}
```

### Verifying Downloads

```bash
# Run data quality validation
python scripts/validate_data_quality.py

# Check database records
python -c "
from src.database.db_manager import DatabaseManager
db = DatabaseManager('data/agriflux.db')
stats = db.get_database_statistics()
print(f'Real imagery count: {stats[\"real_imagery_count\"]}')
print(f'Synthetic imagery count: {stats[\"synthetic_imagery_count\"]}')
"
```

---

## Training Data Preparation

### CNN Training Data Preparation

The CNN model requires spatial patches extracted from real imagery.

#### Running the Preparation Script

```bash
# Prepare CNN training dataset
python scripts/prepare_real_training_data.py \
    --patch-size 64 \
    --stride 32 \
    --samples-per-class 2000 \
    --output-dir data/training
```

#### What the Script Does

1. **Finds Real Imagery**: Scans `data/processed/` for directories with `synthetic=false`
2. **Extracts Patches**: Creates 64x64 pixel patches from each imagery date
3. **Generates Labels**: Uses rule-based classification on NDVI values:
   - Healthy: NDVI > 0.6
   - Moderate: 0.4 < NDVI ≤ 0.6
   - Stressed: 0.2 < NDVI ≤ 0.4
   - Critical: NDVI ≤ 0.2
4. **Balances Dataset**: Ensures equal samples per class (2000 each)
5. **Splits Data**: 80% training, 20% validation
6. **Saves Arrays**: Stores as numpy arrays with metadata

#### Output Files

```
data/training/
├── cnn_X_train_real.npy      # Training patches (N, 64, 64, 4)
├── cnn_y_train_real.npy      # Training labels (N,)
├── cnn_X_val_real.npy        # Validation patches
├── cnn_y_val_real.npy        # Validation labels
└── cnn_metadata_real.json    # Dataset metadata
```

### LSTM Training Data Preparation

The LSTM model requires temporal sequences of vegetation indices.

#### Running the Preparation Script

```bash
# Prepare LSTM training dataset
python scripts/prepare_lstm_training_data.py \
    --sequence-length 10 \
    --samples 1000 \
    --output-dir data/training
```

#### What the Script Does

1. **Loads Temporal Data**: Reads all real imagery sorted by acquisition date
2. **Creates Sequences**: Uses sliding window to generate sequences:
   - Input: 10 consecutive NDVI measurements
   - Target: Next NDVI value (prediction target)
3. **Splits Data**: 80% training, 20% validation
4. **Saves Arrays**: Stores as numpy arrays with metadata

#### Output Files

```
data/training/
├── lstm_X_sequences_real.npy  # Input sequences (N, 10, H, W)
├── lstm_y_targets_real.npy    # Target values (N, H, W)
├── lstm_X_val_real.npy        # Validation sequences
├── lstm_y_val_real.npy        # Validation targets
└── lstm_metadata_real.json    # Dataset metadata
```

### Verifying Training Data

```python
import numpy as np
import json

# Load CNN data
X_train = np.load('data/training/cnn_X_train_real.npy')
y_train = np.load('data/training/cnn_y_train_real.npy')

print(f"CNN Training samples: {X_train.shape[0]}")
print(f"Patch size: {X_train.shape[1:3]}")
print(f"Bands: {X_train.shape[3]}")
print(f"Class distribution: {np.bincount(y_train)}")

# Load metadata
with open('data/training/cnn_metadata_real.json') as f:
    metadata = json.load(f)
    print(f"Data source: {metadata['data_source']}")
    print(f"Imagery dates: {len(metadata['imagery_dates'])}")
```

---

## Model Retraining Instructions

### Prerequisites

1. **Download Real Data**: Complete at least 15 imagery dates
2. **Prepare Training Data**: Run preparation scripts for both CNN and LSTM
3. **Verify Data Quality**: Run validation script to ensure data meets requirements

### Training the CNN Model

#### Step 1: Run Training Script

```bash
# Train CNN on real data
python scripts/train_cnn_on_real_data.py \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --patience 10 \
    --min-accuracy 0.85
```

#### Step 2: Monitor Training

Training logs are saved to `logs/cnn_training.log`:

```
2024-12-09 10:00:00 - INFO - Starting CNN training on real data
2024-12-09 10:00:01 - INFO - Training samples: 8000, Validation samples: 2000
2024-12-09 10:00:05 - INFO - Epoch 1/50 - Loss: 1.234, Acc: 0.456
2024-12-09 10:00:10 - INFO - Epoch 2/50 - Loss: 0.987, Acc: 0.567
...
2024-12-09 10:15:00 - INFO - Best validation accuracy: 0.892
2024-12-09 10:15:01 - INFO - Model saved to models/crop_health_cnn_real.pth
```

#### Step 3: Verify Model Performance

```bash
# Check model metrics
cat models/cnn_model_metrics_real.json
```

Expected output:

```json
{
  "accuracy": 0.892,
  "precision": 0.885,
  "recall": 0.890,
  "f1_score": 0.887,
  "confusion_matrix": [[450, 20, 15, 15], ...],
  "training_date": "2024-12-09T10:15:00Z",
  "trained_on": "real_satellite_data"
}
```

### Training the LSTM Model

#### Step 1: Run Training Script

```bash
# Train LSTM on real temporal data
python scripts/train_lstm_on_real_data.py \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --patience 15 \
    --min-accuracy 0.80
```

#### Step 2: Monitor Training

Training logs are saved to `logs/lstm_training.log`:

```
2024-12-09 11:00:00 - INFO - Starting LSTM training on real temporal data
2024-12-09 11:00:01 - INFO - Training sequences: 800, Validation sequences: 200
2024-12-09 11:00:10 - INFO - Epoch 1/100 - Loss: 0.234, MSE: 0.045
2024-12-09 11:00:20 - INFO - Epoch 2/100 - Loss: 0.198, MSE: 0.038
...
2024-12-09 11:30:00 - INFO - Best validation MSE: 0.012
2024-12-09 11:30:01 - INFO - Model saved to models/crop_health_lstm_real.pth
```

#### Step 3: Verify Model Performance

```bash
# Check model metrics
cat models/lstm_model_metrics_real.json
```

### Deploying Trained Models

#### Step 1: Backup Existing Models

```bash
# Run deployment script (includes automatic backup)
python scripts/deploy_real_trained_models.py
```

The script will:
1. Backup existing models to `models/backups/`
2. Copy real-trained models to production location
3. Update model registry with new metadata
4. Verify models load correctly

#### Step 2: Enable AI Predictions

Update `.env` file:

```bash
# Enable AI models
USE_AI_MODELS=true
AI_MODEL_PATH=models/crop_health_cnn_real.pth
LSTM_MODEL_PATH=models/crop_health_lstm_real.pth
```

#### Step 3: Test Predictions

```python
from src.ai_models.crop_health_predictor import CropHealthPredictor

# Load predictor with real-trained models
predictor = CropHealthPredictor(
    cnn_model_path='models/crop_health_cnn_real.pth',
    lstm_model_path='models/crop_health_lstm_real.pth'
)

# Test prediction
result = predictor.predict_crop_health(
    imagery_data=test_imagery,
    temporal_data=test_sequences
)

print(f"Prediction: {result['health_class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Comparing Model Performance

```bash
# Run comparison script
python scripts/compare_model_performance.py

# View comparison report
cat reports/model_comparison_report.json
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. API Authentication Errors

**Error**: `401 Unauthorized` or `Invalid credentials`

**Causes**:
- Incorrect client ID or secret
- Expired credentials
- Missing environment variables

**Solutions**:
```bash
# Verify credentials are set
echo $SENTINEL_HUB_CLIENT_ID
echo $SENTINEL_HUB_CLIENT_SECRET

# Test authentication
python -c "
from src.data_processing.sentinel_hub_client import create_client_from_env
client = create_client_from_env()
print('Authentication successful!')
"

# If failed, update .env file with correct credentials
```

#### 2. 406 Not Acceptable Errors

**Error**: `406 Not Acceptable` from API

**Causes**:
- Incorrect request format
- Missing or wrong Accept header
- Invalid payload structure

**Solutions**:
- Ensure you're using the fixed API client (version with STAC API support)
- Check that Accept header is set to `application/geo+json`
- Verify payload follows STAC API v1 specification

```python
# Verify API client version
from src.data_processing.sentinel_hub_client import SentinelHubClient
print(SentinelHubClient.__doc__)  # Should mention STAC API support
```

#### 3. Rate Limit Errors

**Error**: `429 Too Many Requests`

**Causes**:
- Exceeded API rate limit
- Too many concurrent requests

**Solutions**:
- System automatically handles rate limits with exponential backoff
- Check logs for retry attempts
- If persistent, reduce `--target-count` or increase delay between requests

```bash
# Check rate limit status in logs
grep "Rate limited" logs/real_data_download.log

# Reduce request rate
python scripts/download_real_satellite_data.py --target-count 10
```

#### 4. No Imagery Available

**Error**: `No imagery found for date range`

**Causes**:
- Cloud coverage too restrictive
- Date range too narrow
- Region has no Sentinel-2 coverage

**Solutions**:
```bash
# Increase cloud threshold
python scripts/download_real_satellite_data.py --cloud-threshold 30.0

# Expand date range
python scripts/download_real_satellite_data.py --days-back 730

# Verify region has Sentinel-2 coverage
# Check: https://apps.sentinel-hub.com/eo-browser/
```

#### 5. Insufficient Training Data

**Error**: `Insufficient imagery dates for training (found X, need 15)`

**Causes**:
- Not enough imagery downloaded
- Too many images filtered out by cloud coverage

**Solutions**:
```bash
# Download more imagery
python scripts/download_real_satellite_data.py --target-count 30

# Check current count
python -c "
from src.database.db_manager import DatabaseManager
db = DatabaseManager('data/agriflux.db')
stats = db.get_database_statistics()
print(f'Real imagery: {stats[\"real_imagery_count\"]}')
"
```

#### 6. Model Accuracy Below Threshold

**Error**: `Model accuracy 0.78 below threshold 0.85`

**Causes**:
- Insufficient training data
- Poor data quality
- Imbalanced dataset
- Suboptimal hyperparameters

**Solutions**:
```bash
# 1. Download more imagery
python scripts/download_real_satellite_data.py --target-count 30

# 2. Verify data quality
python scripts/validate_data_quality.py

# 3. Check class balance
python -c "
import numpy as np
y_train = np.load('data/training/cnn_y_train_real.npy')
print('Class distribution:', np.bincount(y_train))
"

# 4. Adjust hyperparameters
python scripts/train_cnn_on_real_data.py \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --epochs 100
```

#### 7. Out of Memory During Training

**Error**: `CUDA out of memory` or `RuntimeError: out of memory`

**Causes**:
- Batch size too large
- Model too large for available GPU/RAM

**Solutions**:
```bash
# Reduce batch size
python scripts/train_cnn_on_real_data.py --batch-size 16

# Use CPU instead of GPU (slower but more memory)
python scripts/train_cnn_on_real_data.py --device cpu

# Clear GPU cache
python -c "
import torch
torch.cuda.empty_cache()
"
```

#### 8. Database Errors

**Error**: `sqlite3.OperationalError: database is locked`

**Causes**:
- Multiple processes accessing database
- Incomplete transaction

**Solutions**:
```bash
# Check for running processes
ps aux | grep python

# Kill conflicting processes
pkill -f download_real_satellite_data.py

# Verify database integrity
sqlite3 data/agriflux.db "PRAGMA integrity_check;"
```

#### 9. Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'X'`

**Causes**:
- Missing Python packages
- Virtual environment not activated

**Solutions**:
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; import numpy; import rasterio; print('All dependencies installed')"
```

#### 10. File Permission Errors

**Error**: `PermissionError: [Errno 13] Permission denied`

**Causes**:
- Insufficient file permissions
- Directory doesn't exist

**Solutions**:
```bash
# Create required directories
mkdir -p data/processed data/training models logs

# Fix permissions
chmod -R 755 data/ models/ logs/

# Run with appropriate permissions
sudo python scripts/download_real_satellite_data.py  # If necessary
```

### Logging and Debugging

#### Enable Debug Logging

```python
# In your script or .env file
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

#### Check Log Files

```bash
# API requests and responses
tail -f logs/real_data_download.log

# Training progress
tail -f logs/cnn_training.log
tail -f logs/lstm_training.log

# Data quality validation
tail -f logs/data_quality_validation.log

# Pipeline orchestration
tail -f logs/pipeline_orchestration.log
```

#### Verify Pipeline Status

```bash
# Run complete verification
python verify_complete_pipeline.py

# Check specific components
python -c "
from src.database.db_manager import DatabaseManager
from pathlib import Path

db = DatabaseManager('data/agriflux.db')
stats = db.get_database_statistics()

print('=== Pipeline Status ===')
print(f'Real imagery: {stats[\"real_imagery_count\"]}')
print(f'Synthetic imagery: {stats[\"synthetic_imagery_count\"]}')

# Check training data
cnn_train = Path('data/training/cnn_X_train_real.npy')
lstm_train = Path('data/training/lstm_X_sequences_real.npy')
print(f'CNN training data: {\"✓\" if cnn_train.exists() else \"✗\"}')
print(f'LSTM training data: {\"✓\" if lstm_train.exists() else \"✗\"}')

# Check models
cnn_model = Path('models/crop_health_cnn_real.pth')
lstm_model = Path('models/crop_health_lstm_real.pth')
print(f'CNN model: {\"✓\" if cnn_model.exists() else \"✗\"}')
print(f'LSTM model: {\"✓\" if lstm_model.exists() else \"✗\"}')
"
```

### Getting Help

If you encounter issues not covered in this guide:

1. **Check logs**: Review relevant log files for detailed error messages
2. **Verify requirements**: Ensure all requirements from the design document are met
3. **Run validation**: Use validation scripts to identify specific issues
4. **Check documentation**: Review API documentation at https://docs.sentinel-hub.com/
5. **Contact support**: Reach out to Sentinel Hub support for API-specific issues

---

## Additional Resources

- **Sentinel Hub API Documentation**: https://docs.sentinel-hub.com/
- **Sentinel-2 Mission Guide**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- **STAC API Specification**: https://github.com/radiantearth/stac-api-spec
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Hypothesis Testing Library**: https://hypothesis.readthedocs.io/

---

## Appendix: Complete Pipeline Example

Here's a complete example of running the entire pipeline from scratch:

```bash
# Step 1: Set up environment
export SENTINEL_HUB_CLIENT_ID=your_client_id
export SENTINEL_HUB_CLIENT_SECRET=your_client_secret

# Step 2: Download real satellite data
python scripts/download_real_satellite_data.py \
    --region ludhiana \
    --days-back 365 \
    --target-count 20 \
    --cloud-threshold 20.0

# Step 3: Validate data quality
python scripts/validate_data_quality.py

# Step 4: Prepare CNN training data
python scripts/prepare_real_training_data.py \
    --patch-size 64 \
    --samples-per-class 2000

# Step 5: Prepare LSTM training data
python scripts/prepare_lstm_training_data.py \
    --sequence-length 10 \
    --samples 1000

# Step 6: Train CNN model
python scripts/train_cnn_on_real_data.py \
    --epochs 50 \
    --min-accuracy 0.85

# Step 7: Train LSTM model
python scripts/train_lstm_on_real_data.py \
    --epochs 100 \
    --min-accuracy 0.80

# Step 8: Compare model performance
python scripts/compare_model_performance.py

# Step 9: Deploy models
python scripts/deploy_real_trained_models.py

# Step 10: Verify complete pipeline
python verify_complete_pipeline.py

# Step 11: Enable AI predictions
echo "USE_AI_MODELS=true" >> .env

# Step 12: Start dashboard
streamlit run production_dashboard.py
```

This completes the real satellite data pipeline guide.
