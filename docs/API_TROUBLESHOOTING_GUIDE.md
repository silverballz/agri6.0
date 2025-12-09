# Sentinel Hub API Troubleshooting Guide

## Overview

This guide provides detailed troubleshooting steps for common Sentinel Hub API issues encountered during real satellite data download.

---

## Authentication Issues

### Issue 1: 401 Unauthorized

#### Symptoms
```
requests.exceptions.HTTPError: 401 Client Error: Unauthorized
```

#### Root Causes
1. Missing or incorrect credentials
2. Expired OAuth2 token
3. Invalid client ID/secret format
4. Credentials not loaded from environment

#### Diagnostic Steps

```bash
# Step 1: Check environment variables are set
echo "Client ID: $SENTINEL_HUB_CLIENT_ID"
echo "Client Secret: $SENTINEL_HUB_CLIENT_SECRET"

# Step 2: Verify credentials format (should be UUID-like strings)
python -c "
import os
client_id = os.getenv('SENTINEL_HUB_CLIENT_ID', '')
client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET', '')
print(f'Client ID length: {len(client_id)} (should be ~36)')
print(f'Client Secret length: {len(client_secret)} (should be ~36)')
print(f'Client ID format: {client_id[:8]}...')
"

# Step 3: Test authentication directly
python -c "
from src.data_processing.sentinel_hub_client import create_client_from_env
try:
    client = create_client_from_env()
    print('✓ Authentication successful')
except Exception as e:
    print(f'✗ Authentication failed: {e}')
"
```

#### Solutions

**Solution 1: Set credentials in .env file**
```bash
# Edit .env file
cat >> .env << EOF
SENTINEL_HUB_CLIENT_ID=your-client-id-here
SENTINEL_HUB_CLIENT_SECRET=your-client-secret-here
EOF

# Reload environment
source .env  # or restart your terminal
```

**Solution 2: Get new credentials**
1. Go to https://apps.sentinel-hub.com/dashboard/
2. Log in or create account
3. Navigate to "User Settings" → "OAuth clients"
4. Create new OAuth client
5. Copy client ID and secret
6. Update .env file

**Solution 3: Verify token refresh**
```python
from src.data_processing.sentinel_hub_client import SentinelHubClient
from src.data_processing.config import SentinelHubConfig

config = SentinelHubConfig.from_env()
client = SentinelHubClient(config)

# Force token refresh
client._authenticate()
print(f"Token expires at: {client.token_expiry}")
```

---

## Request Format Issues

### Issue 2: 406 Not Acceptable

#### Symptoms
```
requests.exceptions.HTTPError: 406 Client Error: Not Acceptable
```

#### Root Causes
1. Incorrect Accept header
2. Wrong API endpoint
3. Invalid payload structure
4. Using old API version

#### Diagnostic Steps

```python
# Check API client version
from src.data_processing.sentinel_hub_client import SentinelHubClient
import inspect

# Verify query method signature
sig = inspect.signature(SentinelHubClient.query_sentinel_imagery)
print(f"Method signature: {sig}")

# Check if STAC API is used
source = inspect.getsource(SentinelHubClient.query_sentinel_imagery)
if 'catalog/1.0.0/search' in source:
    print("✓ Using STAC API v1")
else:
    print("✗ Using old API - UPDATE REQUIRED")
```

#### Solutions

**Solution 1: Verify API client is updated**
```bash
# Check for STAC API support
grep -n "catalog/1.0.0/search" src/data_processing/sentinel_hub_client.py

# If not found, update the client
git pull origin main  # or update from latest version
```

**Solution 2: Verify request headers**
```python
# Test with correct headers
import requests
import os

url = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
headers = {
    'Authorization': f'Bearer {os.getenv("SENTINEL_HUB_ACCESS_TOKEN")}',
    'Content-Type': 'application/json',
    'Accept': 'application/geo+json'  # CRITICAL: Must be geo+json
}

payload = {
    "bbox": [75.8, 30.9, 75.9, 31.0],
    "datetime": "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z",
    "collections": ["sentinel-2-l2a"],
    "limit": 10
}

response = requests.post(url, json=payload, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")
```

**Solution 3: Validate payload structure**
```python
# Ensure payload follows STAC spec
def validate_stac_payload(payload):
    required_fields = ['bbox', 'datetime', 'collections']
    for field in required_fields:
        if field not in payload:
            print(f"✗ Missing required field: {field}")
            return False
    
    # Validate bbox format
    if not isinstance(payload['bbox'], list) or len(payload['bbox']) != 4:
        print(f"✗ Invalid bbox format: {payload['bbox']}")
        return False
    
    # Validate datetime format
    if '/' not in payload['datetime']:
        print(f"✗ Invalid datetime format: {payload['datetime']}")
        return False
    
    print("✓ Payload structure valid")
    return True

# Test your payload
test_payload = {
    "bbox": [75.8, 30.9, 75.9, 31.0],
    "datetime": "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z",
    "collections": ["sentinel-2-l2a"]
}
validate_stac_payload(test_payload)
```

---

## Rate Limiting Issues

### Issue 3: 429 Too Many Requests

#### Symptoms
```
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
Retry-After: 60
```

#### Root Causes
1. Exceeded requests per second limit
2. Too many concurrent requests
3. Burst of requests without delay

#### Diagnostic Steps

```bash
# Check rate limit status in logs
grep -i "rate limit" logs/real_data_download.log

# Count requests in last minute
grep "$(date -u +%Y-%m-%d\ %H:%M)" logs/real_data_download.log | wc -l

# Check for retry attempts
grep "Retrying" logs/real_data_download.log | tail -20
```

#### Solutions

**Solution 1: Verify automatic retry is working**
```python
# Check retry logic in client
from src.data_processing.sentinel_hub_client import SentinelHubClient
import inspect

source = inspect.getsource(SentinelHubClient._execute_query_with_retry)
if 'Retry-After' in source and 'time.sleep' in source:
    print("✓ Automatic retry logic present")
else:
    print("✗ Retry logic missing - UPDATE REQUIRED")
```

**Solution 2: Reduce request rate**
```bash
# Download fewer images at once
python scripts/download_real_satellite_data.py --target-count 10

# Add delay between requests (if needed)
python scripts/download_real_satellite_data.py --delay 2.0
```

**Solution 3: Monitor rate limit headers**
```python
import requests
import os

# Make test request and check headers
url = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
headers = {'Authorization': f'Bearer {os.getenv("SENTINEL_HUB_ACCESS_TOKEN")}'}
response = requests.post(url, json={}, headers=headers)

print("Rate Limit Headers:")
print(f"  X-RateLimit-Limit: {response.headers.get('X-RateLimit-Limit', 'N/A')}")
print(f"  X-RateLimit-Remaining: {response.headers.get('X-RateLimit-Remaining', 'N/A')}")
print(f"  X-RateLimit-Reset: {response.headers.get('X-RateLimit-Reset', 'N/A')}")
```

---

## Data Availability Issues

### Issue 4: No Imagery Found

#### Symptoms
```
WARNING: No imagery found for date range 2024-01-01 to 2024-12-31
```

#### Root Causes
1. Cloud coverage threshold too restrictive
2. Date range too narrow
3. Region outside Sentinel-2 coverage
4. Incorrect geometry format

#### Diagnostic Steps

```python
# Test with relaxed constraints
from src.data_processing.sentinel_hub_client import create_client_from_env

client = create_client_from_env()

# Test 1: Very relaxed cloud threshold
geometry = {
    "type": "Polygon",
    "coordinates": [[[75.8, 30.9], [75.9, 30.9], [75.9, 31.0], [75.8, 31.0], [75.8, 30.9]]]
}

results = client.query_sentinel_imagery(
    geometry=geometry,
    date_range=("2024-01-01", "2024-12-31"),
    cloud_threshold=100.0,  # Accept any cloud coverage
    max_results=100
)

print(f"Found {len(results)} images with 100% cloud threshold")

# Test 2: Check if any imagery exists for region
if len(results) == 0:
    print("✗ No Sentinel-2 coverage for this region")
else:
    print(f"✓ Region has coverage, try cloud_threshold={max(r['cloud_coverage'] for r in results)}")
```

#### Solutions

**Solution 1: Increase cloud threshold**
```bash
# Try with higher cloud coverage
python scripts/download_real_satellite_data.py --cloud-threshold 30.0

# Or even higher if needed
python scripts/download_real_satellite_data.py --cloud-threshold 50.0
```

**Solution 2: Expand date range**
```bash
# Look back 2 years instead of 1
python scripts/download_real_satellite_data.py --days-back 730

# Or specify custom date range
python -c "
from scripts.download_real_satellite_data import RealDataDownloader
from pathlib import Path

downloader = RealDataDownloader(
    output_dir=Path('data/processed'),
    db_path=Path('data/agriflux.db')
)

# Custom date range
results = downloader.download_timeseries(
    days_back=1095,  # 3 years
    target_count=30,
    cloud_threshold=25.0
)
"
```

**Solution 3: Verify region geometry**
```python
# Validate geometry format
def validate_geometry(geometry):
    if geometry['type'] != 'Polygon':
        print(f"✗ Invalid type: {geometry['type']}")
        return False
    
    coords = geometry['coordinates'][0]
    if len(coords) < 4:
        print(f"✗ Polygon needs at least 4 points, got {len(coords)}")
        return False
    
    if coords[0] != coords[-1]:
        print(f"✗ Polygon not closed: {coords[0]} != {coords[-1]}")
        return False
    
    # Check coordinate order (lon, lat)
    for i, (lon, lat) in enumerate(coords):
        if not (-180 <= lon <= 180):
            print(f"✗ Invalid longitude at point {i}: {lon}")
            return False
        if not (-90 <= lat <= 90):
            print(f"✗ Invalid latitude at point {i}: {lat}")
            return False
    
    print("✓ Geometry valid")
    return True

# Test Ludhiana geometry
ludhiana = {
    "type": "Polygon",
    "coordinates": [[[75.8, 30.9], [75.9, 30.9], [75.9, 31.0], [75.8, 31.0], [75.8, 30.9]]]
}
validate_geometry(ludhiana)
```

**Solution 4: Check Sentinel-2 coverage**
```bash
# Use EO Browser to verify coverage
echo "Check coverage at: https://apps.sentinel-hub.com/eo-browser/"
echo "Coordinates: 30.9°N to 31.0°N, 75.8°E to 75.9°E"
```

---

## Date Validation Issues

### Issue 5: Future Date Error

#### Symptoms
```
ValueError: Date range cannot be in future. Requested: 2025-01-01 to 2025-12-31
```

#### Root Causes
1. Hardcoded future dates in script
2. System clock incorrect
3. Timezone mismatch

#### Diagnostic Steps

```bash
# Check system date
date

# Check if date is correct
python -c "
from datetime import datetime
now = datetime.now()
print(f'System time: {now}')
print(f'UTC time: {datetime.utcnow()}')
"

# Check date validation logic
python -c "
from src.data_processing.sentinel_hub_client import SentinelHubClient
from src.data_processing.config import SentinelHubConfig

config = SentinelHubConfig.from_env()
client = SentinelHubClient(config)

# Test validation
try:
    client._validate_date_range(('2024-01-01', '2024-12-31'))
    print('✓ Past dates accepted')
except ValueError as e:
    print(f'✗ Validation error: {e}')

try:
    client._validate_date_range(('2025-01-01', '2025-12-31'))
    print('✗ Future dates accepted (BUG!)')
except ValueError as e:
    print('✓ Future dates rejected correctly')
"
```

#### Solutions

**Solution 1: Use relative dates**
```python
from datetime import datetime, timedelta

# Always use dates relative to now
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
```

**Solution 2: Fix system clock**
```bash
# On Linux/Mac
sudo ntpdate -s time.nist.gov

# Or use timedatectl
sudo timedatectl set-ntp true

# Verify
date
```

---

## Download and Processing Issues

### Issue 6: Corrupted Band Data

#### Symptoms
```
ValueError: Band data has unexpected shape: (512, 512) expected (512, 512, 1)
```

#### Root Causes
1. Incomplete download
2. API returned error in image data
3. Incorrect band parsing

#### Diagnostic Steps

```python
# Verify downloaded band data
import numpy as np
from pathlib import Path

def check_band_integrity(imagery_dir):
    band_dir = Path(imagery_dir) / 'bands'
    
    for band_file in band_dir.glob('*.tif'):
        try:
            import rasterio
            with rasterio.open(band_file) as src:
                data = src.read(1)
                print(f"{band_file.name}: shape={data.shape}, dtype={data.dtype}, "
                      f"min={data.min()}, max={data.max()}")
                
                # Check for invalid values
                if np.isnan(data).any():
                    print(f"  ✗ Contains NaN values")
                if np.isinf(data).any():
                    print(f"  ✗ Contains Inf values")
                if data.min() == data.max():
                    print(f"  ✗ All values are identical")
        except Exception as e:
            print(f"  ✗ Error reading {band_file.name}: {e}")

# Check latest download
check_band_integrity('data/processed/S2A_43REQ_20240923')
```

#### Solutions

**Solution 1: Re-download corrupted imagery**
```python
from scripts.download_real_satellite_data import RealDataDownloader
from pathlib import Path
import shutil

# Delete corrupted directory
corrupted_dir = Path('data/processed/S2A_43REQ_20240923')
if corrupted_dir.exists():
    shutil.rmtree(corrupted_dir)
    print(f"Deleted {corrupted_dir}")

# Re-download
downloader = RealDataDownloader(
    output_dir=Path('data/processed'),
    db_path=Path('data/agriflux.db')
)

# Download specific date
result = downloader._download_and_process_single_date(
    geometry=downloader._create_ludhiana_geometry(),
    imagery_meta={'acquisition_date': '2024-09-23', 'tile_id': '43REQ', 'cloud_coverage': 5.2}
)
print(f"Re-download result: {result}")
```

**Solution 2: Validate after download**
```python
# Add validation to download script
def validate_downloaded_bands(bands_dict):
    required_bands = ['B02', 'B03', 'B04', 'B08']
    
    for band_name in required_bands:
        if band_name not in bands_dict:
            raise ValueError(f"Missing band: {band_name}")
        
        data = bands_dict[band_name]
        
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"Band {band_name} has zero dimension")
        
        if np.isnan(data).all():
            raise ValueError(f"Band {band_name} is all NaN")
        
        if data.min() == data.max():
            raise ValueError(f"Band {band_name} has no variation")
    
    print("✓ All bands valid")
    return True
```

---

## Training Data Issues

### Issue 7: Insufficient Training Samples

#### Symptoms
```
ValueError: Insufficient training samples. Found 500, need at least 2000 per class
```

#### Root Causes
1. Not enough imagery downloaded
2. Too few valid patches extracted
3. Imbalanced class distribution

#### Diagnostic Steps

```python
# Check available imagery
from src.database.db_manager import DatabaseManager

db = DatabaseManager('data/agriflux.db')
stats = db.get_database_statistics()

print(f"Real imagery count: {stats['real_imagery_count']}")
print(f"Need at least: 15")

if stats['real_imagery_count'] < 15:
    print("✗ Insufficient imagery - download more")
else:
    print("✓ Sufficient imagery")

# Check extracted patches
import numpy as np
from pathlib import Path

train_file = Path('data/training/cnn_X_train_real.npy')
if train_file.exists():
    X = np.load(train_file)
    y = np.load('data/training/cnn_y_train_real.npy')
    
    print(f"\nTraining samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    min_samples = np.bincount(y).min()
    if min_samples < 2000:
        print(f"✗ Insufficient samples per class: {min_samples}")
    else:
        print(f"✓ Sufficient samples per class: {min_samples}")
```

#### Solutions

**Solution 1: Download more imagery**
```bash
# Increase target count
python scripts/download_real_satellite_data.py --target-count 30

# Or expand date range
python scripts/download_real_satellite_data.py --days-back 730
```

**Solution 2: Adjust patch extraction**
```bash
# Reduce stride to extract more patches
python scripts/prepare_real_training_data.py --stride 16

# Or reduce samples per class requirement
python scripts/prepare_real_training_data.py --samples-per-class 1000
```

**Solution 3: Check class distribution**
```python
# Analyze NDVI distribution to understand class imbalance
import numpy as np
from pathlib import Path

def analyze_ndvi_distribution():
    processed_dir = Path('data/processed')
    all_ndvi = []
    
    for img_dir in processed_dir.iterdir():
        if not img_dir.is_dir():
            continue
        
        ndvi_file = img_dir / 'arrays' / 'indices.npy'
        if ndvi_file.exists():
            indices = np.load(ndvi_file, allow_pickle=True).item()
            all_ndvi.append(indices['NDVI'].flatten())
    
    if all_ndvi:
        all_ndvi = np.concatenate(all_ndvi)
        
        print("NDVI Distribution:")
        print(f"  Critical (≤0.2): {(all_ndvi <= 0.2).sum() / len(all_ndvi) * 100:.1f}%")
        print(f"  Stressed (0.2-0.4): {((all_ndvi > 0.2) & (all_ndvi <= 0.4)).sum() / len(all_ndvi) * 100:.1f}%")
        print(f"  Moderate (0.4-0.6): {((all_ndvi > 0.4) & (all_ndvi <= 0.6)).sum() / len(all_ndvi) * 100:.1f}%")
        print(f"  Healthy (>0.6): {(all_ndvi > 0.6).sum() / len(all_ndvi) * 100:.1f}%")

analyze_ndvi_distribution()
```

---

## Model Training Issues

### Issue 8: Training Divergence

#### Symptoms
```
WARNING: Loss increased from 0.234 to 1.567
WARNING: Training may be diverging
```

#### Root Causes
1. Learning rate too high
2. Batch size too small
3. Poor weight initialization
4. Data normalization issues

#### Diagnostic Steps

```python
# Check training history
import json

with open('models/cnn_training_history_real.json') as f:
    history = json.load(f)

losses = history['train_loss']
print(f"Loss trend: {losses[:5]} ... {losses[-5:]}")

# Check for divergence
if losses[-1] > losses[0] * 2:
    print("✗ Training diverged")
else:
    print("✓ Training converged")

# Check data normalization
import numpy as np
X_train = np.load('data/training/cnn_X_train_real.npy')
print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"Data mean: {X_train.mean():.3f}")
print(f"Data std: {X_train.std():.3f}")
```

#### Solutions

**Solution 1: Reduce learning rate**
```bash
# Try lower learning rate
python scripts/train_cnn_on_real_data.py --learning-rate 0.0001

# Or use learning rate scheduler
python scripts/train_cnn_on_real_data.py --use-scheduler
```

**Solution 2: Increase batch size**
```bash
# Larger batches = more stable gradients
python scripts/train_cnn_on_real_data.py --batch-size 64
```

**Solution 3: Normalize data**
```python
# Add normalization to training script
import numpy as np

X_train = np.load('data/training/cnn_X_train_real.npy')

# Normalize to [0, 1]
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())

# Or standardize (mean=0, std=1)
X_train = (X_train - X_train.mean()) / X_train.std()

np.save('data/training/cnn_X_train_real_normalized.npy', X_train)
```

---

## Getting Additional Help

### Enable Debug Logging

```python
# Add to top of script
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)
```

### Collect Diagnostic Information

```bash
# Run diagnostic script
python -c "
import sys
import os
from pathlib import Path

print('=== System Information ===')
print(f'Python version: {sys.version}')
print(f'Platform: {sys.platform}')

print('\n=== Environment Variables ===')
print(f'SENTINEL_HUB_CLIENT_ID: {\"SET\" if os.getenv(\"SENTINEL_HUB_CLIENT_ID\") else \"NOT SET\"}')
print(f'SENTINEL_HUB_CLIENT_SECRET: {\"SET\" if os.getenv(\"SENTINEL_HUB_CLIENT_SECRET\") else \"NOT SET\"}')

print('\n=== File Status ===')
print(f'Processed imagery: {len(list(Path(\"data/processed\").glob(\"*/\")))} directories')
print(f'Training data: {\"EXISTS\" if Path(\"data/training/cnn_X_train_real.npy\").exists() else \"MISSING\"}')
print(f'CNN model: {\"EXISTS\" if Path(\"models/crop_health_cnn_real.pth\").exists() else \"MISSING\"}')
print(f'LSTM model: {\"EXISTS\" if Path(\"models/crop_health_lstm_real.pth\").exists() else \"MISSING\"}')

print('\n=== Database Status ===')
from src.database.db_manager import DatabaseManager
db = DatabaseManager('data/agriflux.db')
stats = db.get_database_statistics()
print(f'Real imagery: {stats[\"real_imagery_count\"]}')
print(f'Synthetic imagery: {stats[\"synthetic_imagery_count\"]}')
"
```

### Contact Support

If issues persist:
1. Collect diagnostic information above
2. Check relevant log files
3. Review Sentinel Hub API status: https://status.sentinel-hub.com/
4. Contact Sentinel Hub support: https://forum.sentinel-hub.com/
