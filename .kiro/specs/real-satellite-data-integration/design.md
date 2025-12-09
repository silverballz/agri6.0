# Design Document

## Overview

This design addresses the critical issue where AgriFlux AI models are trained on synthetic data instead of real Sentinel-2 satellite imagery. The root cause is a bug in the Sentinel Hub API integration that prevents successful data download. This design provides a comprehensive solution to fix the API client, download 15-20 real imagery dates, and retrain both CNN and LSTM models on actual agricultural data.

The solution involves three main components:
1. **API Client Fixes**: Correct date handling, request format, and error handling
2. **Data Pipeline**: Download, process, and store real multi-temporal imagery
3. **Model Retraining**: Train CNN and LSTM on real data with validation

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Sentinel Hub API                          │
│              (Real Satellite Imagery Source)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ OAuth2 + REST API
                     │
┌────────────────────▼────────────────────────────────────────┐
│            Fixed API Client Layer                           │
│  - Correct date handling (no future dates)                  │
│  - Proper request format (STAC API compliant)               │
│  - Robust error handling and retry logic                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Raw Band Data
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Data Processing Pipeline                           │
│  - Calculate vegetation indices (NDVI, SAVI, EVI, NDWI)     │
│  - Generate GeoTIFF and numpy arrays                        │
│  - Mark data as real (synthetic=false)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Processed Imagery
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Database Storage                               │
│  - Store imagery metadata and file paths                    │
│  - Track data provenance (real vs synthetic)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Training Data
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Training Data Preparation                          │
│  - Extract 64x64 patches from real imagery                  │
│  - Generate labels using rule-based classifier              │
│  - Balance dataset across health classes                    │
│  - Split train/validation (80/20)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Prepared Datasets
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────┐         ┌────────▼─────┐
│  CNN Model   │         │  LSTM Model  │
│  Training    │         │  Training    │
│  (Spatial)   │         │  (Temporal)  │
└───────┬──────┘         └────────┬─────┘
        │                         │
        │ Trained Models          │
        │                         │
┌───────▼─────────────────────────▼─────┐
│      Production Model Deployment      │
│  - Replace synthetic-trained models   │
│  - Update model metadata              │
│  - Enable AI predictions              │
└───────────────────────────────────────┘
```

### Data Flow

1. **API Query**: System queries Sentinel Hub for imagery in date range with cloud filter
2. **Download**: For each available date, download B02, B03, B04, B08 bands
3. **Processing**: Calculate vegetation indices and save as GeoTIFF + numpy
4. **Storage**: Insert records into database with real data flag
5. **Preparation**: Extract patches and generate balanced training dataset
6. **Training**: Train CNN (spatial) and LSTM (temporal) models
7. **Deployment**: Replace old models and enable AI features

## Components and Interfaces

### 1. Fixed Sentinel Hub API Client

**File**: `src/data_processing/sentinel_hub_client.py`

**Key Fixes**:

```python
class SentinelHubClient:
    def query_sentinel_imagery(
        self,
        geometry: Dict[str, Any],
        date_range: Tuple[str, str],
        cloud_threshold: float = 20.0,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Query for available imagery with corrected implementation.
        
        Fixes:
        1. Validate date range is in past (not future)
        2. Use correct STAC API endpoint and format
        3. Handle pagination for results > 10
        4. Proper error handling for 406 errors
        """
        # FIX 1: Validate dates
        start_date, end_date = self._validate_date_range(date_range)
        
        # FIX 2: Use correct STAC endpoint
        catalog_url = f"{self.config.base_url}/api/v1/catalog/1.0.0/search"
        
        # FIX 3: Correct payload format
        payload = {
            "bbox": self._geometry_to_bbox(geometry),
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "collections": ["sentinel-2-l2a"],
            "limit": min(max_results, 100),  # API limit
            "query": {
                "eo:cloud_cover": {"lte": cloud_threshold}
            },
            "fields": {
                "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
                "exclude": []
            }
        }
        
        # FIX 4: Proper headers
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/geo+json'  # STAC format
        }
        
        return self._execute_query_with_retry(catalog_url, payload, headers)
    
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
        
        if start > end:
            raise ValueError(f"Start date {start_str} is after end date {end_str}")
        
        return start_str, end_str
```

**Interface**:
- Input: Geometry (GeoJSON), date range, cloud threshold
- Output: List of imagery metadata with acquisition dates, cloud coverage, tile IDs
- Errors: Raises ValueError for invalid dates, RequestException for API failures

### 2. Real Data Download Script

**File**: `scripts/download_real_satellite_data.py`

**Purpose**: Orchestrate the complete download pipeline

```python
class RealDataDownloader:
    """Download and process real Sentinel-2 imagery."""
    
    def __init__(self, output_dir: Path, db_path: Path):
        self.client = create_client_from_env()
        self.output_dir = output_dir
        self.db = DatabaseManager(str(db_path))
        self.calculator = VegetationIndexCalculator()
    
    def download_ludhiana_timeseries(
        self,
        days_back: int = 365,
        target_count: int = 20,
        cloud_threshold: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Download time-series imagery for Ludhiana region.
        
        Returns:
            List of processing results with metadata
        """
        # Define Ludhiana boundary
        geometry = self._create_ludhiana_geometry()
        
        # Calculate date range (past dates only)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Query available imagery
        imagery_list = self.client.query_sentinel_imagery(
            geometry=geometry,
            date_range=(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ),
            cloud_threshold=cloud_threshold,
            max_results=target_count
        )
        
        # Download and process each date
        results = []
        for i, imagery_meta in enumerate(imagery_list, 1):
            logger.info(f"Processing {i}/{len(imagery_list)}: {imagery_meta['acquisition_date']}")
            
            result = self._download_and_process_single_date(
                geometry,
                imagery_meta
            )
            results.append(result)
        
        return results
    
    def _download_and_process_single_date(
        self,
        geometry: Dict,
        imagery_meta: Dict
    ) -> Dict[str, Any]:
        """Download bands, calculate indices, save to disk and database."""
        acquisition_date = imagery_meta['acquisition_date'][:10]
        tile_id = imagery_meta.get('tile_id', '43REQ')
        
        # Download bands
        bands = self.client.download_multispectral_bands(
            geometry=geometry,
            acquisition_date=acquisition_date,
            bands=['B02', 'B03', 'B04', 'B08'],
            resolution=10
        )
        
        # Calculate indices
        indices = {
            'NDVI': self.calculator.calculate_ndvi(bands['B08'], bands['B04']),
            'SAVI': self.calculator.calculate_savi(bands['B08'], bands['B04']),
            'EVI': self.calculator.calculate_evi(bands['B08'], bands['B04'], bands['B02']),
            'NDWI': self.calculator.calculate_ndwi(bands['B03'], bands['B08'])
        }
        
        # Save to disk
        output_path = self._save_processed_data(
            tile_id,
            acquisition_date,
            bands,
            indices,
            imagery_meta
        )
        
        # Save to database
        imagery_id = self._save_to_database(
            tile_id,
            acquisition_date,
            imagery_meta['cloud_coverage'],
            output_path,
            synthetic=False  # CRITICAL: Mark as real data
        )
        
        return {
            'imagery_id': imagery_id,
            'acquisition_date': acquisition_date,
            'tile_id': tile_id,
            'cloud_coverage': imagery_meta['cloud_coverage'],
            'output_path': output_path,
            'success': True
        }
```

**Interface**:
- Input: Date range, target count, cloud threshold
- Output: List of downloaded imagery with file paths and database IDs
- Side Effects: Creates files in `data/processed/`, inserts database records

### 3. Training Data Preparation

**File**: `scripts/prepare_real_training_data.py`

**Purpose**: Extract patches and create balanced datasets from real imagery

```python
class RealDatasetPreparator:
    """Prepare training datasets from real satellite imagery."""
    
    def __init__(self, processed_dir: Path, output_dir: Path):
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.classifier = RuleBasedClassifier()
    
    def prepare_cnn_dataset(
        self,
        patch_size: int = 64,
        stride: int = 32,
        samples_per_class: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare CNN training dataset from real imagery.
        
        Returns:
            Tuple of (X_train, y_train) arrays
        """
        # Load all real imagery dates
        imagery_dirs = self._find_real_imagery_dirs()
        
        all_patches = []
        all_labels = []
        
        # Extract patches from each date
        for img_dir in imagery_dirs:
            patches, labels = self._extract_patches_from_imagery(
                img_dir,
                patch_size,
                stride
            )
            all_patches.append(patches)
            all_labels.append(labels)
        
        # Concatenate all patches
        X = np.concatenate(all_patches, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Balance dataset
        X_balanced, y_balanced = self._balance_dataset(
            X, y, samples_per_class
        )
        
        # Save prepared data
        self._save_training_data(X_balanced, y_balanced, 'cnn')
        
        return X_balanced, y_balanced
    
    def prepare_lstm_dataset(
        self,
        sequence_length: int = 10,
        samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare LSTM training dataset from temporal sequences.
        
        Returns:
            Tuple of (X_sequences, y_targets) arrays
        """
        # Load imagery sorted by date
        imagery_list = self._load_temporal_imagery_sorted()
        
        # Extract temporal sequences
        sequences = []
        targets = []
        
        for i in range(len(imagery_list) - sequence_length):
            # Get sequence of NDVI values
            seq = [
                self._load_ndvi(imagery_list[j])
                for j in range(i, i + sequence_length)
            ]
            
            # Target is next time step
            target = self._load_ndvi(imagery_list[i + sequence_length])
            
            sequences.append(np.stack(seq, axis=0))
            targets.append(target)
        
        X_seq = np.array(sequences)
        y_target = np.array(targets)
        
        # Save prepared data
        self._save_training_data(X_seq, y_target, 'lstm')
        
        return X_seq, y_target
    
    def _find_real_imagery_dirs(self) -> List[Path]:
        """Find all directories containing real (non-synthetic) imagery."""
        real_dirs = []
        
        for img_dir in self.processed_dir.iterdir():
            if not img_dir.is_dir():
                continue
            
            metadata_file = img_dir / 'metadata.json'
            if not metadata_file.exists():
                continue
            
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Only include real data
            if not metadata.get('synthetic', True):
                real_dirs.append(img_dir)
        
        return sorted(real_dirs)
```

**Interface**:
- Input: Processed imagery directory
- Output: Numpy arrays saved to `data/training/`
- Validation: Ensures only real data (synthetic=false) is used

### 4. Model Training Scripts

**Files**: 
- `scripts/train_cnn_on_real_data.py`
- `scripts/train_lstm_on_real_data.py`

**CNN Training**:

```python
def train_cnn_on_real_data():
    """Train CNN model on real satellite imagery."""
    
    # Load real training data
    X_train = np.load('data/training/cnn_X_train_real.npy')
    y_train = np.load('data/training/cnn_y_train_real.npy')
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    # Create model
    model = CropHealthCNN(num_classes=4)
    
    # Train with early stopping
    history, best_acc = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        patience=10,
        min_accuracy=0.85  # Requirement threshold
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)
    
    # Save model with metadata
    save_model_with_metadata(
        model,
        'models/crop_health_cnn_real.pth',
        {
            'trained_on': 'real_satellite_data',
            'accuracy': metrics['accuracy'],
            'training_date': datetime.now().isoformat(),
            'data_source': 'Sentinel-2 via Sentinel Hub API'
        }
    )
    
    return metrics
```

**LSTM Training**:

```python
def train_lstm_on_real_data():
    """Train LSTM model on real temporal sequences."""
    
    # Load real temporal data
    X_seq = np.load('data/training/lstm_X_sequences_real.npy')
    y_target = np.load('data/training/lstm_y_targets_real.npy')
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_target,
        test_size=0.2,
        random_state=42
    )
    
    # Create model
    model = CropHealthLSTM(
        input_size=1,  # NDVI values
        hidden_size=128,
        num_layers=2,
        output_size=1
    )
    
    # Train
    history, best_loss = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        patience=15,
        min_accuracy=0.80  # Requirement threshold
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)
    
    # Save model with metadata
    save_model_with_metadata(
        model,
        'models/crop_health_lstm_real.pth',
        {
            'trained_on': 'real_temporal_sequences',
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'training_date': datetime.now().isoformat(),
            'data_source': 'Sentinel-2 time-series via Sentinel Hub API'
        }
    )
    
    return metrics
```

## Data Models

### Imagery Metadata

```python
@dataclass
class ImageryMetadata:
    """Metadata for processed satellite imagery."""
    acquisition_date: datetime
    tile_id: str
    cloud_coverage: float
    synthetic: bool  # CRITICAL: False for real data
    data_source: str  # "Sentinel Hub API" or "Synthetic Generator"
    bands: List[str]
    indices: List[str]
    file_paths: Dict[str, str]
    statistics: Dict[str, Dict[str, float]]
    processed_at: datetime
```

### Training Dataset Metadata

```python
@dataclass
class TrainingDatasetMetadata:
    """Metadata for training datasets."""
    dataset_type: str  # "cnn" or "lstm"
    data_source: str  # "real" or "synthetic"
    num_samples: int
    num_classes: int
    patch_size: Optional[int]  # For CNN
    sequence_length: Optional[int]  # For LSTM
    class_distribution: Dict[str, int]
    created_at: datetime
    imagery_dates: List[str]  # Source imagery dates
```

### Model Metadata

```python
@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    model_type: str  # "CNN" or "LSTM"
    framework: str  # "PyTorch"
    trained_on: str  # "real_satellite_data" or "synthetic_data"
    data_source: str  # "Sentinel-2 via Sentinel Hub API"
    training_date: datetime
    accuracy: float
    validation_metrics: Dict[str, float]
    training_samples: int
    validation_samples: int
    epochs_trained: int
    model_path: str
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Date validation prevents future queries

*For any* date range provided to the API client, if either the start or end date is in the future, the system should raise a ValueError before making any API request.

**Validates: Requirements 1.2**

### Property 2: Real data is marked correctly

*For any* imagery downloaded from the Sentinel Hub API, the metadata synthetic flag should be set to false, and the data_source field should contain "Sentinel Hub API".

**Validates: Requirements 3.3**

### Property 3: Training data contains only real imagery

*For any* training dataset prepared for model training, all source imagery should have synthetic=false in their metadata.

**Validates: Requirements 4.1**

### Property 4: Model accuracy meets threshold

*For any* trained CNN model on real data, the validation accuracy should be greater than or equal to 0.85.

**Validates: Requirements 5.2**

### Property 5: LSTM accuracy meets threshold

*For any* trained LSTM model on real data, the validation accuracy should be greater than or equal to 0.80.

**Validates: Requirements 6.3**

### Property 6: API retry logic handles rate limits

*For any* API request that receives a 429 (rate limit) response, the system should wait for the duration specified in the Retry-After header before retrying.

**Validates: Requirements 1.5**

### Property 7: Downloaded imagery has all required bands

*For any* successfully downloaded imagery, the returned dictionary should contain all requested bands (B02, B03, B04, B08).

**Validates: Requirements 2.3**

### Property 8: Vegetation indices are within valid ranges

*For any* calculated vegetation index (NDVI, SAVI, EVI, NDWI), all values should be within the mathematically valid range for that index.

**Validates: Requirements 8.2**

### Property 9: Balanced dataset has equal class representation

*For any* balanced training dataset, the number of samples for each crop health class should be equal (within tolerance of ±5%).

**Validates: Requirements 4.3**

### Property 10: Model metadata reflects training data source

*For any* model trained on real data, the model metadata file should contain trained_on="real_satellite_data" and data_source should reference Sentinel Hub API.

**Validates: Requirements 5.4, 6.5**

## Error Handling

### API Errors

1. **406 Not Acceptable**: 
   - Cause: Incorrect request format or headers
   - Handling: Log detailed request/response, validate payload format, check Accept header
   - Recovery: Retry with corrected format

2. **401 Unauthorized**:
   - Cause: Invalid or expired token
   - Handling: Re-authenticate and retry
   - Recovery: Automatic token refresh

3. **429 Rate Limit**:
   - Cause: Too many requests
   - Handling: Respect Retry-After header, implement exponential backoff
   - Recovery: Wait and retry

4. **404 Not Found**:
   - Cause: No imagery available for date/region
   - Handling: Log and skip date, continue with other dates
   - Recovery: Try alternative dates

### Data Processing Errors

1. **Invalid Band Data**:
   - Cause: Corrupted download or parsing error
   - Handling: Validate band dimensions and data types
   - Recovery: Re-download imagery

2. **Index Calculation Errors**:
   - Cause: Division by zero, invalid band values
   - Handling: Use epsilon values, clip results to valid ranges
   - Recovery: Log warning, continue with valid indices

3. **Insufficient Training Data**:
   - Cause: Too few imagery dates downloaded
   - Handling: Check minimum threshold (15 dates)
   - Recovery: Download more dates or adjust date range

### Training Errors

1. **Low Model Accuracy**:
   - Cause: Insufficient or poor quality training data
   - Handling: Log warning, save model anyway with metadata
   - Recovery: Request more training data or adjust hyperparameters

2. **Out of Memory**:
   - Cause: Batch size too large
   - Handling: Reduce batch size automatically
   - Recovery: Resume training with smaller batches

3. **Training Divergence**:
   - Cause: Learning rate too high
   - Handling: Implement early stopping
   - Recovery: Reduce learning rate and restart

## Testing Strategy

### Unit Tests

1. **Date Validation Tests**:
   - Test future date rejection
   - Test invalid date format handling
   - Test start > end date rejection

2. **API Request Format Tests**:
   - Test payload structure matches STAC spec
   - Test header format
   - Test bbox calculation

3. **Data Processing Tests**:
   - Test vegetation index calculations
   - Test GeoTIFF creation
   - Test metadata generation

### Property-Based Tests

Each correctness property will be implemented as a property-based test using Hypothesis:

1. **Property 1 Test**: Generate random date ranges, verify future dates raise ValueError
2. **Property 2 Test**: Generate mock API responses, verify synthetic flag is false
3. **Property 3 Test**: Generate training datasets, verify all source imagery is real
4. **Property 4 Test**: Train on various real datasets, verify accuracy ≥ 0.85
5. **Property 5 Test**: Train LSTM on various sequences, verify accuracy ≥ 0.80
6. **Property 6 Test**: Simulate 429 responses, verify retry logic waits correctly
7. **Property 7 Test**: Generate download responses, verify all bands present
8. **Property 8 Test**: Generate random band data, verify indices in valid ranges
9. **Property 9 Test**: Generate datasets, verify class balance within ±5%
10. **Property 10 Test**: Generate model metadata, verify training source fields correct

### Integration Tests

1. **End-to-End Pipeline Test**:
   - Run complete pipeline on small date range
   - Verify data downloaded, processed, stored
   - Verify models trained and saved

2. **API Integration Test**:
   - Test actual API connection with credentials
   - Verify query returns results
   - Verify download works

3. **Database Integration Test**:
   - Verify imagery records inserted correctly
   - Verify real vs synthetic data queries work
   - Verify latest imagery retrieval prioritizes real data

### Validation Scripts

1. **Data Quality Validator**:
   - Check all imagery has required bands
   - Verify indices in valid ranges
   - Confirm minimum date count

2. **Model Performance Validator**:
   - Evaluate models on held-out test set
   - Compare real-trained vs synthetic-trained
   - Generate performance comparison report

3. **Provenance Validator**:
   - Verify all training data marked as real
   - Check model metadata accuracy
   - Confirm no synthetic data in training pipeline
