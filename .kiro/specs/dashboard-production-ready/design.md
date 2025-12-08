# Design Document

## Overview

The Dashboard Production-Ready Enhancement transforms AgriFlux from a prototype with mock data into a fully functional demonstration system. The design focuses on three critical pillars: (1) Real data processing pipeline from Sentinel-2A to dashboard, (2) Robust error handling ensuring zero-crash demos, and (3) Working AI predictions with graceful fallback. The architecture maintains the existing modular structure while adding data persistence, comprehensive error handling, and demo mode capabilities.

## Architecture

### System Flow

```
[Sentinel-2A SAFE Directory]
         ↓
[Data Processing Pipeline]
  - Parse JP2 files
  - Calculate vegetation indices
  - Generate GeoTIFFs
         ↓
[SQLite Database]
  - Processed imagery
  - Vegetation indices
  - Metadata
  - Alert history
         ↓
[Streamlit Dashboard]
  - Load from database
  - Display visualizations
  - Generate alerts
  - Export reports
```

### Technology Stack

**Core Technologies:**
- Python 3.9+
- Streamlit 1.28.0 for dashboard
- SQLite for data persistence
- Rasterio/GDAL for geospatial processing
- TensorFlow/Keras for AI models (optional)
- Plotly for interactive charts
- Folium for maps

**Key Libraries:**
- pandas, numpy for data manipulation
- scikit-learn for ML utilities
- Pillow for image processing
- pickle for serialization

## Components and Interfaces

### 1. Data Processing Pipeline

**Purpose**: Process Sentinel-2A data and populate database

**Key Functions:**
- `process_sentinel2_safe()`: Main processing orchestrator
- `calculate_all_indices()`: Compute NDVI, SAVI, EVI, NDWI, NDSI
- `export_geotiff()`: Save processed indices as GeoTIFF
- `save_to_database()`: Persist results to SQLite

**Input**: S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE directory

**Output**: 
- SQLite database with processed data
- GeoTIFF files in `data/processed/`
- Metadata JSON files

**Error Handling:**
- Validate SAFE directory structure
- Check for required bands
- Handle missing SCL layer gracefully
- Log all processing steps

### 2. Database Layer

**Purpose**: Persistent storage for processed data and application state

**Schema:**

```python
# processed_imagery table
{
    'id': INTEGER PRIMARY KEY,
    'acquisition_date': TEXT,
    'tile_id': TEXT,
    'cloud_coverage': REAL,
    'ndvi_path': TEXT,  # Path to GeoTIFF
    'savi_path': TEXT,
    'evi_path': TEXT,
    'ndwi_path': TEXT,
    'ndsi_path': TEXT,
    'metadata_json': TEXT,
    'processed_at': TEXT
}

# alerts table
{
    'id': INTEGER PRIMARY KEY,
    'imagery_id': INTEGER,
    'alert_type': TEXT,  # 'vegetation_stress', 'pest_risk', etc.
    'severity': TEXT,  # 'critical', 'high', 'medium', 'low'
    'affected_area': TEXT,  # GeoJSON polygon
    'message': TEXT,
    'recommendation': TEXT,
    'created_at': TEXT,
    'acknowledged': INTEGER,
    'acknowledged_at': TEXT
}

# ai_predictions table
{
    'id': INTEGER PRIMARY KEY,
    'imagery_id': INTEGER,
    'model_version': TEXT,
    'prediction_type': TEXT,  # 'crop_health', 'yield_forecast'
    'predictions_json': TEXT,  # Serialized predictions
    'confidence_scores': TEXT,
    'created_at': TEXT
}
```

**Operations:**
- `init_database()`: Create tables if not exist
- `save_processed_imagery()`: Insert new processed data
- `get_latest_imagery()`: Retrieve most recent data
- `get_temporal_series()`: Get time series for trends
- `save_alert()`: Store generated alerts
- `get_active_alerts()`: Retrieve unacknowledged alerts

### 3. AI Prediction Module

**Purpose**: Generate crop health predictions with fallback logic

**Primary Mode - CNN Model:**
```python
class CropHealthPredictor:
    def __init__(self, model_path='models/crop_health_cnn.h5'):
        try:
            self.model = load_model(model_path)
            self.mode = 'ai'
        except:
            self.mode = 'rule_based'
    
    def predict(self, image_patch):
        if self.mode == 'ai':
            return self._ai_predict(image_patch)
        else:
            return self._rule_based_predict(image_patch)
```

**Fallback Mode - Rule-Based:**
```python
def _rule_based_predict(self, ndvi_values):
    """
    Classification rules:
    - NDVI > 0.7: Healthy (class 0)
    - 0.5 < NDVI <= 0.7: Moderate (class 1)
    - 0.3 < NDVI <= 0.5: Stressed (class 2)
    - NDVI <= 0.3: Critical (class 3)
    """
    predictions = np.zeros_like(ndvi_values, dtype=int)
    predictions[ndvi_values > 0.7] = 0  # healthy
    predictions[(ndvi_values > 0.5) & (ndvi_values <= 0.7)] = 1
    predictions[(ndvi_values > 0.3) & (ndvi_values <= 0.5)] = 2
    predictions[ndvi_values <= 0.3] = 3  # critical
    
    # Generate confidence scores
    confidence = np.abs(ndvi_values - 0.5) / 0.5  # Distance from threshold
    
    return predictions, confidence
```

### 4. Alert Generation System

**Purpose**: Generate actionable alerts based on data analysis

**Alert Rules:**

```python
class AlertGenerator:
    THRESHOLDS = {
        'ndvi_critical': 0.3,
        'ndvi_high': 0.4,
        'ndvi_medium': 0.5,
        'temp_high': 32,
        'humidity_low': 40,
        'humidity_high': 80
    }
    
    def generate_alerts(self, imagery_data, sensor_data=None):
        alerts = []
        
        # Vegetation stress alerts
        ndvi = imagery_data['ndvi']
        if np.mean(ndvi) < self.THRESHOLDS['ndvi_critical']:
            alerts.append({
                'type': 'vegetation_stress',
                'severity': 'critical',
                'message': 'Severe vegetation stress detected',
                'recommendation': 'Immediate irrigation and inspection required'
            })
        
        # Pest risk alerts (if sensor data available)
        if sensor_data:
            temp = sensor_data.get('temperature', 25)
            humidity = sensor_data.get('humidity', 60)
            
            if temp > 28 and humidity > 75:
                alerts.append({
                    'type': 'pest_risk',
                    'severity': 'high',
                    'message': 'High fungal disease risk',
                    'recommendation': 'Monitor for fungal infections, consider preventive treatment'
                })
        
        return alerts
```

### 5. Dashboard Pages

**Page Structure:**

```
dashboard/
├── main.py                 # Entry point, navigation
└── pages/
    ├── overview.py         # Summary metrics, key insights
    ├── field_monitoring.py # Interactive maps, real-time data
    ├── temporal_analysis.py # Time series, trends
    ├── alerts.py           # Alert management
    └── data_export.py      # Export functionality
```

**Overview Page:**
- Key metrics cards (health index, alert count, data quality)
- Recent alerts summary
- Quick field status overview
- System health indicators

**Field Monitoring Page:**
- Folium map with vegetation index layers
- Click interactions for pixel details
- Field boundary overlays
- AI prediction overlay (if available)

**Temporal Analysis Page:**
- Plotly line charts for NDVI trends
- Multi-index comparison
- Seasonal pattern detection
- Anomaly highlighting

**Alerts Page:**
- Active alerts list with severity badges
- Alert history timeline
- Acknowledgment functionality
- Recommendation display

**Data Export Page:**
- GeoTIFF download for each index
- CSV export of time series
- PDF report generation
- Batch export options

### 6. Error Handling Framework

**Purpose**: Ensure dashboard never crashes during demos

**Implementation:**

```python
# Decorator for page functions
def safe_page(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            st.error(f"📁 Data file not found: {e.filename}")
            st.info("💡 Run data processing pipeline first")
        except ImportError as e:
            st.error(f"📦 Missing dependency: {e.name}")
            st.code(f"pip install {e.name}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
            st.info("Please refresh the page or contact support")
            logging.error(f"Dashboard error: {e}", exc_info=True)
    return wrapper

@safe_page
def show_field_monitoring():
    # Page implementation
    pass
```

**Logging Configuration:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
```

### 7. Demo Mode System

**Purpose**: Quick-load pre-configured data for demonstrations

**Implementation:**

```python
class DemoDataManager:
    def __init__(self):
        self.demo_data_path = 'data/demo/'
    
    def load_demo_data(self):
        """Load pre-processed demo data"""
        return {
            'imagery': self._load_demo_imagery(),
            'alerts': self._load_demo_alerts(),
            'predictions': self._load_demo_predictions(),
            'time_series': self._load_demo_timeseries()
        }
    
    def _load_demo_imagery(self):
        # Load 3 scenarios: healthy, stressed, mixed
        scenarios = ['healthy_field', 'stressed_field', 'mixed_field']
        return {s: pickle.load(open(f'{self.demo_data_path}{s}.pkl', 'rb')) 
                for s in scenarios}
```

**Demo Mode UI:**
```python
# In sidebar
if st.sidebar.button("🎬 Load Demo Data"):
    st.session_state.demo_mode = True
    st.session_state.data = demo_manager.load_demo_data()
    st.success("✅ Demo data loaded!")
    st.rerun()
```

## Data Models

### ProcessedImagery
```python
@dataclass
class ProcessedImagery:
    id: int
    acquisition_date: datetime
    tile_id: str
    cloud_coverage: float
    indices: Dict[str, np.ndarray]  # NDVI, SAVI, etc.
    geotiff_paths: Dict[str, Path]
    metadata: Dict[str, Any]
    processed_at: datetime
```

### Alert
```python
@dataclass
class Alert:
    id: int
    imagery_id: int
    alert_type: str
    severity: str
    affected_area: str  # GeoJSON
    message: str
    recommendation: str
    created_at: datetime
    acknowledged: bool
    acknowledged_at: Optional[datetime]
```

### Prediction
```python
@dataclass
class Prediction:
    id: int
    imagery_id: int
    model_version: str
    predictions: np.ndarray
    confidence_scores: np.ndarray
    class_names: List[str]
    created_at: datetime
```

## Error Handling

### Data Processing Errors
- **Missing SAFE directory**: Check path, provide example
- **Corrupted JP2 files**: Skip and log, continue with available bands
- **Insufficient bands**: Warn user, calculate only possible indices
- **Memory errors**: Process in tiles, reduce resolution

### Dashboard Errors
- **Database not found**: Initialize new database, show setup wizard
- **No processed data**: Display "Get Started" guide
- **Import errors**: Check dependencies, show installation commands
- **Visualization errors**: Show placeholder, log error details

### AI Model Errors
- **Model file not found**: Switch to rule-based mode automatically
- **Inference errors**: Fall back to rule-based, log error
- **Invalid input shape**: Resize/reshape automatically
- **Out of memory**: Process in smaller batches

## Testing Strategy

### Unit Testing
- Test vegetation index calculations with known values
- Test alert generation logic with synthetic data
- Test database operations (CRUD)
- Test error handling decorators

### Integration Testing
- Test complete processing pipeline
- Test dashboard page loading
- Test data export functionality
- Test demo mode activation

### Manual Testing Checklist
- [ ] Dashboard loads without errors
- [ ] All pages accessible
- [ ] Real data displays correctly
- [ ] Alerts generate properly
- [ ] Export functions work
- [ ] Demo mode loads quickly
- [ ] Error messages are helpful
- [ ] Mobile responsive

## Performance Considerations

### Optimization Strategies
1. **Caching**: Use `@st.cache_data` for expensive operations
2. **Lazy Loading**: Load data only when page accessed
3. **Database Indexing**: Index on acquisition_date, tile_id
4. **Image Compression**: Store compressed GeoTIFFs
5. **Pagination**: Limit time series to recent 50 points

### Expected Performance
- Dashboard load time: < 3 seconds
- Page navigation: < 1 second
- Data export: < 5 seconds
- Alert generation: < 2 seconds

## Deployment Configuration

### Environment Variables
```bash
# .env file
AGRIFLUX_ENV=production
DATABASE_PATH=data/agriflux.db
PROCESSED_DATA_PATH=data/processed/
MODEL_PATH=models/
LOG_LEVEL=INFO
DEMO_MODE_ENABLED=true
```

### Requirements
```
streamlit==1.28.0
pandas==1.5.3
numpy==1.24.3
plotly==5.17.0
folium==0.14.0
streamlit-folium==0.15.0
rasterio==1.3.8
geopandas==0.13.2
scikit-learn==1.3.0
Pillow==10.0.0
```
