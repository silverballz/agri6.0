# Task 18: Update Database Queries to Prioritize Real Data - COMPLETE ✅

## Overview

Task 18 has been successfully completed. All database query methods now properly prioritize real satellite data over synthetic data and provide comprehensive filtering capabilities.

## Requirements Validated

### Requirement 3.4: System SHALL distinguish between real and synthetic data sources
✅ **IMPLEMENTED** - Multiple query methods distinguish between real and synthetic data:
- `list_processed_imagery(synthetic=True/False/None)` - Filter by data type
- `get_real_imagery()` - Get only real data
- `get_synthetic_imagery()` - Get only synthetic data
- `count_real_imagery()` - Count real records
- `count_synthetic_imagery()` - Count synthetic records
- `get_temporal_series(synthetic=True/False/None)` - Filter temporal data

### Requirement 3.5: System SHALL prioritize real data over synthetic data
✅ **IMPLEMENTED** - Latest imagery retrieval prioritizes real data:
- `get_latest_imagery(prefer_real=True)` - Defaults to preferring real data
- Falls back to synthetic data only if no real data available
- Works with tile_id filtering

## Implementation Details

### 1. Modified `get_latest_imagery()` to Prefer Real Data

**Location**: `src/database/db_manager.py`

```python
def get_latest_imagery(self, tile_id: Optional[str] = None, prefer_real: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get the most recent processed imagery record.
    Prioritizes real data over synthetic data by default.
    
    Args:
        tile_id: Optional tile ID filter
        prefer_real: If True, prioritize real data (synthetic=0) over synthetic data
        
    Returns:
        Dictionary with imagery data or None if not found
    """
```

**Behavior**:
- When `prefer_real=True` (default), queries for real data first (synthetic=0)
- If no real data found, falls back to synthetic data
- Maintains backward compatibility with existing code

### 2. Updated `list_processed_imagery()` to Filter by Synthetic Flag

**Location**: `src/database/db_manager.py`

```python
def list_processed_imagery(self, 
                          tile_id: Optional[str] = None,
                          limit: int = 50,
                          synthetic: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    List processed imagery records with optional filtering.
    
    Args:
        tile_id: Optional tile ID filter
        limit: Maximum number of records to return
        synthetic: Optional filter - True for synthetic only, False for real only, None for all
        
    Returns:
        List of imagery record dictionaries
    """
```

**Behavior**:
- `synthetic=False` - Returns only real data
- `synthetic=True` - Returns only synthetic data
- `synthetic=None` - Returns all data (default)

### 3. Added Query Methods to Distinguish Real vs Synthetic

**New Methods**:

```python
def get_real_imagery(self, tile_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get only real (non-synthetic) imagery records."""
    
def get_synthetic_imagery(self, tile_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get only synthetic imagery records."""
    
def count_real_imagery(self, tile_id: Optional[str] = None) -> int:
    """Count real (non-synthetic) imagery records."""
    
def count_synthetic_imagery(self, tile_id: Optional[str] = None) -> int:
    """Count synthetic imagery records."""
```

**Features**:
- Convenience methods for common queries
- Support optional tile_id filtering
- Clear, explicit naming for data type

### 4. Updated Database Statistics to Show Real Data Count

**Enhanced `get_database_stats()` Method**:

```python
def get_database_stats(self) -> Dict[str, Any]:
    """
    Get database statistics including real vs synthetic data counts.
    
    Returns:
        Dictionary with database statistics
    """
```

**Statistics Included**:
- `imagery_count` - Total imagery records
- `real_imagery_count` - Count of real data records
- `synthetic_imagery_count` - Count of synthetic data records
- `date_range` - Overall date range (earliest/latest)
- `real_date_range` - Date range for real data only (earliest/latest)
- `total_alerts` - Total alert count
- `active_alerts` - Unacknowledged alert count
- `predictions_count` - Total AI predictions

### 5. Enhanced Temporal Series Filtering

**Updated `get_temporal_series()` Method**:

```python
def get_temporal_series(self, 
                       tile_id: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       synthetic: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    Get time series of imagery for temporal analysis.
    
    Args:
        tile_id: Tile identifier
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)
        synthetic: Optional filter - True for synthetic only, False for real only, None for all
        
    Returns:
        List of imagery records ordered by date
    """
```

**Features**:
- Filter by data type (real/synthetic/all)
- Date range filtering
- Ordered by acquisition date (ascending)

## Database Schema

The `processed_imagery` table includes the `synthetic` column:

```sql
CREATE TABLE processed_imagery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    acquisition_date TEXT NOT NULL,
    tile_id TEXT NOT NULL,
    cloud_coverage REAL,
    ndvi_path TEXT,
    savi_path TEXT,
    evi_path TEXT,
    ndwi_path TEXT,
    ndsi_path TEXT,
    metadata_json TEXT,
    processed_at TEXT NOT NULL,
    synthetic INTEGER DEFAULT 1,  -- 0 = real, 1 = synthetic
    UNIQUE(tile_id, acquisition_date)
);

-- Index for efficient filtering
CREATE INDEX idx_imagery_synthetic ON processed_imagery(synthetic);
```

## Testing

### Unit Tests
**File**: `tests/test_database_real_data_priority.py`

**Test Coverage**:
- ✅ 21 tests, all passing
- ✅ `get_latest_imagery()` prioritization
- ✅ `list_processed_imagery()` filtering
- ✅ Real vs synthetic query methods
- ✅ Count methods
- ✅ Database statistics
- ✅ Temporal series filtering
- ✅ Backward compatibility

**Test Results**:
```
tests/test_database_real_data_priority.py::TestGetLatestImagery::test_prefer_real_returns_real_when_available PASSED
tests/test_database_real_data_priority.py::TestGetLatestImagery::test_prefer_real_falls_back_to_synthetic PASSED
tests/test_database_real_data_priority.py::TestGetLatestImagery::test_prefer_real_false_returns_any PASSED
tests/test_database_real_data_priority.py::TestGetLatestImagery::test_prefer_real_with_tile_filter PASSED
tests/test_database_real_data_priority.py::TestListProcessedImagery::test_filter_real_only PASSED
tests/test_database_real_data_priority.py::TestListProcessedImagery::test_filter_synthetic_only PASSED
tests/test_database_real_data_priority.py::TestListProcessedImagery::test_no_filter_returns_all PASSED
tests/test_database_real_data_priority.py::TestListProcessedImagery::test_filter_with_tile_id PASSED
tests/test_database_real_data_priority.py::TestDistinguishRealVsSynthetic::test_get_real_imagery PASSED
tests/test_database_real_data_priority.py::TestDistinguishRealVsSynthetic::test_get_synthetic_imagery PASSED
tests/test_database_real_data_priority.py::TestDistinguishRealVsSynthetic::test_count_real_imagery PASSED
tests/test_database_real_data_priority.py::TestDistinguishRealVsSynthetic::test_count_synthetic_imagery PASSED
tests/test_database_real_data_priority.py::TestDistinguishRealVsSynthetic::test_count_with_tile_filter PASSED
tests/test_database_real_data_priority.py::TestDatabaseStatistics::test_stats_include_real_count PASSED
tests/test_database_real_data_priority.py::TestDatabaseStatistics::test_stats_include_real_date_range PASSED
tests/test_database_real_data_priority.py::TestDatabaseStatistics::test_stats_with_no_real_data PASSED
tests/test_database_real_data_priority.py::TestTemporalSeries::test_temporal_series_real_only PASSED
tests/test_database_real_data_priority.py::TestTemporalSeries::test_temporal_series_synthetic_only PASSED
tests/test_database_real_data_priority.py::TestTemporalSeries::test_temporal_series_all_data PASSED
tests/test_database_real_data_priority.py::TestBackwardCompatibility::test_save_without_synthetic_defaults_to_true PASSED
tests/test_database_real_data_priority.py::TestBackwardCompatibility::test_get_latest_without_prefer_real_still_works PASSED

21 passed in 0.29s
```

### Integration Verification
**File**: `verify_task_18_complete.py`

**Verification Tests**:
- ✅ Test 1: `get_latest_imagery()` prioritizes real data
- ✅ Test 2: `list_processed_imagery()` filters by synthetic flag
- ✅ Test 3: Query methods distinguish real vs synthetic
- ✅ Test 4: Database statistics show real data count
- ✅ Test 5: Temporal series filtering

**All tests passed successfully!**

## Usage Examples

### Example 1: Get Latest Real Data

```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager('data/agriflux.db')

# Get latest real data (preferred)
latest = db.get_latest_imagery(prefer_real=True)
if latest:
    print(f"Latest imagery: {latest['acquisition_date']}")
    print(f"Data type: {'Real' if latest['synthetic'] == 0 else 'Synthetic'}")
```

### Example 2: List Only Real Imagery

```python
# Get only real satellite data
real_imagery = db.list_processed_imagery(synthetic=False, limit=20)
print(f"Found {len(real_imagery)} real imagery records")

# Or use convenience method
real_imagery = db.get_real_imagery(limit=20)
```

### Example 3: Get Database Statistics

```python
stats = db.get_database_stats()
print(f"Total imagery: {stats['imagery_count']}")
print(f"Real imagery: {stats['real_imagery_count']}")
print(f"Synthetic imagery: {stats['synthetic_imagery_count']}")
print(f"Real data date range: {stats['real_date_range']['earliest']} to {stats['real_date_range']['latest']}")
```

### Example 4: Temporal Analysis with Real Data

```python
# Get time series of real data only
time_series = db.get_temporal_series(
    tile_id='43REQ',
    start_date='2024-01-01',
    end_date='2024-12-31',
    synthetic=False  # Real data only
)
print(f"Found {len(time_series)} real imagery dates for temporal analysis")
```

## Backward Compatibility

All changes maintain backward compatibility:

1. **Default Behavior**: `get_latest_imagery()` defaults to `prefer_real=True`, which is the desired behavior
2. **Optional Parameters**: New `synthetic` parameter in queries defaults to `None` (all data)
3. **Existing Code**: Code that doesn't specify `synthetic` parameter continues to work
4. **Database Migration**: `synthetic` column defaults to 1 (synthetic) for backward compatibility

## Impact on Other Components

### Dashboard
The dashboard can now:
- Display real vs synthetic data counts
- Filter visualizations by data type
- Show data provenance in UI

### AI Models
Training pipelines can:
- Query only real data for training
- Verify data source before training
- Track model provenance (trained on real vs synthetic)

### Alerts
Alert system can:
- Generate alerts based on real data only
- Track alert frequency by data type
- Prioritize alerts from real data

## Files Modified

1. ✅ `src/database/db_manager.py` - Enhanced with real data prioritization
2. ✅ `tests/test_database_real_data_priority.py` - Comprehensive test suite
3. ✅ `verify_task_18_complete.py` - Integration verification script

## Task Checklist

- ✅ Modify `get_latest_imagery()` to prefer real data
- ✅ Update `list_processed_imagery()` to filter by synthetic flag
- ✅ Add query methods to distinguish real vs synthetic
- ✅ Update database statistics to show real data count
- ✅ Add indexes for efficient filtering
- ✅ Write comprehensive tests
- ✅ Verify all requirements met
- ✅ Maintain backward compatibility

## Next Steps

Task 18 is complete. The next task in the pipeline is:

**Task 19**: Create deployment script for real-trained models
- Backup existing synthetic-trained models
- Copy real-trained models to production location
- Update model registry with new metadata
- Verify models load correctly
- Update .env to enable AI predictions

## Conclusion

Task 18 successfully implements all required database query enhancements to prioritize real satellite data over synthetic data. The implementation:

- ✅ Meets all requirements (3.4, 3.5)
- ✅ Passes all 21 unit tests
- ✅ Maintains backward compatibility
- ✅ Provides comprehensive filtering capabilities
- ✅ Includes detailed statistics and reporting
- ✅ Ready for production use

The database layer is now fully equipped to handle the transition from synthetic to real satellite data, ensuring that all downstream components (dashboard, AI models, alerts) can prioritize and utilize real data effectively.
