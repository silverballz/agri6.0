"""
Test database queries for real data prioritization.

Tests Requirements 3.4 and 3.5:
- Database queries distinguish between real and synthetic data
- Latest imagery retrieval prioritizes real data over synthetic data
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DatabaseManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseManager(db_path)
    db.init_database()
    
    yield db
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def populated_db(temp_db):
    """Create a database populated with both real and synthetic data."""
    # Add synthetic data (older dates)
    for i in range(5):
        date = (datetime.now() - timedelta(days=20 + i)).strftime('%Y-%m-%d')
        temp_db.save_processed_imagery(
            acquisition_date=date,
            tile_id='43REQ',
            cloud_coverage=10.0 + i,
            geotiff_paths={
                'NDVI': f'/path/to/synthetic_ndvi_{i}.tif',
                'SAVI': f'/path/to/synthetic_savi_{i}.tif',
            },
            metadata={'source': 'synthetic'},
            synthetic=True
        )
    
    # Add real data (more recent dates)
    for i in range(3):
        date = (datetime.now() - timedelta(days=5 + i)).strftime('%Y-%m-%d')
        temp_db.save_processed_imagery(
            acquisition_date=date,
            tile_id='43REQ',
            cloud_coverage=5.0 + i,
            geotiff_paths={
                'NDVI': f'/path/to/real_ndvi_{i}.tif',
                'SAVI': f'/path/to/real_savi_{i}.tif',
            },
            metadata={'source': 'Sentinel Hub API'},
            synthetic=False
        )
    
    return temp_db


class TestGetLatestImagery:
    """Test get_latest_imagery prioritizes real data."""
    
    def test_prefer_real_returns_real_when_available(self, populated_db):
        """
        Requirement 3.5: When querying imagery, system SHALL prioritize real data.
        """
        result = populated_db.get_latest_imagery(prefer_real=True)
        
        assert result is not None
        assert result['synthetic'] == 0, "Should return real data when prefer_real=True"
        assert 'Sentinel Hub' in result['metadata_json'] or 'real' in result['ndvi_path']
    
    def test_prefer_real_falls_back_to_synthetic(self, temp_db):
        """
        When no real data exists, should fall back to synthetic data.
        """
        # Add only synthetic data
        temp_db.save_processed_imagery(
            acquisition_date='2024-01-01',
            tile_id='43REQ',
            cloud_coverage=10.0,
            geotiff_paths={'NDVI': '/path/to/synthetic.tif'},
            metadata={'source': 'synthetic'},
            synthetic=True
        )
        
        result = temp_db.get_latest_imagery(prefer_real=True)
        
        assert result is not None
        assert result['synthetic'] == 1, "Should fall back to synthetic when no real data"
    
    def test_prefer_real_false_returns_any(self, populated_db):
        """
        When prefer_real=False, should return most recent regardless of type.
        """
        result = populated_db.get_latest_imagery(prefer_real=False)
        
        assert result is not None
        # Should return the most recent date (which is real data in our test)
    
    def test_prefer_real_with_tile_filter(self, populated_db):
        """
        Should prioritize real data even with tile_id filter.
        """
        result = populated_db.get_latest_imagery(tile_id='43REQ', prefer_real=True)
        
        assert result is not None
        assert result['synthetic'] == 0
        assert result['tile_id'] == '43REQ'


class TestListProcessedImagery:
    """Test list_processed_imagery filtering by synthetic flag."""
    
    def test_filter_real_only(self, populated_db):
        """
        Requirement 3.4: System SHALL distinguish between real and synthetic data.
        """
        results = populated_db.list_processed_imagery(synthetic=False)
        
        assert len(results) == 3, "Should return only real imagery"
        for record in results:
            assert record['synthetic'] == 0, "All records should be real data"
    
    def test_filter_synthetic_only(self, populated_db):
        """
        Should be able to filter for synthetic data only.
        """
        results = populated_db.list_processed_imagery(synthetic=True)
        
        assert len(results) == 5, "Should return only synthetic imagery"
        for record in results:
            assert record['synthetic'] == 1, "All records should be synthetic data"
    
    def test_no_filter_returns_all(self, populated_db):
        """
        When synthetic=None, should return all data.
        """
        results = populated_db.list_processed_imagery(synthetic=None)
        
        assert len(results) == 8, "Should return all imagery (3 real + 5 synthetic)"
    
    def test_filter_with_tile_id(self, populated_db):
        """
        Should support combining tile_id and synthetic filters.
        """
        results = populated_db.list_processed_imagery(tile_id='43REQ', synthetic=False)
        
        assert len(results) == 3
        for record in results:
            assert record['tile_id'] == '43REQ'
            assert record['synthetic'] == 0


class TestDistinguishRealVsSynthetic:
    """Test methods to distinguish real vs synthetic data."""
    
    def test_get_real_imagery(self, populated_db):
        """
        Requirement 3.4: System SHALL distinguish between real and synthetic data.
        """
        results = populated_db.get_real_imagery()
        
        assert len(results) == 3
        for record in results:
            assert record['synthetic'] == 0
    
    def test_get_synthetic_imagery(self, populated_db):
        """
        Should retrieve only synthetic imagery.
        """
        results = populated_db.get_synthetic_imagery()
        
        assert len(results) == 5
        for record in results:
            assert record['synthetic'] == 1
    
    def test_count_real_imagery(self, populated_db):
        """
        Should accurately count real imagery records.
        """
        count = populated_db.count_real_imagery()
        assert count == 3
    
    def test_count_synthetic_imagery(self, populated_db):
        """
        Should accurately count synthetic imagery records.
        """
        count = populated_db.count_synthetic_imagery()
        assert count == 5
    
    def test_count_with_tile_filter(self, populated_db):
        """
        Should support tile_id filtering in count methods.
        """
        real_count = populated_db.count_real_imagery(tile_id='43REQ')
        synthetic_count = populated_db.count_synthetic_imagery(tile_id='43REQ')
        
        assert real_count == 3
        assert synthetic_count == 5


class TestDatabaseStatistics:
    """Test database statistics show real data count."""
    
    def test_stats_include_real_count(self, populated_db):
        """
        Requirement 3.5: Database statistics SHALL show real data count.
        """
        stats = populated_db.get_database_stats()
        
        assert 'real_imagery_count' in stats
        assert 'synthetic_imagery_count' in stats
        assert stats['real_imagery_count'] == 3
        assert stats['synthetic_imagery_count'] == 5
        assert stats['imagery_count'] == 8
    
    def test_stats_include_real_date_range(self, populated_db):
        """
        Should include date range for real data specifically.
        """
        stats = populated_db.get_database_stats()
        
        assert 'real_date_range' in stats
        assert stats['real_date_range']['earliest'] is not None
        assert stats['real_date_range']['latest'] is not None
    
    def test_stats_with_no_real_data(self, temp_db):
        """
        Should handle case where no real data exists.
        """
        # Add only synthetic data
        temp_db.save_processed_imagery(
            acquisition_date='2024-01-01',
            tile_id='43REQ',
            cloud_coverage=10.0,
            geotiff_paths={'NDVI': '/path/to/synthetic.tif'},
            metadata={'source': 'synthetic'},
            synthetic=True
        )
        
        stats = temp_db.get_database_stats()
        
        assert stats['real_imagery_count'] == 0
        assert stats['synthetic_imagery_count'] == 1


class TestTemporalSeries:
    """Test temporal series filtering by synthetic flag."""
    
    def test_temporal_series_real_only(self, populated_db):
        """
        Should support filtering temporal series by real data only.
        """
        results = populated_db.get_temporal_series(
            tile_id='43REQ',
            synthetic=False
        )
        
        assert len(results) == 3
        for record in results:
            assert record['synthetic'] == 0
        
        # Verify ordering by date
        dates = [record['acquisition_date'] for record in results]
        assert dates == sorted(dates), "Should be ordered by date ascending"
    
    def test_temporal_series_synthetic_only(self, populated_db):
        """
        Should support filtering temporal series by synthetic data only.
        """
        results = populated_db.get_temporal_series(
            tile_id='43REQ',
            synthetic=True
        )
        
        assert len(results) == 5
        for record in results:
            assert record['synthetic'] == 1
    
    def test_temporal_series_all_data(self, populated_db):
        """
        Should return all data when synthetic=None.
        """
        results = populated_db.get_temporal_series(
            tile_id='43REQ',
            synthetic=None
        )
        
        assert len(results) == 8


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_save_without_synthetic_defaults_to_true(self, temp_db):
        """
        For backward compatibility, omitting synthetic should default to True.
        """
        # Call without synthetic parameter
        imagery_id = temp_db.save_processed_imagery(
            acquisition_date='2024-01-01',
            tile_id='43REQ',
            cloud_coverage=10.0,
            geotiff_paths={'NDVI': '/path/to/test.tif'},
            metadata={'test': 'data'}
        )
        
        record = temp_db.get_processed_imagery(imagery_id)
        assert record['synthetic'] == 1, "Should default to synthetic=True"
    
    def test_get_latest_without_prefer_real_still_works(self, populated_db):
        """
        Should work without prefer_real parameter (defaults to True).
        """
        result = populated_db.get_latest_imagery()
        
        assert result is not None
        # Should prioritize real data by default
        assert result['synthetic'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
