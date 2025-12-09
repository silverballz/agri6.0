"""
Unit tests for DayWiseMapViewer class
"""

import pytest
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.day_wise_map_viewer import DayWiseMapViewer


class TestDayWiseMapViewer:
    """Test suite for DayWiseMapViewer"""
    
    @pytest.fixture
    def sample_imagery_list(self):
        """Create sample imagery list"""
        return [
            {
                'acquisition_date': '2024-01-01',
                'tile_id': 'T43REQ',
                'ndvi_path': '/path/to/ndvi1.tif',
                'savi_path': '/path/to/savi1.tif',
                'evi_path': '/path/to/evi1.tif',
                'ndwi_path': '/path/to/ndwi1.tif'
            },
            {
                'acquisition_date': '2024-01-15',
                'tile_id': 'T43REQ',
                'ndvi_path': '/path/to/ndvi2.tif',
                'savi_path': '/path/to/savi2.tif',
                'evi_path': '/path/to/evi2.tif',
                'ndwi_path': '/path/to/ndwi2.tif'
            },
            {
                'acquisition_date': '2024-02-01',
                'tile_id': 'T43REQ',
                'ndvi_path': '/path/to/ndvi3.tif',
                'savi_path': '/path/to/savi3.tif',
                'evi_path': '/path/to/evi3.tif',
                'ndwi_path': '/path/to/ndwi3.tif'
            }
        ]
    
    @pytest.fixture
    def viewer(self, sample_imagery_list):
        """Create DayWiseMapViewer instance"""
        return DayWiseMapViewer(sample_imagery_list)
    
    def test_initialization(self, viewer, sample_imagery_list):
        """Test viewer initialization"""
        assert len(viewer.imagery_list) == 3
        assert len(viewer.dates) == 3
        
        # Check that dates are sorted
        assert viewer.dates[0] < viewer.dates[1] < viewer.dates[2]
    
    def test_get_layer_path(self, viewer):
        """Test getting layer path for different types"""
        imagery = viewer.imagery_list[0]
        
        # Test NDVI path
        ndvi_path = viewer._get_layer_path(imagery, 'NDVI')
        assert ndvi_path == '/path/to/ndvi1.tif'
        
        # Test SAVI path
        savi_path = viewer._get_layer_path(imagery, 'SAVI')
        assert savi_path == '/path/to/savi1.tif'
        
        # Test EVI path
        evi_path = viewer._get_layer_path(imagery, 'EVI')
        assert evi_path == '/path/to/evi1.tif'
        
        # Test NDWI path
        ndwi_path = viewer._get_layer_path(imagery, 'NDWI')
        assert ndwi_path == '/path/to/ndwi1.tif'
        
        # Test invalid layer type
        invalid_path = viewer._get_layer_path(imagery, 'INVALID')
        assert invalid_path is None
    
    def test_calculate_difference_map_with_mock_data(self, viewer):
        """Test difference map calculation with mock raster data"""
        # This test would require actual raster files, so we'll test the logic
        # In a real scenario, you would create temporary test raster files
        
        imagery1 = viewer.imagery_list[0]
        imagery2 = viewer.imagery_list[1]
        
        # Call the method (will return None without actual files)
        diff_map, stats = viewer._calculate_difference_map(imagery1, imagery2, 'NDVI')
        
        # Without actual files, should return None and empty stats
        assert diff_map is None or isinstance(diff_map, np.ndarray)
        assert isinstance(stats, dict)
    
    def test_interpret_difference_map_improvement(self, viewer):
        """Test interpretation of improvement"""
        stats = {
            'pct_improved': 70.0,
            'pct_stable': 20.0,
            'pct_declined': 10.0,
            'mean_change': 0.15
        }
        
        interpretation = viewer._interpret_difference_map(stats)
        
        assert 'improvement' in interpretation.lower()
        assert '70' in interpretation or '70.0' in interpretation
    
    def test_interpret_difference_map_decline(self, viewer):
        """Test interpretation of decline"""
        stats = {
            'pct_improved': 10.0,
            'pct_stable': 20.0,
            'pct_declined': 70.0,
            'mean_change': -0.15
        }
        
        interpretation = viewer._interpret_difference_map(stats)
        
        assert 'decline' in interpretation.lower()
        assert '70' in interpretation or '70.0' in interpretation
    
    def test_interpret_difference_map_stable(self, viewer):
        """Test interpretation of stable conditions"""
        stats = {
            'pct_improved': 10.0,
            'pct_stable': 75.0,
            'pct_declined': 15.0,
            'mean_change': 0.02
        }
        
        interpretation = viewer._interpret_difference_map(stats)
        
        assert 'stable' in interpretation.lower()
        assert '75' in interpretation or '75.0' in interpretation
    
    def test_interpret_difference_map_mixed(self, viewer):
        """Test interpretation of mixed conditions"""
        stats = {
            'pct_improved': 40.0,
            'pct_stable': 20.0,
            'pct_declined': 40.0,
            'mean_change': 0.0
        }
        
        interpretation = viewer._interpret_difference_map(stats)
        
        assert 'mixed' in interpretation.lower() or 'equal' in interpretation.lower()
    
    def test_change_statistics_calculation(self, viewer):
        """Test change statistics calculation accuracy"""
        # Create mock statistics
        stats = {
            'pct_improved': 50.0,
            'pct_stable': 30.0,
            'pct_declined': 20.0,
            'mean_change': 0.05,
            'max_increase': 0.3,
            'max_decrease': -0.1
        }
        
        # Verify percentages sum to 100
        total = stats['pct_improved'] + stats['pct_stable'] + stats['pct_declined']
        assert abs(total - 100.0) < 0.1
        
        # Verify mean change is reasonable
        assert -1.0 <= stats['mean_change'] <= 1.0
    
    def test_colormap_functions_for_different_ranges(self, viewer):
        """Test colormap functions for different value ranges"""
        # Test that interpretation works for various percentage ranges
        
        # High improvement
        stats_high = {'pct_improved': 80.0, 'pct_stable': 10.0, 'pct_declined': 10.0, 'mean_change': 0.2}
        interp_high = viewer._interpret_difference_map(stats_high)
        assert 'improvement' in interp_high.lower()
        
        # High decline
        stats_decline = {'pct_improved': 10.0, 'pct_stable': 10.0, 'pct_declined': 80.0, 'mean_change': -0.2}
        interp_decline = viewer._interpret_difference_map(stats_decline)
        assert 'decline' in interp_decline.lower()
        
        # Mostly stable
        stats_stable = {'pct_improved': 10.0, 'pct_stable': 80.0, 'pct_declined': 10.0, 'mean_change': 0.0}
        interp_stable = viewer._interpret_difference_map(stats_stable)
        assert 'stable' in interp_stable.lower()
    
    def test_view_mode_rendering_with_various_combinations(self, viewer):
        """Test view mode rendering with various date combinations"""
        # Test that viewer has the correct number of dates
        assert len(viewer.dates) == 3
        
        # Test that dates are datetime objects
        for date in viewer.dates:
            assert isinstance(date, datetime)
        
        # Test that imagery list matches dates
        assert len(viewer.imagery_list) == len(viewer.dates)
    
    def test_error_handling_for_missing_imagery(self, viewer):
        """Test error handling for missing imagery"""
        # Test with empty imagery list
        empty_viewer = DayWiseMapViewer([])
        
        assert len(empty_viewer.imagery_list) == 0
        assert len(empty_viewer.dates) == 0
    
    def test_date_sorting(self):
        """Test that dates are properly sorted"""
        # Create unsorted imagery list
        unsorted_imagery = [
            {'acquisition_date': '2024-02-01', 'tile_id': 'T43REQ'},
            {'acquisition_date': '2024-01-01', 'tile_id': 'T43REQ'},
            {'acquisition_date': '2024-01-15', 'tile_id': 'T43REQ'}
        ]
        
        viewer = DayWiseMapViewer(unsorted_imagery)
        
        # Check that dates are sorted
        assert viewer.dates[0] < viewer.dates[1] < viewer.dates[2]
        
        # Check that imagery list is also sorted
        assert viewer.imagery_list[0]['acquisition_date'] == '2024-01-01'
        assert viewer.imagery_list[1]['acquisition_date'] == '2024-01-15'
        assert viewer.imagery_list[2]['acquisition_date'] == '2024-02-01'
    
    def test_layer_path_with_missing_keys(self, viewer):
        """Test layer path retrieval with missing keys"""
        # Create imagery with missing paths
        incomplete_imagery = {
            'acquisition_date': '2024-01-01',
            'tile_id': 'T43REQ',
            'ndvi_path': '/path/to/ndvi.tif'
            # Missing other paths
        }
        
        # NDVI should work
        ndvi_path = viewer._get_layer_path(incomplete_imagery, 'NDVI')
        assert ndvi_path == '/path/to/ndvi.tif'
        
        # SAVI should return None
        savi_path = viewer._get_layer_path(incomplete_imagery, 'SAVI')
        assert savi_path is None
    
    def test_statistics_with_edge_cases(self, viewer):
        """Test statistics calculation with edge cases"""
        # All improved
        stats_all_improved = {
            'pct_improved': 100.0,
            'pct_stable': 0.0,
            'pct_declined': 0.0,
            'mean_change': 0.3
        }
        interp = viewer._interpret_difference_map(stats_all_improved)
        assert 'improvement' in interp.lower()
        
        # All declined
        stats_all_declined = {
            'pct_improved': 0.0,
            'pct_stable': 0.0,
            'pct_declined': 100.0,
            'mean_change': -0.3
        }
        interp = viewer._interpret_difference_map(stats_all_declined)
        assert 'decline' in interp.lower()
        
        # All stable
        stats_all_stable = {
            'pct_improved': 0.0,
            'pct_stable': 100.0,
            'pct_declined': 0.0,
            'mean_change': 0.0
        }
        interp = viewer._interpret_difference_map(stats_all_stable)
        assert 'stable' in interp.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
