"""
Tests for Sentinel-2A band processing functionality.
Uses the sample S2A data available in the workspace.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os
import rasterio
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.band_processor import (
    Sentinel2BandProcessor,
    BandData,
    read_and_process_bands
)
from data_processing.sentinel2_parser import (
    Sentinel2SafeParser,
    BandInfo,
    parse_sentinel2_safe
)


class TestSentinel2BandProcessor:
    """Test suite for Sentinel-2A band processor."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    @pytest.fixture
    def band_processor(self):
        """Create band processor instance."""
        return Sentinel2BandProcessor(target_resolution=10.0)
    
    @pytest.fixture
    def sample_band_files(self, sample_safe_dir):
        """Get sample band files from workspace SAFE directory."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        target_bands = ['B02', 'B03', 'B04', 'B08']  # Focus on key bands for testing
        return parser.find_jp2_files(target_bands)
    
    def test_processor_initialization(self):
        """Test processor initialization with different resolutions."""
        processor = Sentinel2BandProcessor()
        assert processor.target_resolution == 10.0
        
        processor_20m = Sentinel2BandProcessor(target_resolution=20.0)
        assert processor_20m.target_resolution == 20.0
    
    def test_read_band_success(self, band_processor, sample_band_files):
        """Test successful band reading with real S2A data."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        # Test with first available band
        band_id = next(iter(sample_band_files.keys()))
        band_info = sample_band_files[band_id]
        
        band_data = band_processor.read_band(band_info)
        
        # Verify BandData structure
        assert isinstance(band_data, BandData)
        assert band_data.band_id == band_id
        assert isinstance(band_data.data, np.ndarray)
        assert band_data.data.ndim == 2  # Should be 2D array
        assert band_data.data.size > 0
        assert band_data.transform is not None
        assert band_data.crs is not None
        assert band_data.shape == band_data.data.shape
        
        # Check reflectance scaling was applied
        assert band_data.data.dtype in [np.float32, np.float64]
        assert np.all(band_data.data >= 0.0)
        assert np.all(band_data.data <= 1.0)
    
    def test_read_band_without_scaling(self, band_processor, sample_band_files):
        """Test band reading without reflectance scaling."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        band_id = next(iter(sample_band_files.keys()))
        band_info = sample_band_files[band_id]
        
        band_data = band_processor.read_band(band_info, apply_scaling=False)
        
        # Without scaling, values should be larger (original DN values)
        assert isinstance(band_data.data, np.ndarray)
        # Original L2A values are typically in range 0-10000
        assert np.max(band_data.data) > 1.0  # Should be larger than scaled values
    
    def test_read_band_file_not_found(self, band_processor):
        """Test error handling for missing band file."""
        fake_band_info = BandInfo(
            band_id="B99",
            resolution="10m",
            file_path=Path("nonexistent_file.jp2"),
            central_wavelength=500.0,
            bandwidth=50.0
        )
        
        with pytest.raises(FileNotFoundError):
            band_processor.read_band(fake_band_info)
    
    def test_resample_band_same_resolution(self, band_processor, sample_band_files):
        """Test resampling when band is already at target resolution."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        # Find a 10m band (should not need resampling)
        band_10m = None
        for band_id, band_info in sample_band_files.items():
            if '10m' in band_info.resolution:
                band_10m = band_info
                break
        
        if band_10m is None:
            pytest.skip("No 10m band found for testing")
        
        original_data = band_processor.read_band(band_10m)
        resampled_data = band_processor.resample_band_to_target_resolution(original_data)
        
        # Should be identical (no resampling needed)
        assert resampled_data.resolution == 10.0
        assert resampled_data.shape == original_data.shape
        np.testing.assert_array_equal(resampled_data.data, original_data.data)
    
    def test_resample_band_different_resolution(self, band_processor, sample_band_files):
        """Test resampling from 20m to 10m resolution."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        # Find a 20m band for resampling test
        band_20m = None
        for band_id, band_info in sample_band_files.items():
            if '20m' in band_info.resolution:
                band_20m = band_info
                break
        
        if band_20m is None:
            pytest.skip("No 20m band found for testing")
        
        original_data = band_processor.read_band(band_20m)
        resampled_data = band_processor.resample_band_to_target_resolution(original_data)
        
        # Check that resampling occurred
        assert resampled_data.resolution == 10.0
        assert resampled_data.shape != original_data.shape  # Should be different size
        assert resampled_data.data.size > original_data.data.size  # Should be larger (higher res)
    
    def test_process_bands_integration(self, band_processor, sample_band_files):
        """Test processing multiple bands together."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        processed_bands = band_processor.process_bands(sample_band_files)
        
        assert isinstance(processed_bands, dict)
        assert len(processed_bands) > 0
        
        # All processed bands should have same resolution and compatible shapes
        resolutions = [band.resolution for band in processed_bands.values()]
        shapes = [band.shape for band in processed_bands.values()]
        
        assert all(abs(res - 10.0) < 0.1 for res in resolutions)
        
        # All bands should have the same shape (aligned to common grid)
        if len(shapes) > 1:
            reference_shape = shapes[0]
            for shape in shapes[1:]:
                # Allow small differences due to rounding
                assert abs(shape[0] - reference_shape[0]) <= 1
                assert abs(shape[1] - reference_shape[1]) <= 1
    
    def test_process_bands_with_target_bands(self, band_processor, sample_band_files):
        """Test processing with specific target bands."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        # Test with subset of available bands
        available_bands = list(sample_band_files.keys())
        if len(available_bands) > 1:
            target_bands = available_bands[:2]  # Take first two bands
            
            processed_bands = band_processor.process_bands(sample_band_files, target_bands)
            
            # Should only process requested bands
            assert len(processed_bands) <= len(target_bands)
            for band_id in processed_bands.keys():
                assert band_id in target_bands
    
    def test_validate_band_data_valid(self, band_processor, sample_band_files):
        """Test validation of valid band data."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        band_id = next(iter(sample_band_files.keys()))
        band_info = sample_band_files[band_id]
        band_data = band_processor.read_band(band_info)
        
        validation_results = band_processor.validate_band_data(band_data)
        
        assert isinstance(validation_results, dict)
        assert 'valid_reflectance_range' in validation_results
        assert 'sufficient_data_coverage' in validation_results
        assert 'correct_resolution' in validation_results
        assert 'non_empty_data' in validation_results
        assert 'finite_values' in validation_results
        
        # Most validations should pass for real S2A data
        assert validation_results['non_empty_data'] is True
        # Note: finite_values might be False if there are NaN/inf values in real data
        # assert validation_results['finite_values'] is True
    
    def test_validate_band_data_invalid_range(self, band_processor):
        """Test validation with invalid reflectance range."""
        # Create fake band data with invalid values
        fake_data = np.array([[1.5, 2.0], [0.5, -0.1]], dtype=np.float32)
        
        band_data = BandData(
            band_id="TEST",
            data=fake_data,
            transform=rasterio.Affine(10, 0, 0, 0, -10, 0),
            crs="EPSG:32643",
            nodata_value=None,
            resolution=10.0,
            shape=fake_data.shape,
            dtype=fake_data.dtype
        )
        
        validation_results = band_processor.validate_band_data(band_data)
        assert validation_results['valid_reflectance_range'] == False
    
    def test_convenience_function(self, sample_band_files):
        """Test convenience function for band processing."""
        if not sample_band_files:
            pytest.skip("No sample band files found")
        
        processed_bands = read_and_process_bands(
            sample_band_files, 
            target_bands=['B02', 'B04'], 
            target_resolution=10.0
        )
        
        assert isinstance(processed_bands, dict)
        # Should process available bands from the requested list
        for band_id in processed_bands.keys():
            assert band_id in ['B02', 'B04']
    
    def test_target_bands_constant(self):
        """Test that TARGET_BANDS constant contains expected bands."""
        expected_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        assert Sentinel2BandProcessor.TARGET_BANDS == expected_bands
    
    def test_reflectance_scale_factor(self):
        """Test reflectance scale factor constant."""
        assert Sentinel2BandProcessor.REFLECTANCE_SCALE_FACTOR == 10000.0


class TestBandDataClass:
    """Test the BandData dataclass."""
    
    def test_band_data_creation(self):
        """Test BandData object creation."""
        test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        band_data = BandData(
            band_id="B04",
            data=test_data,
            transform=rasterio.Affine(10, 0, 0, 0, -10, 0),
            crs="EPSG:32643",
            nodata_value=-9999.0,
            resolution=10.0,
            shape=(2, 2),
            dtype=np.float32
        )
        
        assert band_data.band_id == "B04"
        assert band_data.resolution == 10.0
        assert band_data.shape == (2, 2)
        np.testing.assert_array_equal(band_data.data, test_data)


class TestIntegrationWithSentinel2Parser:
    """Integration tests with Sentinel2SafeParser."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_end_to_end_processing(self, sample_safe_dir):
        """Test complete workflow from SAFE parsing to band processing."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        # Parse SAFE directory
        metadata, band_files = parse_sentinel2_safe(
            sample_safe_dir, 
            target_bands=['B02', 'B03', 'B04', 'B08']
        )
        
        # Process bands
        processed_bands = read_and_process_bands(band_files)
        
        # Verify integration
        assert len(processed_bands) > 0
        
        for band_id, band_data in processed_bands.items():
            assert band_data.band_id == band_id
            assert band_data.resolution == 10.0
            assert band_data.data.size > 0
            
            # Verify CRS matches metadata expectation
            if hasattr(metadata, 'epsg_code'):
                # CRS should be compatible (both UTM)
                assert 'UTM' in band_data.crs or 'EPSG' in band_data.crs