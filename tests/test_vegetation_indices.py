"""
Tests for vegetation index calculation functionality.
Tests both synthetic data and real Sentinel-2A data from workspace.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os
import rasterio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.vegetation_indices import (
    VegetationIndexCalculator,
    IndexResult,
    calculate_vegetation_indices
)
from data_processing.band_processor import BandData, read_and_process_bands
from data_processing.sentinel2_parser import parse_sentinel2_safe


class TestIndexResult:
    """Test the IndexResult dataclass."""
    
    def test_index_result_creation(self):
        """Test IndexResult object creation."""
        test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        result = IndexResult(
            index_name="NDVI",
            data=test_data,
            valid_range=(-1.0, 1.0),
            description="Test NDVI",
            formula="(NIR - Red) / (NIR + Red)"
        )
        
        assert result.index_name == "NDVI"
        assert result.valid_range == (-1.0, 1.0)
        np.testing.assert_array_equal(result.data, test_data)
    
    def test_get_statistics(self):
        """Test statistics calculation for IndexResult."""
        # Test with valid data
        test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = IndexResult(
            index_name="NDVI",
            data=test_data,
            valid_range=(-1.0, 1.0),
            description="Test",
            formula="Test"
        )
        
        stats = result.get_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'valid_pixels' in stats
        assert 'total_pixels' in stats
        
        assert stats['valid_pixels'] == 4
        assert stats['total_pixels'] == 4
        assert abs(stats['mean'] - 0.25) < 1e-6
        assert abs(stats['min'] - 0.1) < 1e-6
        assert abs(stats['max'] - 0.4) < 1e-6
    
    def test_get_statistics_with_nan(self):
        """Test statistics with NaN values."""
        test_data = np.array([[0.1, np.nan], [0.3, 0.4]])
        result = IndexResult(
            index_name="NDVI",
            data=test_data,
            valid_range=(-1.0, 1.0),
            description="Test",
            formula="Test"
        )
        
        stats = result.get_statistics()
        assert stats['valid_pixels'] == 3
        assert stats['total_pixels'] == 4
        assert abs(stats['mean'] - (0.1 + 0.3 + 0.4) / 3) < 1e-6


class TestVegetationIndexCalculator:
    """Test suite for vegetation index calculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return VegetationIndexCalculator()
    
    @pytest.fixture
    def sample_bands(self):
        """Create sample band data for testing."""
        # Create synthetic reflectance data
        shape = (100, 100)
        
        # Simulate typical vegetation reflectance values
        red_data = np.random.uniform(0.05, 0.15, shape).astype(np.float32)  # Low red reflectance
        nir_data = np.random.uniform(0.3, 0.8, shape).astype(np.float32)    # High NIR reflectance
        green_data = np.random.uniform(0.08, 0.18, shape).astype(np.float32) # Moderate green
        blue_data = np.random.uniform(0.04, 0.12, shape).astype(np.float32)  # Low blue
        swir1_data = np.random.uniform(0.1, 0.3, shape).astype(np.float32)   # SWIR1
        swir2_data = np.random.uniform(0.05, 0.2, shape).astype(np.float32)  # SWIR2
        
        # Create BandData objects
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        bands = {
            'B02': BandData('B02', blue_data, transform, 'EPSG:32643', None, 10.0, shape, blue_data.dtype),
            'B03': BandData('B03', green_data, transform, 'EPSG:32643', None, 10.0, shape, green_data.dtype),
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
            'B11': BandData('B11', swir1_data, transform, 'EPSG:32643', None, 10.0, shape, swir1_data.dtype),
            'B12': BandData('B12', swir2_data, transform, 'EPSG:32643', None, 10.0, shape, swir2_data.dtype),
        }
        
        return bands
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = VegetationIndexCalculator()
        assert calc.nodata_threshold == 0.0001
        
        calc_custom = VegetationIndexCalculator(nodata_threshold=0.001)
        assert calc_custom.nodata_threshold == 0.001
    
    def test_index_definitions(self):
        """Test that all expected indices are defined."""
        expected_indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'GNDVI', 'NDSI']
        
        for index in expected_indices:
            assert index in VegetationIndexCalculator.INDEX_DEFINITIONS
            definition = VegetationIndexCalculator.INDEX_DEFINITIONS[index]
            assert 'name' in definition
            assert 'formula' in definition
            assert 'bands' in definition
            assert 'range' in definition
            assert 'description' in definition
    
    def test_validate_bands_success(self, calculator, sample_bands):
        """Test successful band validation."""
        required_bands = ['B04', 'B08']
        assert calculator._validate_bands(sample_bands, required_bands) is True
    
    def test_validate_bands_missing(self, calculator, sample_bands):
        """Test band validation with missing bands."""
        required_bands = ['B04', 'B99']  # B99 doesn't exist
        assert calculator._validate_bands(sample_bands, required_bands) is False
    
    def test_apply_nodata_mask(self, calculator):
        """Test nodata mask creation."""
        array1 = np.array([[1.0, 2.0], [np.nan, 4.0]])
        array2 = np.array([[1.0, np.inf], [3.0, 4.0]])
        
        mask = calculator._apply_nodata_mask(array1, array2)
        
        # Only [0,0] and [1,1] should be valid
        expected_mask = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected_mask)
    
    def test_calculate_ndvi_success(self, calculator, sample_bands):
        """Test successful NDVI calculation."""
        result = calculator.calculate_ndvi(sample_bands)
        
        assert isinstance(result, IndexResult)
        assert result.index_name == 'NDVI'
        assert result.data.shape == sample_bands['B04'].shape
        assert result.valid_range == (-1.0, 1.0)
        
        # Check that NDVI values are reasonable for vegetation
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0
        assert np.all(valid_data >= -1.0)
        assert np.all(valid_data <= 1.0)
        # For vegetation, NDVI should be positive
        assert np.mean(valid_data) > 0
    
    def test_calculate_ndvi_known_values(self, calculator):
        """Test NDVI calculation with known input/output values."""
        # Create simple test data with known values
        shape = (2, 2)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # NIR = 0.8, Red = 0.1 -> NDVI = (0.8-0.1)/(0.8+0.1) = 0.7/0.9 ≈ 0.778
        nir_data = np.full(shape, 0.8, dtype=np.float32)
        red_data = np.full(shape, 0.1, dtype=np.float32)
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
        }
        
        result = calculator.calculate_ndvi(bands)
        expected_ndvi = (0.8 - 0.1) / (0.8 + 0.1)
        
        assert result is not None
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) == 4  # All pixels should be valid
        np.testing.assert_allclose(valid_data, expected_ndvi, rtol=1e-6)
    
    def test_calculate_savi_success(self, calculator, sample_bands):
        """Test successful SAVI calculation."""
        result = calculator.calculate_savi(sample_bands)
        
        assert isinstance(result, IndexResult)
        assert result.index_name == 'SAVI'
        assert result.data.shape == sample_bands['B04'].shape
        assert result.valid_range == (-1.5, 1.5)
        
        # Check that SAVI values are reasonable
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0
        assert np.all(valid_data >= -1.5)
        assert np.all(valid_data <= 1.5)
    
    def test_calculate_savi_custom_L(self, calculator, sample_bands):
        """Test SAVI calculation with custom L parameter."""
        result_default = calculator.calculate_savi(sample_bands)
        result_custom = calculator.calculate_savi(sample_bands, L=0.25)
        
        # Results should be different with different L values
        assert not np.array_equal(result_default.data, result_custom.data)
        assert "0.25" in result_custom.formula
    
    def test_calculate_evi_success(self, calculator, sample_bands):
        """Test successful EVI calculation."""
        result = calculator.calculate_evi(sample_bands)
        
        assert isinstance(result, IndexResult)
        assert result.index_name == 'EVI'
        assert result.data.shape == sample_bands['B04'].shape
        assert result.valid_range == (-1.0, 1.0)
        
        # Check that EVI values are reasonable
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0
        # EVI can have a wider range but should be mostly within [-1, 1]
        assert np.percentile(valid_data, 5) >= -2.0  # Allow some outliers
        assert np.percentile(valid_data, 95) <= 2.0
    
    def test_calculate_ndwi_success(self, calculator, sample_bands):
        """Test successful NDWI calculation."""
        result = calculator.calculate_ndwi(sample_bands)
        
        assert isinstance(result, IndexResult)
        assert result.index_name == 'NDWI'
        assert result.data.shape == sample_bands['B03'].shape
        assert result.valid_range == (-1.0, 1.0)
        
        # Check that NDWI values are in valid range
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0
        assert np.all(valid_data >= -1.0)
        assert np.all(valid_data <= 1.0)
    
    def test_calculate_gndvi_success(self, calculator, sample_bands):
        """Test successful GNDVI calculation."""
        result = calculator.calculate_gndvi(sample_bands)
        
        assert isinstance(result, IndexResult)
        assert result.index_name == 'GNDVI'
        assert result.data.shape == sample_bands['B03'].shape
        assert result.valid_range == (-1.0, 1.0)
        
        # Check that GNDVI values are in valid range
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0
        assert np.all(valid_data >= -1.0)
        assert np.all(valid_data <= 1.0)
    
    def test_calculate_ndsi_success(self, calculator, sample_bands):
        """Test successful NDSI calculation."""
        result = calculator.calculate_ndsi(sample_bands)
        
        assert isinstance(result, IndexResult)
        assert result.index_name == 'NDSI'
        assert result.data.shape == sample_bands['B11'].shape
        assert result.valid_range == (-1.0, 1.0)
        
        # Check that NDSI values are in valid range
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0
        assert np.all(valid_data >= -1.0)
        assert np.all(valid_data <= 1.0)
    
    def test_calculate_index_missing_bands(self, calculator):
        """Test index calculation with missing required bands."""
        # Create bands without required bands for NDVI
        shape = (10, 10)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        bands = {
            'B02': BandData('B02', np.ones(shape), transform, 'EPSG:32643', None, 10.0, shape, np.float32),
        }
        
        result = calculator.calculate_ndvi(bands)
        assert result is None
    
    def test_calculate_all_indices(self, calculator, sample_bands):
        """Test calculation of all available indices."""
        results = calculator.calculate_all_indices(sample_bands)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should calculate all indices that have required bands
        expected_indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'GNDVI', 'NDSI']
        for index_name in expected_indices:
            assert index_name in results
            assert isinstance(results[index_name], IndexResult)
    
    def test_validate_index_values_valid(self, calculator):
        """Test validation of valid index values."""
        # Create valid NDVI data
        test_data = np.array([[0.1, 0.2], [0.3, 0.8]])
        result = IndexResult(
            index_name="NDVI",
            data=test_data,
            valid_range=(-1.0, 1.0),
            description="Test",
            formula="Test"
        )
        
        validation = calculator.validate_index_values(result)
        
        assert validation['within_expected_range'] is True
        assert validation['has_valid_data'] is True
        assert validation['sufficient_coverage'] is True
        assert validation['coverage_ratio'] == 1.0
    
    def test_validate_index_values_invalid_range(self, calculator):
        """Test validation of invalid index values."""
        # Create invalid NDVI data (outside [-1, 1] range)
        test_data = np.array([[1.5, 2.0], [0.3, 0.8]])
        result = IndexResult(
            index_name="NDVI",
            data=test_data,
            valid_range=(-1.0, 1.0),
            description="Test",
            formula="Test"
        )
        
        validation = calculator.validate_index_values(result)
        
        assert validation['within_expected_range'] is False
        assert validation['has_valid_data'] is True
    
    def test_validate_index_values_no_data(self, calculator):
        """Test validation with no valid data."""
        # Create data with all NaN values
        test_data = np.full((2, 2), np.nan)
        result = IndexResult(
            index_name="NDVI",
            data=test_data,
            valid_range=(-1.0, 1.0),
            description="Test",
            formula="Test"
        )
        
        validation = calculator.validate_index_values(result)
        
        assert validation['within_expected_range'] is False
        assert validation['has_valid_data'] is False
        assert validation['sufficient_coverage'] is False
        assert validation['coverage_ratio'] == 0.0


class TestConvenienceFunction:
    """Test the convenience function for calculating indices."""
    
    @pytest.fixture
    def sample_bands(self):
        """Create sample band data for testing."""
        shape = (50, 50)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # Create realistic vegetation reflectance values
        bands = {
            'B02': BandData('B02', np.random.uniform(0.04, 0.12, shape).astype(np.float32), 
                          transform, 'EPSG:32643', None, 10.0, shape, np.float32),
            'B03': BandData('B03', np.random.uniform(0.08, 0.18, shape).astype(np.float32), 
                          transform, 'EPSG:32643', None, 10.0, shape, np.float32),
            'B04': BandData('B04', np.random.uniform(0.05, 0.15, shape).astype(np.float32), 
                          transform, 'EPSG:32643', None, 10.0, shape, np.float32),
            'B08': BandData('B08', np.random.uniform(0.3, 0.8, shape).astype(np.float32), 
                          transform, 'EPSG:32643', None, 10.0, shape, np.float32),
        }
        return bands
    
    def test_calculate_all_indices_convenience(self, sample_bands):
        """Test convenience function for all indices."""
        results = calculate_vegetation_indices(sample_bands)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should calculate indices that have required bands available
        expected_indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'GNDVI']
        for index_name in expected_indices:
            if index_name in results:  # Some might not be calculated if bands missing
                assert isinstance(results[index_name], IndexResult)
    
    def test_calculate_specific_indices_convenience(self, sample_bands):
        """Test convenience function for specific indices."""
        target_indices = ['NDVI', 'SAVI']
        results = calculate_vegetation_indices(sample_bands, indices=target_indices)
        
        assert isinstance(results, dict)
        assert len(results) <= len(target_indices)
        
        for index_name in results.keys():
            assert index_name in target_indices
            assert isinstance(results[index_name], IndexResult)
    
    def test_calculate_unknown_index_convenience(self, sample_bands):
        """Test convenience function with unknown index."""
        target_indices = ['NDVI', 'UNKNOWN_INDEX']
        results = calculate_vegetation_indices(sample_bands, indices=target_indices)
        
        # Should calculate NDVI but skip unknown index
        assert 'NDVI' in results
        assert 'UNKNOWN_INDEX' not in results


class TestRealSentinel2Data:
    """Integration tests with real Sentinel-2A data."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_vegetation_indices_with_real_data(self, sample_safe_dir):
        """Test vegetation index calculation with real Sentinel-2A data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        # Parse and process bands
        metadata, band_files = parse_sentinel2_safe(
            sample_safe_dir, 
            target_bands=['B02', 'B03', 'B04', 'B08']
        )
        
        processed_bands = read_and_process_bands(band_files)
        
        if len(processed_bands) == 0:
            pytest.skip("No bands were successfully processed")
        
        # Calculate vegetation indices
        results = calculate_vegetation_indices(processed_bands)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Validate each calculated index
        for index_name, result in results.items():
            assert isinstance(result, IndexResult)
            assert result.data.size > 0
            
            # Check that we have some valid data
            valid_data = result.data[np.isfinite(result.data)]
            assert len(valid_data) > 0, f"No valid data for {index_name}"
            
            # Check that values are within reasonable bounds (allowing some outliers)
            min_val, max_val = result.valid_range
            assert np.percentile(valid_data, 1) >= min_val - 0.5, f"{index_name} values too low"
            assert np.percentile(valid_data, 99) <= max_val + 0.5, f"{index_name} values too high"
    
    def test_ndvi_calculation_real_data(self, sample_safe_dir):
        """Test NDVI calculation specifically with real data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        # Parse and process bands
        metadata, band_files = parse_sentinel2_safe(
            sample_safe_dir, 
            target_bands=['B04', 'B08']  # Red and NIR for NDVI
        )
        
        processed_bands = read_and_process_bands(band_files)
        
        if 'B04' not in processed_bands or 'B08' not in processed_bands:
            pytest.skip("Required bands (B04, B08) not available")
        
        # Calculate NDVI
        calculator = VegetationIndexCalculator()
        ndvi_result = calculator.calculate_ndvi(processed_bands)
        
        assert ndvi_result is not None
        assert ndvi_result.index_name == 'NDVI'
        
        # Validate NDVI statistics
        stats = ndvi_result.get_statistics()
        assert stats['valid_pixels'] > 0
        
        # For real vegetation data, NDVI should typically be positive
        # and within reasonable bounds
        assert -1.0 <= stats['min'] <= 1.0
        assert -1.0 <= stats['max'] <= 1.0
        
        # Most vegetation should have positive NDVI
        if stats['valid_pixels'] > 100:  # Only check if we have enough data
            assert stats['mean'] > -0.5  # Allow for some water/bare soil


class TestVegetationIndexEdgeCases:
    """Test edge cases and error conditions for vegetation indices."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return VegetationIndexCalculator()
    
    def test_division_by_zero_handling(self, calculator):
        """Test handling of division by zero in index calculations."""
        shape = (10, 10)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # Create bands where NIR + Red = 0 (should cause division by zero in NDVI)
        nir_data = np.zeros(shape, dtype=np.float32)
        red_data = np.zeros(shape, dtype=np.float32)
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
        }
        
        result = calculator.calculate_ndvi(bands)
        
        # Should handle division by zero gracefully
        assert result is not None
        assert np.all(np.isnan(result.data) | np.isinf(result.data))
    
    def test_extreme_values_handling(self, calculator):
        """Test handling of extreme reflectance values."""
        shape = (5, 5)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # Create bands with extreme values
        nir_data = np.full(shape, 10000.0, dtype=np.float32)  # Extremely high
        red_data = np.full(shape, -1000.0, dtype=np.float32)  # Negative (invalid)
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
        }
        
        result = calculator.calculate_ndvi(bands)
        
        # Should produce result but validation should flag issues
        assert result is not None
        validation = calculator.validate_index_values(result)
        assert not validation['within_expected_range']
    
    def test_mixed_data_types(self, calculator):
        """Test handling of different data types."""
        shape = (5, 5)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # Mix int16 and float32 data types
        nir_data = np.random.randint(3000, 8000, shape).astype(np.int16)
        red_data = np.random.uniform(0.05, 0.15, shape).astype(np.float32)
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
        }
        
        result = calculator.calculate_ndvi(bands)
        
        # Should handle mixed types and produce valid result
        assert result is not None
        assert result.data.dtype == np.float32  # Should be converted to float
        
        # Values should be reasonable after type conversion
        valid_data = result.data[np.isfinite(result.data)]
        if len(valid_data) > 0:
            assert np.all(valid_data >= -1.0)
            assert np.all(valid_data <= 1.0)
    
    def test_nodata_value_handling(self, calculator):
        """Test proper handling of nodata values."""
        shape = (10, 10)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # Create data with nodata values
        nir_data = np.random.uniform(0.3, 0.8, shape).astype(np.float32)
        red_data = np.random.uniform(0.05, 0.15, shape).astype(np.float32)
        
        # Set some pixels to nodata
        nir_data[0:2, 0:2] = -9999.0  # Nodata value
        red_data[8:10, 8:10] = -9999.0  # Different nodata locations
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype, nodata=-9999.0),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype, nodata=-9999.0),
        }
        
        result = calculator.calculate_ndvi(bands)
        
        assert result is not None
        
        # Pixels with nodata in either band should be NaN in result
        assert np.isnan(result.data[0, 0])  # NIR nodata
        assert np.isnan(result.data[1, 1])  # NIR nodata
        assert np.isnan(result.data[8, 8])  # Red nodata
        assert np.isnan(result.data[9, 9])  # Red nodata
        
        # Other pixels should have valid values
        valid_mask = np.isfinite(result.data)
        assert np.sum(valid_mask) > 0
    
    def test_single_pixel_arrays(self, calculator):
        """Test calculation with single pixel arrays."""
        shape = (1, 1)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        nir_data = np.array([[0.8]], dtype=np.float32)
        red_data = np.array([[0.1]], dtype=np.float32)
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
        }
        
        result = calculator.calculate_ndvi(bands)
        
        assert result is not None
        assert result.data.shape == (1, 1)
        
        expected_ndvi = (0.8 - 0.1) / (0.8 + 0.1)
        np.testing.assert_allclose(result.data[0, 0], expected_ndvi, rtol=1e-6)
    
    def test_large_array_memory_efficiency(self, calculator):
        """Test memory efficiency with large arrays."""
        # Create a moderately large array to test memory handling
        shape = (1000, 1000)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        # Use memory-mapped arrays to simulate large data
        nir_data = np.random.uniform(0.3, 0.8, shape).astype(np.float32)
        red_data = np.random.uniform(0.05, 0.15, shape).astype(np.float32)
        
        bands = {
            'B04': BandData('B04', red_data, transform, 'EPSG:32643', None, 10.0, shape, red_data.dtype),
            'B08': BandData('B08', nir_data, transform, 'EPSG:32643', None, 10.0, shape, nir_data.dtype),
        }
        
        # Monitor memory usage during calculation
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        result = calculator.calculate_ndvi(bands)
        
        memory_after = process.memory_info().rss
        memory_increase_mb = (memory_after - memory_before) / (1024 * 1024)
        
        assert result is not None
        assert result.data.shape == shape
        
        # Memory increase should be reasonable (less than 3x the input data size)
        input_size_mb = (nir_data.nbytes + red_data.nbytes) / (1024 * 1024)
        assert memory_increase_mb < input_size_mb * 3, f"Memory usage too high: {memory_increase_mb:.1f} MB"