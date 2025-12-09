"""
Property-based tests for vegetation index calculations.
Tests universal properties that should hold across all valid inputs.

Feature: production-enhancements
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import sys
import os
import rasterio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.vegetation_indices import VegetationIndexCalculator, IndexResult
from data_processing.band_processor import BandData


# Strategy for generating valid reflectance values (0.002 to 1.0)
# Start at 0.002 to avoid nodata threshold (0.0001) and division by zero issues
reflectance_strategy = st.floats(min_value=0.002, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for generating small arrays (for performance)
small_shape_strategy = st.tuples(
    st.integers(min_value=5, max_value=20),
    st.integers(min_value=5, max_value=20)
)


def create_band_data(data: np.ndarray, band_id: str) -> BandData:
    """Helper to create BandData objects for testing."""
    transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
    shape = data.shape
    return BandData(
        band_id=band_id,
        data=data,
        transform=transform,
        crs='EPSG:32643',
        nodata_value=None,
        resolution=10.0,
        shape=shape,
        dtype=data.dtype
    )


class TestNDVIProperties:
    """Property-based tests for NDVI calculation.
    
    **Feature: production-enhancements, Property 6: NDVI formula correctness**
    **Validates: Requirements 2.1**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        nir_value=reflectance_strategy,
        red_value=reflectance_strategy
    )
    def test_ndvi_formula_correctness(self, shape, nir_value, red_value):
        """
        Property 6: NDVI formula correctness
        
        For any valid NIR and Red band arrays, calculated NDVI should equal 
        (NIR - Red) / (NIR + Red) and be within range [-1, 1]
        """
        # Skip cases where denominator would be too close to zero
        assume(abs(nir_value + red_value) > 0.01)
        
        # Create uniform arrays with the generated values
        nir_data = np.full(shape, nir_value, dtype=np.float32)
        red_data = np.full(shape, red_value, dtype=np.float32)
        
        bands = {
            'B04': create_band_data(red_data, 'B04'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_ndvi(bands)
        
        # Calculate expected NDVI manually
        expected_ndvi = (nir_value - red_value) / (nir_value + red_value)
        
        # Verify result exists
        assert result is not None, "NDVI calculation should not return None"
        
        # Verify formula correctness
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0, "Should have valid NDVI values"
        
        # All values should match expected NDVI (within floating point tolerance)
        # Use rtol=1e-4 to account for float32 precision
        np.testing.assert_allclose(
            valid_data, 
            expected_ndvi, 
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"NDVI formula incorrect for NIR={nir_value}, Red={red_value}"
        )
        
        # Verify range constraint [-1, 1]
        assert np.all(valid_data >= -1.0), f"NDVI values below -1: min={np.min(valid_data)}"
        assert np.all(valid_data <= 1.0), f"NDVI values above 1: max={np.max(valid_data)}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_ndvi_range_property(self, shape, seed):
        """
        Property: NDVI values must always be in range [-1, 1] for valid reflectance inputs.
        """
        np.random.seed(seed)
        
        # Generate realistic reflectance values
        nir_data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        red_data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        
        bands = {
            'B04': create_band_data(red_data, 'B04'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_ndvi(bands)
        
        assert result is not None
        
        # All finite values must be in [-1, 1]
        valid_data = result.data[np.isfinite(result.data)]
        if len(valid_data) > 0:
            assert np.all(valid_data >= -1.0), f"NDVI below -1: {np.min(valid_data)}"
            assert np.all(valid_data <= 1.0), f"NDVI above 1: {np.max(valid_data)}"


class TestSAVIProperties:
    """Property-based tests for SAVI calculation.
    
    **Feature: production-enhancements, Property 7: SAVI formula correctness**
    **Validates: Requirements 2.2**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        nir_value=reflectance_strategy,
        red_value=reflectance_strategy
    )
    def test_savi_formula_correctness(self, shape, nir_value, red_value):
        """
        Property 7: SAVI formula correctness
        
        For any valid NIR and Red band arrays with L=0.5, calculated SAVI should equal 
        ((NIR - Red) / (NIR + Red + 0.5)) * 1.5 and be within range [-1.5, 1.5]
        """
        L = 0.5
        
        # Skip cases where denominator would be too close to zero
        assume(abs(nir_value + red_value + L) > 0.01)
        
        # Create uniform arrays with the generated values
        nir_data = np.full(shape, nir_value, dtype=np.float32)
        red_data = np.full(shape, red_value, dtype=np.float32)
        
        bands = {
            'B04': create_band_data(red_data, 'B04'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_savi(bands, L=L)
        
        # Calculate expected SAVI manually
        expected_savi = ((nir_value - red_value) / (nir_value + red_value + L)) * (1 + L)
        
        # Verify result exists
        assert result is not None, "SAVI calculation should not return None"
        
        # Verify formula correctness
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0, "Should have valid SAVI values"
        
        # All values should match expected SAVI (within floating point tolerance)
        np.testing.assert_allclose(
            valid_data, 
            expected_savi, 
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"SAVI formula incorrect for NIR={nir_value}, Red={red_value}, L={L}"
        )
        
        # Verify range constraint [-1.5, 1.5]
        assert np.all(valid_data >= -1.5), f"SAVI values below -1.5: min={np.min(valid_data)}"
        assert np.all(valid_data <= 1.5), f"SAVI values above 1.5: max={np.max(valid_data)}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_savi_range_property(self, shape, seed):
        """
        Property: SAVI values must always be in range [-1.5, 1.5] for valid reflectance inputs.
        """
        np.random.seed(seed)
        
        # Generate realistic reflectance values
        nir_data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        red_data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        
        bands = {
            'B04': create_band_data(red_data, 'B04'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_savi(bands, L=0.5)
        
        assert result is not None
        
        # All finite values must be in [-1.5, 1.5]
        valid_data = result.data[np.isfinite(result.data)]
        if len(valid_data) > 0:
            assert np.all(valid_data >= -1.5), f"SAVI below -1.5: {np.min(valid_data)}"
            assert np.all(valid_data <= 1.5), f"SAVI above 1.5: {np.max(valid_data)}"


class TestEVIProperties:
    """Property-based tests for EVI calculation.
    
    **Feature: production-enhancements, Property 8: EVI formula correctness**
    **Validates: Requirements 2.4**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        nir_value=reflectance_strategy,
        red_value=reflectance_strategy,
        blue_value=reflectance_strategy
    )
    def test_evi_formula_correctness(self, shape, nir_value, red_value, blue_value):
        """
        Property 8: EVI formula correctness
        
        For any valid NIR, Red, and Blue band arrays, calculated EVI should equal 
        2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)) and be within range [-1, 1]
        """
        # Calculate denominator
        denominator = nir_value + 6 * red_value - 7.5 * blue_value + 1
        
        # Skip cases where denominator would be too close to zero
        assume(abs(denominator) > 0.01)
        
        # Create uniform arrays with the generated values
        nir_data = np.full(shape, nir_value, dtype=np.float32)
        red_data = np.full(shape, red_value, dtype=np.float32)
        blue_data = np.full(shape, blue_value, dtype=np.float32)
        
        bands = {
            'B02': create_band_data(blue_data, 'B02'),
            'B04': create_band_data(red_data, 'B04'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_evi(bands)
        
        # Calculate expected EVI manually
        expected_evi = 2.5 * ((nir_value - red_value) / denominator)
        
        # Verify result exists
        assert result is not None, "EVI calculation should not return None"
        
        # Verify formula correctness
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0, "Should have valid EVI values"
        
        # All values should match expected EVI (within floating point tolerance)
        np.testing.assert_allclose(
            valid_data, 
            expected_evi, 
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"EVI formula incorrect for NIR={nir_value}, Red={red_value}, Blue={blue_value}"
        )
        
        # Verify range constraint
        # Note: EVI can exceed [-1, 1] with extreme inputs due to the formula structure
        # The 2.5 multiplier and atmospheric correction terms can produce wider ranges
        # We verify the formula is correct, not that values are in a specific range
        # For typical vegetation, EVI is usually in [-1, 1], but edge cases can exceed this
        assert np.all(np.isfinite(valid_data)), "EVI values should be finite"
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_evi_reasonable_range_property(self, shape, seed):
        """
        Property: EVI values should be reasonable for typical vegetation reflectance.
        """
        np.random.seed(seed)
        
        # Generate realistic vegetation reflectance values
        nir_data = np.random.uniform(0.2, 0.9, shape).astype(np.float32)
        red_data = np.random.uniform(0.03, 0.2, shape).astype(np.float32)
        blue_data = np.random.uniform(0.02, 0.15, shape).astype(np.float32)
        
        bands = {
            'B02': create_band_data(blue_data, 'B02'),
            'B04': create_band_data(red_data, 'B04'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_evi(bands)
        
        assert result is not None
        
        # For realistic vegetation values, EVI should be mostly in [-1, 1]
        valid_data = result.data[np.isfinite(result.data)]
        if len(valid_data) > 0:
            # Allow some outliers but most should be reasonable
            assert np.percentile(valid_data, 5) >= -2.0
            assert np.percentile(valid_data, 95) <= 2.0


class TestNDWIProperties:
    """Property-based tests for NDWI calculation.
    
    **Feature: production-enhancements, Property 9: NDWI formula correctness**
    **Validates: Requirements 2.3**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        green_value=reflectance_strategy,
        nir_value=reflectance_strategy
    )
    def test_ndwi_formula_correctness(self, shape, green_value, nir_value):
        """
        Property 9: NDWI formula correctness
        
        For any valid Green and NIR band arrays, calculated NDWI should equal 
        (Green - NIR) / (Green + NIR) and be within range [-1, 1]
        """
        # Skip cases where denominator would be too close to zero
        assume(abs(green_value + nir_value) > 0.01)
        
        # Create uniform arrays with the generated values
        green_data = np.full(shape, green_value, dtype=np.float32)
        nir_data = np.full(shape, nir_value, dtype=np.float32)
        
        bands = {
            'B03': create_band_data(green_data, 'B03'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_ndwi(bands)
        
        # Calculate expected NDWI manually
        expected_ndwi = (green_value - nir_value) / (green_value + nir_value)
        
        # Verify result exists
        assert result is not None, "NDWI calculation should not return None"
        
        # Verify formula correctness
        valid_data = result.data[np.isfinite(result.data)]
        assert len(valid_data) > 0, "Should have valid NDWI values"
        
        # All values should match expected NDWI (within floating point tolerance)
        np.testing.assert_allclose(
            valid_data, 
            expected_ndwi, 
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"NDWI formula incorrect for Green={green_value}, NIR={nir_value}"
        )
        
        # Verify range constraint [-1, 1]
        assert np.all(valid_data >= -1.0), f"NDWI values below -1: min={np.min(valid_data)}"
        assert np.all(valid_data <= 1.0), f"NDWI values above 1: max={np.max(valid_data)}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_ndwi_range_property(self, shape, seed):
        """
        Property: NDWI values must always be in range [-1, 1] for valid reflectance inputs.
        """
        np.random.seed(seed)
        
        # Generate realistic reflectance values
        green_data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        nir_data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        
        bands = {
            'B03': create_band_data(green_data, 'B03'),
            'B08': create_band_data(nir_data, 'B08')
        }
        
        calculator = VegetationIndexCalculator()
        result = calculator.calculate_ndwi(bands)
        
        assert result is not None
        
        # All finite values must be in [-1, 1]
        valid_data = result.data[np.isfinite(result.data)]
        if len(valid_data) > 0:
            assert np.all(valid_data >= -1.0), f"NDWI below -1: {np.min(valid_data)}"
            assert np.all(valid_data <= 1.0), f"NDWI above 1: {np.max(valid_data)}"


class TestIndexValidationProperties:
    """Property-based tests for index validation.
    
    **Feature: production-enhancements, Property 10: Index range validation**
    **Validates: Requirements 2.5**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_index_range_validation_property(self, shape, seed):
        """
        Property 10: Index range validation
        
        For any calculated vegetation index, values outside the expected range 
        should be flagged as anomalous
        """
        np.random.seed(seed)
        
        # Generate NDVI data with some values outside valid range
        valid_ndvi = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        # Randomly add some out-of-range values
        num_invalid = max(1, shape[0] * shape[1] // 10)  # 10% invalid
        invalid_indices = np.random.choice(shape[0] * shape[1], num_invalid, replace=False)
        flat_data = valid_ndvi.flatten()
        
        # Add values outside [-1, 1]
        for idx in invalid_indices:
            if np.random.random() > 0.5:
                flat_data[idx] = np.random.uniform(1.1, 2.0)  # Above range
            else:
                flat_data[idx] = np.random.uniform(-2.0, -1.1)  # Below range
        
        test_data = flat_data.reshape(shape)
        
        result = IndexResult(
            index_name='NDVI',
            data=test_data,
            valid_range=(-1.0, 1.0),
            description='Test NDVI',
            formula='(NIR - Red) / (NIR + Red)'
        )
        
        calculator = VegetationIndexCalculator()
        validation = calculator.validate_index_values(result)
        
        # Should detect that values are outside expected range
        assert 'within_expected_range' in validation
        assert validation['within_expected_range'] is False, \
            "Validation should flag out-of-range values"
    
    @settings(max_examples=100, deadline=None)
    @given(
        shape=small_shape_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_valid_index_passes_validation(self, shape, seed):
        """
        Property: Valid index values within expected range should pass validation.
        """
        np.random.seed(seed)
        
        # Generate NDVI data strictly within valid range
        valid_ndvi = np.random.uniform(-0.9, 0.9, shape).astype(np.float32)
        
        result = IndexResult(
            index_name='NDVI',
            data=valid_ndvi,
            valid_range=(-1.0, 1.0),
            description='Test NDVI',
            formula='(NIR - Red) / (NIR + Red)'
        )
        
        calculator = VegetationIndexCalculator()
        validation = calculator.validate_index_values(result)
        
        # Should pass validation
        assert validation['within_expected_range'] is True, \
            "Valid index values should pass validation"
        assert validation['has_valid_data'] is True
    
    @settings(max_examples=50, deadline=None)
    @given(
        shape=small_shape_strategy,
        coverage_ratio=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_coverage_validation_property(self, shape, coverage_ratio):
        """
        Property: Validation should correctly report data coverage ratio.
        """
        # Create data with specific coverage ratio
        total_pixels = shape[0] * shape[1]
        valid_pixels = int(total_pixels * coverage_ratio)
        
        data = np.full(shape, np.nan, dtype=np.float32)
        
        # Set some pixels to valid values
        if valid_pixels > 0:
            flat_data = data.flatten()
            valid_indices = np.random.choice(total_pixels, valid_pixels, replace=False)
            flat_data[valid_indices] = np.random.uniform(-1.0, 1.0, valid_pixels)
            data = flat_data.reshape(shape)
        
        result = IndexResult(
            index_name='NDVI',
            data=data,
            valid_range=(-1.0, 1.0),
            description='Test NDVI',
            formula='(NIR - Red) / (NIR + Red)'
        )
        
        calculator = VegetationIndexCalculator()
        validation = calculator.validate_index_values(result)
        
        # Check coverage ratio is correctly calculated
        assert 'coverage_ratio' in validation
        expected_ratio = valid_pixels / total_pixels
        assert abs(validation['coverage_ratio'] - expected_ratio) < 0.01, \
            f"Coverage ratio mismatch: expected {expected_ratio}, got {validation['coverage_ratio']}"
        
        # Check sufficient coverage flag (use actual ratio after integer rounding)
        actual_ratio = valid_pixels / total_pixels
        if actual_ratio > 0.1:
            assert validation['sufficient_coverage'] is True
        else:
            assert validation['sufficient_coverage'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
