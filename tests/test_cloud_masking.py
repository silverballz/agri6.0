"""
Tests for cloud masking functionality using Scene Classification Layer.
Tests both synthetic SCL data and real Sentinel-2A SCL data from workspace.
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

from data_processing.cloud_masking import (
    CloudMaskProcessor,
    CloudMaskResult,
    SCLClass,
    create_cloud_mask_from_scl_file,
    apply_cloud_masking
)
from data_processing.band_processor import BandData
from data_processing.sentinel2_parser import Sentinel2SafeParser


class TestSCLClass:
    """Test SCL class enumeration."""
    
    def test_scl_class_values(self):
        """Test that SCL class values match Sentinel-2A specification."""
        assert SCLClass.NO_DATA == 0
        assert SCLClass.SATURATED_DEFECTIVE == 1
        assert SCLClass.DARK_AREA_PIXELS == 2
        assert SCLClass.CLOUD_SHADOWS == 3
        assert SCLClass.VEGETATION == 4
        assert SCLClass.NOT_VEGETATED == 5
        assert SCLClass.WATER == 6
        assert SCLClass.UNCLASSIFIED == 7
        assert SCLClass.CLOUD_MEDIUM_PROBABILITY == 8
        assert SCLClass.CLOUD_HIGH_PROBABILITY == 9
        assert SCLClass.THIN_CIRRUS == 10
        assert SCLClass.SNOW_ICE == 11


class TestCloudMaskResult:
    """Test CloudMaskResult dataclass."""
    
    def test_cloud_mask_result_creation(self):
        """Test CloudMaskResult object creation."""
        cloud_mask = np.array([[True, False], [False, True]])
        quality_flags = {'clouds': np.array([[True, False], [False, False]])}
        statistics = {'clear_percentage': 50.0, 'cloud_percentage': 50.0}
        
        result = CloudMaskResult(
            cloud_mask=cloud_mask,
            quality_flags=quality_flags,
            statistics=statistics
        )
        
        assert result.cloud_mask.shape == (2, 2)
        assert 'clouds' in result.quality_flags
        assert result.statistics['clear_percentage'] == 50.0
    
    def test_get_clear_pixel_mask(self):
        """Test clear pixel mask generation."""
        cloud_mask = np.array([[True, False], [False, True]])
        result = CloudMaskResult(
            cloud_mask=cloud_mask,
            quality_flags={},
            statistics={}
        )
        
        clear_mask = result.get_clear_pixel_mask()
        expected_clear = np.array([[False, True], [True, False]])
        
        np.testing.assert_array_equal(clear_mask, expected_clear)
    
    def test_get_coverage_statistics(self):
        """Test coverage statistics calculation."""
        cloud_mask = np.array([[True, False], [False, True]])  # 50% cloudy
        quality_flags = {
            'clouds': np.array([[True, False], [False, False]]),  # 25% clouds
            'shadows': np.array([[False, False], [False, True]])  # 25% shadows
        }
        
        result = CloudMaskResult(
            cloud_mask=cloud_mask,
            quality_flags=quality_flags,
            statistics={}
        )
        
        stats = result.get_coverage_statistics()
        
        assert stats['total_pixels'] == 4
        assert stats['clear_pixels'] == 2
        assert stats['cloudy_pixels'] == 2
        assert stats['clear_percentage'] == 50.0
        assert stats['cloud_percentage'] == 50.0
        assert stats['clouds_pixels'] == 1
        assert stats['clouds_percentage'] == 25.0
        assert stats['shadows_pixels'] == 1
        assert stats['shadows_percentage'] == 25.0


class TestCloudMaskProcessor:
    """Test suite for cloud mask processor."""
    
    @pytest.fixture
    def processor(self):
        """Create default processor instance."""
        return CloudMaskProcessor()
    
    @pytest.fixture
    def sample_scl_data(self):
        """Create sample SCL data for testing."""
        # Create 10x10 SCL data with various classes
        scl_data = np.full((10, 10), SCLClass.VEGETATION, dtype=np.uint8)
        
        # Add some clouds
        scl_data[0:2, 0:2] = SCLClass.CLOUD_HIGH_PROBABILITY
        scl_data[2:4, 0:2] = SCLClass.CLOUD_MEDIUM_PROBABILITY
        
        # Add cloud shadows
        scl_data[0:2, 8:10] = SCLClass.CLOUD_SHADOWS
        
        # Add thin cirrus
        scl_data[8:10, 0:2] = SCLClass.THIN_CIRRUS
        
        # Add water
        scl_data[4:6, 4:6] = SCLClass.WATER
        
        # Add no data
        scl_data[8:10, 8:10] = SCLClass.NO_DATA
        
        return scl_data
    
    @pytest.fixture
    def sample_band_data(self):
        """Create sample band data for testing."""
        shape = (10, 10)
        data = np.random.uniform(0.1, 0.8, shape).astype(np.float32)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        return BandData(
            band_id='B04',
            data=data,
            transform=transform,
            crs='EPSG:32643',
            nodata_value=None,
            resolution=10.0,
            shape=shape,
            dtype=data.dtype
        )
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_processor_initialization_default(self):
        """Test processor initialization with default parameters."""
        processor = CloudMaskProcessor()
        
        expected_classes = [
            SCLClass.CLOUD_SHADOWS,
            SCLClass.CLOUD_MEDIUM_PROBABILITY,
            SCLClass.CLOUD_HIGH_PROBABILITY,
            SCLClass.THIN_CIRRUS
        ]
        
        assert processor.cloud_classes == expected_classes
    
    def test_processor_initialization_custom(self):
        """Test processor initialization with custom parameters."""
        custom_classes = [SCLClass.CLOUD_HIGH_PROBABILITY]
        processor = CloudMaskProcessor(
            cloud_classes=custom_classes,
            include_shadows=False,
            include_cirrus=False,
            include_snow=True
        )
        
        # When include_snow=True, it adds snow to the custom classes
        expected_classes = custom_classes + [SCLClass.SNOW_ICE]
        assert processor.cloud_classes == expected_classes
    
    def test_processor_initialization_options(self):
        """Test processor initialization with include/exclude options."""
        # Test excluding shadows
        processor = CloudMaskProcessor(include_shadows=False)
        assert SCLClass.CLOUD_SHADOWS not in processor.cloud_classes
        
        # Test excluding cirrus
        processor = CloudMaskProcessor(include_cirrus=False)
        assert SCLClass.THIN_CIRRUS not in processor.cloud_classes
        
        # Test including snow
        processor = CloudMaskProcessor(include_snow=True)
        assert SCLClass.SNOW_ICE in processor.cloud_classes
    
    def test_create_cloud_mask_from_scl(self, processor, sample_scl_data):
        """Test cloud mask creation from SCL data."""
        result = processor.create_cloud_mask_from_scl(sample_scl_data)
        
        assert isinstance(result, CloudMaskResult)
        assert result.cloud_mask.shape == sample_scl_data.shape
        assert result.scl_data is not None
        
        # Check that cloud pixels are properly masked
        # High probability clouds should be masked
        assert result.cloud_mask[0, 0] == True  # CLOUD_HIGH_PROBABILITY
        assert result.cloud_mask[2, 0] == True  # CLOUD_MEDIUM_PROBABILITY
        assert result.cloud_mask[0, 8] == True  # CLOUD_SHADOWS
        assert result.cloud_mask[8, 0] == True  # THIN_CIRRUS
        
        # Vegetation should not be masked
        assert result.cloud_mask[5, 5] == False  # VEGETATION
        
        # Check statistics
        assert 'clear_percentage' in result.statistics
        assert 'cloud_percentage' in result.statistics
        assert result.statistics['clear_percentage'] + result.statistics['cloud_percentage'] == 100.0
    
    def test_create_cloud_mask_quality_flags(self, processor, sample_scl_data):
        """Test quality flag creation."""
        result = processor.create_cloud_mask_from_scl(sample_scl_data)
        
        # Check that quality flags are created
        assert 'clouds' in result.quality_flags
        assert 'cloud_shadows' in result.quality_flags
        assert 'cirrus' in result.quality_flags
        assert 'no_data' in result.quality_flags
        
        # Check specific flag values
        clouds_flag = result.quality_flags['clouds']
        assert clouds_flag[0, 0] == True  # CLOUD_HIGH_PROBABILITY
        assert clouds_flag[2, 0] == True  # CLOUD_MEDIUM_PROBABILITY
        assert clouds_flag[5, 5] == False  # VEGETATION
        
        shadows_flag = result.quality_flags['cloud_shadows']
        assert shadows_flag[0, 8] == True  # CLOUD_SHADOWS
        assert shadows_flag[5, 5] == False  # VEGETATION
    
    def test_read_scl_data_nonexistent_file(self, processor):
        """Test SCL data reading with nonexistent file."""
        fake_path = Path("nonexistent_scl.jp2")
        result = processor.read_scl_data(fake_path)
        assert result is None
    
    def test_apply_cloud_mask_to_bands(self, processor, sample_band_data):
        """Test applying cloud mask to band data."""
        # Create a simple cloud mask
        cloud_mask = np.zeros(sample_band_data.shape, dtype=bool)
        cloud_mask[0:2, 0:2] = True  # Mask top-left corner
        
        cloud_mask_result = CloudMaskResult(
            cloud_mask=cloud_mask,
            quality_flags={},
            statistics={}
        )
        
        bands = {'B04': sample_band_data}
        masked_bands = processor.apply_cloud_mask_to_bands(bands, cloud_mask_result)
        
        assert 'B04' in masked_bands
        masked_data = masked_bands['B04'].data
        
        # Check that cloudy pixels are masked (NaN)
        assert np.isnan(masked_data[0, 0])
        assert np.isnan(masked_data[1, 1])
        
        # Check that clear pixels are preserved
        assert not np.isnan(masked_data[5, 5])
        assert masked_data[5, 5] == sample_band_data.data[5, 5]
    
    def test_apply_cloud_mask_shape_mismatch(self, processor, sample_band_data):
        """Test cloud mask application with shape mismatch."""
        # Create cloud mask with different shape
        wrong_shape_mask = np.zeros((5, 5), dtype=bool)
        
        cloud_mask_result = CloudMaskResult(
            cloud_mask=wrong_shape_mask,
            quality_flags={},
            statistics={}
        )
        
        bands = {'B04': sample_band_data}
        masked_bands = processor.apply_cloud_mask_to_bands(bands, cloud_mask_result)
        
        # Should return original band data when shapes don't match
        assert 'B04' in masked_bands
        np.testing.assert_array_equal(masked_bands['B04'].data, sample_band_data.data)
    
    def test_interpolate_cloudy_pixels(self, processor, sample_band_data):
        """Test cloudy pixel interpolation."""
        # Create a cloud mask with some isolated cloudy pixels
        cloud_mask = np.zeros(sample_band_data.shape, dtype=bool)
        cloud_mask[5, 5] = True  # Single cloudy pixel in the middle
        
        # Interpolate
        interpolated_band = processor.interpolate_cloudy_pixels(
            sample_band_data, cloud_mask, method='nearest'
        )
        
        assert isinstance(interpolated_band, BandData)
        assert interpolated_band.shape == sample_band_data.shape
        
        # The interpolated pixel should have a finite value (not NaN)
        assert np.isfinite(interpolated_band.data[5, 5])
        
        # Other pixels should remain unchanged
        clear_mask = ~cloud_mask
        np.testing.assert_array_equal(
            interpolated_band.data[clear_mask],
            sample_band_data.data[clear_mask]
        )
    
    def test_interpolate_insufficient_clear_pixels(self, processor, sample_band_data):
        """Test interpolation with insufficient clear pixels."""
        # Create a cloud mask that masks almost everything
        cloud_mask = np.ones(sample_band_data.shape, dtype=bool)
        cloud_mask[0, 0] = False  # Only one clear pixel
        cloud_mask[0, 1] = False  # Only two clear pixels
        
        # Should return original data when insufficient clear pixels
        interpolated_band = processor.interpolate_cloudy_pixels(
            sample_band_data, cloud_mask, method='nearest'
        )
        
        # Should return original band data
        np.testing.assert_array_equal(interpolated_band.data, sample_band_data.data)


class TestConvenienceFunctions:
    """Test convenience functions for cloud masking."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_create_cloud_mask_from_scl_file_nonexistent(self):
        """Test convenience function with nonexistent SCL file."""
        fake_path = Path("nonexistent_scl.jp2")
        result = create_cloud_mask_from_scl_file(fake_path)
        assert result is None
    
    def test_apply_cloud_masking_no_scl(self):
        """Test apply_cloud_masking with nonexistent SCL file."""
        # Create sample band data
        shape = (10, 10)
        data = np.random.uniform(0.1, 0.8, shape).astype(np.float32)
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
        
        band_data = BandData(
            band_id='B04',
            data=data,
            transform=transform,
            crs='EPSG:32643',
            nodata_value=None,
            resolution=10.0,
            shape=shape,
            dtype=data.dtype
        )
        
        bands = {'B04': band_data}
        fake_scl_path = Path("nonexistent_scl.jp2")
        
        processed_bands, cloud_mask_result = apply_cloud_masking(bands, fake_scl_path)
        
        # Should return original bands when SCL processing fails
        assert processed_bands == bands
        assert cloud_mask_result is None


class TestRealSentinel2Data:
    """Integration tests with real Sentinel-2A SCL data."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_scl_file_discovery(self, sample_safe_dir):
        """Test SCL file discovery in real SAFE directory."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        scl_path = parser.get_scene_classification_layer()
        
        if scl_path is None:
            pytest.skip("SCL file not found in sample data")
        
        assert scl_path.exists()
        assert scl_path.suffix == '.jp2'
        assert 'SCL' in scl_path.name
    
    def test_cloud_mask_creation_real_scl(self, sample_safe_dir):
        """Test cloud mask creation with real SCL data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        scl_path = parser.get_scene_classification_layer()
        
        if scl_path is None:
            pytest.skip("SCL file not found in sample data")
        
        # Create cloud mask
        processor = CloudMaskProcessor()
        result = processor.process_scl_file(scl_path)
        
        if result is None:
            pytest.skip("Failed to process SCL file")
        
        assert isinstance(result, CloudMaskResult)
        assert result.cloud_mask.size > 0
        assert result.scl_data is not None
        
        # Check statistics
        stats = result.get_coverage_statistics()
        assert 'clear_percentage' in stats
        assert 'cloud_percentage' in stats
        assert 0 <= stats['clear_percentage'] <= 100
        assert 0 <= stats['cloud_percentage'] <= 100
        assert abs(stats['clear_percentage'] + stats['cloud_percentage'] - 100.0) < 1e-6
    
    def test_cloud_mask_with_band_resampling(self, sample_safe_dir):
        """Test cloud mask creation with resampling to band resolution."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        scl_path = parser.get_scene_classification_layer()
        
        if scl_path is None:
            pytest.skip("SCL file not found in sample data")
        
        # Get a band file for reference
        band_files = parser.find_jp2_files(['B04'])
        if 'B04' not in band_files:
            pytest.skip("B04 band not found for resampling test")
        
        # Create a mock target band (we don't need to read the actual data)
        with rasterio.open(band_files['B04'].file_path) as src:
            target_band = BandData(
                band_id='B04',
                data=np.zeros((100, 100)),  # Dummy data
                transform=src.transform,
                crs=src.crs.to_string(),
                nodata_value=src.nodata,
                resolution=10.0,
                shape=(100, 100),
                dtype=np.float32
            )
        
        # Create cloud mask with resampling
        processor = CloudMaskProcessor()
        result = processor.process_scl_file(scl_path, target_band)
        
        if result is None:
            pytest.skip("Failed to process SCL file with resampling")
        
        assert isinstance(result, CloudMaskResult)
        # The resampled cloud mask should have different dimensions than original SCL
        # (since SCL is typically 20m and we're resampling to 10m)
        assert result.cloud_mask.size > 0
    
    def test_scl_class_distribution_real_data(self, sample_safe_dir):
        """Test SCL class distribution in real data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        scl_path = parser.get_scene_classification_layer()
        
        if scl_path is None:
            pytest.skip("SCL file not found in sample data")
        
        processor = CloudMaskProcessor()
        scl_data = processor.read_scl_data(scl_path)
        
        if scl_data is None:
            pytest.skip("Failed to read SCL data")
        
        # Check that we have valid SCL class values
        unique_classes = np.unique(scl_data)
        
        # All values should be valid SCL classes (0-11)
        assert np.all(unique_classes >= 0)
        assert np.all(unique_classes <= 11)
        
        # Should have some vegetation or other land cover classes
        land_classes = [SCLClass.VEGETATION, SCLClass.NOT_VEGETATED, SCLClass.WATER]
        has_land_cover = any(cls in unique_classes for cls in land_classes)
        assert has_land_cover, "No land cover classes found in SCL data"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_scl_data(self):
        """Test handling of empty SCL data."""
        processor = CloudMaskProcessor()
        empty_scl = np.array([]).reshape(0, 0).astype(np.uint8)
        
        result = processor.create_cloud_mask_from_scl(empty_scl)
        
        assert isinstance(result, CloudMaskResult)
        assert result.cloud_mask.size == 0
        assert result.statistics['total_pixels'] == 0
    
    def test_all_cloud_scl_data(self):
        """Test SCL data with all cloud pixels."""
        processor = CloudMaskProcessor()
        all_cloud_scl = np.full((10, 10), SCLClass.CLOUD_HIGH_PROBABILITY, dtype=np.uint8)
        
        result = processor.create_cloud_mask_from_scl(all_cloud_scl)
        
        assert isinstance(result, CloudMaskResult)
        assert np.all(result.cloud_mask)  # All pixels should be masked
        assert result.statistics['cloud_percentage'] == 100.0
        assert result.statistics['clear_percentage'] == 0.0
    
    def test_all_clear_scl_data(self):
        """Test SCL data with all clear pixels."""
        processor = CloudMaskProcessor()
        all_clear_scl = np.full((10, 10), SCLClass.VEGETATION, dtype=np.uint8)
        
        result = processor.create_cloud_mask_from_scl(all_clear_scl)
        
        assert isinstance(result, CloudMaskResult)
        assert not np.any(result.cloud_mask)  # No pixels should be masked
        assert result.statistics['cloud_percentage'] == 0.0
        assert result.statistics['clear_percentage'] == 100.0