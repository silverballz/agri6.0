"""
Tests for Sentinel-2A SAFE directory parser.
Uses the sample S2A data available in the workspace.
"""

import pytest
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.sentinel2_parser import (
    Sentinel2SafeParser, 
    parse_sentinel2_safe,
    Sentinel2Metadata,
    BandInfo
)


class TestSentinel2SafeParser:
    """Test suite for Sentinel-2A SAFE parser."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_parser_initialization(self, sample_safe_dir):
        """Test parser initialization with valid SAFE directory."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        assert parser.safe_directory == sample_safe_dir
    
    def test_parser_initialization_invalid_path(self):
        """Test parser initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            Sentinel2SafeParser(Path("nonexistent.SAFE"))
    
    def test_parser_initialization_invalid_name(self, tmp_path):
        """Test parser initialization with invalid directory name."""
        invalid_dir = tmp_path / "invalid_name"
        invalid_dir.mkdir()
        
        with pytest.raises(ValueError):
            Sentinel2SafeParser(invalid_dir)
    
    def test_validate_safe_structure(self, sample_safe_dir):
        """Test SAFE directory structure validation."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        assert parser.validate_safe_structure() is True
    
    def test_parse_metadata(self, sample_safe_dir):
        """Test metadata extraction from sample SAFE directory."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        metadata = parser.parse_metadata()
        
        assert isinstance(metadata, Sentinel2Metadata)
        assert metadata.tile_id == "43REQ"
        assert "S2A_MSIL2A" in metadata.product_id
        assert metadata.processing_level == "Level-2A"
        assert metadata.spacecraft_name in ["Sentinel-2A", "Sentinel-2B"]
        assert 0 <= metadata.cloud_coverage <= 100
    
    def test_find_jp2_files(self, sample_safe_dir):
        """Test JP2 file discovery."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        
        # Test finding specific bands
        target_bands = ['B02', 'B03', 'B04', 'B08']
        band_files = parser.find_jp2_files(target_bands)
        
        assert isinstance(band_files, dict)
        
        # Check that we found some bands
        found_bands = list(band_files.keys())
        assert len(found_bands) > 0
        
        # Verify BandInfo objects
        for band_id, band_info in band_files.items():
            assert isinstance(band_info, BandInfo)
            assert band_info.band_id == band_id
            assert band_info.file_path.exists()
            assert band_info.file_path.suffix == '.jp2'
            assert band_info.resolution in ['R10m', 'R20m', 'R60m']
    
    def test_get_scene_classification_layer(self, sample_safe_dir):
        """Test SCL file discovery."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        scl_path = parser.get_scene_classification_layer()
        
        if scl_path is not None:
            assert scl_path.exists()
            assert scl_path.suffix == '.jp2'
            assert 'SCL' in scl_path.name
    
    def test_parse_sentinel2_safe_convenience_function(self, sample_safe_dir):
        """Test convenience function for parsing SAFE directory."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        target_bands = ['B02', 'B03', 'B04', 'B08']
        metadata, band_files = parse_sentinel2_safe(sample_safe_dir, target_bands)
        
        assert isinstance(metadata, Sentinel2Metadata)
        assert isinstance(band_files, dict)
        assert len(band_files) > 0
    
    def test_band_config_completeness(self):
        """Test that band configuration includes all standard S2 bands."""
        expected_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 
                         'B08', 'B8A', 'B09', 'B11', 'B12']
        
        for band in expected_bands:
            assert band in Sentinel2SafeParser.BAND_CONFIG
            config = Sentinel2SafeParser.BAND_CONFIG[band]
            assert 'resolution' in config
            assert 'wavelength' in config
            assert 'bandwidth' in config


class TestSentinel2ParserWithWorkspaceData:
    """Test Sentinel-2A parser with actual workspace sample data."""
    
    @pytest.fixture
    def sample_safe_dir(self):
        """Path to sample SAFE directory in workspace."""
        return Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    def test_metadata_extraction_workspace_data(self, sample_safe_dir):
        """Test metadata extraction from workspace sample data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        metadata = parser.parse_metadata()
        
        # Verify specific metadata from the workspace sample
        assert metadata.tile_id == "43REQ"
        assert metadata.spacecraft_name == "Sentinel-2A"
        assert metadata.processing_level == "Level-2A"
        assert "20240923T053641" in metadata.product_id
        
        # Check acquisition date
        assert metadata.acquisition_date.year == 2024
        assert metadata.acquisition_date.month == 9
        assert metadata.acquisition_date.day == 23
        
        # Cloud coverage should be a valid percentage
        assert 0 <= metadata.cloud_coverage <= 100
    
    def test_band_file_discovery_workspace_data(self, sample_safe_dir):
        """Test band file discovery with workspace sample data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        
        # Test discovery of specific bands that should exist
        target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        band_files = parser.find_jp2_files(target_bands)
        
        # Should find at least some bands
        assert len(band_files) > 0
        
        # Verify band file structure
        for band_id, band_info in band_files.items():
            assert band_id in target_bands
            assert band_info.file_path.exists()
            assert band_info.file_path.suffix == '.jp2'
            assert band_info.resolution in ['R10m', 'R20m', 'R60m']
            
            # Verify file naming convention
            expected_pattern = f"T43REQ_20240923T053641_{band_id}"
            assert expected_pattern in band_info.file_path.name
    
    def test_scl_file_discovery_workspace_data(self, sample_safe_dir):
        """Test Scene Classification Layer file discovery."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        scl_path = parser.get_scene_classification_layer()
        
        if scl_path is not None:
            assert scl_path.exists()
            assert 'SCL' in scl_path.name
            assert scl_path.suffix == '.jp2'
            assert 'T43REQ_20240923T053641' in scl_path.name
    
    def test_directory_structure_validation_workspace_data(self, sample_safe_dir):
        """Test directory structure validation with workspace data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        
        # Should validate successfully
        assert parser.validate_safe_structure() is True
        
        # Check that required directories exist
        granule_dir = sample_safe_dir / "GRANULE"
        assert granule_dir.exists()
        
        # Should have at least one granule subdirectory
        granule_subdirs = [d for d in granule_dir.iterdir() if d.is_dir()]
        assert len(granule_subdirs) > 0
        
        # Check IMG_DATA directory structure
        for granule_subdir in granule_subdirs:
            img_data_dir = granule_subdir / "IMG_DATA"
            if img_data_dir.exists():
                # Should have resolution subdirectories
                resolution_dirs = [d for d in img_data_dir.iterdir() if d.is_dir()]
                assert len(resolution_dirs) > 0
                
                # Check that resolution directories follow naming convention
                valid_resolutions = {'R10m', 'R20m', 'R60m'}
                found_resolutions = {d.name for d in resolution_dirs}
                assert found_resolutions.issubset(valid_resolutions)
    
    def test_metadata_xml_parsing_workspace_data(self, sample_safe_dir):
        """Test XML metadata parsing with workspace data."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        # Check main metadata file exists
        main_metadata = sample_safe_dir / "MTD_MSIL2A.xml"
        if main_metadata.exists():
            parser = Sentinel2SafeParser(sample_safe_dir)
            metadata = parser.parse_metadata()
            
            # Should extract valid coordinate reference system
            assert metadata.crs is not None
            assert 'EPSG' in str(metadata.crs) or 'UTM' in str(metadata.crs)
            
            # Should have valid geometry
            assert metadata.geometry is not None
            assert hasattr(metadata.geometry, 'bounds')
            
            # Bounds should be reasonable for UTM Zone 43N
            bounds = metadata.geometry.bounds
            assert 200000 < bounds[0] < 800000  # min_x
            assert 200000 < bounds[2] < 800000  # max_x
            assert 2000000 < bounds[1] < 4000000  # min_y (northern hemisphere)
            assert 2000000 < bounds[3] < 4000000  # max_y
    
    def test_band_resolution_consistency_workspace_data(self, sample_safe_dir):
        """Test that band resolutions match expected values."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        
        # Test bands with known resolutions
        expected_resolutions = {
            'B02': 'R10m',  # Blue
            'B03': 'R10m',  # Green
            'B04': 'R10m',  # Red
            'B08': 'R10m',  # NIR
            'B05': 'R20m',  # Red Edge 1
            'B06': 'R20m',  # Red Edge 2
            'B07': 'R20m',  # Red Edge 3
            'B8A': 'R20m',  # Red Edge 4
            'B11': 'R20m',  # SWIR 1
            'B12': 'R20m',  # SWIR 2
            'B01': 'R60m',  # Coastal
            'B09': 'R60m',  # Water vapor
        }
        
        band_files = parser.find_jp2_files(list(expected_resolutions.keys()))
        
        for band_id, band_info in band_files.items():
            expected_res = expected_resolutions[band_id]
            assert band_info.resolution == expected_res, \
                f"Band {band_id} should be {expected_res}, found {band_info.resolution}"
    
    def test_file_naming_convention_workspace_data(self, sample_safe_dir):
        """Test that files follow Sentinel-2A naming convention."""
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        band_files = parser.find_jp2_files(['B02', 'B03', 'B04', 'B08'])
        
        for band_id, band_info in band_files.items():
            filename = band_info.file_path.name
            
            # Should contain tile ID
            assert 'T43REQ' in filename
            
            # Should contain acquisition date/time
            assert '20240923T053641' in filename
            
            # Should contain band identifier
            assert band_id in filename
            
            # Should have correct extension
            assert filename.endswith('.jp2')
            
            # Should contain resolution indicator
            assert any(res in filename for res in ['10m', '20m', '60m'])


class TestSentinel2ParserErrorHandling:
    """Test error handling and edge cases for Sentinel-2A parser."""
    
    def test_missing_metadata_file(self, tmp_path):
        """Test handling of missing metadata file."""
        # Create a fake SAFE directory without metadata
        fake_safe = tmp_path / "S2A_MSIL2A_20240101T000000_N0000_R000_T00AAA_20240101T000000.SAFE"
        fake_safe.mkdir()
        
        parser = Sentinel2SafeParser(fake_safe)
        
        # Should handle missing metadata gracefully
        with pytest.raises((FileNotFoundError, ValueError)):
            parser.parse_metadata()
    
    def test_corrupted_directory_structure(self, tmp_path):
        """Test handling of corrupted directory structure."""
        # Create incomplete SAFE directory
        fake_safe = tmp_path / "S2A_MSIL2A_20240101T000000_N0000_R000_T00AAA_20240101T000000.SAFE"
        fake_safe.mkdir()
        
        # Create some directories but not all required ones
        (fake_safe / "GRANULE").mkdir()
        
        parser = Sentinel2SafeParser(fake_safe)
        
        # Should detect invalid structure
        assert parser.validate_safe_structure() is False
    
    def test_empty_band_search(self):
        """Test behavior when no bands are found."""
        sample_safe_dir = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
        
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        
        # Search for non-existent bands
        band_files = parser.find_jp2_files(['B99', 'B100'])
        
        # Should return empty dictionary
        assert isinstance(band_files, dict)
        assert len(band_files) == 0
    
    def test_invalid_band_names(self):
        """Test handling of invalid band names."""
        sample_safe_dir = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
        
        if not sample_safe_dir.exists():
            pytest.skip("Sample SAFE directory not found")
        
        parser = Sentinel2SafeParser(sample_safe_dir)
        
        # Test with invalid band names
        invalid_bands = ['INVALID', '', None, 123]
        
        # Should handle gracefully without crashing
        for invalid_band in invalid_bands:
            try:
                band_files = parser.find_jp2_files([invalid_band])
                assert isinstance(band_files, dict)
            except (TypeError, ValueError):
                # Some invalid inputs may raise exceptions, which is acceptable
                pass