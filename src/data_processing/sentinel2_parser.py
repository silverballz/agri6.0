"""
Sentinel-2A SAFE directory parser for extracting JP2 files and metadata.
Handles the standard SAFE format structure and extracts relevant bands and metadata.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class Sentinel2Metadata:
    """Container for Sentinel-2A metadata extracted from SAFE directory."""
    product_id: str
    acquisition_date: datetime
    tile_id: str
    cloud_coverage: float
    processing_level: str
    spacecraft_name: str
    orbit_number: int
    relative_orbit_number: int
    utm_zone: str
    epsg_code: str
    
    
@dataclass
class BandInfo:
    """Information about a Sentinel-2A spectral band."""
    band_id: str
    resolution: str  # "10m", "20m", "60m"
    file_path: Path
    central_wavelength: float
    bandwidth: float


class Sentinel2SafeParser:
    """Parser for Sentinel-2A SAFE format directories."""
    
    # Standard Sentinel-2A band configurations
    BAND_CONFIG = {
        'B01': {'resolution': '60m', 'wavelength': 443.0, 'bandwidth': 21.0},
        'B02': {'resolution': '10m', 'wavelength': 490.0, 'bandwidth': 66.0},
        'B03': {'resolution': '10m', 'wavelength': 560.0, 'bandwidth': 36.0},
        'B04': {'resolution': '10m', 'wavelength': 665.0, 'bandwidth': 31.0},
        'B05': {'resolution': '20m', 'wavelength': 705.0, 'bandwidth': 15.0},
        'B06': {'resolution': '20m', 'wavelength': 740.0, 'bandwidth': 15.0},
        'B07': {'resolution': '20m', 'wavelength': 783.0, 'bandwidth': 20.0},
        'B08': {'resolution': '10m', 'wavelength': 842.0, 'bandwidth': 106.0},
        'B8A': {'resolution': '20m', 'wavelength': 865.0, 'bandwidth': 21.0},
        'B09': {'resolution': '60m', 'wavelength': 945.0, 'bandwidth': 20.0},
        'B11': {'resolution': '20m', 'wavelength': 1610.0, 'bandwidth': 91.0},
        'B12': {'resolution': '20m', 'wavelength': 2190.0, 'bandwidth': 175.0},
    }
    
    def __init__(self, safe_directory: Path):
        """
        Initialize parser with SAFE directory path.
        
        Args:
            safe_directory: Path to Sentinel-2A SAFE directory
        """
        self.safe_directory = Path(safe_directory)
        if not self.safe_directory.exists():
            raise FileNotFoundError(f"SAFE directory not found: {safe_directory}")
        
        if not self.safe_directory.name.endswith('.SAFE'):
            raise ValueError(f"Invalid SAFE directory name: {safe_directory.name}")
    
    def parse_metadata(self) -> Sentinel2Metadata:
        """
        Extract metadata from MTD_MSIL2A.xml file.
        
        Returns:
            Sentinel2Metadata object with extracted information
        """
        metadata_file = self.safe_directory / "MTD_MSIL2A.xml"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        tree = ET.parse(metadata_file)
        root = tree.getroot()
        
        # Extract product identification
        product_info = root.find('.//Product_Info')
        if product_info is None:
            raise ValueError("Product_Info not found in metadata")
        
        product_id = product_info.find('PRODUCT_URI').text
        processing_level = product_info.find('PROCESSING_LEVEL').text
        
        # Extract acquisition date from product name
        date_match = re.search(r'(\d{8}T\d{6})', product_id)
        if date_match:
            acquisition_date = datetime.strptime(date_match.group(1), '%Y%m%dT%H%M%S')
        else:
            raise ValueError(f"Could not extract acquisition date from {product_id}")
        
        # Extract tile ID from product name
        tile_match = re.search(r'T(\d{2}[A-Z]{3})', product_id)
        if tile_match:
            tile_id = tile_match.group(1)
        else:
            raise ValueError(f"Could not extract tile ID from {product_id}")
        
        # Extract spacecraft name and orbit info
        spacecraft_name = product_info.find('Datatake/SPACECRAFT_NAME').text
        
        # Extract orbit number from SENSING_ORBIT_NUMBER if available
        orbit_element = product_info.find('Datatake/SENSING_ORBIT_NUMBER')
        if orbit_element is not None:
            orbit_number = int(orbit_element.text)
        else:
            # Fallback: extract from datatake identifier or use 0
            orbit_number = 0
        
        # Extract cloud coverage
        quality_info = root.find('.//Quality_Indicators_Info')
        if quality_info is not None:
            cloud_element = quality_info.find('Cloud_Coverage_Assessment')
            if cloud_element is not None:
                cloud_coverage = float(cloud_element.text) * 100  # Convert to percentage
            else:
                cloud_coverage = 0.0
        else:
            cloud_coverage = 0.0
        
        # Extract coordinate reference system info from granule metadata
        # First try to find it in the main metadata
        geocoding = root.find('.//Tile_Geocoding')
        if geocoding is not None:
            epsg_element = geocoding.find('HORIZONTAL_CS_CODE')
            if epsg_element is not None:
                epsg_code = epsg_element.text
            else:
                epsg_code = "EPSG:32643"  # Default for T43REQ
        else:
            # Fallback: try to read from granule metadata
            granule_dir = self.safe_directory / "GRANULE"
            if granule_dir.exists():
                granule_subdirs = [d for d in granule_dir.iterdir() if d.is_dir()]
                if granule_subdirs:
                    granule_metadata = granule_subdirs[0] / "MTD_TL.xml"
                    if granule_metadata.exists():
                        granule_tree = ET.parse(granule_metadata)
                        granule_root = granule_tree.getroot()
                        granule_geocoding = granule_root.find('.//Tile_Geocoding')
                        if granule_geocoding is not None:
                            epsg_element = granule_geocoding.find('HORIZONTAL_CS_CODE')
                            if epsg_element is not None:
                                epsg_code = epsg_element.text
                            else:
                                epsg_code = "EPSG:32643"  # Default
                        else:
                            epsg_code = "EPSG:32643"  # Default
                    else:
                        epsg_code = "EPSG:32643"  # Default
                else:
                    epsg_code = "EPSG:32643"  # Default
            else:
                epsg_code = "EPSG:32643"  # Default
        
        utm_zone = epsg_code.split(':')[-1]  # Extract zone from EPSG code
        
        return Sentinel2Metadata(
            product_id=product_id,
            acquisition_date=acquisition_date,
            tile_id=tile_id,
            cloud_coverage=cloud_coverage,
            processing_level=processing_level,
            spacecraft_name=spacecraft_name,
            orbit_number=orbit_number,
            relative_orbit_number=0,  # Would need additional parsing
            utm_zone=utm_zone,
            epsg_code=epsg_code
        )
    
    def find_jp2_files(self, target_bands: Optional[List[str]] = None) -> Dict[str, BandInfo]:
        """
        Find JP2 band files in the SAFE directory structure.
        
        Args:
            target_bands: List of band IDs to search for (e.g., ['B02', 'B03', 'B04', 'B08'])
                         If None, searches for all available bands
        
        Returns:
            Dictionary mapping band IDs to BandInfo objects
        """
        if target_bands is None:
            target_bands = list(self.BAND_CONFIG.keys())
        
        band_files = {}
        
        # Search in GRANULE directory structure
        granule_dir = self.safe_directory / "GRANULE"
        if not granule_dir.exists():
            raise FileNotFoundError(f"GRANULE directory not found in {self.safe_directory}")
        
        # Find the granule subdirectory (should be only one)
        granule_subdirs = [d for d in granule_dir.iterdir() if d.is_dir()]
        if not granule_subdirs:
            raise FileNotFoundError("No granule subdirectory found")
        
        granule_path = granule_subdirs[0]
        img_data_dir = granule_path / "IMG_DATA"
        
        # Search in resolution subdirectories
        for resolution in ['R10m', 'R20m', 'R60m']:
            res_dir = img_data_dir / resolution
            if not res_dir.exists():
                continue
            
            # Find JP2 files in this resolution directory
            for jp2_file in res_dir.glob("*.jp2"):
                # Extract band ID from filename
                band_match = re.search(r'_(B\d{2}|B8A)_', jp2_file.name)
                if band_match:
                    band_id = band_match.group(1)
                    if band_id in target_bands and band_id in self.BAND_CONFIG:
                        band_config = self.BAND_CONFIG[band_id]
                        band_files[band_id] = BandInfo(
                            band_id=band_id,
                            resolution=resolution,
                            file_path=jp2_file,
                            central_wavelength=band_config['wavelength'],
                            bandwidth=band_config['bandwidth']
                        )
        
        return band_files
    
    def get_scene_classification_layer(self) -> Optional[Path]:
        """
        Find the Scene Classification Layer (SCL) file for cloud masking.
        
        Returns:
            Path to SCL JP2 file, or None if not found
        """
        granule_dir = self.safe_directory / "GRANULE"
        if not granule_dir.exists():
            return None
        
        granule_subdirs = [d for d in granule_dir.iterdir() if d.is_dir()]
        if not granule_subdirs:
            return None
        
        granule_path = granule_subdirs[0]
        
        # SCL is typically in R20m directory
        scl_path = granule_path / "IMG_DATA" / "R20m"
        if scl_path.exists():
            scl_files = list(scl_path.glob("*_SCL_20m.jp2"))
            if scl_files:
                return scl_files[0]
        
        return None
    
    def validate_safe_structure(self) -> bool:
        """
        Validate that the SAFE directory has the expected structure.
        
        Returns:
            True if structure is valid, False otherwise
        """
        required_files = [
            "MTD_MSIL2A.xml",
            "manifest.safe"
        ]
        
        required_dirs = [
            "GRANULE",
            "DATASTRIP"
        ]
        
        # Check required files
        for file_name in required_files:
            if not (self.safe_directory / file_name).exists():
                return False
        
        # Check required directories
        for dir_name in required_dirs:
            if not (self.safe_directory / dir_name).is_dir():
                return False
        
        return True


def parse_sentinel2_safe(safe_directory: Path, target_bands: Optional[List[str]] = None) -> Tuple[Sentinel2Metadata, Dict[str, BandInfo]]:
    """
    Convenience function to parse Sentinel-2A SAFE directory.
    
    Args:
        safe_directory: Path to SAFE directory
        target_bands: List of band IDs to extract
    
    Returns:
        Tuple of (metadata, band_files_dict)
    """
    parser = Sentinel2SafeParser(safe_directory)
    
    if not parser.validate_safe_structure():
        raise ValueError(f"Invalid SAFE directory structure: {safe_directory}")
    
    metadata = parser.parse_metadata()
    band_files = parser.find_jp2_files(target_bands)
    
    return metadata, band_files