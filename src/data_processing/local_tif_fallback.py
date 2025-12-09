"""
Fallback mechanism for loading local TIF files when API is unavailable.
Provides seamless switching between API and local data sources.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

import numpy as np

logger = logging.getLogger(__name__)


class LocalTifFallback:
    """
    Fallback handler for loading local TIF files.
    
    Searches for and loads Sentinel-2 band data from local filesystem
    when API is unavailable.
    """
    
    def __init__(self, search_paths: Optional[List[Path]] = None):
        """
        Initialize local TIF fallback handler.
        
        Args:
            search_paths: List of directories to search for TIF files.
                         If None, uses default paths.
        """
        if search_paths is None:
            search_paths = [
                Path('data/processed'),
                Path('S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE'),
                Path('.')
            ]
        
        self.search_paths = [Path(p) for p in search_paths]
        self._cached_files: Optional[Dict[str, Path]] = None
    
    def discover_local_tif_files(self, bands: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Discover available TIF files in search paths.
        
        Args:
            bands: List of band names to search for (e.g., ['B02', 'B03', 'B04', 'B08'])
                  If None, searches for all bands
        
        Returns:
            Dictionary mapping band names to file paths
        """
        if bands is None:
            bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 
                    'B8A', 'B09', 'B11', 'B12']
        
        found_files = {}
        
        for search_path in self.search_paths:
            if not search_path.exists():
                logger.debug(f"Search path does not exist: {search_path}")
                continue
            
            logger.info(f"Searching for TIF files in: {search_path}")
            
            # Search recursively for TIF/JP2 files
            for pattern in ['**/*.tif', '**/*.TIF', '**/*.jp2', '**/*.JP2']:
                for file_path in search_path.glob(pattern):
                    # Try to extract band name from filename
                    band_name = self._extract_band_name(file_path.name)
                    
                    if band_name and band_name in bands:
                        if band_name not in found_files:
                            found_files[band_name] = file_path
                            logger.info(f"Found {band_name}: {file_path}")
        
        self._cached_files = found_files
        
        logger.info(f"Discovered {len(found_files)} band files locally")
        return found_files
    
    def _extract_band_name(self, filename: str) -> Optional[str]:
        """
        Extract band name from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Band name (e.g., 'B02', 'B8A') or None if not found
        """
        # Common patterns in Sentinel-2 filenames
        patterns = [
            r'_(B\d{2})_',  # _B02_, _B08_
            r'_(B8A)_',     # _B8A_
            r'_(B\d{2})\.',  # _B02.
            r'_(B8A)\.',     # _B8A.
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def load_band_data(self, band_name: str) -> Optional[np.ndarray]:
        """
        Load band data from local TIF file.
        
        Args:
            band_name: Name of band to load (e.g., 'B02')
            
        Returns:
            Numpy array with band data, or None if not found
        """
        # Use cached files if available
        if self._cached_files is None:
            self.discover_local_tif_files()
        
        if band_name not in self._cached_files:
            logger.warning(f"Band {band_name} not found in local files")
            return None
        
        file_path = self._cached_files[band_name]
        
        try:
            # Try to import rasterio (optional dependency)
            try:
                import rasterio
                
                with rasterio.open(file_path) as src:
                    data = src.read(1).astype(np.float32)
                    logger.info(f"Loaded {band_name} from {file_path} using rasterio")
                    return data
            
            except ImportError:
                logger.warning("rasterio not available, trying alternative methods")
                
                # Try PIL for TIFF files
                try:
                    from PIL import Image
                    
                    img = Image.open(file_path)
                    data = np.array(img, dtype=np.float32)
                    logger.info(f"Loaded {band_name} from {file_path} using PIL")
                    return data
                
                except ImportError:
                    logger.error("Neither rasterio nor PIL available for loading TIF files")
                    return None
        
        except Exception as e:
            logger.error(f"Failed to load {band_name} from {file_path}: {e}")
            return None
    
    def load_multiple_bands(self, bands: List[str]) -> Dict[str, np.ndarray]:
        """
        Load multiple bands from local files.
        
        Args:
            bands: List of band names to load
            
        Returns:
            Dictionary mapping band names to numpy arrays
        """
        band_data = {}
        
        for band_name in bands:
            data = self.load_band_data(band_name)
            if data is not None:
                band_data[band_name] = data
        
        logger.info(f"Loaded {len(band_data)}/{len(bands)} bands from local files")
        
        return band_data
    
    def validate_local_files(self, bands: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that required bands are available locally.
        
        Args:
            bands: List of required band names
            
        Returns:
            Tuple of (all_available, missing_bands)
        """
        if self._cached_files is None:
            self.discover_local_tif_files(bands)
        
        missing_bands = [b for b in bands if b not in self._cached_files]
        
        all_available = len(missing_bands) == 0
        
        if all_available:
            logger.info(f"All {len(bands)} required bands available locally")
        else:
            logger.warning(f"Missing {len(missing_bands)} bands: {missing_bands}")
        
        return all_available, missing_bands
    
    def get_local_metadata(self) -> Dict[str, any]:
        """
        Extract metadata from local files.
        
        Returns:
            Dictionary with metadata about local files
        """
        if self._cached_files is None:
            self.discover_local_tif_files()
        
        metadata = {
            'source': 'local_tif',
            'num_bands': len(self._cached_files),
            'bands': list(self._cached_files.keys()),
            'file_paths': {k: str(v) for k, v in self._cached_files.items()}
        }
        
        # Try to extract date from filenames
        if self._cached_files:
            first_file = next(iter(self._cached_files.values()))
            first_file_path = Path(first_file) if not isinstance(first_file, Path) else first_file
            date_match = re.search(r'(\d{8})', first_file_path.name)
            if date_match:
                date_str = date_match.group(1)
                try:
                    acquisition_date = datetime.strptime(date_str, '%Y%m%d')
                    metadata['acquisition_date'] = acquisition_date.isoformat()
                except ValueError:
                    pass
        
        return metadata


def fallback_to_local_tif(
    bands: List[str],
    search_paths: Optional[List[Path]] = None
) -> Tuple[bool, Dict[str, np.ndarray], Dict[str, any]]:
    """
    Convenience function to attempt loading bands from local TIF files.
    
    Args:
        bands: List of band names to load
        search_paths: Optional list of directories to search
        
    Returns:
        Tuple of (success, band_data, metadata)
    """
    fallback = LocalTifFallback(search_paths)
    
    # Validate files are available
    all_available, missing = fallback.validate_local_files(bands)
    
    if not all_available:
        logger.warning(f"Cannot use local fallback - missing bands: {missing}")
        return False, {}, {}
    
    # Load band data
    band_data = fallback.load_multiple_bands(bands)
    
    if len(band_data) < len(bands):
        logger.warning(f"Only loaded {len(band_data)}/{len(bands)} bands")
        return False, band_data, {}
    
    # Get metadata
    metadata = fallback.get_local_metadata()
    
    logger.info("Successfully loaded all bands from local files")
    return True, band_data, metadata
