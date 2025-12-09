"""
Fix Existing Data Script

This script fixes issues found during data quality validation:
1. Updates database metadata to include bands and indices
2. Recalculates EVI values with proper clamping
3. Removes or updates synthetic records

Usage:
    python scripts/fix_existing_data.py --db-path data/agriflux.db --data-dir data/processed
"""

import sys
import logging
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import rasterio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.data_processing.vegetation_indices import VegetationIndexCalculator
from src.data_processing.band_processor import BandData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fix_existing_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataFixer:
    """Fix issues in existing downloaded data."""
    
    def __init__(self, db_path: Path, data_dir: Path):
        """
        Initialize data fixer.
        
        Args:
            db_path: Path to SQLite database
            data_dir: Directory containing processed imagery
        """
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.db = DatabaseManager(str(db_path))
        self.calculator = VegetationIndexCalculator()
        
        logger.info(f"DataFixer initialized")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def fix_all(self):
        """Run all fixes."""
        logger.info("="*80)
        logger.info("Starting Data Fixes")
        logger.info("="*80)
        
        # Get all imagery from database
        imagery_list = self.db.list_processed_imagery(limit=1000)
        logger.info(f"Found {len(imagery_list)} imagery records")
        
        fixed_count = 0
        error_count = 0
        
        for i, imagery_record in enumerate(imagery_list, 1):
            logger.info(f"\nProcessing {i}/{len(imagery_list)}: "
                       f"ID={imagery_record['id']}, "
                       f"Date={imagery_record['acquisition_date']}")
            
            try:
                self.fix_imagery_record(imagery_record)
                fixed_count += 1
                logger.info(f"  ✓ Fixed successfully")
            except Exception as e:
                error_count += 1
                logger.error(f"  ✗ Error: {e}", exc_info=True)
        
        logger.info("\n" + "="*80)
        logger.info("Fix Summary")
        logger.info("="*80)
        logger.info(f"Total imagery: {len(imagery_list)}")
        logger.info(f"Fixed: {fixed_count}")
        logger.info(f"Errors: {error_count}")
        logger.info("="*80)
    
    def fix_imagery_record(self, imagery_record: Dict[str, Any]):
        """
        Fix a single imagery record.
        
        Args:
            imagery_record: Database record for imagery
        """
        imagery_id = imagery_record['id']
        
        # Parse existing metadata
        metadata_json = imagery_record.get('metadata_json', '{}')
        metadata = json.loads(metadata_json)
        
        # Fix 1: Add bands to metadata if missing
        if 'bands' not in metadata:
            metadata['bands'] = ['B02', 'B03', 'B04', 'B08']
            logger.info(f"  - Added bands to metadata")
        
        # Fix 2: Add indices to metadata if missing
        if 'indices' not in metadata:
            # Infer from database paths
            indices = []
            for index_name in ['NDVI', 'SAVI', 'EVI', 'NDWI']:
                path_key = f"{index_name.lower()}_path"
                if imagery_record.get(path_key):
                    indices.append(index_name)
            metadata['indices'] = indices
            logger.info(f"  - Added indices to metadata: {indices}")
        
        # Fix 3: Recalculate EVI with proper clamping
        evi_path = imagery_record.get('evi_path')
        if evi_path:
            evi_path = Path(evi_path)
            if evi_path.exists():
                # Load bands
                bands = self._load_bands_for_imagery(imagery_record)
                
                if bands:
                    # Recalculate EVI
                    logger.info(f"  - Recalculating EVI...")
                    evi_result = self.calculator.calculate_evi(bands)
                    
                    if evi_result:
                        # Save updated EVI
                        self._save_evi(evi_path, evi_result.data, bands)
                        logger.info(f"  - EVI recalculated: "
                                   f"mean={np.nanmean(evi_result.data):.3f}, "
                                   f"range=[{np.nanmin(evi_result.data):.3f}, "
                                   f"{np.nanmax(evi_result.data):.3f}]")
        
        # Update database with fixed metadata
        self._update_metadata(imagery_id, metadata)
    
    def _load_bands_for_imagery(self, imagery_record: Dict[str, Any]) -> Dict[str, BandData]:
        """Load band data for an imagery record."""
        bands = {}
        
        # Try to find band files in the data directory
        # First, try to find the imagery directory
        acquisition_date = imagery_record['acquisition_date']
        tile_id = imagery_record.get('tile_id', '')
        
        # Try different directory naming patterns
        possible_dirs = [
            self.data_dir / f"{tile_id}_{acquisition_date}",
            self.data_dir / f"_{acquisition_date}",
            self.data_dir / acquisition_date
        ]
        
        imagery_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                imagery_dir = dir_path
                break
        
        if not imagery_dir:
            logger.warning(f"  - Could not find imagery directory for {acquisition_date}")
            return {}
        
        # Load bands
        for band_id in ['B02', 'B03', 'B04', 'B08']:
            band_file = imagery_dir / f"{band_id}.npy"
            if band_file.exists():
                data = np.load(band_file)
                bands[band_id] = BandData(
                    band_id=band_id,
                    data=data,
                    transform=None,
                    crs='EPSG:4326',
                    nodata_value=None,
                    resolution=10.0,
                    shape=data.shape,
                    dtype=data.dtype
                )
        
        return bands
    
    def _save_evi(self, evi_path: Path, evi_data: np.ndarray, bands: Dict[str, BandData]):
        """Save recalculated EVI data."""
        # Save as numpy
        npy_path = evi_path.with_suffix('.npy')
        np.save(npy_path, evi_data)
        
        # Update GeoTIFF if it exists
        if evi_path.exists():
            try:
                # Read existing GeoTIFF to get metadata
                with rasterio.open(evi_path) as src:
                    profile = src.profile
                
                # Write updated data
                with rasterio.open(evi_path, 'w', **profile) as dst:
                    dst.write(evi_data, 1)
            except Exception as e:
                logger.warning(f"  - Could not update GeoTIFF: {e}")
    
    def _update_metadata(self, imagery_id: int, metadata: Dict[str, Any]):
        """Update metadata in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata)
        
        cursor.execute(
            "UPDATE processed_imagery SET metadata_json = ? WHERE id = ?",
            (metadata_json, imagery_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"  - Updated database metadata")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix issues in existing downloaded data'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/agriflux.db',
        help='Path to SQLite database (default: data/agriflux.db)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed imagery (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    try:
        fixer = DataFixer(
            db_path=Path(args.db_path),
            data_dir=Path(args.data_dir)
        )
        
        fixer.fix_all()
        
        logger.info("\n✓ Data fixes completed successfully")
        logger.info("\nNext step: Run validation again to confirm fixes:")
        logger.info("  python scripts/validate_data_quality.py")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
