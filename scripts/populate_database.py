#!/usr/bin/env python3
"""
Populate Database Script

Populates the AgriFlux database with processed Sentinel-2A data.
Can work with existing processed GeoTIFF files or trigger new processing.

Usage:
    python scripts/populate_database.py [--reprocess] [--safe-dir <path>]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.data_processing.sentinel2_parser import parse_sentinel2_safe
from scripts.process_sentinel2_data import process_sentinel2_safe_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_population.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_metadata_from_safe(safe_dir: Path) -> Dict[str, Any]:
    """
    Extract metadata from SAFE directory without full processing.
    
    Args:
        safe_dir: Path to SAFE directory
        
    Returns:
        Dictionary with metadata
    """
    logger.info(f"Extracting metadata from {safe_dir}")
    metadata, _ = parse_sentinel2_safe(safe_dir)
    
    return {
        'product_id': metadata.product_id,
        'acquisition_date': metadata.acquisition_date.isoformat(),
        'tile_id': metadata.tile_id,
        'cloud_coverage': metadata.cloud_coverage,
        'processing_level': metadata.processing_level,
        'spacecraft_name': metadata.spacecraft_name,
        'orbit_number': metadata.orbit_number,
        'utm_zone': metadata.utm_zone,
        'epsg_code': metadata.epsg_code
    }


def find_processed_data(processed_dir: Path, tile_id: str, acquisition_date: str) -> Optional[Dict[str, str]]:
    """
    Find processed GeoTIFF files for a given tile and date.
    
    Args:
        processed_dir: Base processed data directory
        tile_id: Tile identifier
        acquisition_date: Acquisition date in YYYYMMDD format
        
    Returns:
        Dictionary mapping index names to file paths, or None if not found
    """
    # Try to find the directory
    date_str = acquisition_date.replace('-', '')[:8]  # Convert to YYYYMMDD
    tile_dir = processed_dir / f"{tile_id}_{date_str}"
    
    if not tile_dir.exists():
        logger.warning(f"Processed data directory not found: {tile_dir}")
        return None
    
    # Look for GeoTIFF files
    geotiff_paths = {}
    for index_name in ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']:
        tif_path = tile_dir / f"{index_name}.tif"
        if tif_path.exists():
            geotiff_paths[index_name] = str(tif_path.absolute())
        else:
            logger.warning(f"Missing {index_name} file: {tif_path}")
    
    if geotiff_paths:
        logger.info(f"Found {len(geotiff_paths)} processed indices in {tile_dir}")
        return geotiff_paths
    
    return None


def populate_from_existing_data(db_manager: DatabaseManager, 
                                safe_dir: Path,
                                processed_dir: Path) -> int:
    """
    Populate database from existing processed data.
    
    Args:
        db_manager: DatabaseManager instance
        safe_dir: Path to SAFE directory
        processed_dir: Path to processed data directory
        
    Returns:
        Imagery record ID
    """
    logger.info("Populating database from existing processed data...")
    
    # Extract metadata from SAFE directory
    metadata = extract_metadata_from_safe(safe_dir)
    
    # Find processed GeoTIFF files
    geotiff_paths = find_processed_data(
        processed_dir,
        metadata['tile_id'],
        metadata['acquisition_date']
    )
    
    if not geotiff_paths:
        raise FileNotFoundError(
            f"No processed data found for tile {metadata['tile_id']} "
            f"on {metadata['acquisition_date']}"
        )
    
    # Save to database (mark as real data from SAFE file)
    imagery_id = db_manager.save_processed_imagery(
        acquisition_date=metadata['acquisition_date'],
        tile_id=metadata['tile_id'],
        cloud_coverage=metadata['cloud_coverage'],
        geotiff_paths=geotiff_paths,
        metadata=metadata,
        synthetic=False  # This is real Sentinel-2 data from SAFE file
    )
    
    logger.info(f"Successfully saved imagery record with ID: {imagery_id}")
    return imagery_id


def populate_with_reprocessing(db_manager: DatabaseManager,
                               safe_dir: Path,
                               processed_dir: Path) -> int:
    """
    Reprocess data and populate database.
    
    Args:
        db_manager: DatabaseManager instance
        safe_dir: Path to SAFE directory
        processed_dir: Path to processed data directory
        
    Returns:
        Imagery record ID
    """
    logger.info("Reprocessing data and populating database...")
    
    # Run full processing pipeline
    results = process_sentinel2_safe_directory(safe_dir, processed_dir)
    
    # Save to database (mark as real data from SAFE file)
    imagery_id = db_manager.save_processed_imagery(
        acquisition_date=results['metadata']['acquisition_date'],
        tile_id=results['metadata']['tile_id'],
        cloud_coverage=results['metadata']['cloud_coverage'],
        geotiff_paths=results['geotiff_paths'],
        metadata=results['metadata'],
        synthetic=False  # This is real Sentinel-2 data from SAFE file
    )
    
    logger.info(f"Successfully saved imagery record with ID: {imagery_id}")
    return imagery_id


def verify_database_integrity(db_manager: DatabaseManager, imagery_id: int):
    """
    Verify data integrity with queries.
    
    Args:
        db_manager: DatabaseManager instance
        imagery_id: Imagery record ID to verify
    """
    logger.info("Verifying database integrity...")
    
    # Test 1: Retrieve the record we just inserted
    record = db_manager.get_processed_imagery(imagery_id)
    if not record:
        raise ValueError(f"Failed to retrieve imagery record {imagery_id}")
    logger.info(f"✓ Successfully retrieved imagery record {imagery_id}")
    
    # Test 2: Verify all expected fields are present
    required_fields = ['id', 'acquisition_date', 'tile_id', 'cloud_coverage', 'processed_at']
    for field in required_fields:
        if field not in record or record[field] is None:
            raise ValueError(f"Missing or null field: {field}")
    logger.info(f"✓ All required fields present")
    
    # Test 3: Verify at least one GeoTIFF path exists
    geotiff_fields = ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']
    has_geotiff = any(record.get(field) for field in geotiff_fields)
    if not has_geotiff:
        raise ValueError("No GeoTIFF paths found in record")
    logger.info(f"✓ GeoTIFF paths verified")
    
    # Test 4: Verify GeoTIFF files actually exist
    for field in geotiff_fields:
        path = record.get(field)
        if path and not Path(path).exists():
            logger.warning(f"⚠ GeoTIFF file not found: {path}")
        elif path:
            logger.info(f"✓ Verified file exists: {Path(path).name}")
    
    # Test 5: Test get_latest_imagery
    latest = db_manager.get_latest_imagery(record['tile_id'])
    if not latest or latest['id'] != imagery_id:
        logger.warning(f"⚠ get_latest_imagery returned unexpected result")
    else:
        logger.info(f"✓ get_latest_imagery working correctly")
    
    # Test 6: Test list_processed_imagery
    imagery_list = db_manager.list_processed_imagery(tile_id=record['tile_id'])
    if not imagery_list or imagery_id not in [img['id'] for img in imagery_list]:
        raise ValueError("Record not found in list_processed_imagery")
    logger.info(f"✓ list_processed_imagery working correctly")
    
    # Test 7: Get database statistics
    stats = db_manager.get_database_stats()
    logger.info(f"✓ Database statistics: {stats}")
    
    logger.info("✅ All integrity checks passed!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Populate AgriFlux database with processed Sentinel-2A data'
    )
    parser.add_argument(
        '--safe-dir',
        type=Path,
        default=Path('S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE'),
        help='Path to Sentinel-2A SAFE directory'
    )
    parser.add_argument(
        '--processed-dir',
        type=Path,
        default=Path('data/processed'),
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('data/agriflux.db'),
        help='Path to database file'
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Reprocess data instead of using existing processed files'
    )
    
    args = parser.parse_args()
    
    # Validate SAFE directory
    if not args.safe_dir.exists():
        logger.error(f"SAFE directory not found: {args.safe_dir}")
        sys.exit(1)
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        db_manager = DatabaseManager(str(args.db_path))
        db_manager.init_database()
        logger.info(f"Database initialized at {args.db_path}")
        
        # Populate database
        if args.reprocess:
            imagery_id = populate_with_reprocessing(
                db_manager,
                args.safe_dir,
                args.processed_dir
            )
        else:
            imagery_id = populate_from_existing_data(
                db_manager,
                args.safe_dir,
                args.processed_dir
            )
        
        # Verify integrity
        verify_database_integrity(db_manager, imagery_id)
        
        # Print summary
        print("\n" + "="*60)
        print("DATABASE POPULATION SUMMARY")
        print("="*60)
        
        record = db_manager.get_processed_imagery(imagery_id)
        print(f"Imagery ID: {record['id']}")
        print(f"Tile ID: {record['tile_id']}")
        print(f"Acquisition Date: {record['acquisition_date']}")
        print(f"Cloud Coverage: {record['cloud_coverage']:.2f}%")
        print(f"Processed At: {record['processed_at']}")
        
        print("\nAvailable Indices:")
        for field in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
            if record.get(field):
                index_name = field.replace('_path', '').upper()
                print(f"  ✓ {index_name}: {Path(record[field]).name}")
        
        stats = db_manager.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total Imagery Records: {stats['imagery_count']}")
        print(f"  Total Alerts: {stats['total_alerts']}")
        print(f"  Active Alerts: {stats['active_alerts']}")
        print(f"  AI Predictions: {stats['predictions_count']}")
        
        print("="*60)
        print("✅ Database population completed successfully!")
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Database population failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
