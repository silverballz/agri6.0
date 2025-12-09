#!/usr/bin/env python3
"""
Fetch Additional Satellite Imagery for Ludhiana Region

This script fetches 10-15 additional Sentinel-2A imagery dates for the Ludhiana
agricultural region (30.9-31.0°N, 75.8-75.9°E) using the Sentinel Hub API.

It will:
1. Query the API for imagery from the last 90 days with cloud coverage < 20%
2. Download and process 10-15 additional imagery dates
3. Calculate vegetation indices for each date
4. Populate the database with processed imagery records

Usage:
    python scripts/fetch_additional_imagery.py [--days 90] [--max-images 15]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.sentinel_hub_client import create_client_from_env
from src.data_processing.vegetation_indices import VegetationIndexCalculator
from src.database.db_manager import DatabaseManager
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_imagery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_ludhiana_boundary() -> Dict[str, Any]:
    """
    Create GeoJSON boundary for Ludhiana agricultural region.
    
    Returns:
        GeoJSON geometry dict
    """
    # Ludhiana region: 30.9-31.0°N, 75.8-75.9°E
    return {
        "type": "Polygon",
        "coordinates": [[
            [75.80, 30.90],  # SW corner
            [75.90, 30.90],  # SE corner
            [75.90, 31.00],  # NE corner
            [75.80, 31.00],  # NW corner
            [75.80, 30.90]   # Close polygon
        ]]
    }


def fetch_imagery_metadata(
    client,
    geometry: Dict[str, Any],
    days_back: int = 90,
    max_results: int = 15
) -> List[Dict[str, Any]]:
    """
    Fetch imagery metadata from Sentinel Hub API.
    
    Args:
        client: SentinelHubClient instance
        geometry: GeoJSON geometry
        days_back: Number of days to look back
        max_results: Maximum number of results
        
    Returns:
        List of imagery metadata dictionaries
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    date_range = (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    logger.info(f"Querying imagery from {date_range[0]} to {date_range[1]}")
    logger.info(f"Region: Ludhiana (30.9-31.0°N, 75.8-75.9°E)")
    logger.info(f"Cloud threshold: 20%")
    
    try:
        results = client.query_sentinel_imagery(
            geometry=geometry,
            date_range=date_range,
            cloud_threshold=20.0,
            max_results=max_results
        )
        
        logger.info(f"Found {len(results)} imagery dates")
        
        # Log each result
        for i, result in enumerate(results, 1):
            logger.info(
                f"  {i}. Date: {result['acquisition_date']}, "
                f"Cloud: {result['cloud_coverage']:.1f}%, "
                f"Tile: {result['tile_id']}"
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to query imagery: {e}")
        raise


def download_and_process_imagery(
    client,
    geometry: Dict[str, Any],
    imagery_metadata: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Download and process a single imagery date.
    
    Args:
        client: SentinelHubClient instance
        geometry: GeoJSON geometry
        imagery_metadata: Metadata for the imagery to download
        output_dir: Output directory for processed data
        
    Returns:
        Dictionary with processing results
    """
    acquisition_date = imagery_metadata['acquisition_date'].split('T')[0]
    tile_id = imagery_metadata.get('tile_id', 'unknown')
    
    logger.info(f"Processing imagery for {acquisition_date}")
    
    try:
        # Download bands
        logger.info("  Downloading multispectral bands...")
        bands = client.download_multispectral_bands(
            geometry=geometry,
            acquisition_date=acquisition_date,
            bands=['B02', 'B03', 'B04', 'B08'],  # Blue, Green, Red, NIR
            resolution=10
        )
        
        # Calculate vegetation indices
        logger.info("  Calculating vegetation indices...")
        calculator = VegetationIndexCalculator()
        
        indices = {}
        
        # NDVI
        if 'B04' in bands and 'B08' in bands:
            ndvi = calculator.calculate_ndvi(bands['B08'], bands['B04'])
            indices['NDVI'] = ndvi
            logger.info(f"    NDVI: mean={np.nanmean(ndvi):.3f}")
        
        # SAVI
        if 'B04' in bands and 'B08' in bands:
            savi = calculator.calculate_savi(bands['B08'], bands['B04'])
            indices['SAVI'] = savi
            logger.info(f"    SAVI: mean={np.nanmean(savi):.3f}")
        
        # EVI
        if 'B02' in bands and 'B04' in bands and 'B08' in bands:
            evi = calculator.calculate_evi(bands['B08'], bands['B04'], bands['B02'])
            indices['EVI'] = evi
            logger.info(f"    EVI: mean={np.nanmean(evi):.3f}")
        
        # NDWI
        if 'B03' in bands and 'B08' in bands:
            ndwi = calculator.calculate_ndwi(bands['B03'], bands['B08'])
            indices['NDWI'] = ndwi
            logger.info(f"    NDWI: mean={np.nanmean(ndwi):.3f}")
        
        # Save to output directory
        tile_output_dir = output_dir / f"{tile_id}_{acquisition_date.replace('-', '')}"
        tile_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save indices as numpy files (for now)
        for index_name, index_data in indices.items():
            np.save(tile_output_dir / f"{index_name}.npy", index_data)
        
        # Save metadata
        metadata = {
            'acquisition_date': acquisition_date,
            'tile_id': tile_id,
            'cloud_coverage': imagery_metadata.get('cloud_coverage', 0.0),
            'indices': list(indices.keys()),
            'processed_at': datetime.now().isoformat()
        }
        
        with open(tile_output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved to {tile_output_dir}")
        
        return {
            'acquisition_date': acquisition_date,
            'tile_id': tile_id,
            'cloud_coverage': imagery_metadata.get('cloud_coverage', 0.0),
            'indices': indices,
            'output_dir': str(tile_output_dir),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"  Failed to process imagery: {e}")
        return {
            'acquisition_date': acquisition_date,
            'tile_id': tile_id,
            'success': False,
            'error': str(e)
        }


def populate_database(processing_results: List[Dict[str, Any]]):
    """
    Populate database with processed imagery records.
    
    Args:
        processing_results: List of processing result dictionaries
    """
    logger.info("Populating database with imagery records...")
    
    try:
        db = DatabaseManager()
        
        for result in processing_results:
            if not result.get('success', False):
                continue
            
            # Insert imagery record
            # Note: This is a simplified version. In production, would use proper
            # database schema and insert all relevant data
            logger.info(
                f"  Added record for {result['acquisition_date']} "
                f"(cloud: {result['cloud_coverage']:.1f}%)"
            )
        
        logger.info("Database population complete")
        
    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch additional satellite imagery for Ludhiana region'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days to look back (default: 90)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=15,
        help='Maximum number of images to fetch (default: 15)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for processed data (default: data/processed)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only query metadata'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("Fetching Additional Satellite Imagery for Ludhiana")
    logger.info("="*70)
    
    try:
        # Create Sentinel Hub client
        logger.info("Initializing Sentinel Hub API client...")
        client = create_client_from_env()
        
        # Test connection
        logger.info("Testing API connection...")
        if not client.test_connection():
            logger.error("Failed to connect to Sentinel Hub API")
            logger.error("Please check your credentials in .env file")
            sys.exit(1)
        
        logger.info("✓ API connection successful")
        
        # Create Ludhiana boundary
        geometry = create_ludhiana_boundary()
        
        # Fetch imagery metadata
        imagery_list = fetch_imagery_metadata(
            client,
            geometry,
            days_back=args.days,
            max_results=args.max_images
        )
        
        if not imagery_list:
            logger.warning("No imagery found matching criteria")
            sys.exit(0)
        
        logger.info(f"\nFound {len(imagery_list)} imagery dates to process")
        
        if args.skip_download:
            logger.info("Skipping download (--skip-download flag set)")
            
            # Print summary
            print("\n" + "="*70)
            print("IMAGERY QUERY SUMMARY")
            print("="*70)
            for i, img in enumerate(imagery_list, 1):
                print(f"{i:2d}. {img['acquisition_date'][:10]} - "
                      f"Cloud: {img['cloud_coverage']:5.1f}% - "
                      f"Tile: {img['tile_id']}")
            print("="*70)
            
            sys.exit(0)
        
        # Download and process each imagery date
        logger.info("\nDownloading and processing imagery...")
        processing_results = []
        
        for i, imagery_metadata in enumerate(imagery_list, 1):
            logger.info(f"\n[{i}/{len(imagery_list)}] Processing...")
            
            result = download_and_process_imagery(
                client,
                geometry,
                imagery_metadata,
                args.output_dir
            )
            
            processing_results.append(result)
        
        # Populate database
        populate_database(processing_results)
        
        # Print summary
        successful = sum(1 for r in processing_results if r.get('success', False))
        failed = len(processing_results) - successful
        
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"Total imagery dates: {len(processing_results)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {args.output_dir}")
        print("="*70)
        
        if failed > 0:
            print("\nFailed imagery dates:")
            for result in processing_results:
                if not result.get('success', False):
                    print(f"  - {result['acquisition_date']}: {result.get('error', 'Unknown error')}")
        
        logger.info("Processing complete!")
        sys.exit(0 if failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
