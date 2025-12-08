#!/usr/bin/env python3
"""
Sentinel-2A Data Processing Script

Orchestrates the complete processing pipeline:
1. Parse SAFE directory and extract metadata
2. Read and process spectral bands
3. Calculate vegetation indices (NDVI, SAVI, EVI, NDWI, NDSI)
4. Export indices as GeoTIFF files
5. Save metadata as JSON

Usage:
    python scripts/process_sentinel2_data.py <SAFE_directory> [--output-dir <path>]
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import rasterio
from rasterio.transform import Affine

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.sentinel2_parser import parse_sentinel2_safe
from src.data_processing.band_processor import read_and_process_bands
from src.data_processing.vegetation_indices import calculate_vegetation_indices


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def export_index_as_geotiff(index_result, output_path: Path, transform: Affine, crs: str):
    """
    Export vegetation index as GeoTIFF file.
    
    Args:
        index_result: IndexResult object from vegetation_indices module
        output_path: Path for output GeoTIFF file
        transform: Rasterio affine transform
        crs: Coordinate reference system string
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=index_result.data.shape[0],
        width=index_result.data.shape[1],
        count=1,
        dtype=index_result.data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw',
        nodata=np.nan
    ) as dst:
        dst.write(index_result.data, 1)
        
        # Add metadata
        dst.update_tags(
            index_name=index_result.index_name,
            description=index_result.description,
            formula=index_result.formula,
            valid_range_min=str(index_result.valid_range[0]),
            valid_range_max=str(index_result.valid_range[1])
        )
    
    logger.info(f"Exported {index_result.index_name} to {output_path}")


def create_metadata_json(metadata, indices_stats: Dict[str, Dict], output_path: Path):
    """
    Create metadata JSON file with processing information.
    
    Args:
        metadata: Sentinel2Metadata object
        indices_stats: Dictionary of index statistics
        output_path: Path for output JSON file
    """
    metadata_dict = {
        'product_id': metadata.product_id,
        'acquisition_date': metadata.acquisition_date.isoformat(),
        'tile_id': metadata.tile_id,
        'cloud_coverage': metadata.cloud_coverage,
        'processing_level': metadata.processing_level,
        'spacecraft_name': metadata.spacecraft_name,
        'orbit_number': metadata.orbit_number,
        'utm_zone': metadata.utm_zone,
        'epsg_code': metadata.epsg_code,
        'processed_at': datetime.now().isoformat(),
        'indices': indices_stats
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    logger.info(f"Saved metadata to {output_path}")


def process_sentinel2_safe_directory(safe_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process Sentinel-2A SAFE directory and generate all outputs.
    
    Args:
        safe_dir: Path to SAFE directory
        output_dir: Path to output directory for processed data
        
    Returns:
        Dictionary with processing results and file paths
    """
    logger.info(f"Starting processing of {safe_dir}")
    
    # Step 1: Parse SAFE directory
    logger.info("Step 1: Parsing SAFE directory...")
    try:
        metadata, band_files = parse_sentinel2_safe(safe_dir)
        logger.info(f"Found {len(band_files)} bands")
        logger.info(f"Acquisition date: {metadata.acquisition_date}")
        logger.info(f"Tile ID: {metadata.tile_id}")
        logger.info(f"Cloud coverage: {metadata.cloud_coverage:.2f}%")
    except Exception as e:
        logger.error(f"Failed to parse SAFE directory: {e}")
        raise
    
    # Step 2: Read and process bands
    logger.info("Step 2: Reading and processing spectral bands...")
    try:
        # Target bands needed for all indices
        target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        processed_bands = read_and_process_bands(band_files, target_bands=target_bands)
        logger.info(f"Processed {len(processed_bands)} bands to 10m resolution")
    except Exception as e:
        logger.error(f"Failed to process bands: {e}")
        raise
    
    # Step 3: Calculate vegetation indices
    logger.info("Step 3: Calculating vegetation indices...")
    try:
        # Calculate all available indices
        indices = calculate_vegetation_indices(processed_bands)
        logger.info(f"Calculated {len(indices)} vegetation indices")
        
        # Log statistics for each index
        indices_stats = {}
        for index_name, index_result in indices.items():
            stats = index_result.get_statistics()
            indices_stats[index_name] = stats
            logger.info(f"  {index_name}: mean={stats['mean']:.4f}, "
                       f"std={stats['std']:.4f}, "
                       f"valid_pixels={stats['valid_pixels']}")
    except Exception as e:
        logger.error(f"Failed to calculate indices: {e}")
        raise
    
    # Step 4: Export indices as GeoTIFF files
    logger.info("Step 4: Exporting indices as GeoTIFF files...")
    geotiff_paths = {}
    try:
        # Get transform and CRS from first processed band
        reference_band = next(iter(processed_bands.values()))
        transform = reference_band.transform
        crs = reference_band.crs
        
        # Create output subdirectory for this tile and date
        tile_output_dir = output_dir / f"{metadata.tile_id}_{metadata.acquisition_date.strftime('%Y%m%d')}"
        tile_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each index
        for index_name, index_result in indices.items():
            output_path = tile_output_dir / f"{index_name}.tif"
            export_index_as_geotiff(index_result, output_path, transform, crs)
            geotiff_paths[index_name] = str(output_path)
    except Exception as e:
        logger.error(f"Failed to export GeoTIFF files: {e}")
        raise
    
    # Step 5: Save metadata as JSON
    logger.info("Step 5: Saving metadata...")
    try:
        metadata_path = tile_output_dir / "metadata.json"
        create_metadata_json(metadata, indices_stats, metadata_path)
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise
    
    logger.info("Processing completed successfully!")
    
    # Return processing results
    return {
        'metadata': {
            'product_id': metadata.product_id,
            'acquisition_date': metadata.acquisition_date.isoformat(),
            'tile_id': metadata.tile_id,
            'cloud_coverage': metadata.cloud_coverage,
            'epsg_code': metadata.epsg_code
        },
        'geotiff_paths': geotiff_paths,
        'metadata_path': str(metadata_path),
        'output_directory': str(tile_output_dir),
        'indices_calculated': list(indices.keys()),
        'indices_stats': indices_stats
    }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process Sentinel-2A SAFE directory and generate vegetation indices'
    )
    parser.add_argument(
        'safe_directory',
        type=Path,
        help='Path to Sentinel-2A SAFE directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for processed data (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.safe_directory.exists():
        logger.error(f"SAFE directory not found: {args.safe_directory}")
        sys.exit(1)
    
    if not args.safe_directory.name.endswith('.SAFE'):
        logger.error(f"Invalid SAFE directory name: {args.safe_directory.name}")
        sys.exit(1)
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    # Process the data
    try:
        results = process_sentinel2_safe_directory(args.safe_directory, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Tile ID: {results['metadata']['tile_id']}")
        print(f"Acquisition Date: {results['metadata']['acquisition_date']}")
        print(f"Cloud Coverage: {results['metadata']['cloud_coverage']:.2f}%")
        print(f"Indices Calculated: {', '.join(results['indices_calculated'])}")
        print(f"Output Directory: {results['output_directory']}")
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
