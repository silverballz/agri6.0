#!/usr/bin/env python3
"""
Generate Synthetic Time Series Data for Training

Since fetching real satellite imagery from Sentinel Hub API has API limitations,
this script generates synthetic time series data by:
1. Using the existing 2024-09-23 imagery as a base
2. Creating realistic temporal variations (10-15 dates over 90 days)
3. Adding seasonal trends and noise
4. Calculating vegetation indices for each synthetic date
5. Saving processed data for training

This provides sufficient data for training AI models while we work on API access.

Usage:
    python scripts/generate_synthetic_time_series.py [--num-dates 12]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

import numpy as np
import rasterio
from rasterio.transform import Affine

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.vegetation_indices import VegetationIndexCalculator
from src.data_processing.band_processor import BandData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/synthetic_time_series.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_base_imagery(crop_size: int = 2000) -> Dict[str, np.ndarray]:
    """
    Load the existing Sentinel-2A imagery from 2024-09-23.
    
    Args:
        crop_size: Size of crop to load (for faster processing)
    
    Returns:
        Dictionary of band_name -> numpy array
    """
    logger.info(f"Loading base imagery from 2024-09-23 (crop size: {crop_size}x{crop_size})...")
    
    base_path = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE/GRANULE/L2A_T43REQ_A048336_20240923T055000/IMG_DATA/R10m")
    
    band_files = {
        'B02': base_path / "T43REQ_20240923T053641_B02_10m.jp2",
        'B03': base_path / "T43REQ_20240923T053641_B03_10m.jp2",
        'B04': base_path / "T43REQ_20240923T053641_B04_10m.jp2",
        'B08': base_path / "T43REQ_20240923T053641_B08_10m.jp2",
    }
    
    bands = {}
    for band_name, file_path in band_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Band file not found: {file_path}")
        
        with rasterio.open(file_path) as src:
            # Read a crop from the center for faster processing
            full_height, full_width = src.height, src.width
            start_row = (full_height - crop_size) // 2
            start_col = (full_width - crop_size) // 2
            
            window = ((start_row, start_row + crop_size), (start_col, start_col + crop_size))
            bands[band_name] = src.read(1, window=window).astype(np.float32)
            logger.info(f"  Loaded {band_name}: shape={bands[band_name].shape}")
    
    return bands


def generate_temporal_variation(
    base_bands: Dict[str, np.ndarray],
    date: datetime,
    base_date: datetime
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic temporal variation for a given date.
    
    Args:
        base_bands: Base imagery bands
        date: Target date for synthetic data
        base_date: Base date of original imagery
        
    Returns:
        Dictionary of band_name -> modified numpy array
    """
    # Calculate days difference
    days_diff = (date - base_date).days
    
    # Seasonal factor (sinusoidal variation)
    # Assume base date (Sept 23) is in growing season
    day_of_year = date.timetuple().tm_yday
    seasonal_factor = 0.1 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Trend factor (gradual change over time)
    trend_factor = 0.002 * days_diff  # 0.2% change per day
    
    # Random noise
    noise_level = 0.03  # 3% noise
    
    modified_bands = {}
    
    for band_name, band_data in base_bands.items():
        # Apply temporal variation
        # NIR (B08) is most sensitive to vegetation changes
        if band_name == 'B08':
            variation = seasonal_factor + trend_factor
        else:
            # Other bands vary less
            variation = seasonal_factor * 0.5 + trend_factor * 0.5
        
        # Add noise
        noise = np.random.normal(0, noise_level, band_data.shape)
        
        # Apply variation
        modified = band_data * (1 + variation + noise)
        
        # Clip to valid range
        modified = np.clip(modified, 0, 10000)
        
        modified_bands[band_name] = modified.astype(np.float32)
    
    return modified_bands


def calculate_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate vegetation indices from bands.
    
    Args:
        bands: Dictionary of band_name -> numpy array
        
    Returns:
        Dictionary of index_name -> numpy array
    """
    # Calculate NDVI
    nir = bands['B08']
    red = bands['B04']
    green = bands['B03']
    blue = bands['B02']
    
    indices = {}
    
    # NDVI
    denominator = nir + red
    ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
    indices['NDVI'] = np.clip(ndvi, -1, 1)
    
    # SAVI (L=0.5)
    L = 0.5
    denominator = nir + red + L
    savi = np.where(denominator != 0, ((nir - red) / denominator) * (1 + L), 0)
    indices['SAVI'] = np.clip(savi, -1.5, 1.5)
    
    # EVI
    denominator = nir + 6 * red - 7.5 * blue + 1
    evi = np.where(denominator != 0, 2.5 * ((nir - red) / denominator), 0)
    indices['EVI'] = np.clip(evi, -1, 1)
    
    # NDWI
    denominator = green + nir
    ndwi = np.where(denominator != 0, (green - nir) / denominator, 0)
    indices['NDWI'] = np.clip(ndwi, -1, 1)
    
    return indices


def save_synthetic_date(
    date: datetime,
    bands: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """
    Save synthetic imagery and indices for a date.
    
    Args:
        date: Date of synthetic imagery
        bands: Band data
        indices: Calculated indices
        output_dir: Output directory
        metadata: Metadata dictionary
    """
    date_str = date.strftime('%Y%m%d')
    tile_dir = output_dir / f"43REQ_{date_str}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    # Save bands as numpy files
    for band_name, band_data in bands.items():
        np.save(tile_dir / f"{band_name}.npy", band_data)
    
    # Save indices as numpy files
    for index_name, index_data in indices.items():
        np.save(tile_dir / f"{index_name}.npy", index_data)
    
    # Save metadata
    metadata_dict = {
        'acquisition_date': date.isoformat(),
        'tile_id': '43REQ',
        'synthetic': True,
        'base_date': '2024-09-23',
        'bands': list(bands.keys()),
        'indices': list(indices.keys()),
        'processed_at': datetime.now().isoformat(),
        **metadata
    }
    
    with open(tile_dir / 'metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    logger.info(f"  Saved to {tile_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic time series data for training'
    )
    parser.add_argument(
        '--num-dates',
        type=int,
        default=12,
        help='Number of synthetic dates to generate (default: 12)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=90,
        help='Number of days to go back from base date (default: 90)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for processed data (default: data/processed)'
    )
    parser.add_argument(
        '--crop-size',
        type=int,
        default=2000,
        help='Size of image crop to process (default: 2000)'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("Generating Synthetic Time Series Data")
    logger.info("="*70)
    
    try:
        # Load base imagery
        base_bands = load_base_imagery(crop_size=args.crop_size)
        base_date = datetime(2024, 9, 23)
        
        logger.info(f"\nGenerating {args.num_dates} synthetic dates...")
        logger.info(f"Date range: {args.days_back} days before {base_date.strftime('%Y-%m-%d')}")
        
        # Generate dates (evenly spaced)
        dates = []
        for i in range(args.num_dates):
            days_offset = int(args.days_back * i / (args.num_dates - 1)) if args.num_dates > 1 else 0
            date = base_date - timedelta(days=days_offset)
            dates.append(date)
        
        # Sort dates chronologically
        dates.sort()
        
        # Generate synthetic data for each date
        results = []
        
        for i, date in enumerate(dates, 1):
            logger.info(f"\n[{i}/{len(dates)}] Generating data for {date.strftime('%Y-%m-%d')}...")
            
            # Generate temporal variation
            modified_bands = generate_temporal_variation(base_bands, date, base_date)
            
            # Calculate indices
            indices = calculate_indices(modified_bands)
            
            # Log statistics
            for index_name, index_data in indices.items():
                mean_val = np.nanmean(index_data)
                logger.info(f"  {index_name}: mean={mean_val:.3f}")
            
            # Save data
            metadata = {
                'days_from_base': (date - base_date).days,
                'statistics': {
                    index_name: {
                        'mean': float(np.nanmean(index_data)),
                        'std': float(np.nanstd(index_data)),
                        'min': float(np.nanmin(index_data)),
                        'max': float(np.nanmax(index_data))
                    }
                    for index_name, index_data in indices.items()
                }
            }
            
            save_synthetic_date(date, modified_bands, indices, args.output_dir, metadata)
            
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'success': True
            })
        
        # Print summary
        print("\n" + "="*70)
        print("SYNTHETIC TIME SERIES GENERATION SUMMARY")
        print("="*70)
        print(f"Base imagery date: {base_date.strftime('%Y-%m-%d')}")
        print(f"Synthetic dates generated: {len(results)}")
        print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"Output directory: {args.output_dir}")
        print("\nGenerated dates:")
        for i, result in enumerate(results, 1):
            print(f"  {i:2d}. {result['date']}")
        print("="*70)
        
        logger.info("\n✅ Synthetic time series generation complete!")
        logger.info("These synthetic dates can now be used for training AI models.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
