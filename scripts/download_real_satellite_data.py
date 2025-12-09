"""
Real Satellite Data Download Script

Downloads real Sentinel-2 imagery for the Ludhiana region from Sentinel Hub API,
processes the data, and stores it in the database with synthetic=false flag.

This script orchestrates the complete download pipeline:
1. Query Sentinel Hub for available imagery
2. Download multispectral bands for each date
3. Calculate vegetation indices
4. Save processed data to disk and database
5. Mark all data as real (synthetic=false)

Usage:
    python scripts/download_real_satellite_data.py --days-back 365 --target-count 20
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import argparse

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.sentinel_hub_client import create_client_from_env, SentinelHubClient
from src.data_processing.vegetation_indices import VegetationIndexCalculator
from src.data_processing.band_processor import BandData
from src.data_processing.geojson_handler import create_ludhiana_sample_geojson, LUDHIANA_BOUNDS
from src.database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealDataDownloader:
    """
    Downloads and processes real Sentinel-2 imagery for the Ludhiana region.
    
    Orchestrates the complete pipeline from API query to database storage,
    ensuring all data is properly marked as real (synthetic=false).
    """
    
    def __init__(self, 
                 output_dir: Path,
                 db_path: Path,
                 client: Optional[SentinelHubClient] = None):
        """
        Initialize real data downloader.
        
        Args:
            output_dir: Directory for processed imagery output
            db_path: Path to SQLite database
            client: Optional SentinelHubClient instance (creates from env if None)
        """
        self.output_dir = Path(output_dir)
        self.db_path = Path(db_path)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.client = client if client is not None else create_client_from_env()
        self.db = DatabaseManager(str(db_path))
        self.calculator = VegetationIndexCalculator()
        
        logger.info(f"RealDataDownloader initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Database: {self.db_path}")
    
    def _create_ludhiana_geometry(self) -> Dict[str, Any]:
        """
        Create geometry for Ludhiana region.
        
        Returns:
            GeoJSON geometry dictionary
        """
        return create_ludhiana_sample_geojson()
    
    def download_ludhiana_timeseries(
        self,
        days_back: int = 365,
        target_count: int = 20,
        cloud_threshold: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Download time-series imagery for Ludhiana region.
        
        Args:
            days_back: Number of days to look back from today
            target_count: Target number of imagery dates to download
            cloud_threshold: Maximum cloud coverage percentage
            
        Returns:
            List of processing results with metadata
            
        Raises:
            ValueError: If date range is invalid
            requests.exceptions.RequestException: If API requests fail
        """
        logger.info("="*80)
        logger.info("Starting Ludhiana time-series download")
        logger.info(f"Parameters: days_back={days_back}, target_count={target_count}, "
                   f"cloud_threshold={cloud_threshold}%")
        logger.info("="*80)
        
        # Define Ludhiana boundary
        geometry = self._create_ludhiana_geometry()
        logger.info(f"Ludhiana region: {LUDHIANA_BOUNDS}")
        
        # Calculate date range (past dates only)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        date_range = (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
        
        # Query available imagery
        logger.info("Querying Sentinel Hub for available imagery...")
        imagery_list = self.client.query_sentinel_imagery(
            geometry=geometry,
            date_range=date_range,
            cloud_threshold=cloud_threshold,
            max_results=target_count
        )
        
        if not imagery_list:
            logger.warning("No imagery found matching criteria")
            return []
        
        logger.info(f"Found {len(imagery_list)} imagery dates")
        
        # Download and process each date
        results = []
        successful = 0
        failed = 0
        
        for i, imagery_meta in enumerate(imagery_list, 1):
            acquisition_date = imagery_meta['acquisition_date'][:10]
            logger.info("")
            logger.info(f"Processing {i}/{len(imagery_list)}: {acquisition_date}")
            logger.info(f"  Cloud coverage: {imagery_meta['cloud_coverage']:.1f}%")
            
            try:
                result = self._download_and_process_single_date(
                    geometry,
                    imagery_meta
                )
                results.append(result)
                successful += 1
                logger.info(f"  ✓ Successfully processed {acquisition_date}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process {acquisition_date}: {e}")
                results.append({
                    'acquisition_date': acquisition_date,
                    'success': False,
                    'error': str(e)
                })
                failed += 1
        
        # Summary
        logger.info("")
        logger.info("="*80)
        logger.info("Download Summary")
        logger.info(f"  Total imagery dates: {len(imagery_list)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info("="*80)
        
        return results
    
    def _download_and_process_single_date(
        self,
        geometry: Dict[str, Any],
        imagery_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Download bands, calculate indices, save to disk and database.
        
        Args:
            geometry: GeoJSON geometry for the region
            imagery_meta: Imagery metadata from query
            
        Returns:
            Processing result dictionary
            
        Raises:
            Exception: If download or processing fails
        """
        acquisition_date = imagery_meta['acquisition_date'][:10]
        tile_id = imagery_meta.get('tile_id', 'UNKNOWN')
        cloud_coverage = imagery_meta.get('cloud_coverage', 0.0)
        
        logger.info(f"  Downloading bands...")
        
        # Download bands
        bands_dict = self.client.download_multispectral_bands(
            geometry=geometry,
            acquisition_date=acquisition_date,
            bands=['B02', 'B03', 'B04', 'B08'],
            resolution=10
        )
        
        logger.info(f"  Downloaded {len(bands_dict)} bands")
        
        # Convert to BandData format for vegetation index calculator
        bands_for_calc = {}
        for band_id, band_array in bands_dict.items():
            bands_for_calc[band_id] = BandData(
                band_id=band_id,
                data=band_array,
                transform=None,  # Will be set when saving
                crs='EPSG:4326',
                nodata_value=None,
                resolution=10.0,
                shape=band_array.shape,
                dtype=band_array.dtype
            )
        
        logger.info(f"  Calculating vegetation indices...")
        
        # Calculate indices
        indices = {}
        
        # NDVI
        ndvi_result = self.calculator.calculate_ndvi(bands_for_calc)
        if ndvi_result:
            indices['NDVI'] = ndvi_result.data
            logger.info(f"    NDVI: mean={np.nanmean(ndvi_result.data):.3f}")
        
        # SAVI
        savi_result = self.calculator.calculate_savi(bands_for_calc)
        if savi_result:
            indices['SAVI'] = savi_result.data
            logger.info(f"    SAVI: mean={np.nanmean(savi_result.data):.3f}")
        
        # EVI
        evi_result = self.calculator.calculate_evi(bands_for_calc)
        if evi_result:
            indices['EVI'] = evi_result.data
            logger.info(f"    EVI: mean={np.nanmean(evi_result.data):.3f}")
        
        # NDWI
        ndwi_result = self.calculator.calculate_ndwi(bands_for_calc)
        if ndwi_result:
            indices['NDWI'] = ndwi_result.data
            logger.info(f"    NDWI: mean={np.nanmean(ndwi_result.data):.3f}")
        
        logger.info(f"  Calculated {len(indices)} indices")
        
        # Save to disk
        logger.info(f"  Saving to disk...")
        output_path, geotiff_paths = self._save_processed_data(
            tile_id,
            acquisition_date,
            bands_dict,
            indices,
            imagery_meta
        )
        
        logger.info(f"  Saved to: {output_path}")
        
        # Save to database with synthetic=false flag
        logger.info(f"  Saving to database...")
        imagery_id = self._save_to_database(
            tile_id,
            acquisition_date,
            cloud_coverage,
            geotiff_paths,
            imagery_meta,
            synthetic=False  # CRITICAL: Mark as real data
        )
        
        logger.info(f"  Database record ID: {imagery_id}")
        
        return {
            'imagery_id': imagery_id,
            'acquisition_date': acquisition_date,
            'tile_id': tile_id,
            'cloud_coverage': cloud_coverage,
            'output_path': str(output_path),
            'geotiff_paths': geotiff_paths,
            'success': True
        }
    
    def _save_processed_data(
        self,
        tile_id: str,
        acquisition_date: str,
        bands: Dict[str, np.ndarray],
        indices: Dict[str, np.ndarray],
        metadata: Dict[str, Any]
    ) -> tuple[Path, Dict[str, str]]:
        """
        Save processed data to disk as GeoTIFF and numpy arrays.
        
        Args:
            tile_id: Tile identifier
            acquisition_date: Acquisition date (YYYY-MM-DD)
            bands: Dictionary of band arrays
            indices: Dictionary of index arrays
            metadata: Imagery metadata
            
        Returns:
            Tuple of (output_directory_path, geotiff_paths_dict)
        """
        # Create output directory for this imagery
        dir_name = f"{tile_id}_{acquisition_date}"
        output_dir = self.output_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get bounds from metadata or use Ludhiana bounds
        bbox = metadata.get('bbox', None)
        
        # If bbox is empty or invalid, use Ludhiana bounds
        if not bbox or len(bbox) < 4:
            bbox = [
                LUDHIANA_BOUNDS['lon_min'],
                LUDHIANA_BOUNDS['lat_min'],
                LUDHIANA_BOUNDS['lon_max'],
                LUDHIANA_BOUNDS['lat_max']
            ]
            logger.debug(f"Using default Ludhiana bounds: {bbox}")
        else:
            logger.debug(f"Using bbox from metadata: {bbox}")
        
        # Get array shape from first band
        first_band = next(iter(bands.values()))
        height, width = first_band.shape
        
        # Create transform
        transform = from_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3],
            width, height
        )
        
        geotiff_paths = {}
        
        # Save bands as GeoTIFF
        for band_id, band_data in bands.items():
            geotiff_path = output_dir / f"{band_id}.tif"
            
            with rasterio.open(
                geotiff_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=band_data.dtype,
                crs='EPSG:4326',
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(band_data, 1)
            
            # Also save as numpy
            np.save(output_dir / f"{band_id}.npy", band_data)
        
        # Save indices as GeoTIFF
        for index_name, index_data in indices.items():
            geotiff_path = output_dir / f"{index_name}.tif"
            
            with rasterio.open(
                geotiff_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=index_data.dtype,
                crs='EPSG:4326',
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(index_data, 1)
            
            # Store path for database
            geotiff_paths[index_name] = str(geotiff_path)
            
            # Also save as numpy
            np.save(output_dir / f"{index_name}.npy", index_data)
        
        # Save metadata
        metadata_enhanced = {
            **metadata,
            'tile_id': tile_id,
            'acquisition_date': acquisition_date,
            'synthetic': False,  # CRITICAL: Mark as real data
            'data_source': 'Sentinel Hub API',
            'processed_at': datetime.now().isoformat(),
            'bands': list(bands.keys()),
            'indices': list(indices.keys()),
            'bbox': bbox,
            'shape': [height, width]
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata_enhanced, f, indent=2)
        
        return output_dir, geotiff_paths
    
    def _save_to_database(
        self,
        tile_id: str,
        acquisition_date: str,
        cloud_coverage: float,
        geotiff_paths: Dict[str, str],
        imagery_meta: Dict[str, Any],
        synthetic: bool = False
    ) -> int:
        """
        Save imagery record to database.
        
        Args:
            tile_id: Tile identifier
            acquisition_date: Acquisition date (YYYY-MM-DD)
            cloud_coverage: Cloud coverage percentage
            geotiff_paths: Dictionary of index name to GeoTIFF path
            imagery_meta: Original imagery metadata
            synthetic: Whether data is synthetic (should be False for real data)
            
        Returns:
            Database record ID
        """
        # Prepare metadata for database
        metadata = {
            **imagery_meta,
            'synthetic': synthetic,
            'data_source': 'Sentinel Hub API' if not synthetic else 'Synthetic Generator',
            'tile_id': tile_id,
            'acquisition_date': acquisition_date,
            'bands': ['B02', 'B03', 'B04', 'B08'],  # Add bands to metadata
            'indices': list(geotiff_paths.keys())  # Add indices from geotiff_paths
        }
        
        # Save to database (mark as real data)
        imagery_id = self.db.save_processed_imagery(
            acquisition_date=acquisition_date,
            tile_id=tile_id,
            cloud_coverage=cloud_coverage,
            geotiff_paths=geotiff_paths,
            metadata=metadata,
            synthetic=False  # This is real satellite data
        )
        
        return imagery_id


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description='Download real Sentinel-2 imagery for Ludhiana region'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=365,
        help='Number of days to look back from today (default: 365)'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=20,
        help='Target number of imagery dates to download (default: 20)'
    )
    parser.add_argument(
        '--cloud-threshold',
        type=float,
        default=20.0,
        help='Maximum cloud coverage percentage (default: 20.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed imagery (default: data/processed)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/agriflux.db',
        help='Path to SQLite database (default: data/agriflux.db)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create downloader
        downloader = RealDataDownloader(
            output_dir=Path(args.output_dir),
            db_path=Path(args.db_path)
        )
        
        # Download time-series
        results = downloader.download_ludhiana_timeseries(
            days_back=args.days_back,
            target_count=args.target_count,
            cloud_threshold=args.cloud_threshold
        )
        
        # Save results summary
        summary_path = Path('logs') / f'download_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'days_back': args.days_back,
                    'target_count': args.target_count,
                    'cloud_threshold': args.cloud_threshold
                },
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results summary saved to: {summary_path}")
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results if not r.get('success', False))
        if failed_count > 0:
            logger.warning(f"Completed with {failed_count} failures")
            sys.exit(1)
        else:
            logger.info("All downloads completed successfully!")
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
