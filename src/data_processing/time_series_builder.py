"""
Time-series dataset builder for multi-date satellite imagery.

Fetches multi-date imagery from Sentinel Hub API and organizes it
into time-series datasets suitable for LSTM training and temporal analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data_processing.sentinel_hub_client import SentinelHubClient
from src.data_processing.vegetation_indices import VegetationIndexCalculator
from src.data_processing.band_processor import BandData

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesDataset:
    """Container for time-series satellite data."""
    dates: List[datetime]
    location: Tuple[float, float]  # (lat, lon)
    ndvi_series: np.ndarray
    savi_series: np.ndarray
    evi_series: np.ndarray
    ndwi_series: np.ndarray
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'date': self.dates,
            'ndvi': self.ndvi_series,
            'savi': self.savi_series,
            'evi': self.evi_series,
            'ndwi': self.ndwi_series,
            'latitude': self.location[0],
            'longitude': self.location[1]
        })
    
    def save(self, filepath: Path):
        """Save time-series dataset to disk."""
        data = {
            'dates': [d.isoformat() for d in self.dates],
            'location': self.location,
            'ndvi_series': self.ndvi_series.tolist(),
            'savi_series': self.savi_series.tolist(),
            'evi_series': self.evi_series.tolist(),
            'ndwi_series': self.ndwi_series.tolist(),
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved time-series dataset to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TimeSeriesDataset':
        """Load time-series dataset from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            dates=[datetime.fromisoformat(d) for d in data['dates']],
            location=tuple(data['location']),
            ndvi_series=np.array(data['ndvi_series']),
            savi_series=np.array(data['savi_series']),
            evi_series=np.array(data['evi_series']),
            ndwi_series=np.array(data['ndwi_series']),
            metadata=data['metadata']
        )


class TimeSeriesBuilder:
    """
    Build time-series datasets from multi-date satellite imagery.
    
    Fetches imagery from Sentinel Hub API across multiple dates and
    organizes it into time-series suitable for temporal analysis and LSTM training.
    """
    
    def __init__(self, sentinel_client: SentinelHubClient):
        """
        Initialize time-series builder.
        
        Args:
            sentinel_client: Configured SentinelHubClient instance
        """
        self.client = sentinel_client
        self.index_calculator = VegetationIndexCalculator()
        logger.info("TimeSeriesBuilder initialized")
    
    def build_time_series(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        temporal_resolution_days: int = 5,
        cloud_threshold: float = 20.0,
        sample_points: Optional[List[Tuple[float, float]]] = None
    ) -> List[TimeSeriesDataset]:
        """
        Build time-series datasets from multi-date imagery.
        
        Args:
            geometry: GeoJSON geometry defining the area of interest
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            temporal_resolution_days: Desired days between observations
            cloud_threshold: Maximum cloud coverage percentage
            sample_points: Optional list of (lat, lon) points to extract time-series for
            
        Returns:
            List of TimeSeriesDataset objects
        """
        logger.info(
            f"Building time-series from {start_date} to {end_date} "
            f"with {temporal_resolution_days}-day resolution"
        )
        
        # Query available imagery
        imagery_list = self.client.query_sentinel_imagery(
            geometry=geometry,
            date_range=(start_date, end_date),
            cloud_threshold=cloud_threshold,
            max_results=100
        )
        
        if not imagery_list:
            logger.warning("No imagery found for specified date range and cloud threshold")
            return []
        
        logger.info(f"Found {len(imagery_list)} available images")
        
        # Filter to desired temporal resolution
        selected_dates = self._select_temporal_samples(
            imagery_list,
            temporal_resolution_days
        )
        
        logger.info(f"Selected {len(selected_dates)} dates for time-series")
        
        # Download and process imagery for each date
        time_series_data = []
        
        for date_info in selected_dates:
            try:
                # Download bands
                bands = self.client.download_multispectral_bands(
                    geometry=geometry,
                    acquisition_date=date_info['acquisition_date'].split('T')[0]
                )
                
                # Calculate vegetation indices
                indices = self._calculate_all_indices(bands)
                
                time_series_data.append({
                    'date': datetime.fromisoformat(date_info['acquisition_date'].split('T')[0]),
                    'indices': indices,
                    'cloud_coverage': date_info['cloud_coverage'],
                    'metadata': date_info
                })
                
                logger.info(
                    f"Processed {date_info['acquisition_date'].split('T')[0]} "
                    f"(cloud: {date_info['cloud_coverage']:.1f}%)"
                )
                
            except Exception as e:
                logger.error(f"Failed to process {date_info['acquisition_date']}: {e}")
                continue
        
        if not time_series_data:
            logger.warning("No imagery successfully processed")
            return []
        
        # Extract time-series for sample points or grid
        if sample_points is None:
            # Create a grid of sample points
            sample_points = self._create_sample_grid(geometry, grid_size=10)
        
        datasets = self._extract_point_time_series(time_series_data, sample_points)
        
        logger.info(f"Created {len(datasets)} time-series datasets")
        
        return datasets
    
    def _select_temporal_samples(
        self,
        imagery_list: List[Dict[str, Any]],
        temporal_resolution_days: int
    ) -> List[Dict[str, Any]]:
        """
        Select imagery at desired temporal resolution.
        
        Args:
            imagery_list: List of available imagery metadata
            temporal_resolution_days: Desired days between observations
            
        Returns:
            List of selected imagery metadata
        """
        if not imagery_list:
            return []
        
        # Sort by date
        sorted_imagery = sorted(
            imagery_list,
            key=lambda x: x['acquisition_date']
        )
        
        selected = [sorted_imagery[0]]
        last_date = datetime.fromisoformat(sorted_imagery[0]['acquisition_date'].split('T')[0])
        
        for img in sorted_imagery[1:]:
            current_date = datetime.fromisoformat(img['acquisition_date'].split('T')[0])
            days_diff = (current_date - last_date).days
            
            if days_diff >= temporal_resolution_days:
                selected.append(img)
                last_date = current_date
        
        return selected
    
    def _calculate_all_indices(
        self,
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all vegetation indices from bands.
        
        Args:
            bands: Dictionary of band arrays
            
        Returns:
            Dictionary of index arrays
        """
        # Convert to BandData format
        band_data = {}
        for band_name, data in bands.items():
            band_data[band_name] = BandData(
                band_id=band_name,
                data=data,
                transform=None,
                crs='EPSG:4326',
                nodata_value=None,
                resolution=10.0,
                shape=data.shape,
                dtype=data.dtype
            )
        
        # Calculate indices
        indices = {}
        
        try:
            ndvi_result = self.index_calculator.calculate_ndvi(band_data)
            indices['ndvi'] = ndvi_result.data if ndvi_result else np.zeros_like(bands['B04'])
        except Exception as e:
            logger.warning(f"Failed to calculate NDVI: {e}")
            indices['ndvi'] = np.zeros_like(bands['B04'])
        
        try:
            savi_result = self.index_calculator.calculate_savi(band_data)
            indices['savi'] = savi_result.data if savi_result else np.zeros_like(bands['B04'])
        except Exception as e:
            logger.warning(f"Failed to calculate SAVI: {e}")
            indices['savi'] = np.zeros_like(bands['B04'])
        
        try:
            evi_result = self.index_calculator.calculate_evi(band_data)
            indices['evi'] = evi_result.data if evi_result else np.zeros_like(bands['B04'])
        except Exception as e:
            logger.warning(f"Failed to calculate EVI: {e}")
            indices['evi'] = np.zeros_like(bands['B04'])
        
        try:
            ndwi_result = self.index_calculator.calculate_ndwi(band_data)
            indices['ndwi'] = ndwi_result.data if ndwi_result else np.zeros_like(bands['B04'])
        except Exception as e:
            logger.warning(f"Failed to calculate NDWI: {e}")
            indices['ndwi'] = np.zeros_like(bands['B04'])
        
        return indices
    
    def _create_sample_grid(
        self,
        geometry: Dict[str, Any],
        grid_size: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Create a grid of sample points within geometry.
        
        Args:
            geometry: GeoJSON geometry
            grid_size: Number of points per dimension
            
        Returns:
            List of (lat, lon) tuples
        """
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            # Create grid
            lon_grid = np.linspace(min_lon, max_lon, grid_size)
            lat_grid = np.linspace(min_lat, max_lat, grid_size)
            
            points = []
            for lat in lat_grid:
                for lon in lon_grid:
                    points.append((lat, lon))
            
            return points
        else:
            raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    def _extract_point_time_series(
        self,
        time_series_data: List[Dict[str, Any]],
        sample_points: List[Tuple[float, float]]
    ) -> List[TimeSeriesDataset]:
        """
        Extract time-series for specific points.
        
        Args:
            time_series_data: List of processed imagery with indices
            sample_points: List of (lat, lon) points
            
        Returns:
            List of TimeSeriesDataset objects
        """
        datasets = []
        
        for point in sample_points:
            lat, lon = point
            
            # Extract values for this point across all dates
            dates = []
            ndvi_series = []
            savi_series = []
            evi_series = []
            ndwi_series = []
            
            for data in time_series_data:
                # For simplicity, use center pixel
                # In production, would convert lat/lon to pixel coordinates
                indices = data['indices']
                h, w = indices['ndvi'].shape
                i, j = h // 2, w // 2
                
                dates.append(data['date'])
                ndvi_series.append(float(indices['ndvi'][i, j]))
                savi_series.append(float(indices['savi'][i, j]))
                evi_series.append(float(indices['evi'][i, j]))
                ndwi_series.append(float(indices['ndwi'][i, j]))
            
            if dates:
                dataset = TimeSeriesDataset(
                    dates=dates,
                    location=point,
                    ndvi_series=np.array(ndvi_series),
                    savi_series=np.array(savi_series),
                    evi_series=np.array(evi_series),
                    ndwi_series=np.array(ndwi_series),
                    metadata={
                        'n_observations': len(dates),
                        'date_range': (dates[0].isoformat(), dates[-1].isoformat())
                    }
                )
                datasets.append(dataset)
        
        return datasets
    
    def build_lstm_training_data(
        self,
        time_series_datasets: List[TimeSeriesDataset],
        sequence_length: int = 10,
        target_index: str = 'ndvi'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training data for LSTM from time-series datasets.
        
        Args:
            time_series_datasets: List of TimeSeriesDataset objects
            sequence_length: Length of input sequences
            target_index: Which index to predict ('ndvi', 'savi', 'evi', 'ndwi')
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        X_sequences = []
        y_targets = []
        
        for dataset in time_series_datasets:
            # Get target series
            if target_index == 'ndvi':
                series = dataset.ndvi_series
            elif target_index == 'savi':
                series = dataset.savi_series
            elif target_index == 'evi':
                series = dataset.evi_series
            elif target_index == 'ndwi':
                series = dataset.ndwi_series
            else:
                raise ValueError(f"Unknown target index: {target_index}")
            
            # Create sequences
            for i in range(len(series) - sequence_length):
                X_sequences.append(series[i:i+sequence_length])
                y_targets.append(series[i+sequence_length])
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        # Reshape for LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        logger.info(
            f"Built LSTM training data: X shape {X.shape}, y shape {y.shape}"
        )
        
        return X, y
