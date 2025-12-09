"""
Data fusion module for integrating sensor data with satellite imagery.

Combines synthetic sensor data with satellite-derived vegetation indices
to create comprehensive datasets for AI model training and analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from src.sensors.synthetic_sensor_generator import SyntheticSensorGenerator, SyntheticSensorReading
from src.data_processing.time_series_builder import TimeSeriesDataset

logger = logging.getLogger(__name__)


@dataclass
class FusedDataPoint:
    """Single fused data point combining satellite and sensor data."""
    timestamp: datetime
    location: Tuple[float, float]
    
    # Satellite-derived indices
    ndvi: float
    savi: float
    evi: float
    ndwi: float
    
    # Sensor measurements
    soil_moisture: float
    temperature: float
    humidity: float
    leaf_wetness: float
    
    # Metadata
    is_synthetic_sensor: bool
    cloud_coverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'latitude': self.location[0],
            'longitude': self.location[1],
            'ndvi': self.ndvi,
            'savi': self.savi,
            'evi': self.evi,
            'ndwi': self.ndwi,
            'soil_moisture': self.soil_moisture,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'leaf_wetness': self.leaf_wetness,
            'is_synthetic_sensor': self.is_synthetic_sensor,
            'cloud_coverage': self.cloud_coverage
        }


class DataFusionEngine:
    """
    Fuse satellite imagery with sensor data for comprehensive analysis.
    
    Integrates multi-spectral satellite data with ground-based (or synthetic)
    sensor measurements to create rich datasets for AI training and monitoring.
    """
    
    def __init__(self, sensor_generator: Optional[SyntheticSensorGenerator] = None):
        """
        Initialize data fusion engine.
        
        Args:
            sensor_generator: Optional SyntheticSensorGenerator for creating synthetic data
        """
        self.sensor_generator = sensor_generator or SyntheticSensorGenerator()
        logger.info("DataFusionEngine initialized")
    
    def fuse_time_series_with_sensors(
        self,
        time_series_dataset: TimeSeriesDataset,
        use_synthetic_sensors: bool = True
    ) -> pd.DataFrame:
        """
        Fuse time-series satellite data with sensor measurements.
        
        Args:
            time_series_dataset: TimeSeriesDataset from satellite imagery
            use_synthetic_sensors: Whether to generate synthetic sensor data
            
        Returns:
            DataFrame with fused data
        """
        logger.info(
            f"Fusing time-series data for location {time_series_dataset.location} "
            f"with {len(time_series_dataset.dates)} observations"
        )
        
        if use_synthetic_sensors:
            # Generate synthetic sensor data correlated with satellite indices
            sensor_data = self._generate_correlated_sensor_data(time_series_dataset)
        else:
            # In production, would load real sensor data here
            raise NotImplementedError("Real sensor data integration not yet implemented")
        
        # Combine into fused data points
        fused_points = []
        
        for i, date in enumerate(time_series_dataset.dates):
            point = FusedDataPoint(
                timestamp=date,
                location=time_series_dataset.location,
                ndvi=time_series_dataset.ndvi_series[i],
                savi=time_series_dataset.savi_series[i],
                evi=time_series_dataset.evi_series[i],
                ndwi=time_series_dataset.ndwi_series[i],
                soil_moisture=sensor_data['soil_moisture'][i].value,
                temperature=sensor_data['temperature'][i].value,
                humidity=sensor_data['humidity'][i].value,
                leaf_wetness=sensor_data['leaf_wetness'][i].value,
                is_synthetic_sensor=True
            )
            fused_points.append(point)
        
        # Convert to DataFrame
        df = pd.DataFrame([p.to_dict() for p in fused_points])
        
        logger.info(f"Created fused dataset with {len(df)} observations")
        
        return df
    
    def _generate_correlated_sensor_data(
        self,
        time_series_dataset: TimeSeriesDataset
    ) -> Dict[str, List[SyntheticSensorReading]]:
        """
        Generate synthetic sensor data correlated with satellite indices.
        
        Args:
            time_series_dataset: TimeSeriesDataset to correlate with
            
        Returns:
            Dictionary of sensor type to list of readings
        """
        n_obs = len(time_series_dataset.dates)
        
        # Generate soil moisture (correlated with NDVI)
        soil_moisture = self.sensor_generator.generate_soil_moisture(
            ndvi_values=time_series_dataset.ndvi_series,
            timestamps=time_series_dataset.dates,
            locations=[time_series_dataset.location] * n_obs
        )
        
        # Generate temperature (seasonal pattern)
        temperature = self.sensor_generator.generate_temperature(
            timestamps=time_series_dataset.dates,
            locations=[time_series_dataset.location] * n_obs,
            location_lat=time_series_dataset.location[0]
        )
        
        # Extract temperature values for humidity generation
        temp_values = np.array([r.value for r in temperature])
        soil_values = np.array([r.value for r in soil_moisture])
        
        # Generate humidity (inversely correlated with temperature)
        humidity = self.sensor_generator.generate_humidity(
            temperature_values=temp_values,
            soil_moisture_values=soil_values,
            timestamps=time_series_dataset.dates,
            locations=[time_series_dataset.location] * n_obs
        )
        
        # Extract humidity values for leaf wetness
        humid_values = np.array([r.value for r in humidity])
        
        # Generate leaf wetness (based on humidity and temperature)
        leaf_wetness = self.sensor_generator.generate_leaf_wetness(
            humidity_values=humid_values,
            temperature_values=temp_values,
            timestamps=time_series_dataset.dates,
            locations=[time_series_dataset.location] * n_obs
        )
        
        return {
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'humidity': humidity,
            'leaf_wetness': leaf_wetness
        }
    
    def create_training_dataset(
        self,
        time_series_datasets: List[TimeSeriesDataset],
        include_sensors: bool = True
    ) -> pd.DataFrame:
        """
        Create comprehensive training dataset from multiple time-series.
        
        Args:
            time_series_datasets: List of TimeSeriesDataset objects
            include_sensors: Whether to include sensor data
            
        Returns:
            Combined DataFrame suitable for model training
        """
        logger.info(f"Creating training dataset from {len(time_series_datasets)} time-series")
        
        all_data = []
        
        for dataset in time_series_datasets:
            if include_sensors:
                df = self.fuse_time_series_with_sensors(dataset)
            else:
                df = dataset.to_dataframe()
            
            all_data.append(df)
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(
            f"Created training dataset with {len(combined_df)} samples "
            f"across {combined_df['timestamp'].nunique()} unique timestamps"
        )
        
        return combined_df
    
    def prepare_cnn_training_data(
        self,
        fused_data: pd.DataFrame,
        target_column: str = 'ndvi',
        classification_thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for CNN training from fused dataset.
        
        Args:
            fused_data: DataFrame with fused satellite and sensor data
            target_column: Column to use for classification
            classification_thresholds: Thresholds for class boundaries
            
        Returns:
            Tuple of (features, labels) for CNN training
        """
        if classification_thresholds is None:
            classification_thresholds = {
                'healthy': 0.7,
                'moderate': 0.5,
                'stressed': 0.3
            }
        
        # Extract features (all indices + sensor data)
        feature_columns = [
            'ndvi', 'savi', 'evi', 'ndwi',
            'soil_moisture', 'temperature', 'humidity', 'leaf_wetness'
        ]
        
        X = fused_data[feature_columns].values
        
        # Create labels based on target column
        target_values = fused_data[target_column].values
        
        labels = np.zeros(len(target_values), dtype=int)
        labels[target_values >= classification_thresholds['healthy']] = 0  # Healthy
        labels[(target_values >= classification_thresholds['moderate']) & 
               (target_values < classification_thresholds['healthy'])] = 1  # Moderate
        labels[(target_values >= classification_thresholds['stressed']) & 
               (target_values < classification_thresholds['moderate'])] = 2  # Stressed
        labels[target_values < classification_thresholds['stressed']] = 3  # Critical
        
        logger.info(
            f"Prepared CNN training data: X shape {X.shape}, "
            f"label distribution: {np.bincount(labels)}"
        )
        
        return X, labels
    
    def prepare_lstm_training_data(
        self,
        fused_data: pd.DataFrame,
        sequence_length: int = 10,
        target_column: str = 'ndvi',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training from fused dataset.
        
        Args:
            fused_data: DataFrame with fused satellite and sensor data
            sequence_length: Length of input sequences
            target_column: Column to predict
            feature_columns: Columns to use as features
            
        Returns:
            Tuple of (X, y) for LSTM training
        """
        if feature_columns is None:
            feature_columns = [
                'ndvi', 'savi', 'evi', 'ndwi',
                'soil_moisture', 'temperature', 'humidity', 'leaf_wetness'
            ]
        
        # Group by location to create sequences
        sequences_X = []
        sequences_y = []
        
        for location in fused_data[['latitude', 'longitude']].drop_duplicates().values:
            lat, lon = location
            
            # Get data for this location
            location_data = fused_data[
                (fused_data['latitude'] == lat) & 
                (fused_data['longitude'] == lon)
            ].sort_values('timestamp')
            
            if len(location_data) < sequence_length + 1:
                continue
            
            # Extract features and target
            features = location_data[feature_columns].values
            target = location_data[target_column].values
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                sequences_X.append(features[i:i+sequence_length])
                sequences_y.append(target[i+sequence_length])
        
        X = np.array(sequences_X)
        y = np.array(sequences_y)
        
        logger.info(
            f"Prepared LSTM training data: X shape {X.shape}, y shape {y.shape}"
        )
        
        return X, y
    
    def calculate_correlation_metrics(
        self,
        fused_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate correlation metrics between satellite and sensor data.
        
        Args:
            fused_data: DataFrame with fused data
            
        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # NDVI - Soil Moisture correlation
        correlations['ndvi_soil_moisture'] = fused_data['ndvi'].corr(
            fused_data['soil_moisture']
        )
        
        # Temperature - Humidity correlation
        correlations['temperature_humidity'] = fused_data['temperature'].corr(
            fused_data['humidity']
        )
        
        # NDWI - Soil Moisture correlation
        correlations['ndwi_soil_moisture'] = fused_data['ndwi'].corr(
            fused_data['soil_moisture']
        )
        
        # Leaf Wetness - Humidity correlation
        correlations['leaf_wetness_humidity'] = fused_data['leaf_wetness'].corr(
            fused_data['humidity']
        )
        
        logger.info("Calculated correlation metrics:")
        for key, value in correlations.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return correlations
