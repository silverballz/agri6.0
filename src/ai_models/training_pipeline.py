"""
Training pipeline for LSTM temporal trend analysis.

This module provides utilities for preparing data, training models,
and evaluating performance for vegetation index time series analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import joblib

from .temporal_lstm import TemporalLSTM, LSTMConfig, AnomalyDetector
from ..models.index_timeseries import IndexTimeSeries
from ..database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class LSTMTrainingPipeline:
    """
    Complete training pipeline for LSTM temporal analysis.
    
    Handles data preparation, model training, validation, and persistence.
    """
    
    def __init__(self, 
                 config: LSTMConfig = None,
                 model_save_path: str = "models/lstm_temporal"):
        """
        Initialize training pipeline.
        
        Args:
            config: LSTM configuration
            model_save_path: Path to save trained models
        """
        self.config = config or LSTMConfig()
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.lstm_model = TemporalLSTM(self.config)
        self.anomaly_detector = AnomalyDetector()
        
    def load_training_data(self, 
                          zone_ids: List[str] = None,
                          start_date: datetime = None,
                          end_date: datetime = None,
                          index_types: List[str] = None) -> pd.DataFrame:
        """
        Load training data from database.
        
        Args:
            zone_ids: List of monitoring zone IDs
            start_date: Start date for data
            end_date: End date for data
            index_types: Types of vegetation indices to include
            
        Returns:
            DataFrame with time series data
        """
        if index_types is None:
            index_types = ['ndvi', 'savi', 'evi', 'ndwi']
        
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
        
        db = DatabaseConnection()
        
        # Load index time series data
        query = """
        SELECT 
            its.zone_id,
            its.timestamp,
            its.index_type,
            its.mean_value,
            its.std_deviation,
            its.quality_score,
            mz.crop_type
        FROM index_timeseries its
        JOIN monitoring_zones mz ON its.zone_id = mz.id
        WHERE its.timestamp BETWEEN %s AND %s
        """
        
        params = [start_date, end_date]
        
        if zone_ids:
            query += " AND its.zone_id = ANY(%s)"
            params.append(zone_ids)
        
        if index_types:
            query += " AND its.index_type = ANY(%s)"
            params.append(index_types)
        
        query += " ORDER BY its.zone_id, its.timestamp"
        
        with db.get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
        
        logger.info(f"Loaded {len(df)} time series records")
        return df
    
    def load_environmental_data(self,
                               zone_ids: List[str] = None,
                               start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
        """
        Load environmental sensor data for training.
        
        Args:
            zone_ids: List of monitoring zone IDs
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with environmental data
        """
        db = DatabaseConnection()
        
        query = """
        SELECT 
            zone_id,
            timestamp,
            temperature,
            humidity,
            soil_moisture,
            precipitation,
            wind_speed,
            solar_radiation
        FROM sensor_readings
        WHERE timestamp BETWEEN %s AND %s
        """
        
        params = [start_date, end_date]
        
        if zone_ids:
            query += " AND zone_id = ANY(%s)"
            params.append(zone_ids)
        
        query += " ORDER BY zone_id, timestamp"
        
        with db.get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
        
        logger.info(f"Loaded {len(df)} environmental records")
        return df
    
    def prepare_training_dataset(self,
                                index_data: pd.DataFrame,
                                environmental_data: pd.DataFrame,
                                target_index: str = 'ndvi') -> pd.DataFrame:
        """
        Prepare combined dataset for LSTM training.
        
        Args:
            index_data: Vegetation index time series data
            environmental_data: Environmental sensor data
            target_index: Target vegetation index to predict
            
        Returns:
            Combined DataFrame ready for training
        """
        # Pivot index data to have columns for each index type
        index_pivot = index_data.pivot_table(
            index=['zone_id', 'timestamp'],
            columns='index_type',
            values='mean_value',
            aggfunc='mean'
        ).reset_index()
        
        # Merge with environmental data
        combined = pd.merge(
            index_pivot,
            environmental_data,
            on=['zone_id', 'timestamp'],
            how='inner'
        )
        
        # Sort by zone and timestamp
        combined = combined.sort_values(['zone_id', 'timestamp'])
        
        # Forward fill missing values within each zone
        combined = combined.groupby('zone_id').ffill()
        
        # Drop rows with remaining NaN values
        combined = combined.dropna()
        
        logger.info(f"Prepared dataset with {len(combined)} records and {len(combined.columns)} features")
        return combined
    
    def create_zone_datasets(self, 
                           combined_data: pd.DataFrame,
                           min_sequence_length: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Create separate datasets for each monitoring zone.
        
        Args:
            combined_data: Combined time series data
            min_sequence_length: Minimum sequence length for training
            
        Returns:
            Dictionary mapping zone_id to DataFrame
        """
        zone_datasets = {}
        
        for zone_id in combined_data['zone_id'].unique():
            zone_data = combined_data[combined_data['zone_id'] == zone_id].copy()
            zone_data = zone_data.sort_values('timestamp')
            
            if len(zone_data) >= min_sequence_length:
                # Set timestamp as index
                zone_data.set_index('timestamp', inplace=True)
                zone_data.drop('zone_id', axis=1, inplace=True)
                
                zone_datasets[zone_id] = zone_data
                logger.info(f"Zone {zone_id}: {len(zone_data)} records")
            else:
                logger.warning(f"Zone {zone_id} has insufficient data ({len(zone_data)} records)")
        
        return zone_datasets
    
    def train_model(self,
                   training_data: pd.DataFrame,
                   target_column: str = 'ndvi',
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train LSTM model on prepared data.
        
        Args:
            training_data: Prepared training data
            target_column: Target column to predict
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        # Prepare sequences
        X, y = self.lstm_model.prepare_training_data(
            training_data,
            target_column=target_column
        )
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} sequences")
        logger.info(f"Validation set: {len(X_val)} sequences")
        
        # Train model
        history = self.lstm_model.train(
            X_train, y_train,
            validation_data=(X_val, y_val)
        )
        
        # Evaluate on validation set
        val_metrics = self.lstm_model.evaluate(X_val, y_val)
        
        # Fit anomaly detector on training data
        train_predictions = self.lstm_model.predict(X_train, return_confidence=False)
        self.anomaly_detector.fit_baseline(pd.Series(train_predictions.predictions))
        
        results = {
            'history': history,
            'validation_metrics': val_metrics,
            'training_sequences': len(X_train),
            'validation_sequences': len(X_val)
        }
        
        return results
    
    def train_multi_zone_model(self,
                              zone_datasets: Dict[str, pd.DataFrame],
                              target_column: str = 'ndvi') -> Dict[str, Any]:
        """
        Train model on data from multiple zones.
        
        Args:
            zone_datasets: Dictionary of zone datasets
            target_column: Target column to predict
            
        Returns:
            Training results
        """
        # Combine all zone data
        all_data = []
        for zone_id, zone_data in zone_datasets.items():
            zone_data_copy = zone_data.copy()
            zone_data_copy['zone_id'] = zone_id
            all_data.append(zone_data_copy)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.drop('zone_id', axis=1)
        
        logger.info(f"Training multi-zone model with {len(combined_data)} total records")
        
        return self.train_model(combined_data, target_column)
    
    def save_pipeline(self, suffix: str = None):
        """
        Save trained model and components.
        
        Args:
            suffix: Optional suffix for model files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = suffix or timestamp
        
        # Save LSTM model
        model_path = self.model_save_path / f"lstm_model_{suffix}.h5"
        self.lstm_model.save_model(str(model_path))
        
        # Save scaler
        scaler_path = self.model_save_path / f"scaler_{suffix}.pkl"
        joblib.dump(self.lstm_model.scaler, scaler_path)
        
        # Save anomaly detector
        anomaly_path = self.model_save_path / f"anomaly_detector_{suffix}.pkl"
        joblib.dump(self.anomaly_detector, anomaly_path)
        
        # Save configuration
        config_path = self.model_save_path / f"config_{suffix}.pkl"
        joblib.dump(self.config, config_path)
        
        logger.info(f"Pipeline saved with suffix: {suffix}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'anomaly_path': str(anomaly_path),
            'config_path': str(config_path)
        }
    
    def load_pipeline(self, suffix: str):
        """
        Load trained model and components.
        
        Args:
            suffix: Suffix of model files to load
        """
        # Load LSTM model
        model_path = self.model_save_path / f"lstm_model_{suffix}.h5"
        self.lstm_model.load_model(str(model_path))
        
        # Load scaler
        scaler_path = self.model_save_path / f"scaler_{suffix}.pkl"
        self.lstm_model.scaler = joblib.load(scaler_path)
        
        # Load anomaly detector
        anomaly_path = self.model_save_path / f"anomaly_detector_{suffix}.pkl"
        self.anomaly_detector = joblib.load(anomaly_path)
        
        # Load configuration
        config_path = self.model_save_path / f"config_{suffix}.pkl"
        self.config = joblib.load(config_path)
        
        logger.info(f"Pipeline loaded with suffix: {suffix}")
    
    def validate_model_performance(self,
                                  test_data: pd.DataFrame,
                                  target_column: str = 'ndvi') -> Dict[str, Any]:
        """
        Validate model performance on test data.
        
        Args:
            test_data: Test dataset
            target_column: Target column to predict
            
        Returns:
            Validation metrics and results
        """
        # Prepare test sequences
        X_test, y_test = self.lstm_model.prepare_training_data(
            test_data,
            target_column=target_column
        )
        
        # Make predictions
        predictions = self.lstm_model.predict(X_test)
        
        # Evaluate metrics
        metrics = self.lstm_model.evaluate(X_test, y_test)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(predictions.predictions)
        
        validation_results = {
            'metrics': metrics,
            'predictions': predictions,
            'anomalies': anomalies,
            'anomaly_rate': np.mean(anomalies),
            'test_sequences': len(X_test)
        }
        
        logger.info(f"Model validation completed. Anomaly rate: {validation_results['anomaly_rate']:.3f}")
        
        return validation_results


def create_sample_training_data(n_zones: int = 3, 
                               n_days: int = 365,
                               save_to_db: bool = False) -> pd.DataFrame:
    """
    Create sample training data for testing the LSTM pipeline.
    
    Args:
        n_zones: Number of monitoring zones
        n_days: Number of days of data
        save_to_db: Whether to save data to database
        
    Returns:
        Sample training data DataFrame
    """
    np.random.seed(42)
    
    # Generate time series
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    data = []
    for zone_id in range(1, n_zones + 1):
        # Generate base NDVI trend with seasonality
        t = np.arange(n_days)
        seasonal = 0.3 * np.sin(2 * np.pi * t / 365)  # Annual cycle
        trend = 0.1 * np.sin(2 * np.pi * t / 30)      # Monthly variation
        noise = np.random.normal(0, 0.05, n_days)
        
        base_ndvi = 0.6 + seasonal + trend + noise
        base_ndvi = np.clip(base_ndvi, 0, 1)
        
        # Generate other indices correlated with NDVI
        savi = base_ndvi * 0.8 + np.random.normal(0, 0.02, n_days)
        evi = base_ndvi * 1.2 + np.random.normal(0, 0.03, n_days)
        ndwi = 0.4 - base_ndvi * 0.3 + np.random.normal(0, 0.02, n_days)
        
        # Generate environmental data
        temp_base = 20 + 10 * np.sin(2 * np.pi * t / 365)
        temperature = temp_base + np.random.normal(0, 2, n_days)
        
        humidity = 60 + 20 * np.sin(2 * np.pi * t / 365 + np.pi/2) + np.random.normal(0, 5, n_days)
        humidity = np.clip(humidity, 0, 100)
        
        soil_moisture = 0.3 + 0.2 * np.sin(2 * np.pi * t / 365 + np.pi) + np.random.normal(0, 0.05, n_days)
        soil_moisture = np.clip(soil_moisture, 0, 1)
        
        for i, date in enumerate(dates):
            data.append({
                'zone_id': f'zone_{zone_id}',
                'timestamp': date,
                'ndvi': base_ndvi[i],
                'savi': savi[i],
                'evi': evi[i],
                'ndwi': ndwi[i],
                'temperature': temperature[i],
                'humidity': humidity[i],
                'soil_moisture': soil_moisture[i],
                'precipitation': max(0, np.random.normal(2, 3)),
                'wind_speed': max(0, np.random.normal(5, 2)),
                'solar_radiation': max(0, np.random.normal(200, 50))
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated sample data: {len(df)} records for {n_zones} zones")
    
    return df