"""
Tests for LSTM temporal trend analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from src.ai_models.temporal_lstm import TemporalLSTM, LSTMConfig, AnomalyDetector, PredictionResult
from src.ai_models.training_pipeline import LSTMTrainingPipeline, create_sample_training_data


class TestLSTMConfig:
    """Test LSTM configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMConfig()
        
        assert config.sequence_length == 30
        assert config.lstm_units == 64
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.validation_split == 0.2
        assert config.early_stopping_patience == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LSTMConfig(
            sequence_length=20,
            lstm_units=32,
            dropout_rate=0.3,
            learning_rate=0.01
        )
        
        assert config.sequence_length == 20
        assert config.lstm_units == 32
        assert config.dropout_rate == 0.3
        assert config.learning_rate == 0.01


class TestTemporalLSTM:
    """Test LSTM model for temporal analysis."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range('2022-01-01', periods=100, freq='D')
        
        # Generate synthetic NDVI data with trend and seasonality
        t = np.arange(100)
        ndvi = 0.6 + 0.2 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.05, 100)
        ndvi = np.clip(ndvi, 0, 1)
        
        # Add environmental features
        temp = 20 + 5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 2, 100)
        humidity = 60 + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 5, 100)
        soil_moisture = 0.3 + 0.1 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.02, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'ndvi': ndvi,
            'temperature': temp,
            'humidity': humidity,
            'soil_moisture': soil_moisture
        }).set_index('timestamp')
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model with test configuration."""
        config = LSTMConfig(
            sequence_length=10,
            lstm_units=16,
            epochs=5,
            batch_size=8
        )
        return TemporalLSTM(config)
    
    def test_model_initialization(self, lstm_model):
        """Test model initialization."""
        assert lstm_model.config.sequence_length == 10
        assert lstm_model.config.lstm_units == 16
        assert lstm_model.model is None
        assert not lstm_model.is_trained
    
    def test_prepare_training_data(self, lstm_model, sample_data):
        """Test training data preparation."""
        X, y = lstm_model.prepare_training_data(sample_data)
        
        # Check shapes
        expected_sequences = len(sample_data) - lstm_model.config.sequence_length
        assert X.shape[0] == expected_sequences
        assert X.shape[1] == lstm_model.config.sequence_length
        assert X.shape[2] == len(sample_data.columns)  # Number of features
        assert y.shape[0] == expected_sequences
        
        # Check that data is scaled
        assert X.min() >= 0
        assert X.max() <= 1
    
    def test_model_training(self, lstm_model, sample_data):
        """Test model training."""
        X, y = lstm_model.prepare_training_data(sample_data)
        
        # Train model
        history = lstm_model.train(X, y)
        
        # Check training completed
        assert lstm_model.is_trained
        assert lstm_model.model is not None
        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) > 0
    
    def test_model_prediction(self, lstm_model, sample_data):
        """Test model prediction."""
        X, y = lstm_model.prepare_training_data(sample_data)
        
        # Train model
        lstm_model.train(X, y)
        
        # Make predictions
        result = lstm_model.predict(X[:10])
        
        # Check prediction result
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == 10
        assert result.confidence_intervals is not None
        assert len(result.anomaly_scores) == 10
        assert result.trend_direction in ['increasing', 'decreasing', 'stable']
        assert 0 <= result.trend_strength <= 1
    
    def test_model_evaluation(self, lstm_model, sample_data):
        """Test model evaluation."""
        X, y = lstm_model.prepare_training_data(sample_data)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train and evaluate
        lstm_model.train(X_train, y_train)
        metrics = lstm_model.evaluate(X_test, y_test)
        
        # Check metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_model_save_load(self, lstm_model, sample_data):
        """Test model saving and loading."""
        X, y = lstm_model.prepare_training_data(sample_data)
        lstm_model.train(X, y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Save model
            lstm_model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new model and load
            new_model = TemporalLSTM(lstm_model.config)
            new_model.scaler = lstm_model.scaler  # Copy scaler
            new_model.load_model(model_path)
            
            # Test loaded model
            assert new_model.is_trained
            predictions1 = lstm_model.predict(X[:5], return_confidence=False)
            predictions2 = new_model.predict(X[:5], return_confidence=False)
            
            # Predictions should be very similar
            np.testing.assert_allclose(
                predictions1.predictions, 
                predictions2.predictions, 
                rtol=1e-5
            )


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series for anomaly detection."""
        np.random.seed(42)
        # Normal data with some outliers
        normal_data = np.random.normal(0.6, 0.1, 100)
        # Add some anomalies
        normal_data[20] = 0.9  # High anomaly
        normal_data[50] = 0.2  # Low anomaly
        normal_data[80] = 0.95  # High anomaly
        
        return pd.Series(normal_data)
    
    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = AnomalyDetector(threshold_std=2.5)
        assert detector.threshold_std == 2.5
        assert detector.baseline_stats == {}
    
    def test_fit_baseline(self, sample_series):
        """Test fitting baseline statistics."""
        detector = AnomalyDetector()
        detector.fit_baseline(sample_series)
        
        # Check baseline statistics
        assert 'mean' in detector.baseline_stats
        assert 'std' in detector.baseline_stats
        assert 'median' in detector.baseline_stats
        assert 'q25' in detector.baseline_stats
        assert 'q75' in detector.baseline_stats
        
        # Check values are reasonable
        assert 0.4 < detector.baseline_stats['mean'] < 0.8
        assert detector.baseline_stats['std'] > 0
    
    def test_statistical_anomaly_detection(self, sample_series):
        """Test statistical anomaly detection."""
        detector = AnomalyDetector(threshold_std=2.0)
        detector.fit_baseline(sample_series)
        
        anomalies = detector.detect_anomalies(sample_series.values, method='statistical')
        
        # Should detect some anomalies
        assert anomalies.sum() > 0
        assert anomalies.sum() < len(sample_series) * 0.1  # Less than 10%
        
        # Check that known anomalies are detected
        assert anomalies[20] or anomalies[50] or anomalies[80]  # At least one should be detected
    
    def test_iqr_anomaly_detection(self, sample_series):
        """Test IQR-based anomaly detection."""
        detector = AnomalyDetector()
        detector.fit_baseline(sample_series)
        
        anomalies = detector.detect_anomalies(sample_series.values, method='iqr')
        
        # Should detect some anomalies
        assert anomalies.sum() > 0
        assert anomalies.sum() < len(sample_series) * 0.1  # Less than 10%


class TestLSTMTrainingPipeline:
    """Test LSTM training pipeline."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return create_sample_training_data(n_zones=2, n_days=100)
    
    @pytest.fixture
    def training_pipeline(self):
        """Create training pipeline with test configuration."""
        config = LSTMConfig(
            sequence_length=10,
            lstm_units=16,
            epochs=3,
            batch_size=8
        )
        return LSTMTrainingPipeline(config)
    
    def test_pipeline_initialization(self, training_pipeline):
        """Test pipeline initialization."""
        assert training_pipeline.config.sequence_length == 10
        assert training_pipeline.lstm_model is not None
        assert training_pipeline.anomaly_detector is not None
    
    def test_prepare_training_dataset(self, training_pipeline, sample_training_data):
        """Test training dataset preparation."""
        # Split data into index and environmental parts
        index_cols = ['zone_id', 'timestamp', 'ndvi', 'savi', 'evi', 'ndwi']
        env_cols = ['zone_id', 'timestamp', 'temperature', 'humidity', 'soil_moisture', 
                   'precipitation', 'wind_speed', 'solar_radiation']
        
        index_data = sample_training_data[index_cols].melt(
            id_vars=['zone_id', 'timestamp'],
            var_name='index_type',
            value_name='mean_value'
        )
        
        env_data = sample_training_data[env_cols]
        
        # Prepare dataset
        combined = training_pipeline.prepare_training_dataset(index_data, env_data)
        
        # Check result
        assert len(combined) > 0
        assert 'zone_id' in combined.columns
        assert 'timestamp' in combined.columns
        assert 'ndvi' in combined.columns
        assert 'temperature' in combined.columns
    
    def test_create_zone_datasets(self, training_pipeline, sample_training_data):
        """Test zone dataset creation."""
        zone_datasets = training_pipeline.create_zone_datasets(
            sample_training_data, 
            min_sequence_length=50
        )
        
        # Check results
        assert len(zone_datasets) > 0
        for zone_id, zone_data in zone_datasets.items():
            assert isinstance(zone_data, pd.DataFrame)
            assert len(zone_data) >= 50
            assert 'ndvi' in zone_data.columns
    
    def test_train_model(self, training_pipeline, sample_training_data):
        """Test model training."""
        # Use single zone data
        zone_data = sample_training_data[sample_training_data['zone_id'] == 'zone_1'].copy()
        zone_data.set_index('timestamp', inplace=True)
        zone_data.drop('zone_id', axis=1, inplace=True)
        
        # Train model
        results = training_pipeline.train_model(zone_data, validation_split=0.3)
        
        # Check results
        assert 'history' in results
        assert 'validation_metrics' in results
        assert 'training_sequences' in results
        assert 'validation_sequences' in results
        
        assert training_pipeline.lstm_model.is_trained
        assert results['training_sequences'] > 0
        assert results['validation_sequences'] > 0
    
    def test_save_load_pipeline(self, training_pipeline, sample_training_data):
        """Test pipeline saving and loading."""
        # Train model first
        zone_data = sample_training_data[sample_training_data['zone_id'] == 'zone_1'].copy()
        zone_data.set_index('timestamp', inplace=True)
        zone_data.drop('zone_id', axis=1, inplace=True)
        
        training_pipeline.train_model(zone_data, validation_split=0.3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update save path
            training_pipeline.model_save_path = temp_dir
            
            # Save pipeline
            paths = training_pipeline.save_pipeline(suffix='test')
            
            # Check files exist
            assert os.path.exists(paths['model_path'])
            assert os.path.exists(paths['scaler_path'])
            assert os.path.exists(paths['anomaly_path'])
            assert os.path.exists(paths['config_path'])
            
            # Create new pipeline and load
            new_pipeline = LSTMTrainingPipeline()
            new_pipeline.model_save_path = temp_dir
            new_pipeline.load_pipeline('test')
            
            # Check loaded pipeline
            assert new_pipeline.lstm_model.is_trained


class TestSampleDataGeneration:
    """Test sample data generation utilities."""
    
    def test_create_sample_training_data(self):
        """Test sample training data creation."""
        data = create_sample_training_data(n_zones=3, n_days=100)
        
        # Check structure
        assert len(data) == 3 * 100  # 3 zones * 100 days
        assert 'zone_id' in data.columns
        assert 'timestamp' in data.columns
        assert 'ndvi' in data.columns
        assert 'temperature' in data.columns
        
        # Check data ranges
        assert data['ndvi'].min() >= 0
        assert data['ndvi'].max() <= 1
        assert data['temperature'].min() > -10
        assert data['temperature'].max() < 50
        assert data['humidity'].min() >= 0
        assert data['humidity'].max() <= 100
        
        # Check zones
        unique_zones = data['zone_id'].unique()
        assert len(unique_zones) == 3
        assert all(zone.startswith('zone_') for zone in unique_zones)
    
    def test_sample_data_temporal_structure(self):
        """Test temporal structure of sample data."""
        data = create_sample_training_data(n_zones=2, n_days=50)
        
        # Check each zone has correct number of records
        for zone_id in data['zone_id'].unique():
            zone_data = data[data['zone_id'] == zone_id]
            assert len(zone_data) == 50
            
            # Check timestamps are sequential
            timestamps = zone_data['timestamp'].sort_values()
            time_diffs = timestamps.diff().dropna()
            assert all(diff.days == 1 for diff in time_diffs)  # Daily data