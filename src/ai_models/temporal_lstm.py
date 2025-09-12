"""
LSTM model for temporal trend analysis of vegetation indices.

This module implements LSTM neural networks for analyzing time series data
of vegetation indices and environmental conditions to detect trends and anomalies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    sequence_length: int = 30  # Number of time steps to look back
    lstm_units: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10


@dataclass
class PredictionResult:
    """Result of LSTM prediction."""
    predictions: np.ndarray
    confidence_intervals: np.ndarray
    anomaly_scores: np.ndarray
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1 scale


class TemporalLSTM:
    """
    LSTM model for temporal trend analysis of vegetation indices.
    
    This class implements bidirectional LSTM with attention mechanism
    for analyzing vegetation index time series and predicting trends.
    """
    
    def __init__(self, config: LSTMConfig = None):
        """Initialize LSTM model with configuration."""
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_names = []
        
    def _create_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Create LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Bidirectional LSTM layers
            layers.Bidirectional(
                layers.LSTM(
                    self.config.lstm_units,
                    return_sequences=True,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                ),
                input_shape=input_shape
            ),
            layers.Bidirectional(
                layers.LSTM(
                    self.config.lstm_units // 2,
                    return_sequences=False,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                )
            ),
            
            # Dense layers for prediction
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')  # Single output for next value
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data: Time series data array
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(data)):
            X.append(data[i - self.config.sequence_length:i])
            y.append(data[i, 0])  # Predict first feature (main vegetation index)
            
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, 
                            time_series_data: pd.DataFrame,
                            target_column: str = 'ndvi',
                            feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for LSTM training.
        
        Args:
            time_series_data: DataFrame with time series data
            target_column: Column to predict
            feature_columns: Additional feature columns to include
            
        Returns:
            Tuple of (X, y) training data
        """
        if feature_columns is None:
            feature_columns = [target_column, 'temperature', 'humidity', 'soil_moisture']
        
        # Ensure target column is first
        if target_column not in feature_columns:
            feature_columns = [target_column] + feature_columns
        elif feature_columns[0] != target_column:
            feature_columns = [target_column] + [col for col in feature_columns if col != target_column]
        
        self.feature_names = feature_columns
        
        # Select and sort data
        data = time_series_data[feature_columns].sort_index()
        
        # Handle missing values
        data = data.ffill().bfill()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data.values)
        
        # Create sequences
        X, y = self._prepare_sequences(scaled_data)
        
        logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Input sequences
            y: Target values
            validation_data: Optional validation data
            
        Returns:
            Training history
        """
        # Create model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self._create_model(input_shape)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split if validation_data is None else 0,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("LSTM model training completed")
        
        return history.history
    
    def predict(self, 
                X: np.ndarray,
                return_confidence: bool = True) -> PredictionResult:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input sequences
            return_confidence: Whether to calculate confidence intervals
            
        Returns:
            PredictionResult with predictions and metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate confidence intervals using Monte Carlo dropout
        confidence_intervals = None
        if return_confidence:
            confidence_intervals = self._calculate_confidence_intervals(X)
        
        # Calculate anomaly scores
        anomaly_scores = self._calculate_anomaly_scores(X, predictions)
        
        # Determine trend
        trend_direction, trend_strength = self._analyze_trend(predictions)
        
        return PredictionResult(
            predictions=predictions.flatten(),
            confidence_intervals=confidence_intervals,
            anomaly_scores=anomaly_scores,
            trend_direction=trend_direction,
            trend_strength=trend_strength
        )
    
    def _calculate_confidence_intervals(self, 
                                     X: np.ndarray, 
                                     n_samples: int = 100) -> np.ndarray:
        """
        Calculate confidence intervals using Monte Carlo dropout.
        
        Args:
            X: Input sequences
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Confidence intervals array
        """
        # Enable dropout during inference
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate percentiles
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)
        
        return np.column_stack([lower.flatten(), upper.flatten()])
    
    def _calculate_anomaly_scores(self, 
                                X: np.ndarray, 
                                predictions: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on prediction errors.
        
        Args:
            X: Input sequences
            predictions: Model predictions
            
        Returns:
            Anomaly scores array
        """
        # Use last value of each sequence as ground truth
        actual_values = X[:, -1, 0]  # First feature (target) of last time step
        
        # Calculate prediction errors
        errors = np.abs(actual_values - predictions.flatten())
        
        # Normalize errors to 0-1 scale (anomaly scores)
        if len(errors) > 1:
            anomaly_scores = (errors - np.min(errors)) / (np.max(errors) - np.min(errors) + 1e-8)
        else:
            anomaly_scores = np.array([0.0])
        
        return anomaly_scores
    
    def _analyze_trend(self, predictions: np.ndarray) -> Tuple[str, float]:
        """
        Analyze trend direction and strength from predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Tuple of (trend_direction, trend_strength)
        """
        if len(predictions) < 2:
            return 'stable', 0.0
        
        # Calculate linear trend
        x = np.arange(len(predictions))
        slope, _ = np.polyfit(x, predictions.flatten(), 1)
        
        # Determine direction
        if slope > 0.001:
            direction = 'increasing'
        elif slope < -0.001:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        # Calculate strength (normalized absolute slope)
        strength = min(abs(slope) * 100, 1.0)  # Scale and cap at 1.0
        
        return direction, strength
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class AnomalyDetector:
    """
    Anomaly detection for vegetation index time series.
    
    Uses statistical methods and LSTM predictions to identify anomalies.
    """
    
    def __init__(self, threshold_std: float = 2.0):
        """
        Initialize anomaly detector.
        
        Args:
            threshold_std: Standard deviation threshold for anomaly detection
        """
        self.threshold_std = threshold_std
        self.baseline_stats = {}
    
    def fit_baseline(self, time_series: pd.Series):
        """
        Fit baseline statistics for anomaly detection.
        
        Args:
            time_series: Historical time series data
        """
        self.baseline_stats = {
            'mean': time_series.mean(),
            'std': time_series.std(),
            'median': time_series.median(),
            'q25': time_series.quantile(0.25),
            'q75': time_series.quantile(0.75)
        }
        
        logger.info(f"Baseline statistics fitted: {self.baseline_stats}")
    
    def detect_anomalies(self, 
                        values: np.ndarray,
                        method: str = 'statistical') -> np.ndarray:
        """
        Detect anomalies in time series values.
        
        Args:
            values: Time series values to check
            method: Detection method ('statistical' or 'iqr')
            
        Returns:
            Boolean array indicating anomalies
        """
        if not self.baseline_stats:
            raise ValueError("Must fit baseline statistics first")
        
        if method == 'statistical':
            # Z-score based detection
            z_scores = np.abs((values - self.baseline_stats['mean']) / self.baseline_stats['std'])
            anomalies = z_scores > self.threshold_std
            
        elif method == 'iqr':
            # Interquartile range based detection
            iqr = self.baseline_stats['q75'] - self.baseline_stats['q25']
            lower_bound = self.baseline_stats['q25'] - 1.5 * iqr
            upper_bound = self.baseline_stats['q75'] + 1.5 * iqr
            anomalies = (values < lower_bound) | (values > upper_bound)
            
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        return anomalies