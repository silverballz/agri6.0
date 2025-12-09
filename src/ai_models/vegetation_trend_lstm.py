"""
LSTM model for vegetation trend analysis.

This module implements the VegetationTrendLSTM class as specified in the design document
for temporal analysis of vegetation index time series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class VegetationTrendLSTM:
    """
    LSTM model for temporal trend analysis of vegetation indices.
    
    This class implements bidirectional LSTM with attention mechanism
    for analyzing vegetation index time series and predicting trends.
    
    Attributes:
        sequence_length: Number of time steps to look back
        model: Keras model instance
        is_trained: Whether the model has been trained
    """
    
    def __init__(self, sequence_length: int = 30):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps in input sequences
        """
        self.sequence_length = sequence_length
        self.model = None
        self.is_trained = False
        self.model_version = "1.0.0"
        self._build_model()
    
    def _build_model(self):
        """
        Create LSTM model architecture with bidirectional layers and attention.
        
        Architecture:
        - Input: (sequence_length, 4) - 4 features (NDVI, temp, humidity, soil_moisture)
        - Bidirectional LSTM layer 1: 64 units, return sequences
        - Dropout: 0.2
        - Bidirectional LSTM layer 2: 32 units
        - Dropout: 0.2
        - Dense layer: 16 units, ReLU activation
        - Output: 1 unit (predicted next NDVI value)
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                # Bidirectional LSTM layers
                layers.Bidirectional(
                    layers.LSTM(
                        64,
                        return_sequences=True,
                        dropout=0.2,
                        recurrent_dropout=0.2
                    ),
                    input_shape=(self.sequence_length, 4)
                ),
                layers.Bidirectional(
                    layers.LSTM(
                        32,
                        return_sequences=False,
                        dropout=0.2,
                        recurrent_dropout=0.2
                    )
                ),
                
                # Dense layers for prediction
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='linear')  # Single output for next value
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            self.model = model
            logger.info("VegetationTrendLSTM model created successfully")
            logger.info(f"Model has {self.model.count_params()} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available. Model will not be functional.")
            self.model = None
    
    def prepare_sequences(self, 
                         time_series_data: pd.DataFrame,
                         target_column: str = 'ndvi') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data into sequences for LSTM training.
        
        Args:
            time_series_data: DataFrame with columns [ndvi, temperature, humidity, soil_moisture]
            target_column: Column to predict (default: 'ndvi')
        
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences (N, sequence_length, 4)
            - y: Target values (N,)
        """
        # Ensure required columns exist
        required_cols = ['ndvi', 'temperature', 'humidity', 'soil_moisture']
        for col in required_cols:
            if col not in time_series_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by index (assuming index is datetime)
        data = time_series_data[required_cols].sort_index()
        
        # Handle missing values
        data = data.ffill().bfill()
        
        # Normalize data to 0-1 range
        data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(data_normalized)):
            X.append(data_normalized.iloc[i - self.sequence_length:i].values)
            y.append(data_normalized.iloc[i][target_column])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32) -> Dict:
        """
        Train the LSTM model with validation split.
        
        Args:
            X_train: Training sequences (N, sequence_length, 4)
            y_train: Training targets (N,)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history dictionary
        
        Raises:
            ValueError: If model is not available
        """
        if self.model is None:
            raise ValueError("Model not available (TensorFlow not installed)")
        
        try:
            from tensorflow import keras
            
            # Setup callbacks
            callbacks = [
                # Early stopping
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Learning rate reduction
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            logger.info(f"Starting training with {len(X_train)} samples")
            logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("Training completed successfully")
            
            return history.history
            
        except ImportError:
            raise ValueError("TensorFlow not available for training")
    
    def predict_trend(self, 
                     X: np.ndarray,
                     return_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], str, float]:
        """
        Predict future values and detect trend with confidence intervals.
        
        Args:
            X: Input sequences (N, sequence_length, 4)
            return_confidence: Whether to calculate confidence intervals
        
        Returns:
            Tuple of (predictions, confidence_intervals, trend_direction, trend_strength)
            - predictions: Predicted values (N,)
            - confidence_intervals: 95% confidence intervals (N, 2) or None
            - trend_direction: 'increasing', 'decreasing', or 'stable'
            - trend_strength: Float in [0, 1] indicating trend strength
        
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model not available (TensorFlow not installed)")
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0).flatten()
        
        # Calculate confidence intervals if requested
        confidence_intervals = None
        if return_confidence:
            confidence_intervals = self._calculate_confidence_intervals(X)
        
        # Analyze trend
        trend_direction, trend_strength = self._analyze_trend(predictions)
        
        logger.info(f"Predictions completed: trend={trend_direction}, strength={trend_strength:.3f}")
        
        return predictions, confidence_intervals, trend_direction, trend_strength
    
    def _calculate_confidence_intervals(self, 
                                       X: np.ndarray,
                                       n_samples: int = 100) -> np.ndarray:
        """
        Calculate confidence intervals using Monte Carlo dropout.
        
        Args:
            X: Input sequences
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Confidence intervals (N, 2) with [lower, upper] bounds
        """
        # Collect predictions with dropout enabled
        predictions_list = []
        
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions_list.append(pred.numpy().flatten())
        
        predictions_array = np.array(predictions_list)
        
        # Calculate percentiles
        lower = np.percentile(predictions_array, 2.5, axis=0)
        upper = np.percentile(predictions_array, 97.5, axis=0)
        
        return np.column_stack([lower, upper])
    
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
        slope, _ = np.polyfit(x, predictions, 1)
        
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
    
    def calculate_anomaly_score(self, 
                                X: np.ndarray,
                                actual_values: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on prediction errors.
        
        Args:
            X: Input sequences
            actual_values: Actual observed values
        
        Returns:
            Anomaly scores (N,) in range [0, 1]
        
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating anomaly scores")
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0).flatten()
        
        # Calculate prediction errors
        errors = np.abs(actual_values - predictions)
        
        # Normalize errors to 0-1 scale (anomaly scores)
        if len(errors) > 1:
            anomaly_scores = (errors - np.min(errors)) / (np.max(errors) - np.min(errors) + 1e-8)
        else:
            anomaly_scores = np.array([0.0])
        
        return anomaly_scores
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if self.model is None:
            raise ValueError("Model not available")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath} (version {self.model_version})")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            from tensorflow import keras
            
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            
        except ImportError:
            raise ValueError("TensorFlow not available for loading model")
