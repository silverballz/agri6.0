"""
Crop health prediction module with AI model and rule-based fallback.

This module provides a unified interface for crop health prediction that automatically
falls back to rule-based classification when AI models are unavailable.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union
import os

from .rule_based_classifier import RuleBasedClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class CropHealthPredictor:
    """
    Unified crop health predictor with AI model and rule-based fallback.
    
    This class attempts to load a trained CNN model for crop health prediction.
    If the model is unavailable or fails to load, it automatically falls back
    to rule-based classification using NDVI thresholds.
    
    Attributes:
        mode: Current prediction mode ('ai' or 'rule_based')
        model: Loaded AI model (if available)
        rule_classifier: Rule-based classifier instance
        model_version: Version string of loaded model
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the crop health predictor.
        
        Args:
            model_path: Path to trained model file (.h5, .keras, or .pkl)
                       If None, uses default path from environment or 'models/crop_health_cnn.h5'
        """
        self.mode = 'rule_based'  # Default to rule-based
        self.model = None
        self.model_version = 'rule_based_v1.0'
        self.rule_classifier = RuleBasedClassifier()
        
        # Determine model path
        if model_path is None:
            model_path = os.environ.get('MODEL_PATH', 'models/')
            if os.path.isdir(model_path):
                model_path = os.path.join(model_path, 'crop_health_cnn.h5')
        
        self.model_path = Path(model_path)
        
        # Try to load AI model
        self._try_load_model()
    
    def _try_load_model(self):
        """
        Attempt to load the AI model with comprehensive error handling.
        
        Falls back to rule-based mode if loading fails for any reason.
        """
        if not self.model_path.exists():
            logger.info(f"Model file not found at {self.model_path}. Using rule-based classification.")
            return
        
        try:
            # Try importing TensorFlow/Keras
            try:
                from tensorflow import keras
                load_model = keras.models.load_model
                logger.info("Using TensorFlow/Keras for model loading")
            except ImportError:
                logger.warning("TensorFlow not available. Trying alternative model formats...")
                try:
                    import joblib
                    # For scikit-learn models
                    self.model = joblib.load(self.model_path)
                    self.mode = 'ai'
                    self.model_version = f'sklearn_{self.model_path.stem}'
                    logger.info(f"Successfully loaded scikit-learn model from {self.model_path}")
                    return
                except ImportError:
                    logger.warning("No ML libraries available. Using rule-based classification.")
                    return
            
            # Load Keras/TensorFlow model
            self.model = load_model(str(self.model_path))
            self.mode = 'ai'
            self.model_version = f'cnn_{self.model_path.stem}'
            logger.info(f"Successfully loaded AI model from {self.model_path}")
            logger.info(f"Model summary: {self.model.summary() if hasattr(self.model, 'summary') else 'N/A'}")
            
        except Exception as e:
            logger.warning(f"Failed to load AI model: {e}. Using rule-based classification.")
            self.mode = 'rule_based'
            self.model = None
    
    def predict(self, ndvi_values: np.ndarray, 
                additional_features: Optional[np.ndarray] = None) -> ClassificationResult:
        """
        Predict crop health for given NDVI values.
        
        Automatically uses AI model if available, otherwise falls back to rule-based.
        
        Args:
            ndvi_values: Array of NDVI values (any shape)
            additional_features: Optional additional features for AI model
                               (e.g., other vegetation indices, temperature, etc.)
        
        Returns:
            ClassificationResult with predictions and confidence scores
        
        Raises:
            ValueError: If ndvi_values is invalid
        """
        if self.mode == 'ai' and self.model is not None:
            try:
                return self._ai_predict(ndvi_values, additional_features)
            except Exception as e:
                logger.error(f"AI prediction failed: {e}. Falling back to rule-based.")
                # Fall through to rule-based prediction
        
        return self._rule_based_predict(ndvi_values)
    
    def _ai_predict(self, ndvi_values: np.ndarray,
                   additional_features: Optional[np.ndarray] = None) -> ClassificationResult:
        """
        Perform AI-based prediction using loaded model.
        
        Args:
            ndvi_values: Array of NDVI values
            additional_features: Optional additional features
        
        Returns:
            ClassificationResult with AI predictions
        """
        original_shape = ndvi_values.shape
        
        # Prepare input for model
        # Most CNN models expect (batch, height, width, channels)
        if len(ndvi_values.shape) == 2:
            # Add batch and channel dimensions
            model_input = ndvi_values[np.newaxis, :, :, np.newaxis]
        elif len(ndvi_values.shape) == 1:
            # Flatten case - reshape to 2D then add dimensions
            side_len = int(np.sqrt(len(ndvi_values)))
            if side_len * side_len == len(ndvi_values):
                model_input = ndvi_values.reshape(side_len, side_len)
                model_input = model_input[np.newaxis, :, :, np.newaxis]
            else:
                # Can't reshape nicely, use rule-based
                logger.warning("Cannot reshape 1D array to 2D for CNN. Using rule-based.")
                return self._rule_based_predict(ndvi_values)
        else:
            model_input = ndvi_values
        
        # Run inference
        predictions_proba = self.model.predict(model_input, verbose=0)
        
        # Get class predictions and confidence
        predictions = np.argmax(predictions_proba, axis=-1)
        confidence_scores = np.max(predictions_proba, axis=-1)
        
        # Reshape back to original shape
        predictions = predictions.reshape(original_shape)
        confidence_scores = confidence_scores.reshape(original_shape)
        
        logger.debug(f"AI prediction completed. Shape: {predictions.shape}")
        
        return ClassificationResult(
            predictions=predictions,
            confidence_scores=confidence_scores,
            class_names=RuleBasedClassifier.CLASS_NAMES,
            method='ai'
        )
    
    def _rule_based_predict(self, ndvi_values: np.ndarray) -> ClassificationResult:
        """
        Perform rule-based prediction using threshold classifier.
        
        Args:
            ndvi_values: Array of NDVI values
        
        Returns:
            ClassificationResult with rule-based predictions
        """
        logger.debug("Using rule-based classification")
        return self.rule_classifier.classify(ndvi_values)
    
    def get_mode(self) -> str:
        """
        Get current prediction mode.
        
        Returns:
            'ai' or 'rule_based'
        """
        return self.mode
    
    def get_model_info(self) -> dict:
        """
        Get information about the current prediction setup.
        
        Returns:
            Dictionary with mode, version, and availability info
        """
        return {
            'mode': self.mode,
            'model_version': self.model_version,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'model_loaded': self.model is not None,
            'fallback_available': True  # Rule-based always available
        }
    
    def reload_model(self, model_path: Optional[str] = None):
        """
        Attempt to reload the AI model.
        
        Useful if model becomes available after initialization.
        
        Args:
            model_path: Optional new path to model file
        """
        if model_path is not None:
            self.model_path = Path(model_path)
        
        logger.info(f"Attempting to reload model from {self.model_path}")
        self._try_load_model()
        logger.info(f"Reload complete. Current mode: {self.mode}")
