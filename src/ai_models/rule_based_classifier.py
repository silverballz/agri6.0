"""
Rule-based classification module for crop health assessment.

This module provides threshold-based classification of crop health using NDVI values
as a fallback when AI models are unavailable. It implements the classification rules
defined in the design document.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of crop health classification.
    
    Attributes:
        predictions: Array of class labels (0=Healthy, 1=Moderate, 2=Stressed, 3=Critical)
        confidence_scores: Array of confidence values (0-1)
        class_names: List of class name strings
        method: Classification method used ('rule_based' or 'ai')
    """
    predictions: np.ndarray
    confidence_scores: np.ndarray
    class_names: list
    method: str = 'rule_based'


class RuleBasedClassifier:
    """
    Rule-based crop health classifier using NDVI thresholds.
    
    Classification rules:
    - NDVI > 0.7: Healthy (class 0)
    - 0.5 < NDVI <= 0.7: Moderate (class 1)
    - 0.3 < NDVI <= 0.5: Stressed (class 2)
    - NDVI <= 0.3: Critical (class 3)
    
    Confidence scores are calculated based on distance from threshold boundaries.
    """
    
    # Class definitions
    CLASS_HEALTHY = 0
    CLASS_MODERATE = 1
    CLASS_STRESSED = 2
    CLASS_CRITICAL = 3
    
    CLASS_NAMES = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    
    # NDVI thresholds
    THRESHOLD_HEALTHY = 0.7
    THRESHOLD_MODERATE = 0.5
    THRESHOLD_STRESSED = 0.3
    
    def __init__(self):
        """Initialize the rule-based classifier."""
        logger.info("Initialized RuleBasedClassifier")
    
    def classify(self, ndvi_values: np.ndarray) -> ClassificationResult:
        """
        Classify crop health based on NDVI values.
        
        Args:
            ndvi_values: Array of NDVI values (typically -1 to 1, but focused on 0-1 range)
        
        Returns:
            ClassificationResult containing predictions and confidence scores
        
        Raises:
            ValueError: If ndvi_values is empty or not a numpy array
        """
        # Validate input
        if not isinstance(ndvi_values, np.ndarray):
            raise ValueError("ndvi_values must be a numpy array")
        
        if ndvi_values.size == 0:
            raise ValueError("ndvi_values cannot be empty")
        
        # Initialize predictions array
        predictions = np.zeros_like(ndvi_values, dtype=int)
        
        # Apply classification rules
        predictions[ndvi_values > self.THRESHOLD_HEALTHY] = self.CLASS_HEALTHY
        predictions[(ndvi_values > self.THRESHOLD_MODERATE) & 
                   (ndvi_values <= self.THRESHOLD_HEALTHY)] = self.CLASS_MODERATE
        predictions[(ndvi_values > self.THRESHOLD_STRESSED) & 
                   (ndvi_values <= self.THRESHOLD_MODERATE)] = self.CLASS_STRESSED
        predictions[ndvi_values <= self.THRESHOLD_STRESSED] = self.CLASS_CRITICAL
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(ndvi_values, predictions)
        
        logger.debug(f"Classified {ndvi_values.size} pixels: "
                    f"Healthy={np.sum(predictions == 0)}, "
                    f"Moderate={np.sum(predictions == 1)}, "
                    f"Stressed={np.sum(predictions == 2)}, "
                    f"Critical={np.sum(predictions == 3)}")
        
        return ClassificationResult(
            predictions=predictions,
            confidence_scores=confidence_scores,
            class_names=self.CLASS_NAMES,
            method='rule_based'
        )
    
    def _calculate_confidence(self, ndvi_values: np.ndarray, 
                            predictions: np.ndarray) -> np.ndarray:
        """
        Calculate confidence scores based on distance from thresholds.
        
        Confidence is higher when NDVI values are far from class boundaries.
        
        Args:
            ndvi_values: Array of NDVI values
            predictions: Array of predicted class labels
        
        Returns:
            Array of confidence scores (0-1)
        """
        confidence = np.zeros_like(ndvi_values, dtype=float)
        
        # For each class, calculate distance from nearest threshold
        for i in range(len(ndvi_values)):
            ndvi = ndvi_values.flat[i]
            pred = predictions.flat[i]
            
            if pred == self.CLASS_HEALTHY:
                # Distance from 0.7 threshold, normalized
                distance = ndvi - self.THRESHOLD_HEALTHY
                # Confidence increases as we move away from threshold
                confidence.flat[i] = min(1.0, distance / 0.3 + 0.5)
            
            elif pred == self.CLASS_MODERATE:
                # Distance from nearest boundary (0.5 or 0.7)
                dist_to_lower = ndvi - self.THRESHOLD_MODERATE
                dist_to_upper = self.THRESHOLD_HEALTHY - ndvi
                min_distance = min(dist_to_lower, dist_to_upper)
                # Confidence is highest in the middle of the range
                confidence.flat[i] = min(1.0, min_distance / 0.1 + 0.5)
            
            elif pred == self.CLASS_STRESSED:
                # Distance from nearest boundary (0.3 or 0.5)
                dist_to_lower = ndvi - self.THRESHOLD_STRESSED
                dist_to_upper = self.THRESHOLD_MODERATE - ndvi
                min_distance = min(dist_to_lower, dist_to_upper)
                confidence.flat[i] = min(1.0, min_distance / 0.1 + 0.5)
            
            else:  # CLASS_CRITICAL
                # Distance below 0.3 threshold
                distance = self.THRESHOLD_STRESSED - ndvi
                confidence.flat[i] = min(1.0, distance / 0.3 + 0.5)
        
        # Ensure confidence is in valid range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def get_class_statistics(self, result: ClassificationResult) -> dict:
        """
        Calculate statistics for classification results.
        
        Args:
            result: ClassificationResult from classify()
        
        Returns:
            Dictionary with class counts and percentages
        """
        total_pixels = result.predictions.size
        stats = {}
        
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            count = np.sum(result.predictions == class_idx)
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
            stats[class_name] = {
                'count': int(count),
                'percentage': float(percentage),
                'avg_confidence': float(np.mean(
                    result.confidence_scores[result.predictions == class_idx]
                )) if count > 0 else 0.0
            }
        
        return stats
