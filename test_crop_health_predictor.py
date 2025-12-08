"""
Test script for crop health predictor with fallback logic.
"""

import numpy as np
import sys
import logging
sys.path.insert(0, 'src')

from ai_models.crop_health_predictor import CropHealthPredictor

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_crop_health_predictor():
    """Test the crop health predictor with fallback logic."""
    
    print("=" * 60)
    print("Testing Crop Health Predictor")
    print("=" * 60)
    
    # Test 1: Initialize predictor (should fall back to rule-based)
    print("\nTest 1: Initialize predictor (no model available)")
    predictor = CropHealthPredictor()
    
    info = predictor.get_model_info()
    print(f"Mode: {info['mode']}")
    print(f"Model version: {info['model_version']}")
    print(f"Model path: {info['model_path']}")
    print(f"Model exists: {info['model_exists']}")
    print(f"Model loaded: {info['model_loaded']}")
    print(f"Fallback available: {info['fallback_available']}")
    
    # Test 2: Predict with sample data
    print("\nTest 2: Predict with sample NDVI data")
    sample_ndvi = np.array([
        [0.85, 0.75, 0.65],
        [0.55, 0.45, 0.35],
        [0.25, 0.15, 0.05]
    ])
    
    result = predictor.predict(sample_ndvi)
    print(f"Input shape: {sample_ndvi.shape}")
    print(f"Predictions shape: {result.predictions.shape}")
    print(f"Method used: {result.method}")
    print(f"Predictions:\n{result.predictions}")
    print(f"Class names: {result.class_names}")
    print(f"Confidence scores:\n{result.confidence_scores}")
    
    # Test 3: 1D array
    print("\nTest 3: Predict with 1D NDVI array")
    ndvi_1d = np.array([0.8, 0.6, 0.4, 0.2])
    result_1d = predictor.predict(ndvi_1d)
    print(f"Input: {ndvi_1d}")
    print(f"Predictions: {result_1d.predictions}")
    print(f"Classes: {[result_1d.class_names[p] for p in result_1d.predictions]}")
    print(f"Method: {result_1d.method}")
    
    # Test 4: Large array (simulating real imagery)
    print("\nTest 4: Predict with larger array (100x100)")
    large_ndvi = np.random.uniform(0.2, 0.9, size=(100, 100))
    result_large = predictor.predict(large_ndvi)
    print(f"Input shape: {large_ndvi.shape}")
    print(f"Output shape: {result_large.predictions.shape}")
    print(f"Unique classes: {np.unique(result_large.predictions)}")
    print(f"Class distribution:")
    for i, class_name in enumerate(result_large.class_names):
        count = np.sum(result_large.predictions == i)
        pct = (count / result_large.predictions.size) * 100
        print(f"  {class_name}: {count} pixels ({pct:.1f}%)")
    
    # Test 5: Get mode
    print("\nTest 5: Check prediction mode")
    mode = predictor.get_mode()
    print(f"Current mode: {mode}")
    assert mode == 'rule_based', "Should be in rule_based mode"
    
    # Test 6: Try to reload model (will still fail, but tests the method)
    print("\nTest 6: Test model reload functionality")
    predictor.reload_model('models/nonexistent_model.h5')
    print(f"Mode after reload attempt: {predictor.get_mode()}")
    
    # Test 7: Custom model path
    print("\nTest 7: Initialize with custom model path")
    predictor2 = CropHealthPredictor(model_path='custom/path/model.h5')
    info2 = predictor2.get_model_info()
    print(f"Custom model path: {info2['model_path']}")
    print(f"Mode: {info2['mode']}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Predictor correctly falls back to rule-based classification")
    print("when AI model is unavailable.")
    print("=" * 60)

if __name__ == "__main__":
    test_crop_health_predictor()
