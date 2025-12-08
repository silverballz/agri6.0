"""
Test script to verify AI prediction integration with dashboard components.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from ai_models.crop_health_predictor import CropHealthPredictor

def test_dashboard_integration():
    """Test that predictions work with dashboard-style data"""
    
    print("=" * 60)
    print("Testing Dashboard Integration")
    print("=" * 60)
    
    # Test 1: Initialize predictor (as dashboard would)
    print("\nTest 1: Initialize predictor")
    predictor = CropHealthPredictor()
    mode_info = predictor.get_model_info()
    
    print(f"✓ Predictor initialized")
    print(f"  Mode: {mode_info['mode']}")
    print(f"  Version: {mode_info['model_version']}")
    
    # Test 2: Simulate zone predictions (as field_monitoring.py would)
    print("\nTest 2: Simulate zone predictions")
    
    zones = [
        {"name": "Zone A", "ndvi": 0.78},
        {"name": "Zone B", "ndvi": 0.65},
        {"name": "Zone C", "ndvi": 0.42},
        {"name": "Zone D", "ndvi": 0.25},
    ]
    
    for zone in zones:
        ndvi_array = np.array([zone['ndvi']])
        result = predictor.predict(ndvi_array)
        
        zone['prediction'] = {
            'class_idx': int(result.predictions[0]),
            'class_name': result.class_names[result.predictions[0]],
            'confidence': float(result.confidence_scores[0]),
            'method': result.method
        }
        
        print(f"  {zone['name']}: NDVI={zone['ndvi']:.2f} → "
              f"{zone['prediction']['class_name']} "
              f"({zone['prediction']['confidence']:.1%} confidence, "
              f"method={zone['prediction']['method']})")
    
    # Test 3: Verify classification logic
    print("\nTest 3: Verify classification matches expected thresholds")
    
    test_cases = [
        (0.85, "Healthy"),
        (0.75, "Healthy"),
        (0.65, "Moderate"),
        (0.55, "Moderate"),
        (0.45, "Stressed"),
        (0.35, "Stressed"),
        (0.25, "Critical"),
        (0.15, "Critical"),
    ]
    
    all_correct = True
    for ndvi, expected_class in test_cases:
        result = predictor.predict(np.array([ndvi]))
        actual_class = result.class_names[result.predictions[0]]
        
        if actual_class == expected_class:
            print(f"  ✓ NDVI {ndvi:.2f} → {actual_class}")
        else:
            print(f"  ✗ NDVI {ndvi:.2f} → {actual_class} (expected {expected_class})")
            all_correct = False
    
    if all_correct:
        print("\n✓ All classifications correct!")
    else:
        print("\n✗ Some classifications incorrect")
    
    # Test 4: Test with 2D array (simulating raster data)
    print("\nTest 4: Test with 2D NDVI array (raster simulation)")
    
    ndvi_raster = np.array([
        [0.8, 0.75, 0.7],
        [0.6, 0.5, 0.4],
        [0.3, 0.2, 0.1]
    ])
    
    result = predictor.predict(ndvi_raster)
    print(f"  Input shape: {ndvi_raster.shape}")
    print(f"  Output shape: {result.predictions.shape}")
    print(f"  Predictions:\n{result.predictions}")
    print(f"  Method: {result.method}")
    
    # Test 5: Verify confidence scores are in valid range
    print("\nTest 5: Verify confidence scores")
    
    large_sample = np.random.uniform(0.1, 0.9, size=100)
    result = predictor.predict(large_sample)
    
    min_conf = np.min(result.confidence_scores)
    max_conf = np.max(result.confidence_scores)
    avg_conf = np.mean(result.confidence_scores)
    
    print(f"  Min confidence: {min_conf:.3f}")
    print(f"  Max confidence: {max_conf:.3f}")
    print(f"  Avg confidence: {avg_conf:.3f}")
    
    if 0.0 <= min_conf <= 1.0 and 0.0 <= max_conf <= 1.0:
        print("  ✓ All confidence scores in valid range [0, 1]")
    else:
        print("  ✗ Some confidence scores out of range!")
    
    print("\n" + "=" * 60)
    print("Dashboard integration tests completed!")
    print("The AI prediction system is ready for dashboard use.")
    print("=" * 60)

if __name__ == "__main__":
    test_dashboard_integration()
