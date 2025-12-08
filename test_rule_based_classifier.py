"""
Test script for rule-based classifier with sample NDVI data.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from ai_models.rule_based_classifier import RuleBasedClassifier, ClassificationResult

def test_rule_based_classifier():
    """Test the rule-based classifier with sample NDVI data."""
    
    print("=" * 60)
    print("Testing Rule-Based Classifier")
    print("=" * 60)
    
    # Create classifier
    classifier = RuleBasedClassifier()
    
    # Test 1: Sample NDVI values covering all classes
    print("\nTest 1: Sample NDVI values")
    sample_ndvi = np.array([
        0.85,  # Healthy
        0.75,  # Healthy
        0.65,  # Moderate
        0.55,  # Moderate
        0.45,  # Stressed
        0.35,  # Stressed
        0.25,  # Critical
        0.15   # Critical
    ])
    
    result = classifier.classify(sample_ndvi)
    
    print(f"NDVI values: {sample_ndvi}")
    print(f"Predictions: {result.predictions}")
    print(f"Class names: {[result.class_names[p] for p in result.predictions]}")
    print(f"Confidence scores: {result.confidence_scores}")
    print(f"Method: {result.method}")
    
    # Test 2: Statistics
    print("\nTest 2: Classification statistics")
    stats = classifier.get_class_statistics(result)
    for class_name, class_stats in stats.items():
        print(f"{class_name}: {class_stats['count']} pixels "
              f"({class_stats['percentage']:.1f}%), "
              f"avg confidence: {class_stats['avg_confidence']:.3f}")
    
    # Test 3: 2D array (simulating image patch)
    print("\nTest 3: 2D NDVI array (image patch)")
    ndvi_patch = np.array([
        [0.8, 0.75, 0.7],
        [0.6, 0.5, 0.4],
        [0.3, 0.2, 0.1]
    ])
    
    result_2d = classifier.classify(ndvi_patch)
    print(f"Input shape: {ndvi_patch.shape}")
    print(f"Predictions shape: {result_2d.predictions.shape}")
    print(f"Predictions:\n{result_2d.predictions}")
    print(f"Confidence scores:\n{result_2d.confidence_scores}")
    
    # Test 4: Edge cases
    print("\nTest 4: Edge cases")
    edge_cases = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    result_edge = classifier.classify(edge_cases)
    print(f"NDVI values: {edge_cases}")
    print(f"Predictions: {result_edge.predictions}")
    print(f"Classes: {[result_edge.class_names[p] for p in result_edge.predictions]}")
    
    # Test 5: Error handling
    print("\nTest 5: Error handling")
    try:
        classifier.classify(np.array([]))
        print("ERROR: Should have raised ValueError for empty array")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    try:
        classifier.classify([0.5, 0.6])  # List instead of numpy array
        print("ERROR: Should have raised ValueError for non-numpy array")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_rule_based_classifier()
