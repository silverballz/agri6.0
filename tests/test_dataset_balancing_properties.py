"""
Property-based tests for dataset balancing validation.

**Feature: real-satellite-data-integration, Property 9: Balanced dataset has equal class representation**
**Validates: Requirements 4.3**
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_real_training_data import RealDatasetPreparator
from src.ai_models.rule_based_classifier import RuleBasedClassifier


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_mock_imagery_dir(
    parent_dir: Path,
    tile_id: str,
    acquisition_date: str,
    is_synthetic: bool
) -> Path:
    """
    Create a mock imagery directory with metadata and band files.
    
    Args:
        parent_dir: Parent directory for imagery
        tile_id: Tile identifier
        acquisition_date: Acquisition date string
        is_synthetic: Whether this is synthetic data
        
    Returns:
        Path to created imagery directory
    """
    # Create directory name
    dir_name = f"{tile_id}_{acquisition_date}"
    img_dir = parent_dir / dir_name
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        'tile_id': tile_id,
        'acquisition_date': acquisition_date,
        'synthetic': is_synthetic,
        'data_source': 'Synthetic Generator' if is_synthetic else 'Sentinel Hub API',
        'bands': ['B02', 'B03', 'B04', 'B08'],
        'indices': ['NDVI', 'SAVI', 'EVI', 'NDWI']
    }
    
    with open(img_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Create mock band files (small arrays for testing)
    for band in ['B02', 'B03', 'B04', 'B08']:
        band_data = np.random.uniform(0, 10000, size=(100, 100)).astype(np.uint16)
        np.save(img_dir / f"{band}.npy", band_data)
    
    # Create mock NDVI file with diverse values to generate different classes
    ndvi_data = np.random.uniform(-1, 1, size=(100, 100)).astype(np.float32)
    np.save(img_dir / 'NDVI.npy', ndvi_data)
    
    return img_dir


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(
    samples_per_class=st.integers(min_value=10, max_value=100),
    num_imagery=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_balanced_dataset_equal_class_representation_property(
    samples_per_class,
    num_imagery
):
    """
    Property 9: Balanced dataset has equal class representation
    
    For any balanced training dataset, the number of samples for each crop health
    class should be equal (within tolerance of ±5%).
    
    This property validates Requirements 4.3.
    """
    # Create temporary directories for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create mock real imagery directories
        for i in range(num_imagery):
            create_mock_imagery_dir(
                temp_processed,
                tile_id='43REQ',
                acquisition_date=f'2024-09-{i+1:02d}',
                is_synthetic=False
            )
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Create unbalanced dataset with random class distribution
        # Generate more samples than needed to ensure we have enough for balancing
        total_samples = samples_per_class * len(preparator.classifier.CLASS_NAMES) * 2
        
        # Create random patches (4 channels, 64x64)
        X = np.random.rand(total_samples, 64, 64, 4).astype(np.float32)
        
        # Create random labels with potentially unbalanced distribution
        y = np.random.randint(0, len(preparator.classifier.CLASS_NAMES), size=total_samples)
        
        # Ensure we have at least samples_per_class for each class
        # This is necessary for the balancing to work properly
        for class_idx in range(len(preparator.classifier.CLASS_NAMES)):
            class_count = np.sum(y == class_idx)
            if class_count < samples_per_class:
                # Add more samples of this class
                needed = samples_per_class - class_count
                additional_X = np.random.rand(needed, 64, 64, 4).astype(np.float32)
                additional_y = np.full(needed, class_idx)
                X = np.concatenate([X, additional_X], axis=0)
                y = np.concatenate([y, additional_y], axis=0)
        
        # Balance the dataset
        X_balanced, y_balanced = preparator._balance_dataset(X, y, samples_per_class)
        
        # Property 1: Total samples should equal samples_per_class * num_classes
        num_classes = len(preparator.classifier.CLASS_NAMES)
        expected_total = samples_per_class * num_classes
        
        assert len(X_balanced) == expected_total, \
            f"Expected {expected_total} total samples, got {len(X_balanced)}"
        
        assert len(y_balanced) == expected_total, \
            f"Expected {expected_total} total labels, got {len(y_balanced)}"
        
        # Property 2: Each class should have exactly samples_per_class samples
        # (within ±5% tolerance as specified in requirements)
        tolerance = 0.05  # 5% tolerance
        
        for class_idx in range(num_classes):
            class_count = np.sum(y_balanced == class_idx)
            
            # Calculate acceptable range
            min_acceptable = int(samples_per_class * (1 - tolerance))
            max_acceptable = int(samples_per_class * (1 + tolerance))
            
            assert min_acceptable <= class_count <= max_acceptable, \
                f"Class {class_idx} ({preparator.classifier.CLASS_NAMES[class_idx]}): " \
                f"Expected {samples_per_class} samples (±5%), got {class_count}. " \
                f"Acceptable range: [{min_acceptable}, {max_acceptable}]"
        
        # Property 3: All classes should have approximately equal representation
        class_counts = [np.sum(y_balanced == i) for i in range(num_classes)]
        
        # Check that max and min counts are within tolerance
        min_count = min(class_counts)
        max_count = max(class_counts)
        
        # The difference between max and min should be small relative to samples_per_class
        max_difference = samples_per_class * tolerance * 2  # Allow up to 10% difference
        
        assert (max_count - min_count) <= max_difference, \
            f"Class imbalance detected: min={min_count}, max={max_count}, " \
            f"difference={max_count - min_count}, max_allowed={max_difference}"


@given(
    samples_per_class=st.integers(min_value=50, max_value=200)
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_balanced_dataset_maintains_data_integrity_property(samples_per_class):
    """
    Property: Balanced dataset maintains data integrity.
    
    For any balanced dataset, the data should maintain proper shapes and types,
    and all samples should be valid.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Create dataset with enough samples
        num_classes = len(preparator.classifier.CLASS_NAMES)
        total_samples = samples_per_class * num_classes * 2
        
        X = np.random.rand(total_samples, 64, 64, 4).astype(np.float32)
        y = np.random.randint(0, num_classes, size=total_samples)
        
        # Ensure sufficient samples per class
        for class_idx in range(num_classes):
            class_count = np.sum(y == class_idx)
            if class_count < samples_per_class:
                needed = samples_per_class - class_count
                additional_X = np.random.rand(needed, 64, 64, 4).astype(np.float32)
                additional_y = np.full(needed, class_idx)
                X = np.concatenate([X, additional_X], axis=0)
                y = np.concatenate([y, additional_y], axis=0)
        
        # Balance dataset
        X_balanced, y_balanced = preparator._balance_dataset(X, y, samples_per_class)
        
        # Property 1: Shape consistency
        assert X_balanced.shape[0] == y_balanced.shape[0], \
            "X and y should have same number of samples"
        
        assert X_balanced.shape[1:] == (64, 64, 4), \
            f"X should have shape (N, 64, 64, 4), got {X_balanced.shape}"
        
        # Property 2: Data type consistency
        assert X_balanced.dtype == np.float32, \
            f"X should be float32, got {X_balanced.dtype}"
        
        assert y_balanced.dtype in [np.int32, np.int64], \
            f"y should be integer type, got {y_balanced.dtype}"
        
        # Property 3: Valid label range
        assert np.all(y_balanced >= 0), "All labels should be non-negative"
        assert np.all(y_balanced < num_classes), \
            f"All labels should be less than {num_classes}"
        
        # Property 4: No NaN or Inf values
        assert not np.any(np.isnan(X_balanced)), "X should not contain NaN values"
        assert not np.any(np.isinf(X_balanced)), "X should not contain Inf values"


@given(
    samples_per_class=st.integers(min_value=20, max_value=100)
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_balanced_dataset_handles_oversampling_property(samples_per_class):
    """
    Property: Balanced dataset handles oversampling correctly.
    
    When a class has fewer samples than samples_per_class, the balancing
    should oversample (with replacement) to reach the target count.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        num_classes = len(preparator.classifier.CLASS_NAMES)
        
        # Create dataset where some classes have fewer samples than target
        # Class 0: fewer samples (needs oversampling)
        # Other classes: more samples (needs undersampling)
        few_samples = max(5, samples_per_class // 2)
        many_samples = samples_per_class * 2
        
        X_class0 = np.random.rand(few_samples, 64, 64, 4).astype(np.float32)
        y_class0 = np.zeros(few_samples, dtype=np.int32)
        
        X_others = []
        y_others = []
        for class_idx in range(1, num_classes):
            X_cls = np.random.rand(many_samples, 64, 64, 4).astype(np.float32)
            y_cls = np.full(many_samples, class_idx, dtype=np.int32)
            X_others.append(X_cls)
            y_others.append(y_cls)
        
        X = np.concatenate([X_class0] + X_others, axis=0)
        y = np.concatenate([y_class0] + y_others, axis=0)
        
        # Balance dataset
        X_balanced, y_balanced = preparator._balance_dataset(X, y, samples_per_class)
        
        # Property 1: Class 0 should have samples_per_class samples (oversampled)
        class0_count = np.sum(y_balanced == 0)
        tolerance = int(samples_per_class * 0.05)
        
        assert abs(class0_count - samples_per_class) <= tolerance, \
            f"Class 0 should have ~{samples_per_class} samples after oversampling, " \
            f"got {class0_count}"
        
        # Property 2: All other classes should also have samples_per_class samples
        for class_idx in range(1, num_classes):
            class_count = np.sum(y_balanced == class_idx)
            assert abs(class_count - samples_per_class) <= tolerance, \
                f"Class {class_idx} should have ~{samples_per_class} samples, " \
                f"got {class_count}"


@given(
    samples_per_class=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_balanced_dataset_handles_undersampling_property(samples_per_class):
    """
    Property: Balanced dataset handles undersampling correctly.
    
    When a class has more samples than samples_per_class, the balancing
    should undersample (without replacement) to reach the target count.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        num_classes = len(preparator.classifier.CLASS_NAMES)
        
        # Create dataset where all classes have more samples than target
        many_samples = samples_per_class * 3
        
        X_list = []
        y_list = []
        for class_idx in range(num_classes):
            X_cls = np.random.rand(many_samples, 64, 64, 4).astype(np.float32)
            y_cls = np.full(many_samples, class_idx, dtype=np.int32)
            X_list.append(X_cls)
            y_list.append(y_cls)
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Balance dataset
        X_balanced, y_balanced = preparator._balance_dataset(X, y, samples_per_class)
        
        # Property: All classes should have exactly samples_per_class samples (undersampled)
        tolerance = int(samples_per_class * 0.05)
        
        for class_idx in range(num_classes):
            class_count = np.sum(y_balanced == class_idx)
            assert abs(class_count - samples_per_class) <= tolerance, \
                f"Class {class_idx} should have ~{samples_per_class} samples after undersampling, " \
                f"got {class_count}"


def test_balanced_dataset_concrete_example():
    """
    Concrete example test: Balance a specific dataset and verify results.
    
    This test uses a concrete example to validate the balancing behavior.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        num_classes = len(preparator.classifier.CLASS_NAMES)
        samples_per_class = 100
        
        # Create imbalanced dataset
        # Class 0: 50 samples
        # Class 1: 200 samples
        # Class 2: 150 samples
        # Class 3: 80 samples
        class_counts = [50, 200, 150, 80]
        
        X_list = []
        y_list = []
        for class_idx, count in enumerate(class_counts):
            X_cls = np.random.rand(count, 64, 64, 4).astype(np.float32)
            y_cls = np.full(count, class_idx, dtype=np.int32)
            X_list.append(X_cls)
            y_list.append(y_cls)
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Balance dataset
        X_balanced, y_balanced = preparator._balance_dataset(X, y, samples_per_class)
        
        # Verify balanced counts
        for class_idx in range(num_classes):
            class_count = np.sum(y_balanced == class_idx)
            # Allow ±5% tolerance
            assert 95 <= class_count <= 105, \
                f"Class {class_idx} should have ~100 samples, got {class_count}"
        
        # Verify total count
        expected_total = samples_per_class * num_classes
        assert len(X_balanced) == expected_total, \
            f"Expected {expected_total} total samples, got {len(X_balanced)}"


def test_balanced_dataset_empty_class_handling():
    """
    Test that balancing handles missing classes gracefully.
    
    When a class has no samples, it should be skipped.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        samples_per_class = 50
        
        # Create dataset with only 2 classes (missing classes 2 and 3)
        X_class0 = np.random.rand(100, 64, 64, 4).astype(np.float32)
        y_class0 = np.zeros(100, dtype=np.int32)
        
        X_class1 = np.random.rand(100, 64, 64, 4).astype(np.float32)
        y_class1 = np.ones(100, dtype=np.int32)
        
        X = np.concatenate([X_class0, X_class1], axis=0)
        y = np.concatenate([y_class0, y_class1], axis=0)
        
        # Balance dataset
        X_balanced, y_balanced = preparator._balance_dataset(X, y, samples_per_class)
        
        # Verify that only classes 0 and 1 are present
        unique_classes = np.unique(y_balanced)
        assert len(unique_classes) == 2, \
            f"Expected 2 classes, got {len(unique_classes)}"
        assert 0 in unique_classes and 1 in unique_classes, \
            "Classes 0 and 1 should be present"
        
        # Verify balanced counts for present classes
        for class_idx in [0, 1]:
            class_count = np.sum(y_balanced == class_idx)
            # Allow ±5% tolerance
            assert 47 <= class_count <= 53, \
                f"Class {class_idx} should have ~50 samples, got {class_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
