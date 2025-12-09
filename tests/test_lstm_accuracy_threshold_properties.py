"""
Property-based tests for LSTM model accuracy threshold on real data.

**Feature: real-satellite-data-integration, Property 5: LSTM accuracy meets threshold**
**Validates: Requirements 6.3**

This test validates that any LSTM model trained on real temporal satellite data
achieves at least 80% validation accuracy, as specified in the requirements.
"""

import sys
import json
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the LSTM model from training script
try:
    from scripts.train_lstm_on_real_data import CropHealthLSTM
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    pytest.skip("PyTorch or training script not available", allow_module_level=True)


def load_trained_lstm_model(model_path: Path) -> Tuple[nn.Module, dict]:
    """
    Load a trained LSTM model and its metadata.
    
    Args:
        model_path: Path to the trained model checkpoint
        
    Returns:
        Tuple of (model, metadata)
    """
    if not model_path.exists():
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load metadata
    metadata_path = model_path.parent / 'lstm_model_metrics_real.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Create model with correct architecture
    model = CropHealthLSTM(
        input_size=1,  # NDVI values
        hidden_size=128,
        num_layers=2,
        output_size=1
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, metadata


def test_lstm_accuracy_threshold_from_trained_model():
    """
    Property 5: LSTM accuracy meets threshold
    
    For any trained LSTM model on real data, the validation accuracy should be
    greater than or equal to 0.80 (80%).
    
    This property validates Requirements 6.3.
    
    This is a concrete test that checks the actual trained model's accuracy
    against the threshold specified in the requirements.
    """
    model_path = Path('models/crop_health_lstm_real.pth')
    
    # Check if trained model exists
    if not model_path.exists():
        pytest.skip(
            "Trained LSTM model not found. "
            "Run 'python scripts/train_lstm_on_real_data.py' first."
        )
    
    # Load model and metadata
    model, metadata = load_trained_lstm_model(model_path)
    
    if model is None or not metadata:
        pytest.skip("Could not load trained model or metadata")
    
    # Verify this is a model trained on real data (Requirement 6.1)
    assert metadata.get('trained_on') == 'real_temporal_sequences', \
        f"Model should be trained on real data, got: {metadata.get('trained_on')}"
    
    assert 'Sentinel Hub API' in metadata.get('data_source', ''), \
        f"Model should be trained on Sentinel Hub data, got: {metadata.get('data_source')}"
    
    # Get validation accuracy from metadata
    accuracy = metadata.get('metrics', {}).get('accuracy')
    
    if accuracy is None:
        pytest.skip("Accuracy metric not found in model metadata")
    
    # Property: Validation accuracy must be >= 0.80 (Requirement 6.3)
    min_accuracy = 0.80
    assert accuracy >= min_accuracy, \
        f"LSTM model accuracy {accuracy:.4f} is below required threshold {min_accuracy:.4f}. " \
        f"Requirements 6.3 specifies minimum 80% validation accuracy for real data."
    
    print(f"\n✓ LSTM model accuracy: {accuracy:.4f} (>= {min_accuracy:.4f})")
    print(f"  Trained on: {metadata.get('trained_on')}")
    print(f"  Data source: {metadata.get('data_source')}")
    print(f"  Training date: {metadata.get('training_date')}")


@given(
    sequence_length=st.integers(min_value=5, max_value=20),
    batch_size=st.integers(min_value=1, max_value=8),
    ndvi_mean=st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False),
    ndvi_std=st.floats(min_value=0.05, max_value=0.2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_lstm_accuracy_property_with_synthetic_validation(
    sequence_length, batch_size, ndvi_mean, ndvi_std
):
    """
    Property: LSTM model should maintain reasonable accuracy on validation data.
    
    This property test generates synthetic temporal sequences and validates that
    a trained LSTM model produces predictions within reasonable bounds.
    
    Note: This is a complementary test that validates the model's behavior
    on various input distributions, not the primary accuracy threshold test.
    """
    model_path = Path('models/crop_health_lstm_real.pth')
    
    if not model_path.exists():
        pytest.skip("Trained LSTM model not found")
    
    # Load model
    model, metadata = load_trained_lstm_model(model_path)
    
    if model is None:
        pytest.skip("Could not load trained model")
    
    # Generate synthetic temporal sequences
    # Shape: [batch_size, sequence_length, 1]
    sequences = np.random.normal(
        loc=ndvi_mean,
        scale=ndvi_std,
        size=(batch_size, sequence_length, 1)
    )
    
    # Clip to valid NDVI range [-1, 1]
    sequences = np.clip(sequences, -1.0, 1.0)
    
    # Convert to tensor
    X = torch.FloatTensor(sequences)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X)
    
    predictions = predictions.cpu().numpy().flatten()
    
    # Property 1: Predictions should be in valid NDVI range
    assert np.all(predictions >= -1.0) and np.all(predictions <= 1.0), \
        f"Predictions outside valid NDVI range [-1, 1]: " \
        f"min={np.min(predictions):.4f}, max={np.max(predictions):.4f}"
    
    # Property 2: Predictions should not all be identical (model is learning)
    if batch_size > 1:
        pred_std = np.std(predictions)
        # Allow for some variation unless inputs are very similar
        input_std = np.std(sequences)
        if input_std > 0.01:  # Only check if inputs vary
            assert pred_std > 0.0, \
                "Model predictions should vary for different inputs"
    
    # Property 3: Predictions should be reasonably close to input distribution
    # (for a well-trained temporal model)
    pred_mean = np.mean(predictions)
    # Allow predictions to be within reasonable range of input mean
    assert abs(pred_mean - ndvi_mean) < 0.5, \
        f"Prediction mean {pred_mean:.4f} too far from input mean {ndvi_mean:.4f}"


def test_lstm_metadata_completeness():
    """
    Test that trained LSTM model has complete metadata including accuracy.
    
    This ensures that the model metadata contains all required fields for
    validating the accuracy threshold property.
    """
    model_path = Path('models/crop_health_lstm_real.pth')
    metadata_path = Path('models/lstm_model_metrics_real.json')
    
    if not model_path.exists():
        pytest.skip("Trained LSTM model not found")
    
    if not metadata_path.exists():
        pytest.skip("Model metadata not found")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check required fields for accuracy validation
    required_fields = [
        'trained_on',
        'data_source',
        'training_date',
        'metrics'
    ]
    
    for field in required_fields:
        assert field in metadata, f"Missing required metadata field: {field}"
    
    # Check metrics subfields
    required_metrics = ['accuracy', 'mae', 'mse', 'rmse', 'r2_score']
    
    for metric in required_metrics:
        assert metric in metadata['metrics'], \
            f"Missing required metric: {metric}"
    
    # Validate accuracy is a valid number
    accuracy = metadata['metrics']['accuracy']
    assert isinstance(accuracy, (int, float)), \
        f"Accuracy should be numeric, got {type(accuracy)}"
    assert 0.0 <= accuracy <= 1.0, \
        f"Accuracy should be in range [0, 1], got {accuracy}"
    
    print(f"\n✓ Model metadata is complete")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAE: {metadata['metrics']['mae']:.6f}")
    print(f"  RMSE: {metadata['metrics']['rmse']:.6f}")
    print(f"  R² Score: {metadata['metrics']['r2_score']:.6f}")


def test_lstm_training_history_accuracy_trend():
    """
    Test that LSTM training history shows improving or stable accuracy.
    
    This validates that the model training process was successful and
    converged to a good solution.
    """
    history_path = Path('models/lstm_training_history_real.json')
    
    if not history_path.exists():
        pytest.skip("Training history not found")
    
    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Check that validation loss decreased or stabilized
    val_losses = history.get('val_loss', [])
    
    if len(val_losses) < 2:
        pytest.skip("Not enough training history")
    
    # Check that final validation loss is better than or close to initial
    initial_loss = val_losses[0]
    final_loss = val_losses[-1]
    
    assert final_loss <= initial_loss * 1.1, \
        f"Validation loss should improve or stay stable during training. " \
        f"Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
    
    # Check that best validation loss was achieved
    best_loss = min(val_losses)
    
    # Final loss should be close to best (within 10% due to early stopping)
    assert final_loss <= best_loss * 1.1, \
        f"Final validation loss should be close to best. " \
        f"Best: {best_loss:.6f}, Final: {final_loss:.6f}"
    
    print(f"\n✓ Training history shows successful convergence")
    print(f"  Initial val loss: {initial_loss:.6f}")
    print(f"  Best val loss: {best_loss:.6f}")
    print(f"  Final val loss: {final_loss:.6f}")
    print(f"  Epochs trained: {history.get('epochs_trained', 'unknown')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
