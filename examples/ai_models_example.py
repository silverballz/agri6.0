"""
Example demonstrating the AI analysis models for agricultural monitoring.

This example shows how to use the LSTM temporal analysis, CNN spatial analysis,
and risk prediction models together for comprehensive crop monitoring.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI models
from src.ai_models.temporal_lstm import TemporalLSTM, LSTMConfig
from src.ai_models.spatial_cnn import SpatialCNN, CNNConfig, ImagePatchExtractor
from src.ai_models.risk_prediction import EnsembleRiskPredictor, RiskPredictionConfig
from src.ai_models.training_pipeline import create_sample_training_data
from src.ai_models.cnn_training_pipeline import CNNTrainingPipeline
from src.ai_models.risk_prediction import create_sample_risk_data


def demonstrate_temporal_lstm():
    """Demonstrate LSTM temporal trend analysis."""
    print("\n" + "="*60)
    print("LSTM TEMPORAL TREND ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create LSTM model with small configuration for demo
    config = LSTMConfig(
        sequence_length=15,
        lstm_units=32,
        epochs=5,
        batch_size=16,
        early_stopping_patience=3
    )
    
    lstm_model = TemporalLSTM(config)
    
    # Generate sample time series data
    print("Generating sample time series data...")
    dates = pd.date_range('2023-01-01', periods=120, freq='D')
    
    # Create realistic vegetation index time series with seasonal pattern
    t = np.arange(120)
    seasonal_pattern = 0.3 * np.sin(2 * np.pi * t / 365)
    monthly_variation = 0.1 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 0.05, 120)
    
    ndvi = 0.6 + seasonal_pattern + monthly_variation + noise
    ndvi = np.clip(ndvi, 0, 1)
    
    # Add correlated environmental data
    temperature = 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, 120)
    humidity = 60 + 20 * np.sin(2 * np.pi * t / 365 + np.pi/2) + np.random.normal(0, 5, 120)
    soil_moisture = 0.3 + 0.2 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.03, 120)
    
    time_series_data = pd.DataFrame({
        'timestamp': dates,
        'ndvi': ndvi,
        'temperature': temperature,
        'humidity': np.clip(humidity, 0, 100),
        'soil_moisture': np.clip(soil_moisture, 0, 1)
    }).set_index('timestamp')
    
    print(f"Created time series with {len(time_series_data)} records")
    print(f"NDVI range: {ndvi.min():.3f} - {ndvi.max():.3f}")
    
    # Prepare training data
    print("Preparing training sequences...")
    X, y = lstm_model.prepare_training_data(time_series_data, target_column='ndvi')
    print(f"Created {len(X)} training sequences")
    
    # Split data for training and testing
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    print("Training LSTM model...")
    history = lstm_model.train(X_train, y_train, validation_data=(X_test, y_test))
    
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
    print(f"Validation loss: {history['val_loss'][-1]:.4f}")
    
    # Make predictions
    print("Making predictions...")
    result = lstm_model.predict(X_test)
    
    print(f"Predicted {len(result.predictions)} values")
    print(f"Trend direction: {result.trend_direction}")
    print(f"Trend strength: {result.trend_strength:.3f}")
    print(f"Mean confidence: {np.mean(result.confidence_intervals[:, 1] - result.confidence_intervals[:, 0]):.3f}")
    print(f"Anomaly rate: {np.mean(result.anomaly_scores > 0.7):.3f}")
    
    # Evaluate model
    metrics = lstm_model.evaluate(X_test, y_test)
    print(f"Model R²: {metrics['r2']:.3f}")
    print(f"Model RMSE: {metrics['rmse']:.4f}")


def demonstrate_spatial_cnn():
    """Demonstrate CNN spatial analysis."""
    print("\n" + "="*60)
    print("CNN SPATIAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create CNN model with small configuration for demo
    config = CNNConfig(
        input_shape=(32, 32, 6),
        num_classes=4,
        conv_filters=[16, 32, 64],
        epochs=3,
        batch_size=16
    )
    
    # Create training pipeline
    pipeline = CNNTrainingPipeline(config, patch_size=32)
    
    # Generate synthetic training data
    print("Generating synthetic spatial training data...")
    patches, labels = pipeline.create_synthetic_training_data(n_samples=400, patch_size=32)
    
    print(f"Created {len(patches)} training patches")
    print(f"Patch shape: {patches.shape[1:]}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Train model
    print("Training CNN model...")
    results = pipeline.train_model(patches, labels, validation_split=0.25)
    
    print(f"Training completed.")
    print(f"Training patches: {results['training_patches']}")
    print(f"Validation patches: {results['validation_patches']}")
    print(f"Final accuracy: {results['validation_metrics']['accuracy']:.3f}")
    
    # Test prediction on new data
    print("Testing predictions...")
    test_patches, test_labels = pipeline.create_synthetic_training_data(n_samples=50, patch_size=32)
    
    evaluation_results = pipeline.evaluate_model_performance(test_patches, test_labels)
    
    print(f"Test accuracy: {evaluation_results['metrics']['accuracy']:.3f}")
    print(f"Mean confidence: {evaluation_results['mean_confidence']:.3f}")
    if evaluation_results['mean_uncertainty']:
        print(f"Mean uncertainty: {evaluation_results['mean_uncertainty']:.3f}")
    
    # Demonstrate full image prediction
    print("Testing full image prediction...")
    test_image = np.random.randint(0, 10000, (128, 128, 6)).astype(np.float32)
    
    prediction_map, confidence_map = pipeline.cnn_model.predict_image(test_image, patch_size=32)
    
    print(f"Prediction map shape: {prediction_map.shape}")
    print(f"Unique predictions: {np.unique(prediction_map)}")
    print(f"Mean confidence: {np.mean(confidence_map):.3f}")


def demonstrate_risk_prediction():
    """Demonstrate ensemble risk prediction."""
    print("\n" + "="*60)
    print("ENSEMBLE RISK PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Create ensemble predictor
    config = RiskPredictionConfig(
        n_estimators=20,
        max_depth=8,
        cv_folds=3
    )
    
    ensemble = EnsembleRiskPredictor(config)
    
    # Generate sample risk data
    print("Generating sample risk data...")
    env_data, spectral_data, pest_labels, disease_labels = create_sample_risk_data(
        n_zones=5, n_days=200
    )
    
    print(f"Environmental data: {env_data.shape}")
    print(f"Spectral data: {spectral_data.shape}")
    print(f"Pest outbreak rate: {pest_labels.mean():.3f}")
    print(f"Disease occurrence rate: {disease_labels.mean():.3f}")
    
    # Train ensemble model
    print("Training ensemble risk prediction model...")
    results = ensemble.train(env_data, spectral_data, pest_labels, disease_labels)
    
    print("Training completed.")
    print(f"Pest model CV AUC: {results['pest_model']['cv_auc_mean']:.3f}")
    print(f"Disease model CV AUC: {results['disease_model']['cv_auc_mean']:.3f}")
    print(f"Ensemble weights: Pest={results['ensemble_weights']['pest']:.3f}, Disease={results['ensemble_weights']['disease']:.3f}")
    
    # Make risk predictions for different scenarios
    print("\nTesting risk predictions...")
    
    # Test scenario 1: Normal conditions
    normal_env = env_data.iloc[:10].copy()
    normal_spectral = spectral_data.iloc[:10].copy()
    
    risk_normal = ensemble.predict_risk(normal_env, normal_spectral)
    
    print(f"\nNormal conditions:")
    print(f"  Overall risk: {risk_normal.overall_risk_score:.3f} ({risk_normal.risk_level})")
    print(f"  Pest risk: {risk_normal.pest_risk_probability:.3f}")
    print(f"  Disease risk: {risk_normal.disease_risk_probability:.3f}")
    print(f"  Confidence: {risk_normal.confidence_score:.3f}")
    print(f"  Top recommendations: {risk_normal.recommendations[:2]}")
    
    # Test scenario 2: High-risk conditions
    high_risk_env = env_data.iloc[:10].copy()
    high_risk_env['temperature'] = 35  # Very high temperature
    high_risk_env['humidity'] = 95     # Very high humidity
    
    high_risk_spectral = spectral_data.iloc[:10].copy()
    high_risk_spectral['ndvi'] = 0.3   # Low NDVI (stressed vegetation)
    
    risk_high = ensemble.predict_risk(high_risk_env, high_risk_spectral)
    
    print(f"\nHigh-risk conditions:")
    print(f"  Overall risk: {risk_high.overall_risk_score:.3f} ({risk_high.risk_level})")
    print(f"  Pest risk: {risk_high.pest_risk_probability:.3f}")
    print(f"  Disease risk: {risk_high.disease_risk_probability:.3f}")
    print(f"  Confidence: {risk_high.confidence_score:.3f}")
    print(f"  Top recommendations: {risk_high.recommendations[:3]}")
    
    # Show top risk factors
    if risk_high.risk_factors:
        top_factors = sorted(risk_high.risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top risk factors:")
        for factor, importance in top_factors:
            print(f"    {factor}: {importance:.3f}")


def main():
    """Run all AI model demonstrations."""
    print("AGRICULTURAL MONITORING AI MODELS DEMONSTRATION")
    print("This example demonstrates the three main AI components:")
    print("1. LSTM Temporal Trend Analysis")
    print("2. CNN Spatial Analysis") 
    print("3. Ensemble Risk Prediction")
    
    try:
        # Demonstrate each component
        demonstrate_temporal_lstm()
        demonstrate_spatial_cnn()
        demonstrate_risk_prediction()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe AI models are ready for integration into the agricultural")
        print("monitoring platform. They provide:")
        print("- Temporal trend analysis and anomaly detection")
        print("- Spatial crop health classification")
        print("- Comprehensive risk assessment and recommendations")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()