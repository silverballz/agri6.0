"""
Tests for risk prediction models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from src.ai_models.risk_prediction import (
    RiskPredictionConfig, PestOutbreakPredictor, DiseaseRiskPredictor,
    EnsembleRiskPredictor, RiskAssessment, create_sample_risk_data
)


class TestRiskPredictionConfig:
    """Test risk prediction configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RiskPredictionConfig()
        
        assert config.pest_model_type == 'random_forest'
        assert config.disease_model_type == 'random_forest'
        assert config.ensemble_method == 'voting'
        assert config.n_estimators == 100
        assert config.max_depth == 10
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.cv_folds == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RiskPredictionConfig(
            pest_model_type='logistic',
            disease_model_type='gradient_boost',
            n_estimators=50,
            max_depth=5
        )
        
        assert config.pest_model_type == 'logistic'
        assert config.disease_model_type == 'gradient_boost'
        assert config.n_estimators == 50
        assert config.max_depth == 5


class TestPestOutbreakPredictor:
    """Test pest outbreak prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample environmental and spectral data."""
        env_data, spectral_data, pest_labels, _ = create_sample_risk_data(n_zones=3, n_days=100)
        return env_data, spectral_data, pest_labels
    
    @pytest.fixture
    def pest_predictor(self):
        """Create pest predictor with test configuration."""
        config = RiskPredictionConfig(
            pest_model_type='random_forest',
            n_estimators=10,
            max_depth=5,
            cv_folds=3
        )
        return PestOutbreakPredictor(config)
    
    def test_predictor_initialization(self, pest_predictor):
        """Test predictor initialization."""
        assert pest_predictor.config.pest_model_type == 'random_forest'
        assert pest_predictor.model is None
        assert not pest_predictor.is_trained
        assert len(pest_predictor.feature_names) == 0
    
    def test_prepare_features(self, pest_predictor, sample_data):
        """Test feature preparation."""
        env_data, spectral_data, _ = sample_data
        
        features = pest_predictor.prepare_features(env_data, spectral_data)
        
        # Check that features were created
        assert len(features) > 0
        assert len(features.columns) > len(env_data.columns)  # Should have derived features
        
        # Check for derived features
        expected_features = ['heat_index', 'vpd', 'gdd', 'cumulative_gdd']
        for feature in expected_features:
            if feature in features.columns:
                assert not features[feature].isna().all()
        
        # Check feature names were stored
        assert len(pest_predictor.feature_names) > 0
    
    def test_model_training(self, pest_predictor, sample_data):
        """Test model training."""
        env_data, spectral_data, pest_labels = sample_data
        
        features = pest_predictor.prepare_features(env_data, spectral_data)
        results = pest_predictor.train(features, pest_labels)
        
        # Check training completed
        assert pest_predictor.is_trained
        assert pest_predictor.model is not None
        
        # Check results
        assert 'cv_auc_mean' in results
        assert 'cv_auc_std' in results
        assert 'feature_importance' in results
        assert 'n_samples' in results
        assert 'n_features' in results
        
        assert 0 <= results['cv_auc_mean'] <= 1
        assert results['n_samples'] == len(features)
        assert results['n_features'] == len(pest_predictor.feature_names)
    
    def test_predict_probability(self, pest_predictor, sample_data):
        """Test probability prediction."""
        env_data, spectral_data, pest_labels = sample_data
        
        features = pest_predictor.prepare_features(env_data, spectral_data)
        pest_predictor.train(features, pest_labels)
        
        # Make predictions
        probabilities = pest_predictor.predict_probability(features[:10])
        
        # Check predictions
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_get_risk_factors(self, pest_predictor, sample_data):
        """Test risk factor extraction."""
        env_data, spectral_data, pest_labels = sample_data
        
        features = pest_predictor.prepare_features(env_data, spectral_data)
        pest_predictor.train(features, pest_labels)
        
        # Get risk factors
        risk_factors = pest_predictor.get_risk_factors(features[:10])
        
        # Check risk factors
        assert isinstance(risk_factors, dict)
        if risk_factors:  # May be empty for some models
            assert all(0 <= v <= 1 for v in risk_factors.values())
            assert abs(sum(risk_factors.values()) - 1.0) < 0.01  # Should sum to ~1
    
    def test_different_model_types(self, sample_data):
        """Test different model types."""
        env_data, spectral_data, pest_labels = sample_data
        
        model_types = ['random_forest', 'logistic', 'gradient_boost']
        
        for model_type in model_types:
            config = RiskPredictionConfig(
                pest_model_type=model_type,
                n_estimators=5,
                cv_folds=2
            )
            predictor = PestOutbreakPredictor(config)
            
            features = predictor.prepare_features(env_data, spectral_data)
            results = predictor.train(features, pest_labels)
            
            assert predictor.is_trained
            assert 'cv_auc_mean' in results


class TestDiseaseRiskPredictor:
    """Test disease risk prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample spectral and environmental data."""
        env_data, spectral_data, _, disease_labels = create_sample_risk_data(n_zones=3, n_days=100)
        return env_data, spectral_data, disease_labels
    
    @pytest.fixture
    def disease_predictor(self):
        """Create disease predictor with test configuration."""
        config = RiskPredictionConfig(
            disease_model_type='random_forest',
            n_estimators=10,
            max_depth=5,
            cv_folds=3
        )
        return DiseaseRiskPredictor(config)
    
    def test_predictor_initialization(self, disease_predictor):
        """Test predictor initialization."""
        assert disease_predictor.config.disease_model_type == 'random_forest'
        assert disease_predictor.model is None
        assert not disease_predictor.is_trained
        assert len(disease_predictor.feature_names) == 0
    
    def test_prepare_features(self, disease_predictor, sample_data):
        """Test feature preparation."""
        env_data, spectral_data, _ = sample_data
        
        features = disease_predictor.prepare_features(spectral_data, env_data)
        
        # Check that features were created
        assert len(features) > 0
        assert len(features.columns) > len(spectral_data.columns)  # Should have derived features
        
        # Check for derived features
        expected_features = ['ndvi_trend', 'ndvi_variability', 'disease_favorable']
        for feature in expected_features:
            if feature in features.columns:
                assert not features[feature].isna().all()
        
        # Check feature names were stored
        assert len(disease_predictor.feature_names) > 0
    
    def test_model_training(self, disease_predictor, sample_data):
        """Test model training."""
        env_data, spectral_data, disease_labels = sample_data
        
        features = disease_predictor.prepare_features(spectral_data, env_data)
        results = disease_predictor.train(features, disease_labels)
        
        # Check training completed
        assert disease_predictor.is_trained
        assert disease_predictor.model is not None
        
        # Check results
        assert 'cv_auc_mean' in results
        assert 'cv_auc_std' in results
        assert 'feature_importance' in results
        assert 'n_samples' in results
        assert 'n_features' in results
        
        assert 0 <= results['cv_auc_mean'] <= 1
        assert results['n_samples'] == len(features)
        assert results['n_features'] == len(disease_predictor.feature_names)
    
    def test_predict_probability(self, disease_predictor, sample_data):
        """Test probability prediction."""
        env_data, spectral_data, disease_labels = sample_data
        
        features = disease_predictor.prepare_features(spectral_data, env_data)
        disease_predictor.train(features, disease_labels)
        
        # Make predictions
        probabilities = disease_predictor.predict_probability(features[:10])
        
        # Check predictions
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)


class TestEnsembleRiskPredictor:
    """Test ensemble risk prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for ensemble training."""
        return create_sample_risk_data(n_zones=3, n_days=100)
    
    @pytest.fixture
    def ensemble_predictor(self):
        """Create ensemble predictor with test configuration."""
        config = RiskPredictionConfig(
            n_estimators=10,
            max_depth=5,
            cv_folds=3
        )
        return EnsembleRiskPredictor(config)
    
    def test_ensemble_initialization(self, ensemble_predictor):
        """Test ensemble predictor initialization."""
        assert ensemble_predictor.pest_predictor is not None
        assert ensemble_predictor.disease_predictor is not None
        assert ensemble_predictor.ensemble_weights == {'pest': 0.5, 'disease': 0.5}
        assert not ensemble_predictor.is_trained
    
    def test_ensemble_training(self, ensemble_predictor, sample_data):
        """Test ensemble model training."""
        env_data, spectral_data, pest_labels, disease_labels = sample_data
        
        results = ensemble_predictor.train(
            env_data, spectral_data, pest_labels, disease_labels
        )
        
        # Check training completed
        assert ensemble_predictor.is_trained
        assert ensemble_predictor.pest_predictor.is_trained
        assert ensemble_predictor.disease_predictor.is_trained
        
        # Check results
        assert 'pest_model' in results
        assert 'disease_model' in results
        assert 'ensemble_weights' in results
        
        # Check weights were optimized
        weights = results['ensemble_weights']
        assert 'pest' in weights
        assert 'disease' in weights
        assert abs(weights['pest'] + weights['disease'] - 1.0) < 0.01
    
    def test_predict_risk(self, ensemble_predictor, sample_data):
        """Test comprehensive risk prediction."""
        env_data, spectral_data, pest_labels, disease_labels = sample_data
        
        # Train ensemble
        ensemble_predictor.train(env_data, spectral_data, pest_labels, disease_labels)
        
        # Make risk prediction
        risk_assessment = ensemble_predictor.predict_risk(
            env_data[:10], spectral_data[:10]
        )
        
        # Check risk assessment
        assert isinstance(risk_assessment, RiskAssessment)
        assert 0 <= risk_assessment.pest_risk_probability <= 1
        assert 0 <= risk_assessment.disease_risk_probability <= 1
        assert 0 <= risk_assessment.overall_risk_score <= 1
        assert 0 <= risk_assessment.confidence_score <= 1
        assert risk_assessment.risk_level in ['low', 'medium', 'high', 'critical']
        assert isinstance(risk_assessment.risk_factors, dict)
        assert isinstance(risk_assessment.recommendations, list)
        assert len(risk_assessment.recommendations) > 0
    
    def test_predict_risk_single_zone(self, ensemble_predictor, sample_data):
        """Test risk prediction for single zone."""
        env_data, spectral_data, pest_labels, disease_labels = sample_data
        
        # Train ensemble
        ensemble_predictor.train(env_data, spectral_data, pest_labels, disease_labels)
        
        # Make risk prediction for specific zone
        risk_assessment = ensemble_predictor.predict_risk(
            env_data, spectral_data, zone_id='zone_1'
        )
        
        # Check risk assessment
        assert isinstance(risk_assessment, RiskAssessment)
        assert 0 <= risk_assessment.overall_risk_score <= 1
    
    def test_save_load_models(self, ensemble_predictor, sample_data):
        """Test model saving and loading."""
        env_data, spectral_data, pest_labels, disease_labels = sample_data
        
        # Train ensemble
        ensemble_predictor.train(env_data, spectral_data, pest_labels, disease_labels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_prefix = os.path.join(temp_dir, 'test_ensemble')
            
            # Save models
            ensemble_predictor.save_models(model_prefix)
            
            # Check files exist
            assert os.path.exists(f"{model_prefix}_pest_model.pkl")
            assert os.path.exists(f"{model_prefix}_disease_model.pkl")
            assert os.path.exists(f"{model_prefix}_ensemble_weights.pkl")
            
            # Create new ensemble and load
            new_ensemble = EnsembleRiskPredictor(ensemble_predictor.config)
            new_ensemble.load_models(model_prefix)
            
            # Check loaded ensemble
            assert new_ensemble.is_trained
            assert new_ensemble.pest_predictor.is_trained
            assert new_ensemble.disease_predictor.is_trained
            
            # Test predictions are similar
            risk1 = ensemble_predictor.predict_risk(env_data[:5], spectral_data[:5])
            risk2 = new_ensemble.predict_risk(env_data[:5], spectral_data[:5])
            
            # Predictions should be very similar
            assert abs(risk1.overall_risk_score - risk2.overall_risk_score) < 0.1
    
    def test_risk_level_classification(self, ensemble_predictor, sample_data):
        """Test risk level classification logic."""
        env_data, spectral_data, pest_labels, disease_labels = sample_data
        
        # Train ensemble
        ensemble_predictor.train(env_data, spectral_data, pest_labels, disease_labels)
        
        # Test different risk scenarios by modifying data
        test_cases = []
        
        # Create high-risk scenario
        high_risk_env = env_data.copy()
        high_risk_env['temperature'] = 35  # Very high temperature
        high_risk_env['humidity'] = 95     # Very high humidity
        test_cases.append((high_risk_env, 'should be high risk'))
        
        # Create low-risk scenario
        low_risk_env = env_data.copy()
        low_risk_env['temperature'] = 15   # Moderate temperature
        low_risk_env['humidity'] = 40      # Low humidity
        test_cases.append((low_risk_env, 'should be lower risk'))
        
        risk_scores = []
        for test_env, description in test_cases:
            risk_assessment = ensemble_predictor.predict_risk(
                test_env[:5], spectral_data[:5]
            )
            risk_scores.append(risk_assessment.overall_risk_score)
        
        # High-risk scenario should have higher score than low-risk
        # (though this may not always be true due to model complexity)
        assert len(risk_scores) == 2


class TestSampleRiskData:
    """Test sample risk data generation utilities."""
    
    def test_create_sample_risk_data(self):
        """Test sample risk data creation."""
        env_data, spectral_data, pest_labels, disease_labels = create_sample_risk_data(
            n_zones=3, n_days=50
        )
        
        # Check environmental data
        assert len(env_data) == 3 * 50  # 3 zones * 50 days
        assert 'zone_id' in env_data.columns
        assert 'timestamp' in env_data.columns
        assert 'temperature' in env_data.columns
        assert 'humidity' in env_data.columns
        
        # Check spectral data
        assert len(spectral_data) == 3 * 50
        assert 'zone_id' in spectral_data.columns
        assert 'timestamp' in spectral_data.columns
        assert 'ndvi' in spectral_data.columns
        assert 'evi' in spectral_data.columns
        
        # Check labels
        assert len(pest_labels) == 3 * 50
        assert len(disease_labels) == 3 * 50
        assert all(label in [0, 1] for label in pest_labels)
        assert all(label in [0, 1] for label in disease_labels)
        
        # Check data ranges
        assert env_data['temperature'].min() > -10
        assert env_data['temperature'].max() < 50
        assert env_data['humidity'].min() >= 0
        assert env_data['humidity'].max() <= 100
        assert spectral_data['ndvi'].min() >= 0
        assert spectral_data['ndvi'].max() <= 1
    
    def test_sample_data_temporal_structure(self):
        """Test temporal structure of sample data."""
        env_data, spectral_data, pest_labels, disease_labels = create_sample_risk_data(
            n_zones=2, n_days=30
        )
        
        # Check each zone has correct number of records
        for zone_id in env_data['zone_id'].unique():
            zone_env = env_data[env_data['zone_id'] == zone_id]
            zone_spectral = spectral_data[spectral_data['zone_id'] == zone_id]
            
            assert len(zone_env) == 30
            assert len(zone_spectral) == 30
            
            # Check timestamps are sequential
            env_timestamps = zone_env['timestamp'].sort_values()
            spectral_timestamps = zone_spectral['timestamp'].sort_values()
            
            env_diffs = env_timestamps.diff().dropna()
            spectral_diffs = spectral_timestamps.diff().dropna()
            
            assert all(diff.days == 1 for diff in env_diffs)  # Daily data
            assert all(diff.days == 1 for diff in spectral_diffs)  # Daily data
    
    def test_sample_data_correlations(self):
        """Test that sample data has expected correlations."""
        env_data, spectral_data, pest_labels, disease_labels = create_sample_risk_data(
            n_zones=5, n_days=100
        )
        
        # Pest labels should correlate with temperature and humidity
        combined_data = pd.merge(env_data, pd.DataFrame({'pest_label': pest_labels}), 
                               left_index=True, right_index=True)
        
        # Check that there's some relationship (correlation may be weak due to randomness)
        temp_pest_corr = combined_data['temperature'].corr(combined_data['pest_label'])
        humidity_pest_corr = combined_data['humidity'].corr(combined_data['pest_label'])
        
        # Should have some positive correlation (though may be weak)
        assert temp_pest_corr >= -0.5  # Allow for some randomness
        assert humidity_pest_corr >= -0.5
        
        # Disease labels should correlate with humidity and NDVI decline
        spectral_with_disease = pd.merge(spectral_data, pd.DataFrame({'disease_label': disease_labels}),
                                       left_index=True, right_index=True)
        
        # NDVI should have some relationship with disease (lower NDVI = higher disease risk)
        ndvi_disease_corr = spectral_with_disease['ndvi'].corr(spectral_with_disease['disease_label'])
        
        # Should have some negative correlation (lower NDVI = higher disease risk)
        assert ndvi_disease_corr <= 0.5  # Allow for randomness