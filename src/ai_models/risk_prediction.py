"""
Risk prediction models for agricultural monitoring.

This module implements models for predicting pest outbreaks, disease risks,
and other agricultural threats using environmental conditions and spectral data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)


@dataclass
class RiskPredictionConfig:
    """Configuration for risk prediction models."""
    pest_model_type: str = 'random_forest'  # 'random_forest', 'logistic', 'gradient_boost'
    disease_model_type: str = 'random_forest'
    ensemble_method: str = 'voting'  # 'voting', 'stacking', 'weighted_average'
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    pest_risk_probability: float
    disease_risk_probability: float
    overall_risk_score: float
    risk_factors: Dict[str, float]
    confidence_score: float
    recommendations: List[str]
    risk_level: str  # 'low', 'medium', 'high', 'critical'


class PestOutbreakPredictor:
    """
    Model for predicting pest outbreak probability.
    
    Uses environmental conditions and historical data to predict
    the likelihood of pest outbreaks in agricultural fields.
    """
    
    def __init__(self, config: RiskPredictionConfig = None):
        """Initialize pest outbreak predictor."""
        self.config = config or RiskPredictionConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def _create_model(self):
        """Create pest prediction model based on configuration."""
        if self.config.pest_model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        elif self.config.pest_model_type == 'logistic':
            return LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        elif self.config.pest_model_type == 'gradient_boost':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.pest_model_type}")
    
    def prepare_features(self, 
                        environmental_data: pd.DataFrame,
                        spectral_data: pd.DataFrame = None,
                        historical_outbreaks: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare features for pest outbreak prediction.
        
        Args:
            environmental_data: Environmental sensor data
            spectral_data: Spectral indices data
            historical_outbreaks: Historical outbreak data
            
        Returns:
            Feature DataFrame
        """
        features = environmental_data.copy()
        
        # Add derived environmental features
        if 'temperature' in features.columns and 'humidity' in features.columns:
            # Heat index (simplified)
            features['heat_index'] = features['temperature'] + 0.5 * features['humidity']
            
            # Vapor pressure deficit (simplified)
            features['vpd'] = features['temperature'] - features['humidity'] / 100 * features['temperature']
        
        # Add temperature and humidity ranges
        if 'temperature' in features.columns:
            features['temp_range'] = features.groupby('zone_id')['temperature'].transform(
                lambda x: x.rolling(window=7, min_periods=1).max() - x.rolling(window=7, min_periods=1).min()
            )
        
        # Add growing degree days (simplified)
        if 'temperature' in features.columns:
            base_temp = 10  # Base temperature for pest development
            features['gdd'] = np.maximum(features['temperature'] - base_temp, 0)
            features['cumulative_gdd'] = features.groupby('zone_id')['gdd'].cumsum()
        
        # Add spectral features if available
        if spectral_data is not None:
            # Merge spectral data
            features = pd.merge(
                features,
                spectral_data,
                on=['zone_id', 'timestamp'],
                how='left'
            )
            
            # Add spectral stress indicators
            if 'ndvi' in features.columns:
                features['ndvi_decline'] = features.groupby('zone_id')['ndvi'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).apply(
                        lambda y: y.iloc[0] - y.iloc[-1] if len(y) > 1 else 0
                    )
                )
        
        # Add seasonal features
        if 'timestamp' in features.columns:
            features['month'] = pd.to_datetime(features['timestamp']).dt.month
            features['day_of_year'] = pd.to_datetime(features['timestamp']).dt.dayofyear
            features['season'] = pd.to_datetime(features['timestamp']).dt.month % 12 // 3 + 1
        
        # Add historical outbreak features if available
        if historical_outbreaks is not None:
            # Count recent outbreaks
            features['recent_outbreaks'] = 0  # Placeholder - would need proper implementation
        
        # Remove non-numeric columns for modeling
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Store feature names
        self.feature_names = list(numeric_features.columns)
        
        logger.info(f"Prepared {len(self.feature_names)} features for pest prediction")
        return numeric_features
    
    def train(self, 
              features: pd.DataFrame,
              outbreak_labels: pd.Series) -> Dict[str, Any]:
        """
        Train pest outbreak prediction model.
        
        Args:
            features: Feature DataFrame
            outbreak_labels: Binary labels (1 = outbreak, 0 = no outbreak)
            
        Returns:
            Training results and metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        y = outbreak_labels.values
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, 
            cv=self.config.cv_folds, 
            scoring='roc_auc'
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            feature_importance = {}
        
        self.is_trained = True
        
        results = {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'feature_importance': feature_importance,
            'n_samples': len(X_scaled),
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Pest model trained. CV AUC: {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f}")
        return results
    
    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict pest outbreak probability.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Array of outbreak probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of outbreak
        
        return probabilities
    
    def get_risk_factors(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Get risk factors contributing to pest outbreak probability.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary of risk factors and their contributions
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Calculate mean feature values
        mean_values = features.mean()
        
        # Combine importance with current values
        risk_factors = {}
        for i, feature in enumerate(self.feature_names):
            if feature in mean_values:
                risk_factors[feature] = float(importance[i] * abs(mean_values[feature]))
        
        # Normalize to sum to 1
        total = sum(risk_factors.values())
        if total > 0:
            risk_factors = {k: v / total for k, v in risk_factors.items()}
        
        return risk_factors


class DiseaseRiskPredictor:
    """
    Model for predicting disease risk based on spectral signatures.
    
    Uses vegetation indices and environmental conditions to assess
    disease risk in agricultural crops.
    """
    
    def __init__(self, config: RiskPredictionConfig = None):
        """Initialize disease risk predictor."""
        self.config = config or RiskPredictionConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def _create_model(self):
        """Create disease prediction model based on configuration."""
        if self.config.disease_model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        elif self.config.disease_model_type == 'logistic':
            return LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        elif self.config.disease_model_type == 'gradient_boost':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.disease_model_type}")
    
    def prepare_features(self,
                        spectral_data: pd.DataFrame,
                        environmental_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare features for disease risk prediction.
        
        Args:
            spectral_data: Spectral indices data
            environmental_data: Environmental sensor data
            
        Returns:
            Feature DataFrame
        """
        features = spectral_data.copy()
        
        # Add spectral stress indicators
        if 'ndvi' in features.columns:
            # NDVI trend and variability
            features['ndvi_trend'] = features.groupby('zone_id')['ndvi'].transform(
                lambda x: x.rolling(window=7, min_periods=1).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
            features['ndvi_variability'] = features.groupby('zone_id')['ndvi'].transform(
                lambda x: x.rolling(window=7, min_periods=1).std()
            )
        
        # Add red-edge indices if available
        if 'evi' in features.columns and 'ndvi' in features.columns:
            features['evi_ndvi_ratio'] = features['evi'] / (features['ndvi'] + 1e-8)
        
        # Add water stress indicators
        if 'ndwi' in features.columns:
            features['ndwi_trend'] = features.groupby('zone_id')['ndwi'].transform(
                lambda x: x.rolling(window=7, min_periods=1).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
        
        # Add environmental features if available
        if environmental_data is not None:
            features = pd.merge(
                features,
                environmental_data,
                on=['zone_id', 'timestamp'],
                how='left'
            )
            
            # Add disease-favorable conditions
            if 'humidity' in features.columns and 'temperature' in features.columns:
                # High humidity + moderate temperature = disease risk
                features['disease_favorable'] = (
                    (features['humidity'] > 80) & 
                    (features['temperature'].between(15, 25))
                ).astype(int)
                
                # Leaf wetness duration (simplified)
                features['leaf_wetness_hours'] = features['humidity'] / 10  # Simplified proxy
        
        # Add temporal features
        if 'timestamp' in features.columns:
            features['month'] = pd.to_datetime(features['timestamp']).dt.month
            features['day_of_year'] = pd.to_datetime(features['timestamp']).dt.dayofyear
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Store feature names
        self.feature_names = list(numeric_features.columns)
        
        logger.info(f"Prepared {len(self.feature_names)} features for disease prediction")
        return numeric_features
    
    def train(self,
              features: pd.DataFrame,
              disease_labels: pd.Series) -> Dict[str, Any]:
        """
        Train disease risk prediction model.
        
        Args:
            features: Feature DataFrame
            disease_labels: Binary labels (1 = disease, 0 = healthy)
            
        Returns:
            Training results and metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        y = disease_labels.values
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=self.config.cv_folds,
            scoring='roc_auc'
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            feature_importance = {}
        
        self.is_trained = True
        
        results = {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'feature_importance': feature_importance,
            'n_samples': len(X_scaled),
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Disease model trained. CV AUC: {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f}")
        return results
    
    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict disease risk probability.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Array of disease risk probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of disease
        
        return probabilities


class EnsembleRiskPredictor:
    """
    Ensemble model combining multiple risk prediction models.
    
    Combines pest outbreak and disease risk predictions with
    environmental and spectral data for comprehensive risk assessment.
    """
    
    def __init__(self, config: RiskPredictionConfig = None):
        """Initialize ensemble risk predictor."""
        self.config = config or RiskPredictionConfig()
        self.pest_predictor = PestOutbreakPredictor(config)
        self.disease_predictor = DiseaseRiskPredictor(config)
        self.ensemble_weights = {'pest': 0.5, 'disease': 0.5}
        self.is_trained = False
    
    def train(self,
              environmental_data: pd.DataFrame,
              spectral_data: pd.DataFrame,
              pest_labels: pd.Series,
              disease_labels: pd.Series) -> Dict[str, Any]:
        """
        Train ensemble risk prediction model.
        
        Args:
            environmental_data: Environmental sensor data
            spectral_data: Spectral indices data
            pest_labels: Pest outbreak labels
            disease_labels: Disease occurrence labels
            
        Returns:
            Training results for both models
        """
        # Train pest predictor
        pest_features = self.pest_predictor.prepare_features(
            environmental_data, spectral_data
        )
        pest_results = self.pest_predictor.train(pest_features, pest_labels)
        
        # Train disease predictor
        disease_features = self.disease_predictor.prepare_features(
            spectral_data, environmental_data
        )
        disease_results = self.disease_predictor.train(disease_features, disease_labels)
        
        # Optimize ensemble weights based on validation performance
        self._optimize_ensemble_weights(
            pest_features, disease_features,
            pest_labels, disease_labels
        )
        
        self.is_trained = True
        
        results = {
            'pest_model': pest_results,
            'disease_model': disease_results,
            'ensemble_weights': self.ensemble_weights
        }
        
        logger.info("Ensemble risk predictor trained successfully")
        return results
    
    def _optimize_ensemble_weights(self,
                                  pest_features: pd.DataFrame,
                                  disease_features: pd.DataFrame,
                                  pest_labels: pd.Series,
                                  disease_labels: pd.Series):
        """
        Optimize ensemble weights based on model performance.
        
        Args:
            pest_features: Pest prediction features
            disease_features: Disease prediction features
            pest_labels: Pest labels
            disease_labels: Disease labels
        """
        # Get model predictions
        pest_probs = self.pest_predictor.predict_probability(pest_features)
        disease_probs = self.disease_predictor.predict_probability(disease_features)
        
        # Calculate individual model AUC scores
        pest_auc = roc_auc_score(pest_labels, pest_probs)
        disease_auc = roc_auc_score(disease_labels, disease_probs)
        
        # Weight by relative performance
        total_auc = pest_auc + disease_auc
        if total_auc > 0:
            self.ensemble_weights = {
                'pest': pest_auc / total_auc,
                'disease': disease_auc / total_auc
            }
        
        logger.info(f"Optimized ensemble weights: {self.ensemble_weights}")
    
    def predict_risk(self,
                    environmental_data: pd.DataFrame,
                    spectral_data: pd.DataFrame,
                    zone_id: str = None) -> RiskAssessment:
        """
        Predict comprehensive risk assessment.
        
        Args:
            environmental_data: Environmental sensor data
            spectral_data: Spectral indices data
            zone_id: Optional zone ID for filtering
            
        Returns:
            RiskAssessment with comprehensive risk analysis
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before prediction")
        
        # Filter data for specific zone if provided
        if zone_id:
            environmental_data = environmental_data[environmental_data['zone_id'] == zone_id]
            spectral_data = spectral_data[spectral_data['zone_id'] == zone_id]
        
        # Prepare features
        pest_features = self.pest_predictor.prepare_features(
            environmental_data, spectral_data
        )
        disease_features = self.disease_predictor.prepare_features(
            spectral_data, environmental_data
        )
        
        # Get individual predictions
        pest_prob = np.mean(self.pest_predictor.predict_probability(pest_features))
        disease_prob = np.mean(self.disease_predictor.predict_probability(disease_features))
        
        # Calculate overall risk score
        overall_risk = (
            self.ensemble_weights['pest'] * pest_prob +
            self.ensemble_weights['disease'] * disease_prob
        )
        
        # Get risk factors
        pest_factors = self.pest_predictor.get_risk_factors(pest_features)
        
        # Combine risk factors
        risk_factors = {
            **{f"pest_{k}": v * self.ensemble_weights['pest'] for k, v in pest_factors.items()},
            'disease_spectral_stress': disease_prob * self.ensemble_weights['disease']
        }
        
        # Determine risk level
        if overall_risk < 0.25:
            risk_level = 'low'
        elif overall_risk < 0.5:
            risk_level = 'medium'
        elif overall_risk < 0.75:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            pest_prob, disease_prob, overall_risk, risk_factors
        )
        
        # Calculate confidence score
        confidence_score = 1.0 - abs(0.5 - overall_risk) * 2  # Higher confidence for extreme values
        
        return RiskAssessment(
            pest_risk_probability=float(pest_prob),
            disease_risk_probability=float(disease_prob),
            overall_risk_score=float(overall_risk),
            risk_factors=risk_factors,
            confidence_score=float(confidence_score),
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _generate_recommendations(self,
                                pest_prob: float,
                                disease_prob: float,
                                overall_risk: float,
                                risk_factors: Dict[str, float]) -> List[str]:
        """
        Generate actionable recommendations based on risk assessment.
        
        Args:
            pest_prob: Pest outbreak probability
            disease_prob: Disease risk probability
            overall_risk: Overall risk score
            risk_factors: Contributing risk factors
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # High-level recommendations based on overall risk
        if overall_risk > 0.75:
            recommendations.append("CRITICAL: Immediate field inspection recommended")
            recommendations.append("Consider emergency treatment protocols")
        elif overall_risk > 0.5:
            recommendations.append("HIGH RISK: Increase monitoring frequency")
            recommendations.append("Prepare preventive treatments")
        elif overall_risk > 0.25:
            recommendations.append("MODERATE RISK: Continue regular monitoring")
        else:
            recommendations.append("LOW RISK: Maintain standard monitoring schedule")
        
        # Specific recommendations based on risk type
        if pest_prob > 0.6:
            recommendations.append("High pest risk detected - consider pest traps and scouting")
            recommendations.append("Monitor for early signs of pest damage")
        
        if disease_prob > 0.6:
            recommendations.append("High disease risk detected - check for symptoms")
            recommendations.append("Consider fungicide application if conditions persist")
        
        # Recommendations based on top risk factors
        top_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for factor, importance in top_factors:
            if importance > 0.2:  # Significant factor
                if 'temperature' in factor:
                    recommendations.append("Monitor temperature conditions closely")
                elif 'humidity' in factor:
                    recommendations.append("High humidity detected - improve ventilation if possible")
                elif 'ndvi' in factor:
                    recommendations.append("Vegetation stress detected - check irrigation and nutrition")
        
        return recommendations
    
    def save_models(self, filepath_prefix: str):
        """Save ensemble models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        # Save pest model
        pest_path = f"{filepath_prefix}_pest_model.pkl"
        joblib.dump({
            'model': self.pest_predictor.model,
            'scaler': self.pest_predictor.scaler,
            'feature_names': self.pest_predictor.feature_names
        }, pest_path)
        
        # Save disease model
        disease_path = f"{filepath_prefix}_disease_model.pkl"
        joblib.dump({
            'model': self.disease_predictor.model,
            'scaler': self.disease_predictor.scaler,
            'feature_names': self.disease_predictor.feature_names
        }, disease_path)
        
        # Save ensemble weights
        weights_path = f"{filepath_prefix}_ensemble_weights.pkl"
        joblib.dump(self.ensemble_weights, weights_path)
        
        logger.info(f"Ensemble models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load ensemble models."""
        # Load pest model
        pest_path = f"{filepath_prefix}_pest_model.pkl"
        pest_data = joblib.load(pest_path)
        self.pest_predictor.model = pest_data['model']
        self.pest_predictor.scaler = pest_data['scaler']
        self.pest_predictor.feature_names = pest_data['feature_names']
        self.pest_predictor.is_trained = True
        
        # Load disease model
        disease_path = f"{filepath_prefix}_disease_model.pkl"
        disease_data = joblib.load(disease_path)
        self.disease_predictor.model = disease_data['model']
        self.disease_predictor.scaler = disease_data['scaler']
        self.disease_predictor.feature_names = disease_data['feature_names']
        self.disease_predictor.is_trained = True
        
        # Load ensemble weights
        weights_path = f"{filepath_prefix}_ensemble_weights.pkl"
        self.ensemble_weights = joblib.load(weights_path)
        
        self.is_trained = True
        logger.info(f"Ensemble models loaded from prefix: {filepath_prefix}")


def create_sample_risk_data(n_zones: int = 5, 
                           n_days: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create sample data for risk prediction model testing.
    
    Args:
        n_zones: Number of monitoring zones
        n_days: Number of days of data
        
    Returns:
        Tuple of (environmental_data, spectral_data, pest_labels, disease_labels)
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    environmental_data = []
    spectral_data = []
    pest_labels = []
    disease_labels = []
    
    for zone_id in range(1, n_zones + 1):
        for i, date in enumerate(dates):
            zone_name = f'zone_{zone_id}'
            
            # Generate environmental data with seasonal patterns
            temp_base = 20 + 10 * np.sin(2 * np.pi * i / 365)
            temperature = temp_base + np.random.normal(0, 3)
            
            humidity = 60 + 20 * np.sin(2 * np.pi * i / 365 + np.pi/2) + np.random.normal(0, 10)
            humidity = np.clip(humidity, 0, 100)
            
            soil_moisture = 0.3 + 0.2 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 0.05)
            soil_moisture = np.clip(soil_moisture, 0, 1)
            
            environmental_data.append({
                'zone_id': zone_name,
                'timestamp': date,
                'temperature': temperature,
                'humidity': humidity,
                'soil_moisture': soil_moisture,
                'precipitation': max(0, np.random.normal(2, 3)),
                'wind_speed': max(0, np.random.normal(5, 2))
            })
            
            # Generate spectral data
            base_ndvi = 0.6 + 0.2 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 0.05)
            base_ndvi = np.clip(base_ndvi, 0, 1)
            
            spectral_data.append({
                'zone_id': zone_name,
                'timestamp': date,
                'ndvi': base_ndvi,
                'evi': base_ndvi * 1.2 + np.random.normal(0, 0.02),
                'savi': base_ndvi * 0.8 + np.random.normal(0, 0.02),
                'ndwi': 0.4 - base_ndvi * 0.3 + np.random.normal(0, 0.02)
            })
            
            # Generate risk labels based on conditions
            # Pest risk increases with temperature and humidity
            pest_risk_score = (temperature - 15) / 20 + humidity / 100
            pest_outbreak = 1 if pest_risk_score > 1.2 and np.random.random() > 0.7 else 0
            pest_labels.append(pest_outbreak)
            
            # Disease risk increases with high humidity and NDVI decline
            ndvi_decline = max(0, 0.7 - base_ndvi)  # Decline from healthy level
            disease_risk_score = humidity / 100 + ndvi_decline * 2
            disease_occurrence = 1 if disease_risk_score > 1.0 and np.random.random() > 0.8 else 0
            disease_labels.append(disease_occurrence)
    
    environmental_df = pd.DataFrame(environmental_data)
    spectral_df = pd.DataFrame(spectral_data)
    pest_labels_series = pd.Series(pest_labels)
    disease_labels_series = pd.Series(disease_labels)
    
    logger.info(f"Created sample risk data: {len(environmental_df)} records")
    logger.info(f"Pest outbreak rate: {pest_labels_series.mean():.3f}")
    logger.info(f"Disease occurrence rate: {disease_labels_series.mean():.3f}")
    
    return environmental_df, spectral_df, pest_labels_series, disease_labels_series