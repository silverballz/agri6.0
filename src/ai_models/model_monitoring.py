"""
Model performance monitoring and retraining pipeline.

This module provides automated monitoring of model performance,
triggers for retraining, and model versioning capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import joblib
import hashlib
from abc import ABC, abstractmethod

from .temporal_lstm import TemporalLSTM, LSTMConfig, AnomalyDetector
from .spatial_cnn import SpatialCNN, CNNConfig
from .training_pipeline import LSTMTrainingPipeline
from ..database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_id: str
    model_type: str  # 'lstm' or 'cnn'
    timestamp: datetime
    accuracy: float
    loss: float
    mae: Optional[float] = None
    mse: Optional[float] = None
    r2: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    data_hash: str = ""
    sample_count: int = 0


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_type: str
    created_at: datetime
    config: Dict[str, Any]
    metrics: ModelMetrics
    file_path: str
    is_active: bool = False
    parent_version: Optional[str] = None
    notes: str = ""


@dataclass
class RetrainingTrigger:
    """Configuration for retraining triggers."""
    performance_threshold: float = 0.1  # Trigger if performance drops by this amount
    data_drift_threshold: float = 0.2   # Trigger if data distribution changes
    time_threshold_days: int = 30       # Trigger after this many days
    min_new_samples: int = 100          # Minimum new samples needed for retraining
    enable_auto_retrain: bool = True    # Whether to automatically retrain


class ModelMonitor(ABC):
    """Abstract base class for model monitoring."""
    
    @abstractmethod
    def evaluate_performance(self, model: Any, test_data: Any) -> ModelMetrics:
        """Evaluate model performance on test data."""
        pass
    
    @abstractmethod
    def detect_data_drift(self, 
                         baseline_data: Any, 
                         new_data: Any) -> Tuple[bool, float]:
        """Detect if data distribution has drifted."""
        pass


class LSTMMonitor(ModelMonitor):
    """Monitor for LSTM temporal models."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        
    def evaluate_performance(self, 
                           model: TemporalLSTM, 
                           test_data: Tuple[np.ndarray, np.ndarray]) -> ModelMetrics:
        """Evaluate LSTM model performance."""
        X_test, y_test = test_data
        
        # Get model predictions
        predictions = model.predict(X_test, return_confidence=False)
        
        # Calculate metrics
        metrics_dict = model.evaluate(X_test, y_test)
        
        # Create data hash for drift detection
        data_hash = hashlib.md5(X_test.tobytes()).hexdigest()
        
        return ModelMetrics(
            model_id=self.model_id,
            model_type='lstm',
            timestamp=datetime.now(),
            accuracy=1.0 - metrics_dict['mae'],  # Convert MAE to accuracy-like metric
            loss=metrics_dict['mse'],
            mae=metrics_dict['mae'],
            mse=metrics_dict['mse'],
            r2=metrics_dict['r2'],
            data_hash=data_hash,
            sample_count=len(X_test)
        )
    
    def detect_data_drift(self, 
                         baseline_data: np.ndarray, 
                         new_data: np.ndarray) -> Tuple[bool, float]:
        """Detect data drift using statistical tests."""
        # Calculate feature-wise KL divergence
        drift_scores = []
        
        for feature_idx in range(baseline_data.shape[-1]):
            baseline_feature = baseline_data[:, :, feature_idx].flatten()
            new_feature = new_data[:, :, feature_idx].flatten()
            
            # Calculate histograms
            bins = np.linspace(
                min(baseline_feature.min(), new_feature.min()),
                max(baseline_feature.max(), new_feature.max()),
                50
            )
            
            baseline_hist, _ = np.histogram(baseline_feature, bins=bins, density=True)
            new_hist, _ = np.histogram(new_feature, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            baseline_hist += 1e-8
            new_hist += 1e-8
            
            # Calculate KL divergence
            kl_div = np.sum(new_hist * np.log(new_hist / baseline_hist))
            drift_scores.append(kl_div)
        
        # Average drift score across features
        avg_drift = np.mean(drift_scores)
        
        # Threshold for drift detection (can be tuned)
        drift_threshold = 0.5
        has_drifted = avg_drift > drift_threshold
        
        logger.info(f"Data drift score: {avg_drift:.3f}, Drifted: {has_drifted}")
        return has_drifted, avg_drift


class CNNMonitor(ModelMonitor):
    """Monitor for CNN spatial models."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        
    def evaluate_performance(self, 
                           model: SpatialCNN, 
                           test_data: Tuple[np.ndarray, np.ndarray]) -> ModelMetrics:
        """Evaluate CNN model performance."""
        X_test, y_test = test_data
        
        # Get model predictions and metrics
        metrics_dict = model.evaluate(X_test, y_test)
        
        # Create data hash for drift detection
        data_hash = hashlib.md5(X_test.tobytes()).hexdigest()
        
        return ModelMetrics(
            model_id=self.model_id,
            model_type='cnn',
            timestamp=datetime.now(),
            accuracy=metrics_dict['accuracy'],
            loss=metrics_dict['classification_report']['macro avg']['f1-score'],
            f1_score=metrics_dict['classification_report']['macro avg']['f1-score'],
            precision=metrics_dict['classification_report']['macro avg']['precision'],
            recall=metrics_dict['classification_report']['macro avg']['recall'],
            data_hash=data_hash,
            sample_count=len(X_test)
        )
    
    def detect_data_drift(self, 
                         baseline_data: np.ndarray, 
                         new_data: np.ndarray) -> Tuple[bool, float]:
        """Detect data drift in image data."""
        # Calculate channel-wise statistics
        drift_scores = []
        
        for channel in range(baseline_data.shape[-1]):
            baseline_channel = baseline_data[:, :, :, channel].flatten()
            new_channel = new_data[:, :, :, channel].flatten()
            
            # Calculate statistical moments
            baseline_stats = {
                'mean': np.mean(baseline_channel),
                'std': np.std(baseline_channel),
                'skew': self._calculate_skewness(baseline_channel),
                'kurt': self._calculate_kurtosis(baseline_channel)
            }
            
            new_stats = {
                'mean': np.mean(new_channel),
                'std': np.std(new_channel),
                'skew': self._calculate_skewness(new_channel),
                'kurt': self._calculate_kurtosis(new_channel)
            }
            
            # Calculate normalized differences
            stat_diffs = []
            for stat_name in baseline_stats:
                baseline_val = baseline_stats[stat_name]
                new_val = new_stats[stat_name]
                
                if baseline_val != 0:
                    diff = abs(new_val - baseline_val) / abs(baseline_val)
                else:
                    diff = abs(new_val)
                
                stat_diffs.append(diff)
            
            drift_scores.append(np.mean(stat_diffs))
        
        # Average drift score across channels
        avg_drift = np.mean(drift_scores)
        
        # Threshold for drift detection
        drift_threshold = 0.3
        has_drifted = avg_drift > drift_threshold
        
        logger.info(f"Image data drift score: {avg_drift:.3f}, Drifted: {has_drifted}")
        return has_drifted, avg_drift
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3


class ModelVersionManager:
    """Manages model versions and rollback capabilities."""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_path / "versions.json"
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load model versions from file."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                versions = {}
                for version_id, version_data in data.items():
                    # Convert datetime strings back to datetime objects
                    version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                    version_data['metrics']['timestamp'] = datetime.fromisoformat(
                        version_data['metrics']['timestamp']
                    )
                    
                    # Reconstruct ModelVersion and ModelMetrics objects
                    metrics = ModelMetrics(**version_data['metrics'])
                    version_data['metrics'] = metrics
                    versions[version_id] = ModelVersion(**version_data)
                
                return versions
        return {}
    
    def _save_versions(self):
        """Save model versions to file."""
        data = {}
        for version_id, version in self.versions.items():
            version_dict = asdict(version)
            # Convert datetime objects to strings for JSON serialization
            version_dict['created_at'] = version.created_at.isoformat()
            version_dict['metrics']['timestamp'] = version.metrics.timestamp.isoformat()
            data[version_id] = version_dict
        
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(self,
                      model_type: str,
                      config: Dict[str, Any],
                      metrics: ModelMetrics,
                      file_path: str,
                      parent_version: str = None,
                      notes: str = "") -> str:
        """Create a new model version."""
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_type}_{timestamp}"
        
        # Deactivate previous active version of same type
        for version in self.versions.values():
            if version.model_type == model_type and version.is_active:
                version.is_active = False
        
        # Create new version
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=datetime.now(),
            config=config,
            metrics=metrics,
            file_path=file_path,
            is_active=True,
            parent_version=parent_version,
            notes=notes
        )
        
        self.versions[version_id] = version
        self._save_versions()
        
        logger.info(f"Created model version: {version_id}")
        return version_id
    
    def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model type."""
        for version in self.versions.values():
            if version.model_type == model_type and version.is_active:
                return version
        return None
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a specific model version."""
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        target_version = self.versions[version_id]
        
        # Deactivate current active version
        for version in self.versions.values():
            if version.model_type == target_version.model_type and version.is_active:
                version.is_active = False
        
        # Activate target version
        target_version.is_active = True
        self._save_versions()
        
        logger.info(f"Rolled back to version: {version_id}")
        return True
    
    def list_versions(self, model_type: str = None) -> List[ModelVersion]:
        """List all versions, optionally filtered by model type."""
        versions = list(self.versions.values())
        
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def cleanup_old_versions(self, keep_count: int = 5):
        """Remove old model versions, keeping only the most recent ones."""
        for model_type in ['lstm', 'cnn']:
            versions = self.list_versions(model_type)
            
            if len(versions) > keep_count:
                versions_to_remove = versions[keep_count:]
                
                for version in versions_to_remove:
                    # Don't remove active version
                    if not version.is_active:
                        # Remove model file
                        model_file = Path(version.file_path)
                        if model_file.exists():
                            model_file.unlink()
                        
                        # Remove from versions dict
                        del self.versions[version.version_id]
                        logger.info(f"Removed old version: {version.version_id}")
                
                self._save_versions()


class ModelRetrainingPipeline:
    """Automated model retraining pipeline."""
    
    def __init__(self, 
                 trigger_config: RetrainingTrigger = None,
                 base_path: str = "models"):
        self.trigger_config = trigger_config or RetrainingTrigger()
        self.version_manager = ModelVersionManager(base_path)
        self.monitors = {
            'lstm': LSTMMonitor('lstm_model'),
            'cnn': CNNMonitor('cnn_model')
        }
        self.baseline_metrics = {}
        self.baseline_data = {}
        
    def register_baseline(self,
                         model_type: str,
                         metrics: ModelMetrics,
                         baseline_data: Any):
        """Register baseline metrics and data for comparison."""
        self.baseline_metrics[model_type] = metrics
        self.baseline_data[model_type] = baseline_data
        logger.info(f"Registered baseline for {model_type} model")
    
    def check_retraining_triggers(self,
                                 model_type: str,
                                 current_metrics: ModelMetrics,
                                 new_data: Any) -> Dict[str, Any]:
        """Check if retraining should be triggered."""
        triggers = {
            'performance_degradation': False,
            'data_drift': False,
            'time_threshold': False,
            'sufficient_data': False,
            'should_retrain': False
        }
        
        if model_type not in self.baseline_metrics:
            logger.warning(f"No baseline metrics for {model_type}")
            return triggers
        
        baseline = self.baseline_metrics[model_type]
        
        # Check performance degradation
        performance_drop = baseline.accuracy - current_metrics.accuracy
        if performance_drop > self.trigger_config.performance_threshold:
            triggers['performance_degradation'] = True
            logger.info(f"Performance degradation detected: {performance_drop:.3f}")
        
        # Check data drift
        if model_type in self.baseline_data:
            monitor = self.monitors[model_type]
            has_drift, drift_score = monitor.detect_data_drift(
                self.baseline_data[model_type], 
                new_data
            )
            if has_drift:
                triggers['data_drift'] = True
                logger.info(f"Data drift detected: {drift_score:.3f}")
        
        # Check time threshold
        time_since_baseline = datetime.now() - baseline.timestamp
        if time_since_baseline.days > self.trigger_config.time_threshold_days:
            triggers['time_threshold'] = True
            logger.info(f"Time threshold exceeded: {time_since_baseline.days} days")
        
        # Check sufficient new data
        if current_metrics.sample_count >= self.trigger_config.min_new_samples:
            triggers['sufficient_data'] = True
        
        # Determine if retraining should be triggered
        triggers['should_retrain'] = (
            self.trigger_config.enable_auto_retrain and
            triggers['sufficient_data'] and
            (triggers['performance_degradation'] or 
             triggers['data_drift'] or 
             triggers['time_threshold'])
        )
        
        return triggers
    
    def retrain_lstm_model(self,
                          training_data: pd.DataFrame,
                          validation_data: pd.DataFrame = None,
                          config: LSTMConfig = None) -> str:
        """Retrain LSTM model with new data."""
        logger.info("Starting LSTM model retraining")
        
        # Initialize training pipeline
        pipeline = LSTMTrainingPipeline(config)
        
        # Train model
        results = pipeline.train_model(training_data)
        
        # Evaluate on validation data
        if validation_data is not None:
            validation_results = pipeline.validate_model_performance(validation_data)
            metrics = ModelMetrics(
                model_id='lstm_retrained',
                model_type='lstm',
                timestamp=datetime.now(),
                accuracy=1.0 - validation_results['metrics']['mae'],
                loss=validation_results['metrics']['mse'],
                mae=validation_results['metrics']['mae'],
                mse=validation_results['metrics']['mse'],
                r2=validation_results['metrics']['r2'],
                sample_count=validation_results['test_sequences']
            )
        else:
            # Use training metrics if no validation data
            metrics = ModelMetrics(
                model_id='lstm_retrained',
                model_type='lstm',
                timestamp=datetime.now(),
                accuracy=1.0 - results['validation_metrics']['mae'],
                loss=results['validation_metrics']['mse'],
                mae=results['validation_metrics']['mae'],
                mse=results['validation_metrics']['mse'],
                r2=results['validation_metrics']['r2'],
                sample_count=results['validation_sequences']
            )
        
        # Save model
        model_paths = pipeline.save_pipeline()
        
        # Create version
        current_version = self.version_manager.get_active_version('lstm')
        parent_version = current_version.version_id if current_version else None
        
        version_id = self.version_manager.create_version(
            model_type='lstm',
            config=asdict(pipeline.config),
            metrics=metrics,
            file_path=model_paths['model_path'],
            parent_version=parent_version,
            notes="Automated retraining"
        )
        
        logger.info(f"LSTM model retrained successfully: {version_id}")
        return version_id
    
    def retrain_cnn_model(self,
                         training_data: Tuple[np.ndarray, np.ndarray],
                         validation_data: Tuple[np.ndarray, np.ndarray] = None,
                         config: CNNConfig = None) -> str:
        """Retrain CNN model with new data."""
        logger.info("Starting CNN model retraining")
        
        # Initialize model
        model = SpatialCNN(config)
        
        # Prepare training data
        X_train, y_train = training_data
        X_train, y_train = model.prepare_training_data(X_train, y_train)
        
        # Train model
        history = model.train(X_train, y_train, validation_data)
        
        # Evaluate model
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val, y_val = model.prepare_training_data(X_val, y_val)
            metrics_dict = model.evaluate(X_val, y_val)
            
            metrics = ModelMetrics(
                model_id='cnn_retrained',
                model_type='cnn',
                timestamp=datetime.now(),
                accuracy=metrics_dict['accuracy'],
                loss=metrics_dict['classification_report']['macro avg']['f1-score'],
                f1_score=metrics_dict['classification_report']['macro avg']['f1-score'],
                precision=metrics_dict['classification_report']['macro avg']['precision'],
                recall=metrics_dict['classification_report']['macro avg']['recall'],
                sample_count=len(X_val)
            )
        else:
            # Use training metrics
            metrics = ModelMetrics(
                model_id='cnn_retrained',
                model_type='cnn',
                timestamp=datetime.now(),
                accuracy=0.9,  # Placeholder - would need actual validation
                loss=0.1,
                sample_count=len(X_train)
            )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/cnn_model_{timestamp}.h5"
        model.save_model(model_path)
        
        # Create version
        current_version = self.version_manager.get_active_version('cnn')
        parent_version = current_version.version_id if current_version else None
        
        version_id = self.version_manager.create_version(
            model_type='cnn',
            config=asdict(model.config),
            metrics=metrics,
            file_path=model_path,
            parent_version=parent_version,
            notes="Automated retraining"
        )
        
        logger.info(f"CNN model retrained successfully: {version_id}")
        return version_id
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle for all models."""
        results = {
            'timestamp': datetime.now(),
            'models_checked': [],
            'retraining_triggered': [],
            'errors': []
        }
        
        # Check each model type
        for model_type in ['lstm', 'cnn']:
            try:
                # Get current active version
                current_version = self.version_manager.get_active_version(model_type)
                if not current_version:
                    logger.warning(f"No active version found for {model_type}")
                    continue
                
                # Load test data (this would be implemented based on your data pipeline)
                test_data = self._load_test_data(model_type)
                if test_data is None:
                    continue
                
                # Load model and evaluate
                if model_type == 'lstm':
                    model = TemporalLSTM()
                    model.load_model(current_version.file_path)
                    monitor = self.monitors['lstm']
                else:
                    model = SpatialCNN()
                    model.load_model(current_version.file_path)
                    monitor = self.monitors['cnn']
                
                # Evaluate current performance
                current_metrics = monitor.evaluate_performance(model, test_data)
                
                # Check retraining triggers
                triggers = self.check_retraining_triggers(
                    model_type, 
                    current_metrics, 
                    test_data[0]  # Input data for drift detection
                )
                
                results['models_checked'].append({
                    'model_type': model_type,
                    'version_id': current_version.version_id,
                    'metrics': asdict(current_metrics),
                    'triggers': triggers
                })
                
                # Trigger retraining if needed
                if triggers['should_retrain']:
                    logger.info(f"Triggering retraining for {model_type}")
                    
                    # Load training data (this would be implemented)
                    training_data = self._load_training_data(model_type)
                    
                    if model_type == 'lstm':
                        new_version_id = self.retrain_lstm_model(training_data)
                    else:
                        new_version_id = self.retrain_cnn_model(training_data)
                    
                    results['retraining_triggered'].append({
                        'model_type': model_type,
                        'old_version': current_version.version_id,
                        'new_version': new_version_id
                    })
                
            except Exception as e:
                logger.error(f"Error monitoring {model_type}: {str(e)}")
                results['errors'].append({
                    'model_type': model_type,
                    'error': str(e)
                })
        
        return results
    
    def _load_test_data(self, model_type: str) -> Optional[Any]:
        """Load test data for model evaluation."""
        # This would be implemented based on your data pipeline
        # For now, return None to indicate no test data available
        logger.info(f"Loading test data for {model_type} (placeholder)")
        return None
    
    def _load_training_data(self, model_type: str) -> Any:
        """Load training data for model retraining."""
        # This would be implemented based on your data pipeline
        logger.info(f"Loading training data for {model_type} (placeholder)")
        return None