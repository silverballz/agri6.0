"""
Tests for model monitoring and retraining pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from src.ai_models.model_monitoring import (
    ModelMetrics, ModelVersion, RetrainingTrigger,
    LSTMMonitor, CNNMonitor, ModelVersionManager,
    ModelRetrainingPipeline
)
from src.ai_models.retraining_scheduler import (
    RetrainingScheduler, ScheduleConfig
)
from src.ai_models.temporal_lstm import TemporalLSTM, LSTMConfig
from src.ai_models.spatial_cnn import SpatialCNN, CNNConfig


class TestModelMetrics:
    """Test ModelMetrics dataclass."""
    
    def test_model_metrics_creation(self):
        """Test creating ModelMetrics instance."""
        metrics = ModelMetrics(
            model_id="test_model",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            mae=0.02,
            sample_count=1000
        )
        
        assert metrics.model_id == "test_model"
        assert metrics.model_type == "lstm"
        assert metrics.accuracy == 0.95
        assert metrics.sample_count == 1000


class TestLSTMMonitor:
    """Test LSTM model monitoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = LSTMMonitor("test_lstm")
        
        # Create mock LSTM model
        self.mock_model = Mock(spec=TemporalLSTM)
        self.mock_model.predict.return_value = Mock(predictions=np.array([0.8, 0.7, 0.9]))
        self.mock_model.evaluate.return_value = {
            'mae': 0.05,
            'mse': 0.003,
            'r2': 0.92
        }
    
    def test_evaluate_performance(self):
        """Test LSTM performance evaluation."""
        # Create test data
        X_test = np.random.random((100, 30, 4))
        y_test = np.random.random(100)
        test_data = (X_test, y_test)
        
        # Evaluate performance
        metrics = self.monitor.evaluate_performance(self.mock_model, test_data)
        
        assert metrics.model_type == "lstm"
        assert metrics.accuracy == 0.95  # 1.0 - 0.05 (MAE)
        assert metrics.mae == 0.05
        assert metrics.mse == 0.003
        assert metrics.r2 == 0.92
        assert metrics.sample_count == 100
    
    def test_detect_data_drift(self):
        """Test data drift detection."""
        # Create baseline and new data
        baseline_data = np.random.normal(0, 1, (100, 30, 4))
        new_data = np.random.normal(0.5, 1.2, (100, 30, 4))  # Shifted distribution
        
        has_drift, drift_score = self.monitor.detect_data_drift(baseline_data, new_data)
        
        assert isinstance(has_drift, bool)
        assert isinstance(drift_score, float)
        assert drift_score >= 0
    
    def test_no_data_drift(self):
        """Test when there's no significant data drift."""
        # Create similar data
        baseline_data = np.random.normal(0, 1, (100, 30, 4))
        new_data = np.random.normal(0.01, 1.01, (100, 30, 4))  # Very similar
        
        has_drift, drift_score = self.monitor.detect_data_drift(baseline_data, new_data)
        
        # Should detect minimal drift
        assert drift_score < 1.0


class TestCNNMonitor:
    """Test CNN model monitoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = CNNMonitor("test_cnn")
        
        # Create mock CNN model
        self.mock_model = Mock(spec=SpatialCNN)
        self.mock_model.evaluate.return_value = {
            'accuracy': 0.88,
            'classification_report': {
                'macro avg': {
                    'f1-score': 0.85,
                    'precision': 0.87,
                    'recall': 0.83
                }
            }
        }
    
    def test_evaluate_performance(self):
        """Test CNN performance evaluation."""
        # Create test data
        X_test = np.random.randint(0, 10000, (50, 64, 64, 6))
        y_test = np.random.randint(0, 4, 50)
        test_data = (X_test, y_test)
        
        # Evaluate performance
        metrics = self.monitor.evaluate_performance(self.mock_model, test_data)
        
        assert metrics.model_type == "cnn"
        assert metrics.accuracy == 0.88
        assert metrics.f1_score == 0.85
        assert metrics.precision == 0.87
        assert metrics.recall == 0.83
        assert metrics.sample_count == 50
    
    def test_detect_data_drift_images(self):
        """Test data drift detection for image data."""
        # Create baseline and new image data
        baseline_data = np.random.randint(0, 5000, (50, 64, 64, 6))
        new_data = np.random.randint(2000, 8000, (50, 64, 64, 6))  # Different range
        
        has_drift, drift_score = self.monitor.detect_data_drift(baseline_data, new_data)
        
        assert isinstance(has_drift, bool)
        assert isinstance(drift_score, float)
        assert drift_score >= 0


class TestModelVersionManager:
    """Test model version management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = ModelVersionManager(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_version(self):
        """Test creating a new model version."""
        metrics = ModelMetrics(
            model_id="test_model",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        config = {"param1": "value1"}
        file_path = "test_model.h5"
        
        version_id = self.version_manager.create_version(
            model_type="lstm",
            config=config,
            metrics=metrics,
            file_path=file_path,
            notes="Test version"
        )
        
        assert version_id.startswith("lstm_")
        assert version_id in self.version_manager.versions
        
        version = self.version_manager.versions[version_id]
        assert version.model_type == "lstm"
        assert version.is_active == True
        assert version.notes == "Test version"
    
    def test_get_active_version(self):
        """Test getting active version."""
        # Create a version
        metrics = ModelMetrics(
            model_id="test_model",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        version_id = self.version_manager.create_version(
            model_type="lstm",
            config={},
            metrics=metrics,
            file_path="test.h5"
        )
        
        # Get active version
        active_version = self.version_manager.get_active_version("lstm")
        
        assert active_version is not None
        assert active_version.version_id == version_id
        assert active_version.is_active == True
    
    def test_rollback_to_version(self):
        """Test rolling back to a previous version."""
        # Create two versions
        metrics1 = ModelMetrics(
            model_id="test_model_1",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.90,
            loss=0.10,
            sample_count=1000
        )
        
        version_id_1 = self.version_manager.create_version(
            model_type="lstm",
            config={},
            metrics=metrics1,
            file_path="test1.h5"
        )
        
        metrics2 = ModelMetrics(
            model_id="test_model_2",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        version_id_2 = self.version_manager.create_version(
            model_type="lstm",
            config={},
            metrics=metrics2,
            file_path="test2.h5"
        )
        
        # Version 2 should be active
        active = self.version_manager.get_active_version("lstm")
        assert active.version_id == version_id_2
        
        # Rollback to version 1
        success = self.version_manager.rollback_to_version(version_id_1)
        assert success == True
        
        # Version 1 should now be active
        active = self.version_manager.get_active_version("lstm")
        assert active.version_id == version_id_1
    
    def test_list_versions(self):
        """Test listing versions."""
        # Create versions for different model types
        lstm_metrics = ModelMetrics(
            model_id="lstm_model",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        cnn_metrics = ModelMetrics(
            model_id="cnn_model",
            model_type="cnn",
            timestamp=datetime.now(),
            accuracy=0.88,
            loss=0.12,
            sample_count=500
        )
        
        lstm_version = self.version_manager.create_version(
            model_type="lstm",
            config={},
            metrics=lstm_metrics,
            file_path="lstm.h5"
        )
        
        cnn_version = self.version_manager.create_version(
            model_type="cnn",
            config={},
            metrics=cnn_metrics,
            file_path="cnn.h5"
        )
        
        # List all versions
        all_versions = self.version_manager.list_versions()
        assert len(all_versions) == 2
        
        # List LSTM versions only
        lstm_versions = self.version_manager.list_versions("lstm")
        assert len(lstm_versions) == 1
        assert lstm_versions[0].model_type == "lstm"


class TestModelRetrainingPipeline:
    """Test model retraining pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        trigger_config = RetrainingTrigger(
            performance_threshold=0.1,
            data_drift_threshold=0.2,
            time_threshold_days=30,
            min_new_samples=50,
            enable_auto_retrain=True
        )
        
        self.pipeline = ModelRetrainingPipeline(
            trigger_config=trigger_config,
            base_path=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_register_baseline(self):
        """Test registering baseline metrics."""
        metrics = ModelMetrics(
            model_id="baseline_model",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        baseline_data = np.random.random((100, 30, 4))
        
        self.pipeline.register_baseline("lstm", metrics, baseline_data)
        
        assert "lstm" in self.pipeline.baseline_metrics
        assert "lstm" in self.pipeline.baseline_data
        assert self.pipeline.baseline_metrics["lstm"] == metrics
    
    def test_check_retraining_triggers_performance(self):
        """Test performance degradation trigger."""
        # Register baseline
        baseline_metrics = ModelMetrics(
            model_id="baseline",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        baseline_data = np.random.random((100, 30, 4))
        self.pipeline.register_baseline("lstm", baseline_metrics, baseline_data)
        
        # Create current metrics with degraded performance
        current_metrics = ModelMetrics(
            model_id="current",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.80,  # Dropped by 0.15 (> threshold of 0.1)
            loss=0.20,
            sample_count=100
        )
        
        new_data = np.random.random((100, 30, 4))
        
        triggers = self.pipeline.check_retraining_triggers(
            "lstm", current_metrics, new_data
        )
        
        assert triggers['performance_degradation'] == True
        assert triggers['sufficient_data'] == True
        assert triggers['should_retrain'] == True
    
    def test_check_retraining_triggers_time(self):
        """Test time threshold trigger."""
        # Register old baseline
        old_timestamp = datetime.now() - timedelta(days=35)
        baseline_metrics = ModelMetrics(
            model_id="baseline",
            model_type="lstm",
            timestamp=old_timestamp,
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        baseline_data = np.random.random((100, 30, 4))
        self.pipeline.register_baseline("lstm", baseline_metrics, baseline_data)
        
        # Current metrics (similar performance)
        current_metrics = ModelMetrics(
            model_id="current",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.94,  # Similar performance
            loss=0.06,
            sample_count=100
        )
        
        new_data = np.random.random((100, 30, 4))
        
        triggers = self.pipeline.check_retraining_triggers(
            "lstm", current_metrics, new_data
        )
        
        assert triggers['time_threshold'] == True
        assert triggers['sufficient_data'] == True
        assert triggers['should_retrain'] == True
    
    def test_check_retraining_triggers_insufficient_data(self):
        """Test insufficient data prevents retraining."""
        # Register baseline
        baseline_metrics = ModelMetrics(
            model_id="baseline",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.95,
            loss=0.05,
            sample_count=1000
        )
        
        baseline_data = np.random.random((100, 30, 4))
        self.pipeline.register_baseline("lstm", baseline_metrics, baseline_data)
        
        # Current metrics with insufficient samples
        current_metrics = ModelMetrics(
            model_id="current",
            model_type="lstm",
            timestamp=datetime.now(),
            accuracy=0.80,  # Poor performance
            loss=0.20,
            sample_count=30  # Below threshold of 50
        )
        
        new_data = np.random.random((30, 30, 4))
        
        triggers = self.pipeline.check_retraining_triggers(
            "lstm", current_metrics, new_data
        )
        
        assert triggers['performance_degradation'] == True
        assert triggers['sufficient_data'] == False
        assert triggers['should_retrain'] == False


class TestRetrainingScheduler:
    """Test retraining scheduler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        config = ScheduleConfig(
            monitoring_interval_hours=1,  # Short interval for testing
            max_concurrent_retraining=1,
            enable_scheduler=True,
            quiet_hours_start=22,
            quiet_hours_end=6
        )
        
        # Mock pipeline
        self.mock_pipeline = Mock(spec=ModelRetrainingPipeline)
        
        self.scheduler = RetrainingScheduler(config, self.mock_pipeline)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
        if self.scheduler.is_running:
            self.scheduler.stop()
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.config.monitoring_interval_hours == 1
        assert self.scheduler.config.max_concurrent_retraining == 1
        assert self.scheduler.is_running == False
        assert len(self.scheduler.active_retraining) == 0
    
    def test_start_stop_scheduler(self):
        """Test starting and stopping scheduler."""
        # Start scheduler
        self.scheduler.start()
        assert self.scheduler.is_running == True
        assert self.scheduler.scheduler_thread is not None
        
        # Stop scheduler
        self.scheduler.stop()
        assert self.scheduler.is_running == False
    
    def test_is_allowed_time(self):
        """Test quiet hours checking."""
        # Mock current time to be during quiet hours
        with patch('src.ai_models.retraining_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = Mock(hour=23)  # 11 PM
            assert self.scheduler._is_allowed_time() == False
            
            mock_datetime.now.return_value = Mock(hour=10)  # 10 AM
            assert self.scheduler._is_allowed_time() == True
    
    def test_get_status(self):
        """Test getting scheduler status."""
        status = self.scheduler.get_status()
        
        assert 'is_running' in status
        assert 'active_retraining' in status
        assert 'config' in status
        assert status['is_running'] == False
        assert status['active_retraining'] == []
    
    def test_add_notification_callback(self):
        """Test adding notification callback."""
        callback_called = []
        
        def test_callback(notification):
            callback_called.append(notification)
        
        self.scheduler.add_notification_callback(test_callback)
        
        # Send a test notification
        self.scheduler._send_notification("info", "Test message")
        
        assert len(callback_called) == 1
        assert callback_called[0]['level'] == "info"
        assert callback_called[0]['message'] == "Test message"
    
    @patch('src.ai_models.retraining_scheduler.schedule')
    def test_force_monitoring_cycle(self, mock_schedule):
        """Test forcing immediate monitoring cycle."""
        # Mock pipeline results
        mock_results = {
            'timestamp': datetime.now(),
            'models_checked': [{'model_type': 'lstm'}],
            'retraining_triggered': [],
            'errors': []
        }
        
        self.mock_pipeline.run_monitoring_cycle.return_value = mock_results
        
        # Force monitoring cycle
        results = self.scheduler.force_monitoring_cycle()
        
        assert results == mock_results
        self.mock_pipeline.run_monitoring_cycle.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])