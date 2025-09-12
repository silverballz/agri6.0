"""
Tests for system scalability features including batch processing and monitoring.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.data_processing.batch_processor import (
    BatchConfig, BatchProgress, ProgressTracker, MemoryOptimizer,
    BatchProcessor, SatelliteImageBatchProcessor, BatchExecutor
)
from src.monitoring.system_monitor import (
    SystemMetrics, ProcessMetrics, AlertThresholds, SystemAlert,
    MetricsCollector, MetricsStorage, AlertManager, SystemMonitor
)


class TestBatchConfig:
    """Test batch configuration."""
    
    def test_default_config(self):
        """Test default batch configuration."""
        config = BatchConfig()
        
        assert config.batch_size == 10
        assert config.memory_limit_gb == 8.0
        assert config.enable_progress_tracking == True
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom batch configuration."""
        config = BatchConfig(
            batch_size=5,
            memory_limit_gb=4.0,
            max_workers=2,
            enable_progress_tracking=False
        )
        
        assert config.batch_size == 5
        assert config.memory_limit_gb == 4.0
        assert config.max_workers == 2
        assert config.enable_progress_tracking == False


class TestBatchProgress:
    """Test batch progress tracking."""
    
    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = BatchProgress(total_items=100)
        
        assert progress.total_items == 100
        assert progress.completed_items == 0
        assert progress.failed_items == 0
        assert progress.completion_percentage == 0.0
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = BatchProgress(total_items=100, completed_items=25)
        
        assert progress.completion_percentage == 25.0
    
    def test_items_per_second(self):
        """Test processing rate calculation."""
        progress = BatchProgress(total_items=100, completed_items=50)
        progress.start_time = datetime.now() - timedelta(seconds=10)
        
        rate = progress.items_per_second
        assert rate == 5.0  # 50 items in 10 seconds
    
    def test_estimate_completion_time(self):
        """Test completion time estimation."""
        progress = BatchProgress(total_items=100, completed_items=25)
        progress.start_time = datetime.now() - timedelta(seconds=10)
        
        progress.estimate_completion_time()
        
        assert progress.estimated_completion is not None
        assert progress.estimated_completion > datetime.now()


class TestProgressTracker:
    """Test progress tracker."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ProgressTracker(100)
        
        assert tracker.progress.total_items == 100
        assert len(tracker.callbacks) == 0
    
    def test_update_progress(self):
        """Test progress updates."""
        tracker = ProgressTracker(100)
        
        tracker.update(completed=10, failed=2)
        
        progress = tracker.get_progress()
        assert progress.completed_items == 10
        assert progress.failed_items == 2
    
    def test_thread_safety(self):
        """Test thread-safe progress updates."""
        tracker = ProgressTracker(1000)
        
        def update_worker(worker_id):
            for i in range(10):
                tracker.update(completed=1)
                time.sleep(0.001)  # Small delay
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        progress = tracker.get_progress()
        assert progress.completed_items == 50  # 5 threads * 10 updates each
    
    def test_progress_callbacks(self):
        """Test progress callbacks."""
        tracker = ProgressTracker(100)
        callback_calls = []
        
        def test_callback(progress):
            callback_calls.append(progress.completed_items)
        
        tracker.add_callback(test_callback)
        
        tracker.update(completed=10)
        tracker.update(completed=5)
        
        assert len(callback_calls) == 2
        assert callback_calls[0] == 10
        assert callback_calls[1] == 15


class TestMemoryOptimizer:
    """Test memory optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = MemoryOptimizer(memory_limit_gb=4.0)
        
        assert optimizer.memory_limit_gb == 4.0
        assert optimizer.memory_limit_bytes == 4.0 * (1024**3)
    
    def test_check_memory_usage(self):
        """Test memory usage checking."""
        optimizer = MemoryOptimizer(memory_limit_gb=1.0)  # Very low limit
        
        usage_gb, is_over_limit = optimizer.check_memory_usage()
        
        assert isinstance(usage_gb, float)
        assert isinstance(is_over_limit, bool)
        assert usage_gb > 0
    
    def test_calculate_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        optimizer = MemoryOptimizer()
        
        # Test with a large array shape
        array_shape = (1000, 1000, 6)  # 1000x1000 pixels, 6 channels
        dtype = np.float32
        
        chunk_height, chunk_width = optimizer.calculate_optimal_chunk_size(
            array_shape, dtype, target_memory_mb=50
        )
        
        assert chunk_height > 0
        assert chunk_width > 0
        assert chunk_height <= array_shape[0]
        assert chunk_width <= array_shape[1]
    
    def test_process_in_chunks(self):
        """Test chunked processing."""
        optimizer = MemoryOptimizer()
        
        # Create test array
        test_array = np.random.random((100, 100, 3))
        
        # Simple processing function (add 1)
        def add_one(chunk):
            return chunk + 1
        
        # Process in chunks
        result = optimizer.process_in_chunks(
            test_array,
            add_one,
            chunk_size=(50, 50)
        )
        
        # Verify result
        assert result.shape == test_array.shape
        np.testing.assert_array_almost_equal(result, test_array + 1)


class MockBatchProcessor(BatchProcessor):
    """Mock batch processor for testing."""
    
    def __init__(self, processing_time: float = 0.1, failure_rate: float = 0.0):
        self.processing_time = processing_time
        self.failure_rate = failure_rate
        self.processed_items = []
    
    def process_item(self, item: str) -> str:
        """Process a single item (simulate work)."""
        time.sleep(self.processing_time)
        
        # Simulate random failures
        if np.random.random() < self.failure_rate:
            raise Exception(f"Simulated failure for item: {item}")
        
        result = f"processed_{item}"
        self.processed_items.append(result)
        return result
    
    def get_item_id(self, item: str) -> str:
        """Get item ID."""
        return item


class TestBatchExecutor:
    """Test batch executor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BatchConfig(
            batch_size=3,
            max_workers=2,
            intermediate_dir=self.temp_dir,
            max_retries=2
        )
        self.processor = MockBatchProcessor(processing_time=0.01)
        self.executor = BatchExecutor(self.processor, self.config)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_sequential_execution(self):
        """Test sequential batch execution."""
        items = ["item1", "item2", "item3", "item4", "item5"]
        
        results, progress = self.executor.execute_batch(items, use_multiprocessing=False)
        
        assert len(results) == 5
        assert progress.completed_items == 5
        assert progress.failed_items == 0
        assert progress.completion_percentage == 100.0
        
        # Check that all items were processed
        processed_results = [r for r in results if r is not None]
        assert len(processed_results) == 5
    
    def test_parallel_execution(self):
        """Test parallel batch execution."""
        items = ["item1", "item2", "item3", "item4", "item5"]
        
        results, progress = self.executor.execute_batch(items, use_multiprocessing=True)
        
        assert len(results) == 5
        assert progress.completed_items == 5
        assert progress.failed_items == 0
        
        # Check that all items were processed
        processed_results = [r for r in results if r is not None]
        assert len(processed_results) == 5
    
    def test_failure_handling(self):
        """Test handling of processing failures."""
        # Create processor with high failure rate
        failing_processor = MockBatchProcessor(processing_time=0.01, failure_rate=0.5)
        executor = BatchExecutor(failing_processor, self.config)
        
        items = ["item1", "item2", "item3", "item4", "item5"]
        
        results, progress = executor.execute_batch(items, use_multiprocessing=False)
        
        assert len(results) == 5
        assert progress.failed_items > 0
        assert progress.completed_items + progress.failed_items == 5
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def test_callback(progress):
            callback_calls.append(progress.completed_items)
        
        self.executor.add_progress_callback(test_callback)
        
        items = ["item1", "item2", "item3"]
        results, progress = self.executor.execute_batch(items, use_multiprocessing=False)
        
        # Should have received progress updates
        assert len(callback_calls) > 0
        assert callback_calls[-1] == 3  # Final count should be 3


class TestSystemMetrics:
    """Test system metrics."""
    
    def test_metrics_creation(self):
        """Test creating system metrics."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.process_count == 150
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['timestamp'] == timestamp.isoformat()
        assert metrics_dict['cpu_percent'] == 50.0
        assert metrics_dict['memory_percent'] == 60.0


class TestMetricsCollector:
    """Test metrics collector."""
    
    def test_collect_system_metrics(self):
        """Test collecting system metrics."""
        collector = MetricsCollector()
        
        metrics = collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_usage_percent >= 0
        assert metrics.process_count > 0
    
    def test_collect_process_metrics(self):
        """Test collecting process metrics."""
        collector = MetricsCollector()
        
        metrics = collector.collect_process_metrics()
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        for metric in metrics:
            assert isinstance(metric, ProcessMetrics)
            assert metric.pid > 0
            assert metric.name is not None


class TestMetricsStorage:
    """Test metrics storage."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_metrics.db"
        self.storage = MetricsStorage(str(self.db_path))
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_system_metrics(self):
        """Test storing system metrics."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        # Should not raise exception
        self.storage.store_system_metrics(metrics)
    
    def test_store_process_metrics(self):
        """Test storing process metrics."""
        metrics = [
            ProcessMetrics(
                pid=1234,
                name="test_process",
                cpu_percent=25.0,
                memory_percent=10.0,
                memory_rss_mb=100.0,
                memory_vms_mb=200.0,
                num_threads=4,
                status="running",
                create_time=datetime.now()
            )
        ]
        
        # Should not raise exception
        self.storage.store_process_metrics(metrics)
    
    def test_get_system_metrics(self):
        """Test retrieving system metrics."""
        # Store some test metrics
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_percent=50.0 + i,
                memory_percent=60.0,
                memory_used_gb=4.0,
                memory_available_gb=4.0,
                disk_usage_percent=70.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000,
                network_bytes_recv=2000,
                process_count=150
            )
            self.storage.store_system_metrics(metrics)
        
        # Retrieve metrics
        retrieved = self.storage.get_system_metrics(limit=3)
        
        assert len(retrieved) == 3
        assert all('timestamp' in m for m in retrieved)
        assert all('cpu_percent' in m for m in retrieved)


class TestAlertManager:
    """Test alert manager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.thresholds = AlertThresholds(
            cpu_percent=80.0,
            memory_percent=85.0,
            disk_usage_percent=90.0
        )
        self.alert_manager = AlertManager(self.thresholds)
    
    def test_cpu_alert(self):
        """Test CPU usage alert."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above threshold
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        alerts = self.alert_manager.check_system_alerts(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == 'cpu_high'
        assert alerts[0].severity == 'warning'
        assert alerts[0].current_value == 85.0
    
    def test_memory_alert(self):
        """Test memory usage alert."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=90.0,  # Above threshold
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        alerts = self.alert_manager.check_system_alerts(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == 'memory_high'
        assert alerts[0].severity == 'warning'
    
    def test_no_alerts(self):
        """Test when no alerts should be generated."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,  # Below threshold
            memory_percent=60.0,  # Below threshold
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,  # Below threshold
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        alerts = self.alert_manager.check_system_alerts(metrics)
        
        assert len(alerts) == 0
    
    def test_alert_callbacks(self):
        """Test alert callback functionality."""
        callback_calls = []
        
        def test_callback(alert):
            callback_calls.append(alert.alert_type)
        
        self.alert_manager.add_alert_callback(test_callback)
        
        # Trigger an alert
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above threshold
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150
        )
        
        self.alert_manager.check_system_alerts(metrics)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == 'cpu_high'


class TestSystemMonitor:
    """Test system monitor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_monitor.db"
        
        self.monitor = SystemMonitor(
            collection_interval=1,  # 1 second for testing
            storage_path=str(self.db_path)
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.monitor.is_running:
            self.monitor.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_monitor_start_stop(self):
        """Test starting and stopping monitor."""
        assert self.monitor.is_running == False
        
        self.monitor.start()
        assert self.monitor.is_running == True
        
        self.monitor.stop()
        assert self.monitor.is_running == False
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        metrics = self.monitor.get_current_metrics()
        
        assert 'system' in metrics
        assert 'processes' in metrics
        assert 'active_alerts' in metrics
        
        system_metrics = metrics['system']
        assert 'cpu_percent' in system_metrics
        assert 'memory_percent' in system_metrics
        assert 'disk_usage_percent' in system_metrics
    
    def test_alert_callbacks(self):
        """Test alert callback registration."""
        callback_calls = []
        
        def test_callback(alert):
            callback_calls.append(alert.alert_type)
        
        self.monitor.add_alert_callback(test_callback)
        
        # Verify callback was added
        assert len(self.monitor.alert_manager.alert_callbacks) == 1


if __name__ == "__main__":
    pytest.main([__file__])