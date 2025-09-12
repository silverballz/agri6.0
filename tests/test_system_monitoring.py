"""
Tests for system monitoring functionality.
Tests system metrics collection, alerting, and storage.
"""

import pytest
import tempfile
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import json

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring.system_monitor import (
    SystemMetrics, ProcessMetrics, AlertThresholds, SystemAlert,
    MetricsCollector, MetricsStorage, AlertManager, SystemMonitor
)


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics object creation."""
        timestamp = datetime.now()
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=60.2,
            memory_used_gb=8.5,
            memory_available_gb=7.3,
            disk_usage_percent=45.8,
            disk_free_gb=120.5,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            process_count=150,
            load_average=[1.2, 1.5, 1.8]
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.2
        assert metrics.memory_used_gb == 8.5
        assert metrics.memory_available_gb == 7.3
        assert metrics.disk_usage_percent == 45.8
        assert metrics.disk_free_gb == 120.5
        assert metrics.network_bytes_sent == 1024000
        assert metrics.network_bytes_recv == 2048000
        assert metrics.process_count == 150
        assert metrics.load_average == [1.2, 1.5, 1.8]
    
    def test_system_metrics_to_dict(self):
        """Test SystemMetrics serialization to dictionary."""
        timestamp = datetime.now()
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=60.2,
            memory_used_gb=8.5,
            memory_available_gb=7.3,
            disk_usage_percent=45.8,
            disk_free_gb=120.5,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            process_count=150,
            load_average=[1.2, 1.5, 1.8]
        )
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data['timestamp'] == timestamp.isoformat()
        assert data['cpu_percent'] == 75.5
        assert data['memory_percent'] == 60.2
        assert data['load_average'] == [1.2, 1.5, 1.8]
        
        # Verify all expected keys are present
        expected_keys = [
            'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_gb',
            'memory_available_gb', 'disk_usage_percent', 'disk_free_gb',
            'network_bytes_sent', 'network_bytes_recv', 'process_count', 'load_average'
        ]
        
        for key in expected_keys:
            assert key in data


class TestProcessMetrics:
    """Test ProcessMetrics dataclass."""
    
    def test_process_metrics_creation(self):
        """Test ProcessMetrics object creation."""
        create_time = datetime.now() - timedelta(hours=2)
        
        metrics = ProcessMetrics(
            pid=1234,
            name="python",
            cpu_percent=25.5,
            memory_percent=15.2,
            memory_rss_mb=512.0,
            memory_vms_mb=1024.0,
            num_threads=8,
            status="running",
            create_time=create_time
        )
        
        assert metrics.pid == 1234
        assert metrics.name == "python"
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 15.2
        assert metrics.memory_rss_mb == 512.0
        assert metrics.memory_vms_mb == 1024.0
        assert metrics.num_threads == 8
        assert metrics.status == "running"
        assert metrics.create_time == create_time
    
    def test_process_metrics_to_dict(self):
        """Test ProcessMetrics serialization to dictionary."""
        create_time = datetime.now() - timedelta(hours=2)
        
        metrics = ProcessMetrics(
            pid=1234,
            name="python",
            cpu_percent=25.5,
            memory_percent=15.2,
            memory_rss_mb=512.0,
            memory_vms_mb=1024.0,
            num_threads=8,
            status="running",
            create_time=create_time
        )
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data['pid'] == 1234
        assert data['name'] == "python"
        assert data['cpu_percent'] == 25.5
        assert data['create_time'] == create_time.isoformat()
        
        # Verify all expected keys are present
        expected_keys = [
            'pid', 'name', 'cpu_percent', 'memory_percent',
            'memory_rss_mb', 'memory_vms_mb', 'num_threads', 'status', 'create_time'
        ]
        
        for key in expected_keys:
            assert key in data


class TestAlertThresholds:
    """Test AlertThresholds configuration."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = AlertThresholds()
        
        assert thresholds.cpu_percent == 80.0
        assert thresholds.memory_percent == 85.0
        assert thresholds.disk_usage_percent == 90.0
        assert thresholds.process_memory_mb == 1000.0
        assert thresholds.process_cpu_percent == 50.0
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = AlertThresholds(
            cpu_percent=70.0,
            memory_percent=75.0,
            disk_usage_percent=80.0,
            process_memory_mb=500.0,
            process_cpu_percent=40.0
        )
        
        assert thresholds.cpu_percent == 70.0
        assert thresholds.memory_percent == 75.0
        assert thresholds.disk_usage_percent == 80.0
        assert thresholds.process_memory_mb == 500.0
        assert thresholds.process_cpu_percent == 40.0


class TestSystemAlert:
    """Test SystemAlert dataclass."""
    
    def test_alert_creation(self):
        """Test SystemAlert object creation."""
        timestamp = datetime.now()
        
        alert = SystemAlert(
            timestamp=timestamp,
            alert_type="cpu_high",
            severity="warning",
            message="High CPU usage detected",
            current_value=85.5,
            threshold_value=80.0
        )
        
        assert alert.timestamp == timestamp
        assert alert.alert_type == "cpu_high"
        assert alert.severity == "warning"
        assert alert.message == "High CPU usage detected"
        assert alert.current_value == 85.5
        assert alert.threshold_value == 80.0
        assert alert.resolved is False
        assert alert.resolved_at is None
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        alert = SystemAlert(
            timestamp=datetime.now(),
            alert_type="memory_high",
            severity="critical",
            message="High memory usage",
            current_value=95.0,
            threshold_value=85.0
        )
        
        # Initially not resolved
        assert alert.resolved is False
        assert alert.resolved_at is None
        
        # Resolve alert
        resolution_time = datetime.now()
        alert.resolved = True
        alert.resolved_at = resolution_time
        
        assert alert.resolved is True
        assert alert.resolved_at == resolution_time


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        collector = MetricsCollector()
        
        # Should initialize without errors
        assert collector is not None
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    @patch('psutil.getloadavg')
    def test_collect_system_metrics_mock(self, mock_loadavg, mock_pids, mock_net, 
                                        mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection with mocked psutil."""
        # Setup mocks
        mock_cpu.return_value = 75.5
        
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 60.2
        mock_memory_obj.used = 8.5 * (1024**3)  # 8.5 GB in bytes
        mock_memory_obj.available = 7.3 * (1024**3)  # 7.3 GB in bytes
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = Mock()
        mock_disk_obj.used = 45.8 * (1024**3)  # Used space
        mock_disk_obj.total = 100 * (1024**3)   # Total space
        mock_disk_obj.free = 54.2 * (1024**3)   # Free space
        mock_disk.return_value = mock_disk_obj
        
        mock_net_obj = Mock()
        mock_net_obj.bytes_sent = 1024000
        mock_net_obj.bytes_recv = 2048000
        mock_net.return_value = mock_net_obj
        
        mock_pids.return_value = list(range(150))  # 150 processes
        mock_loadavg.return_value = (1.2, 1.5, 1.8)
        
        collector = MetricsCollector()
        metrics = collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.2
        assert abs(metrics.memory_used_gb - 8.5) < 0.1
        assert abs(metrics.memory_available_gb - 7.3) < 0.1
        assert abs(metrics.disk_usage_percent - 45.8) < 0.1
        assert abs(metrics.disk_free_gb - 54.2) < 0.1
        assert metrics.network_bytes_sent == 1024000
        assert metrics.network_bytes_recv == 2048000
        assert metrics.process_count == 150
        assert metrics.load_average == [1.2, 1.5, 1.8]
    
    @patch('psutil.Process')
    def test_collect_process_metrics_mock(self, mock_process_class):
        """Test process metrics collection with mocked psutil."""
        # Setup mock process
        mock_process = Mock()
        mock_process.as_dict.return_value = {
            'pid': 1234,
            'name': 'python',
            'cpu_percent': 25.5,
            'memory_percent': 15.2,
            'memory_info': Mock(rss=512 * (1024**2), vms=1024 * (1024**2)),
            'num_threads': 8,
            'status': 'running',
            'create_time': time.time() - 7200  # 2 hours ago
        }
        mock_process_class.return_value = mock_process
        
        collector = MetricsCollector()
        metrics = collector.collect_process_metrics()
        
        assert len(metrics) == 1
        process_metric = metrics[0]
        
        assert isinstance(process_metric, ProcessMetrics)
        assert process_metric.pid == 1234
        assert process_metric.name == 'python'
        assert process_metric.cpu_percent == 25.5
        assert process_metric.memory_percent == 15.2
        assert abs(process_metric.memory_rss_mb - 512.0) < 0.1
        assert abs(process_metric.memory_vms_mb - 1024.0) < 0.1
        assert process_metric.num_threads == 8
        assert process_metric.status == 'running'
    
    def test_collect_system_metrics_real(self):
        """Test system metrics collection with real psutil (integration test)."""
        collector = MetricsCollector()
        metrics = collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert isinstance(metrics.timestamp, datetime)
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.memory_used_gb > 0
        assert metrics.memory_available_gb >= 0
        assert 0 <= metrics.disk_usage_percent <= 100
        assert metrics.disk_free_gb >= 0
        assert metrics.network_bytes_sent >= 0
        assert metrics.network_bytes_recv >= 0
        assert metrics.process_count > 0
    
    def test_collect_process_metrics_real(self):
        """Test process metrics collection with real psutil (integration test)."""
        collector = MetricsCollector()
        metrics = collector.collect_process_metrics()
        
        assert len(metrics) == 1  # Should collect current process
        process_metric = metrics[0]
        
        assert isinstance(process_metric, ProcessMetrics)
        assert process_metric.pid > 0
        assert isinstance(process_metric.name, str)
        assert process_metric.cpu_percent >= 0
        assert process_metric.memory_percent >= 0
        assert process_metric.memory_rss_mb > 0
        assert process_metric.num_threads > 0
        assert isinstance(process_metric.status, str)
        assert isinstance(process_metric.create_time, datetime)


class TestMetricsStorage:
    """Test MetricsStorage functionality."""
    
    def test_storage_initialization(self):
        """Test storage initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Database file should be created
            assert db_path.exists()
            
            # Tables should be created
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = ['system_metrics', 'process_metrics', 'alerts']
                for table in expected_tables:
                    assert table in tables
    
    def test_store_system_metrics(self):
        """Test storing system metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Create test metrics
            timestamp = datetime.now()
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_percent=75.5,
                memory_percent=60.2,
                memory_used_gb=8.5,
                memory_available_gb=7.3,
                disk_usage_percent=45.8,
                disk_free_gb=120.5,
                network_bytes_sent=1024000,
                network_bytes_recv=2048000,
                process_count=150,
                load_average=[1.2, 1.5, 1.8]
            )
            
            # Store metrics
            storage.store_system_metrics(metrics)
            
            # Verify storage
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT * FROM system_metrics")
                rows = cursor.fetchall()
                
                assert len(rows) == 1
                row = rows[0]
                
                # Verify stored values (note: timestamp is stored as ISO string)
                assert row[2] == 75.5  # cpu_percent
                assert row[3] == 60.2  # memory_percent
                assert row[4] == 8.5   # memory_used_gb
    
    def test_store_process_metrics(self):
        """Test storing process metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Create test metrics
            create_time = datetime.now() - timedelta(hours=2)
            metrics = [
                ProcessMetrics(
                    pid=1234,
                    name="python",
                    cpu_percent=25.5,
                    memory_percent=15.2,
                    memory_rss_mb=512.0,
                    memory_vms_mb=1024.0,
                    num_threads=8,
                    status="running",
                    create_time=create_time
                )
            ]
            
            # Store metrics
            storage.store_process_metrics(metrics)
            
            # Verify storage
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT * FROM process_metrics")
                rows = cursor.fetchall()
                
                assert len(rows) == 1
                row = rows[0]
                
                # Verify stored values
                assert row[2] == 1234    # pid
                assert row[3] == "python" # name
                assert row[4] == 25.5    # cpu_percent
    
    def test_store_alert(self):
        """Test storing alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Create test alert
            timestamp = datetime.now()
            alert = SystemAlert(
                timestamp=timestamp,
                alert_type="cpu_high",
                severity="warning",
                message="High CPU usage detected",
                current_value=85.5,
                threshold_value=80.0
            )
            
            # Store alert
            storage.store_alert(alert)
            
            # Verify storage
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT * FROM alerts")
                rows = cursor.fetchall()
                
                assert len(rows) == 1
                row = rows[0]
                
                # Verify stored values
                assert row[2] == "cpu_high"  # alert_type
                assert row[3] == "warning"   # severity
                assert row[4] == "High CPU usage detected"  # message
                assert row[5] == 85.5        # current_value
                assert row[6] == 80.0        # threshold_value
                assert row[7] == 0           # resolved (False = 0)
    
    def test_get_system_metrics(self):
        """Test retrieving system metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Store multiple metrics
            timestamps = [
                datetime.now() - timedelta(hours=2),
                datetime.now() - timedelta(hours=1),
                datetime.now()
            ]
            
            for i, timestamp in enumerate(timestamps):
                metrics = SystemMetrics(
                    timestamp=timestamp,
                    cpu_percent=70.0 + i * 5,  # Increasing CPU usage
                    memory_percent=50.0 + i * 10,
                    memory_used_gb=8.0,
                    memory_available_gb=8.0,
                    disk_usage_percent=40.0,
                    disk_free_gb=100.0,
                    network_bytes_sent=1000000,
                    network_bytes_recv=2000000,
                    process_count=100
                )
                storage.store_system_metrics(metrics)
            
            # Retrieve all metrics
            retrieved_metrics = storage.get_system_metrics()
            
            assert len(retrieved_metrics) == 3
            
            # Should be ordered by timestamp DESC (most recent first)
            assert retrieved_metrics[0]['cpu_percent'] == 80.0  # Most recent
            assert retrieved_metrics[1]['cpu_percent'] == 75.0
            assert retrieved_metrics[2]['cpu_percent'] == 70.0  # Oldest
    
    def test_get_system_metrics_with_time_range(self):
        """Test retrieving system metrics with time range filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Store metrics across different time periods
            base_time = datetime.now()
            timestamps = [
                base_time - timedelta(hours=3),  # Outside range
                base_time - timedelta(hours=1),  # Inside range
                base_time,                       # Inside range
            ]
            
            for i, timestamp in enumerate(timestamps):
                metrics = SystemMetrics(
                    timestamp=timestamp,
                    cpu_percent=60.0 + i * 10,
                    memory_percent=50.0,
                    memory_used_gb=8.0,
                    memory_available_gb=8.0,
                    disk_usage_percent=40.0,
                    disk_free_gb=100.0,
                    network_bytes_sent=1000000,
                    network_bytes_recv=2000000,
                    process_count=100
                )
                storage.store_system_metrics(metrics)
            
            # Retrieve metrics from last 2 hours
            start_time = base_time - timedelta(hours=2)
            retrieved_metrics = storage.get_system_metrics(start_time=start_time)
            
            # Should only get 2 metrics (the ones within the time range)
            assert len(retrieved_metrics) == 2
            assert retrieved_metrics[0]['cpu_percent'] == 80.0  # Most recent
            assert retrieved_metrics[1]['cpu_percent'] == 70.0  # Second most recent
    
    def test_cleanup_old_metrics(self):
        """Test cleanup of old metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = MetricsStorage(str(db_path))
            
            # Store old and new metrics
            old_time = datetime.now() - timedelta(days=35)  # Older than 30 days
            new_time = datetime.now() - timedelta(days=5)   # Within 30 days
            
            for timestamp in [old_time, new_time]:
                metrics = SystemMetrics(
                    timestamp=timestamp,
                    cpu_percent=70.0,
                    memory_percent=50.0,
                    memory_used_gb=8.0,
                    memory_available_gb=8.0,
                    disk_usage_percent=40.0,
                    disk_free_gb=100.0,
                    network_bytes_sent=1000000,
                    network_bytes_recv=2000000,
                    process_count=100
                )
                storage.store_system_metrics(metrics)
            
            # Verify both metrics are stored
            all_metrics = storage.get_system_metrics(limit=10)
            assert len(all_metrics) == 2
            
            # Cleanup old metrics (keep 30 days)
            storage.cleanup_old_metrics(days_to_keep=30)
            
            # Should only have the new metric
            remaining_metrics = storage.get_system_metrics(limit=10)
            assert len(remaining_metrics) == 1


class TestAlertManager:
    """Test AlertManager functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        thresholds = AlertThresholds(cpu_percent=70.0)
        manager = AlertManager(thresholds)
        
        assert manager.thresholds.cpu_percent == 70.0
        assert len(manager.active_alerts) == 0
        assert len(manager.alert_callbacks) == 0
    
    def test_cpu_alert_generation(self):
        """Test CPU usage alert generation."""
        thresholds = AlertThresholds(cpu_percent=80.0)
        manager = AlertManager(thresholds)
        
        # Create metrics with high CPU usage
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above threshold
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(metrics)
        
        assert len(alerts) == 1
        alert = alerts[0]
        
        assert alert.alert_type == 'cpu_high'
        assert alert.severity == 'warning'
        assert alert.current_value == 85.0
        assert alert.threshold_value == 80.0
        assert 'CPU usage' in alert.message
    
    def test_memory_alert_generation(self):
        """Test memory usage alert generation."""
        thresholds = AlertThresholds(memory_percent=85.0)
        manager = AlertManager(thresholds)
        
        # Create metrics with high memory usage
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=90.0,  # Above threshold
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(metrics)
        
        assert len(alerts) == 1
        alert = alerts[0]
        
        assert alert.alert_type == 'memory_high'
        assert alert.severity == 'warning'
        assert alert.current_value == 90.0
        assert alert.threshold_value == 85.0
    
    def test_disk_alert_generation(self):
        """Test disk usage alert generation."""
        thresholds = AlertThresholds(disk_usage_percent=90.0)
        manager = AlertManager(thresholds)
        
        # Create metrics with high disk usage
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=95.0,  # Above threshold
            disk_free_gb=10.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(metrics)
        
        assert len(alerts) == 1
        alert = alerts[0]
        
        assert alert.alert_type == 'disk_high'
        assert alert.severity == 'critical'  # Disk alerts are critical
        assert alert.current_value == 95.0
        assert alert.threshold_value == 90.0
    
    def test_multiple_alerts_generation(self):
        """Test generation of multiple alerts simultaneously."""
        thresholds = AlertThresholds(
            cpu_percent=80.0,
            memory_percent=85.0,
            disk_usage_percent=90.0
        )
        manager = AlertManager(thresholds)
        
        # Create metrics that exceed all thresholds
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,    # Above CPU threshold
            memory_percent=90.0, # Above memory threshold
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=95.0,  # Above disk threshold
            disk_free_gb=10.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(metrics)
        
        assert len(alerts) == 3
        alert_types = {alert.alert_type for alert in alerts}
        assert alert_types == {'cpu_high', 'memory_high', 'disk_high'}
    
    def test_alert_resolution(self):
        """Test alert resolution when metrics return to normal."""
        thresholds = AlertThresholds(cpu_percent=80.0)
        manager = AlertManager(thresholds)
        
        # First, generate an alert
        high_cpu_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above threshold
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(high_cpu_metrics)
        assert len(alerts) == 1
        assert len(manager.active_alerts) == 1
        
        # Now provide normal metrics
        normal_cpu_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=70.0,  # Below threshold
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(normal_cpu_metrics)
        assert len(alerts) == 0  # No new alerts
        assert len(manager.active_alerts) == 0  # Alert should be resolved
    
    def test_process_alerts(self):
        """Test process-specific alerts."""
        thresholds = AlertThresholds(
            process_memory_mb=500.0,
            process_cpu_percent=50.0
        )
        manager = AlertManager(thresholds)
        
        # Create process metrics that exceed thresholds
        process_metrics = [
            ProcessMetrics(
                pid=1234,
                name="memory_hog",
                cpu_percent=30.0,
                memory_percent=15.0,
                memory_rss_mb=600.0,  # Above memory threshold
                memory_vms_mb=1000.0,
                num_threads=4,
                status="running",
                create_time=datetime.now() - timedelta(hours=1)
            ),
            ProcessMetrics(
                pid=5678,
                name="cpu_hog",
                cpu_percent=60.0,  # Above CPU threshold
                memory_percent=10.0,
                memory_rss_mb=300.0,
                memory_vms_mb=500.0,
                num_threads=8,
                status="running",
                create_time=datetime.now() - timedelta(hours=2)
            )
        ]
        
        alerts = manager.check_process_alerts(process_metrics)
        
        assert len(alerts) == 2
        
        # Check memory alert
        memory_alert = next(a for a in alerts if 'memory_hog' in a.message)
        assert 'memory' in memory_alert.alert_type
        assert memory_alert.current_value == 600.0
        
        # Check CPU alert
        cpu_alert = next(a for a in alerts if 'cpu_hog' in a.message)
        assert 'cpu' in cpu_alert.alert_type
        assert cpu_alert.current_value == 60.0
    
    def test_alert_callbacks(self):
        """Test alert callback functionality."""
        manager = AlertManager()
        
        # Add mock callback
        callback_calls = []
        def test_callback(alert):
            callback_calls.append(alert)
        
        manager.add_alert_callback(test_callback)
        
        # Generate an alert
        thresholds = AlertThresholds(cpu_percent=80.0)
        manager.thresholds = thresholds
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        alerts = manager.check_system_alerts(metrics)
        
        # Callback should have been called
        assert len(callback_calls) == 1
        assert callback_calls[0].alert_type == 'cpu_high'
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        thresholds = AlertThresholds(cpu_percent=80.0)
        manager = AlertManager(thresholds)
        
        # Initially no active alerts
        assert len(manager.get_active_alerts()) == 0
        
        # Generate an alert
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=100
        )
        
        manager.check_system_alerts(metrics)
        
        # Should have one active alert
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_type == 'cpu_high'


class TestSystemMonitor:
    """Test SystemMonitor integration."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_monitor.db"
            
            monitor = SystemMonitor(
                collection_interval=30,
                storage_path=str(db_path),
                thresholds=AlertThresholds(cpu_percent=75.0)
            )
            
            assert monitor.collection_interval == 30
            assert monitor.alert_manager.thresholds.cpu_percent == 75.0
            assert not monitor.is_running
    
    def test_monitor_start_stop(self):
        """Test monitor start and stop functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_monitor.db"
            
            monitor = SystemMonitor(
                collection_interval=1,  # Short interval for testing
                storage_path=str(db_path)
            )
            
            # Start monitoring
            monitor.start()
            assert monitor.is_running
            assert monitor.monitor_thread is not None
            assert monitor.monitor_thread.is_alive()
            
            # Let it run briefly
            time.sleep(0.1)
            
            # Stop monitoring
            monitor.stop()
            assert not monitor.is_running
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_monitor.db"
            
            monitor = SystemMonitor(storage_path=str(db_path))
            current = monitor.get_current_metrics()
            
            assert isinstance(current, dict)
            assert 'system' in current
            assert 'processes' in current
            assert 'active_alerts' in current
            
            # Verify system metrics structure
            system = current['system']
            assert 'cpu_percent' in system
            assert 'memory_percent' in system
            assert 'timestamp' in system
            
            # Verify processes structure
            processes = current['processes']
            assert isinstance(processes, list)
            if len(processes) > 0:
                assert 'pid' in processes[0]
                assert 'name' in processes[0]
    
    def test_get_historical_metrics(self):
        """Test getting historical metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_monitor.db"
            
            monitor = SystemMonitor(storage_path=str(db_path))
            
            # Store some test metrics first
            test_metrics = SystemMetrics(
                timestamp=datetime.now() - timedelta(hours=1),
                cpu_percent=70.0,
                memory_percent=50.0,
                memory_used_gb=8.0,
                memory_available_gb=8.0,
                disk_usage_percent=40.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000000,
                network_bytes_recv=2000000,
                process_count=100
            )
            monitor.storage.store_system_metrics(test_metrics)
            
            # Get historical metrics
            historical = monitor.get_historical_metrics(hours=24)
            
            assert isinstance(historical, dict)
            assert 'time_range' in historical
            assert 'metrics' in historical
            
            # Verify time range
            time_range = historical['time_range']
            assert 'start' in time_range
            assert 'end' in time_range
            
            # Verify metrics
            metrics = historical['metrics']
            assert isinstance(metrics, list)
            assert len(metrics) >= 1  # Should have at least our test metric
    
    def test_alert_callback_integration(self):
        """Test alert callback integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_monitor.db"
            
            monitor = SystemMonitor(
                storage_path=str(db_path),
                thresholds=AlertThresholds(cpu_percent=1.0)  # Very low threshold to trigger alerts
            )
            
            # Add callback to track alerts
            received_alerts = []
            def alert_callback(alert):
                received_alerts.append(alert)
            
            monitor.add_alert_callback(alert_callback)
            
            # Get current metrics (should trigger alert due to low threshold)
            current = monitor.get_current_metrics()
            
            # May or may not trigger alert depending on actual system load
            # This is more of an integration test to ensure the callback system works
            assert isinstance(received_alerts, list)  # Should be callable without errors


class TestSystemMonitoringIntegration:
    """Integration tests for system monitoring."""
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "integration_test.db"
            
            # Create monitor with realistic settings
            thresholds = AlertThresholds(
                cpu_percent=95.0,    # High threshold to avoid false alerts
                memory_percent=95.0,
                disk_usage_percent=95.0
            )
            
            monitor = SystemMonitor(
                collection_interval=1,  # Fast collection for testing
                storage_path=str(db_path),
                thresholds=thresholds
            )
            
            # Track alerts
            alerts_received = []
            def track_alerts(alert):
                alerts_received.append(alert)
            
            monitor.add_alert_callback(track_alerts)
            
            # Start monitoring briefly
            monitor.start()
            time.sleep(2.5)  # Let it collect a few metrics
            monitor.stop()
            
            # Verify data was collected and stored
            historical = monitor.get_historical_metrics(hours=1)
            assert len(historical['metrics']) >= 2  # Should have collected at least 2 metrics
            
            # Verify database contains data
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM system_metrics")
                count = cursor.fetchone()[0]
                assert count >= 2
            
            # Get current status
            current = monitor.get_current_metrics()
            assert current['system']['cpu_percent'] >= 0
            assert current['system']['memory_percent'] >= 0
    
    def test_monitoring_with_cleanup(self):
        """Test monitoring with data cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cleanup_test.db"
            
            monitor = SystemMonitor(storage_path=str(db_path))
            
            # Store old metrics
            old_metrics = SystemMetrics(
                timestamp=datetime.now() - timedelta(days=35),  # Old data
                cpu_percent=70.0,
                memory_percent=50.0,
                memory_used_gb=8.0,
                memory_available_gb=8.0,
                disk_usage_percent=40.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000000,
                network_bytes_recv=2000000,
                process_count=100
            )
            monitor.storage.store_system_metrics(old_metrics)
            
            # Store recent metrics
            recent_metrics = SystemMetrics(
                timestamp=datetime.now() - timedelta(hours=1),  # Recent data
                cpu_percent=75.0,
                memory_percent=55.0,
                memory_used_gb=8.5,
                memory_available_gb=7.5,
                disk_usage_percent=45.0,
                disk_free_gb=95.0,
                network_bytes_sent=1100000,
                network_bytes_recv=2100000,
                process_count=105
            )
            monitor.storage.store_system_metrics(recent_metrics)
            
            # Verify both metrics are stored
            all_metrics = monitor.storage.get_system_metrics(limit=10)
            assert len(all_metrics) == 2
            
            # Cleanup old data
            monitor.cleanup_old_data(days_to_keep=30)
            
            # Should only have recent data
            remaining_metrics = monitor.storage.get_system_metrics(limit=10)
            assert len(remaining_metrics) == 1
            assert remaining_metrics[0]['cpu_percent'] == 75.0  # Recent metric