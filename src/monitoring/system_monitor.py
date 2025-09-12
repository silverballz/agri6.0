"""
System monitoring and resource tracking for agricultural monitoring platform.

This module provides comprehensive system monitoring including CPU, memory,
disk usage, and performance metrics for scalable operations.
"""

import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import queue
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Optional[List[float]] = None  # Unix systems only
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_available_gb': self.memory_available_gb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'load_average': self.load_average
        }


@dataclass
class ProcessMetrics:
    """Metrics for a specific process."""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    num_threads: int
    status: str
    create_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pid': self.pid,
            'name': self.name,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_rss_mb': self.memory_rss_mb,
            'memory_vms_mb': self.memory_vms_mb,
            'num_threads': self.num_threads,
            'status': self.status,
            'create_time': self.create_time.isoformat()
        }


@dataclass
class AlertThresholds:
    """Thresholds for system alerts."""
    cpu_percent: float = 80.0
    memory_percent: float = 85.0
    disk_usage_percent: float = 90.0
    process_memory_mb: float = 1000.0
    process_cpu_percent: float = 50.0


@dataclass
class SystemAlert:
    """System alert information."""
    timestamp: datetime
    alert_type: str
    severity: str  # 'warning', 'critical'
    message: str
    current_value: float
    threshold_value: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Collects system and process metrics."""
    
    def __init__(self):
        self.network_counters_baseline = None
        self._initialize_network_baseline()
    
    def _initialize_network_baseline(self):
        """Initialize network counters baseline."""
        try:
            self.network_counters_baseline = psutil.net_io_counters()
        except Exception as e:
            logger.warning(f"Could not initialize network counters: {str(e)}")
            self.network_counters_baseline = None
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk usage (for root partition)
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Network usage
        network_bytes_sent = 0
        network_bytes_recv = 0
        try:
            net_counters = psutil.net_io_counters()
            if net_counters:
                network_bytes_sent = net_counters.bytes_sent
                network_bytes_recv = net_counters.bytes_recv
        except Exception as e:
            logger.warning(f"Could not collect network metrics: {str(e)}")
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix systems only)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except (AttributeError, OSError):
            pass  # Not available on Windows
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            process_count=process_count,
            load_average=load_average
        )
    
    def collect_process_metrics(self, pid: int = None) -> List[ProcessMetrics]:
        """Collect metrics for specific process or all processes."""
        metrics = []
        
        try:
            if pid is not None:
                # Collect for specific process
                processes = [psutil.Process(pid)]
            else:
                # Collect for current process
                processes = [psutil.Process()]
            
            for proc in processes:
                try:
                    # Get process info
                    proc_info = proc.as_dict([
                        'pid', 'name', 'cpu_percent', 'memory_percent',
                        'memory_info', 'num_threads', 'status', 'create_time'
                    ])
                    
                    # Convert memory info
                    memory_info = proc_info.get('memory_info')
                    memory_rss_mb = memory_info.rss / (1024**2) if memory_info else 0
                    memory_vms_mb = memory_info.vms / (1024**2) if memory_info else 0
                    
                    # Convert create time
                    create_time = datetime.fromtimestamp(proc_info.get('create_time', 0))
                    
                    process_metrics = ProcessMetrics(
                        pid=proc_info.get('pid', 0),
                        name=proc_info.get('name', 'unknown'),
                        cpu_percent=proc_info.get('cpu_percent', 0.0),
                        memory_percent=proc_info.get('memory_percent', 0.0),
                        memory_rss_mb=memory_rss_mb,
                        memory_vms_mb=memory_vms_mb,
                        num_threads=proc_info.get('num_threads', 0),
                        status=proc_info.get('status', 'unknown'),
                        create_time=create_time
                    )
                    
                    metrics.append(process_metrics)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            logger.error(f"Error collecting process metrics: {str(e)}")
        
        return metrics


class MetricsStorage:
    """Storage backend for metrics data."""
    
    def __init__(self, db_path: str = "monitoring/metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage."""
        with self._get_connection() as conn:
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_gb REAL,
                    memory_available_gb REAL,
                    disk_usage_percent REAL,
                    disk_free_gb REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    process_count INTEGER,
                    load_average TEXT
                )
            """)
            
            # Process metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS process_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pid INTEGER,
                    name TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_rss_mb REAL,
                    memory_vms_mb REAL,
                    num_threads INTEGER,
                    status TEXT,
                    create_time TEXT
                )
            """)
            
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
            
            # Create indices for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_process_timestamp ON process_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, memory_percent, memory_used_gb,
                    memory_available_gb, disk_usage_percent, disk_free_gb,
                    network_bytes_sent, network_bytes_recv, process_count, load_average
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_used_gb,
                metrics.memory_available_gb,
                metrics.disk_usage_percent,
                metrics.disk_free_gb,
                metrics.network_bytes_sent,
                metrics.network_bytes_recv,
                metrics.process_count,
                json.dumps(metrics.load_average) if metrics.load_average else None
            ))
    
    def store_process_metrics(self, metrics: List[ProcessMetrics], timestamp: datetime = None):
        """Store process metrics in database."""
        if not metrics:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._get_connection() as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO process_metrics (
                        timestamp, pid, name, cpu_percent, memory_percent,
                        memory_rss_mb, memory_vms_mb, num_threads, status, create_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat(),
                    metric.pid,
                    metric.name,
                    metric.cpu_percent,
                    metric.memory_percent,
                    metric.memory_rss_mb,
                    metric.memory_vms_mb,
                    metric.num_threads,
                    metric.status,
                    metric.create_time.isoformat()
                ))
    
    def store_alert(self, alert: SystemAlert):
        """Store alert in database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO alerts (
                    timestamp, alert_type, severity, message,
                    current_value, threshold_value, resolved, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.current_value,
                alert.threshold_value,
                1 if alert.resolved else 0,
                alert.resolved_at.isoformat() if alert.resolved_at else None
            ))
    
    def get_system_metrics(self, 
                          start_time: datetime = None,
                          end_time: datetime = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve system metrics from database."""
        query = "SELECT * FROM system_metrics"
        params = []
        
        conditions = []
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON fields
                if result['load_average']:
                    result['load_average'] = json.loads(result['load_average'])
                results.append(result)
            
            return results
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Remove old metrics data to save space."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_connection() as conn:
            # Clean up system metrics
            cursor = conn.execute(
                "DELETE FROM system_metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            system_deleted = cursor.rowcount
            
            # Clean up process metrics
            cursor = conn.execute(
                "DELETE FROM process_metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            process_deleted = cursor.rowcount
            
            # Clean up resolved alerts
            cursor = conn.execute(
                "DELETE FROM alerts WHERE timestamp < ? AND resolved = 1",
                (cutoff_date.isoformat(),)
            )
            alerts_deleted = cursor.rowcount
            
            logger.info(f"Cleaned up old metrics: {system_deleted} system, "
                       f"{process_deleted} process, {alerts_deleted} alerts")


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, thresholds: AlertThresholds = None):
        self.thresholds = thresholds or AlertThresholds()
        self.active_alerts = {}  # alert_type -> SystemAlert
        self.alert_callbacks = []
    
    def check_system_alerts(self, metrics: SystemMetrics) -> List[SystemAlert]:
        """Check system metrics against thresholds and generate alerts."""
        alerts = []
        
        # Check CPU usage
        if metrics.cpu_percent > self.thresholds.cpu_percent:
            alert = self._create_alert(
                'cpu_high',
                'warning' if metrics.cpu_percent < 95 else 'critical',
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                metrics.cpu_percent,
                self.thresholds.cpu_percent
            )
            alerts.append(alert)
        else:
            self._resolve_alert('cpu_high')
        
        # Check memory usage
        if metrics.memory_percent > self.thresholds.memory_percent:
            alert = self._create_alert(
                'memory_high',
                'warning' if metrics.memory_percent < 95 else 'critical',
                f"High memory usage: {metrics.memory_percent:.1f}%",
                metrics.memory_percent,
                self.thresholds.memory_percent
            )
            alerts.append(alert)
        else:
            self._resolve_alert('memory_high')
        
        # Check disk usage
        if metrics.disk_usage_percent > self.thresholds.disk_usage_percent:
            alert = self._create_alert(
                'disk_high',
                'critical',
                f"High disk usage: {metrics.disk_usage_percent:.1f}%",
                metrics.disk_usage_percent,
                self.thresholds.disk_usage_percent
            )
            alerts.append(alert)
        else:
            self._resolve_alert('disk_high')
        
        return alerts
    
    def check_process_alerts(self, metrics: List[ProcessMetrics]) -> List[SystemAlert]:
        """Check process metrics against thresholds."""
        alerts = []
        
        for proc_metric in metrics:
            # Check process memory usage
            if proc_metric.memory_rss_mb > self.thresholds.process_memory_mb:
                alert = self._create_alert(
                    f'process_memory_{proc_metric.pid}',
                    'warning',
                    f"High memory usage by process {proc_metric.name} (PID {proc_metric.pid}): "
                    f"{proc_metric.memory_rss_mb:.1f} MB",
                    proc_metric.memory_rss_mb,
                    self.thresholds.process_memory_mb
                )
                alerts.append(alert)
            
            # Check process CPU usage
            if proc_metric.cpu_percent > self.thresholds.process_cpu_percent:
                alert = self._create_alert(
                    f'process_cpu_{proc_metric.pid}',
                    'warning',
                    f"High CPU usage by process {proc_metric.name} (PID {proc_metric.pid}): "
                    f"{proc_metric.cpu_percent:.1f}%",
                    proc_metric.cpu_percent,
                    self.thresholds.process_cpu_percent
                )
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, 
                     alert_type: str,
                     severity: str,
                     message: str,
                     current_value: float,
                     threshold_value: float) -> SystemAlert:
        """Create a new alert."""
        alert = SystemAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value
        )
        
        # Store as active alert
        self.active_alerts[alert_type] = alert
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
        
        return alert
    
    def _resolve_alert(self, alert_type: str):
        """Resolve an active alert."""
        if alert_type in self.active_alerts:
            alert = self.active_alerts[alert_type]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[alert_type]
            
            logger.info(f"Resolved alert: {alert_type}")
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get list of currently active alerts."""
        return list(self.active_alerts.values())


class SystemMonitor:
    """Main system monitoring class."""
    
    def __init__(self,
                 collection_interval: int = 60,
                 storage_path: str = "monitoring/metrics.db",
                 thresholds: AlertThresholds = None):
        self.collection_interval = collection_interval
        self.collector = MetricsCollector()
        self.storage = MetricsStorage(storage_path)
        self.alert_manager = AlertManager(thresholds)
        
        self.is_running = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
    
    def start(self):
        """Start system monitoring."""
        if self.is_running:
            logger.warning("System monitor is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"System monitoring started (interval: {self.collection_interval}s)")
    
    def stop(self):
        """Stop system monitoring."""
        if not self.is_running:
            logger.warning("System monitor is not running")
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self.collector.collect_system_metrics()
                
                # Collect process metrics (current process only)
                process_metrics = self.collector.collect_process_metrics()
                
                # Store metrics
                self.storage.store_system_metrics(system_metrics)
                self.storage.store_process_metrics(process_metrics)
                
                # Check for alerts
                system_alerts = self.alert_manager.check_system_alerts(system_metrics)
                process_alerts = self.alert_manager.check_process_alerts(process_metrics)
                
                # Store alerts
                for alert in system_alerts + process_alerts:
                    self.storage.store_alert(alert)
                
                # Log summary
                if system_alerts or process_alerts:
                    logger.warning(f"Generated {len(system_alerts + process_alerts)} alerts")
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.collection_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        system_metrics = self.collector.collect_system_metrics()
        process_metrics = self.collector.collect_process_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'system': system_metrics.to_dict(),
            'processes': [p.to_dict() for p in process_metrics],
            'active_alerts': [
                {
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ]
        }
    
    def get_historical_metrics(self,
                              hours: int = 24) -> Dict[str, Any]:
        """Get historical metrics for the specified time period."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.storage.get_system_metrics(start_time, end_time)
        
        return {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'metrics': metrics
        }
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add alert notification callback."""
        self.alert_manager.add_alert_callback(callback)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old monitoring data."""
        self.storage.cleanup_old_metrics(days_to_keep)


# Example alert callback functions
def log_alert_callback(alert: SystemAlert):
    """Log alert to console."""
    logger.warning(f"ALERT [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")


def save_alert_callback(alert: SystemAlert, filepath: str = "monitoring/alerts.log"):
    """Save alert to log file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'a') as f:
            f.write(f"{alert.timestamp.isoformat()} [{alert.severity.upper()}] "
                   f"{alert.alert_type}: {alert.message}\n")
    except Exception as e:
        logger.error(f"Failed to save alert to file: {str(e)}")


if __name__ == "__main__":
    # Example usage
    thresholds = AlertThresholds(
        cpu_percent=75.0,
        memory_percent=80.0,
        disk_usage_percent=85.0
    )
    
    monitor = SystemMonitor(
        collection_interval=30,  # 30 seconds
        thresholds=thresholds
    )
    
    # Add alert callbacks
    monitor.add_alert_callback(log_alert_callback)
    monitor.add_alert_callback(lambda alert: save_alert_callback(alert, "alerts.log"))
    
    try:
        monitor.start()
        logger.info("System monitor started, press Ctrl+C to stop")
        
        while True:
            time.sleep(60)
            
            # Print current status
            current = monitor.get_current_metrics()
            system = current['system']
            logger.info(f"Status - CPU: {system['cpu_percent']:.1f}%, "
                       f"Memory: {system['memory_percent']:.1f}%, "
                       f"Disk: {system['disk_usage_percent']:.1f}%")
            
    except KeyboardInterrupt:
        logger.info("Stopping system monitor...")
        monitor.stop()