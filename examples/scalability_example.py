"""
Example demonstrating scalability features of the agricultural monitoring platform.

This example shows how to use batch processing, system monitoring, and
model retraining pipeline for large-scale operations.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Import scalability components
from src.data_processing.batch_processor import (
    BatchConfig, create_satellite_batch_processor,
    log_progress_callback, save_progress_callback
)
from src.monitoring.system_monitor import (
    SystemMonitor, AlertThresholds, log_alert_callback, save_alert_callback
)
from src.ai_models.model_monitoring import (
    ModelRetrainingPipeline, RetrainingTrigger, ModelMetrics
)
from src.ai_models.retraining_scheduler import (
    RetrainingScheduler, ScheduleConfig, create_default_scheduler
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_batch_processing():
    """Demonstrate large-scale batch processing capabilities."""
    logger.info("=== Batch Processing Demonstration ===")
    
    # Configure batch processing for large datasets
    config = BatchConfig(
        batch_size=5,                    # Process 5 SAFE directories at a time
        max_workers=4,                   # Use 4 parallel workers
        memory_limit_gb=8.0,            # 8GB memory limit
        chunk_size_mb=200,              # 200MB chunks for raster processing
        enable_progress_tracking=True,   # Enable progress tracking
        save_intermediate=True,          # Save intermediate results
        max_retries=3                   # Retry failed items up to 3 times
    )
    
    # Create batch processor
    executor = create_satellite_batch_processor(config)
    
    # Add progress callbacks
    executor.add_progress_callback(log_progress_callback)
    executor.add_progress_callback(
        lambda p: save_progress_callback(p, "batch_progress.json")
    )
    
    # Simulate a list of SAFE directories to process
    safe_directories = [
        f"data/sentinel2/S2A_MSIL2A_20240{i:02d}01_SAFE"
        for i in range(1, 13)  # 12 months of data
    ]
    
    logger.info(f"Starting batch processing of {len(safe_directories)} SAFE directories")
    
    # Execute batch processing
    start_time = time.time()
    results, final_progress = executor.execute_batch(
        safe_directories,
        use_multiprocessing=True
    )
    end_time = time.time()
    
    # Log results
    processing_time = end_time - start_time
    successful_results = [r for r in results if r is not None]
    
    logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
    logger.info(f"Successfully processed: {len(successful_results)} items")
    logger.info(f"Failed items: {final_progress.failed_items}")
    logger.info(f"Processing rate: {final_progress.items_per_second:.2f} items/sec")
    logger.info(f"Peak memory usage: {final_progress.peak_memory_usage_gb:.2f} GB")
    
    return results, final_progress


def demonstrate_system_monitoring():
    """Demonstrate system monitoring and alerting."""
    logger.info("=== System Monitoring Demonstration ===")
    
    # Configure alert thresholds
    thresholds = AlertThresholds(
        cpu_percent=75.0,           # Alert if CPU > 75%
        memory_percent=80.0,        # Alert if memory > 80%
        disk_usage_percent=85.0,    # Alert if disk > 85%
        process_memory_mb=500.0,    # Alert if process uses > 500MB
        process_cpu_percent=40.0    # Alert if process uses > 40% CPU
    )
    
    # Create system monitor
    monitor = SystemMonitor(
        collection_interval=10,  # Collect metrics every 10 seconds
        thresholds=thresholds
    )
    
    # Add alert callbacks
    monitor.add_alert_callback(log_alert_callback)
    monitor.add_alert_callback(
        lambda alert: save_alert_callback(alert, "system_alerts.log")
    )
    
    # Start monitoring
    monitor.start()
    logger.info("System monitoring started")
    
    # Monitor for a period of time
    monitoring_duration = 60  # Monitor for 1 minute
    logger.info(f"Monitoring system for {monitoring_duration} seconds...")
    
    for i in range(monitoring_duration // 10):
        time.sleep(10)
        
        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        system = current_metrics['system']
        
        logger.info(f"System Status - CPU: {system['cpu_percent']:.1f}%, "
                   f"Memory: {system['memory_percent']:.1f}%, "
                   f"Disk: {system['disk_usage_percent']:.1f}%")
        
        # Check for active alerts
        active_alerts = current_metrics['active_alerts']
        if active_alerts:
            logger.warning(f"Active alerts: {len(active_alerts)}")
            for alert in active_alerts:
                logger.warning(f"  - {alert['type']}: {alert['message']}")
    
    # Get historical metrics
    historical = monitor.get_historical_metrics(hours=1)
    logger.info(f"Collected {len(historical['metrics'])} historical data points")
    
    # Stop monitoring
    monitor.stop()
    logger.info("System monitoring stopped")
    
    # Cleanup old data (keep last 7 days)
    monitor.cleanup_old_data(days_to_keep=7)
    
    return monitor


def demonstrate_model_retraining():
    """Demonstrate automated model retraining pipeline."""
    logger.info("=== Model Retraining Demonstration ===")
    
    # Configure retraining triggers
    trigger_config = RetrainingTrigger(
        performance_threshold=0.1,      # Retrain if accuracy drops by 10%
        data_drift_threshold=0.2,       # Retrain if data drift > 0.2
        time_threshold_days=30,         # Retrain after 30 days
        min_new_samples=100,           # Need at least 100 new samples
        enable_auto_retrain=True       # Enable automatic retraining
    )
    
    # Create retraining pipeline
    pipeline = ModelRetrainingPipeline(trigger_config)
    
    # Simulate baseline model metrics
    baseline_metrics = ModelMetrics(
        model_id="lstm_baseline",
        model_type="lstm",
        timestamp=datetime.now() - timedelta(days=35),  # Old baseline
        accuracy=0.92,
        loss=0.08,
        mae=0.05,
        mse=0.003,
        r2=0.89,
        sample_count=1000
    )
    
    # Register baseline
    baseline_data = np.random.normal(0, 1, (1000, 30, 4))  # Simulated baseline data
    pipeline.register_baseline("lstm", baseline_metrics, baseline_data)
    
    # Simulate current model performance (degraded)
    current_metrics = ModelMetrics(
        model_id="lstm_current",
        model_type="lstm",
        timestamp=datetime.now(),
        accuracy=0.80,  # Dropped by 0.12 (> threshold of 0.1)
        loss=0.20,
        mae=0.15,
        mse=0.025,
        r2=0.75,
        sample_count=150
    )
    
    # Simulate new data with drift
    new_data = np.random.normal(0.3, 1.5, (150, 30, 4))  # Drifted data
    
    # Check retraining triggers
    triggers = pipeline.check_retraining_triggers("lstm", current_metrics, new_data)
    
    logger.info("Retraining trigger analysis:")
    logger.info(f"  Performance degradation: {triggers['performance_degradation']}")
    logger.info(f"  Data drift detected: {triggers['data_drift']}")
    logger.info(f"  Time threshold exceeded: {triggers['time_threshold']}")
    logger.info(f"  Sufficient data available: {triggers['sufficient_data']}")
    logger.info(f"  Should retrain: {triggers['should_retrain']}")
    
    if triggers['should_retrain']:
        logger.info("Retraining would be triggered automatically")
        # In a real scenario, this would trigger actual model retraining
        # For demo purposes, we'll just log the decision
    
    return pipeline


def demonstrate_automated_scheduling():
    """Demonstrate automated retraining scheduler."""
    logger.info("=== Automated Scheduling Demonstration ===")
    
    # Create scheduler with custom configuration
    config = ScheduleConfig(
        monitoring_interval_hours=1,    # Check every hour (for demo)
        max_concurrent_retraining=1,    # Only one retraining at a time
        enable_scheduler=True,          # Enable scheduling
        quiet_hours_start=22,          # No retraining 10 PM - 6 AM
        quiet_hours_end=6,
        max_retrain_attempts=3         # Retry failed retraining up to 3 times
    )
    
    # Create scheduler
    scheduler = create_default_scheduler()
    scheduler.config = config
    
    # Add notification callback
    def notification_callback(notification):
        logger.info(f"Scheduler notification [{notification['level']}]: {notification['message']}")
    
    scheduler.add_notification_callback(notification_callback)
    
    # Start scheduler
    scheduler.start()
    logger.info("Automated scheduler started")
    
    # Get scheduler status
    status = scheduler.get_status()
    logger.info(f"Scheduler status: {status}")
    
    # Force a monitoring cycle for demonstration
    logger.info("Forcing immediate monitoring cycle...")
    try:
        # This would normally check actual models and trigger retraining if needed
        # For demo purposes, we'll just show the mechanism
        logger.info("Monitoring cycle would check all registered models")
        logger.info("Any models meeting retraining criteria would be queued for retraining")
    except Exception as e:
        logger.info(f"Monitoring cycle simulation: {str(e)}")
    
    # Get retraining history
    history = scheduler.get_retraining_history(limit=10)
    logger.info(f"Retraining history: {len(history)} entries")
    
    # Stop scheduler
    scheduler.stop()
    logger.info("Automated scheduler stopped")
    
    return scheduler


def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques."""
    logger.info("=== Memory Optimization Demonstration ===")
    
    from src.data_processing.batch_processor import MemoryOptimizer
    
    # Create memory optimizer
    optimizer = MemoryOptimizer(memory_limit_gb=4.0)
    
    # Check current memory usage
    usage_gb, is_over_limit = optimizer.check_memory_usage()
    logger.info(f"Current memory usage: {usage_gb:.2f} GB (limit: 4.0 GB)")
    logger.info(f"Over limit: {is_over_limit}")
    
    # Demonstrate chunked processing
    logger.info("Demonstrating chunked processing for large arrays...")
    
    # Create a large test array (simulating satellite imagery)
    large_array = np.random.randint(0, 10000, (2000, 2000, 6), dtype=np.uint16)
    logger.info(f"Created test array: {large_array.shape}, "
               f"Size: {large_array.nbytes / (1024**2):.1f} MB")
    
    # Calculate optimal chunk size
    chunk_height, chunk_width = optimizer.calculate_optimal_chunk_size(
        large_array.shape, 
        large_array.dtype,
        target_memory_mb=100
    )
    logger.info(f"Optimal chunk size: {chunk_height} x {chunk_width}")
    
    # Define a processing function (NDVI calculation simulation)
    def calculate_ndvi_chunk(chunk):
        """Simulate NDVI calculation on a chunk."""
        if chunk.shape[-1] >= 4:
            # Simulate NIR (band 4) and Red (band 3) for NDVI
            nir = chunk[:, :, 3].astype(np.float32)
            red = chunk[:, :, 2].astype(np.float32)
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # Return as single channel
            return np.expand_dims(ndvi, axis=-1)
        else:
            return chunk[:, :, :1]  # Return first channel only
    
    # Process in chunks
    start_time = time.time()
    result = optimizer.process_in_chunks(
        large_array,
        calculate_ndvi_chunk,
        chunk_size=(chunk_height, chunk_width)
    )
    end_time = time.time()
    
    logger.info(f"Chunked processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Result shape: {result.shape}")
    
    # Check memory usage after processing
    final_usage_gb, _ = optimizer.check_memory_usage()
    logger.info(f"Final memory usage: {final_usage_gb:.2f} GB")
    
    # Force memory optimization
    optimizer.optimize_memory()
    
    return optimizer


def main():
    """Run all scalability demonstrations."""
    logger.info("Starting Agricultural Monitoring Platform Scalability Demonstration")
    logger.info("=" * 70)
    
    try:
        # 1. Batch Processing
        batch_results, batch_progress = demonstrate_batch_processing()
        
        # 2. System Monitoring
        system_monitor = demonstrate_system_monitoring()
        
        # 3. Model Retraining
        retraining_pipeline = demonstrate_model_retraining()
        
        # 4. Automated Scheduling
        scheduler = demonstrate_automated_scheduling()
        
        # 5. Memory Optimization
        memory_optimizer = demonstrate_memory_optimization()
        
        logger.info("=" * 70)
        logger.info("All scalability demonstrations completed successfully!")
        
        # Summary
        logger.info("\nSummary of demonstrated features:")
        logger.info("✓ Batch processing with parallel execution and progress tracking")
        logger.info("✓ System monitoring with real-time alerts and historical data")
        logger.info("✓ Automated model retraining with performance monitoring")
        logger.info("✓ Scheduled operations with configurable triggers")
        logger.info("✓ Memory optimization for large-scale data processing")
        
    except Exception as e:
        logger.error(f"Error in scalability demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    main()