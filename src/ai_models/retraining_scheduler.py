"""
Automated scheduler for model retraining pipeline.

This module provides scheduling capabilities for automated model monitoring
and retraining based on configurable intervals and triggers.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import schedule

from .model_monitoring import ModelRetrainingPipeline, RetrainingTrigger

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for retraining scheduler."""
    monitoring_interval_hours: int = 24  # How often to check models
    max_concurrent_retraining: int = 1   # Max models to retrain simultaneously
    enable_scheduler: bool = True        # Whether scheduler is active
    quiet_hours_start: int = 22          # Start of quiet hours (no retraining)
    quiet_hours_end: int = 6             # End of quiet hours
    max_retrain_attempts: int = 3        # Max retry attempts for failed retraining
    notification_webhook: Optional[str] = None  # Webhook for notifications


class RetrainingScheduler:
    """
    Automated scheduler for model monitoring and retraining.
    
    Provides scheduling, concurrency control, and notification capabilities
    for the model retraining pipeline.
    """
    
    def __init__(self, 
                 config: ScheduleConfig = None,
                 pipeline: ModelRetrainingPipeline = None):
        """
        Initialize retraining scheduler.
        
        Args:
            config: Scheduler configuration
            pipeline: Model retraining pipeline
        """
        self.config = config or ScheduleConfig()
        self.pipeline = pipeline or ModelRetrainingPipeline()
        
        self.is_running = False
        self.scheduler_thread = None
        self.active_retraining = set()  # Track active retraining jobs
        self.retraining_history = []
        self.notification_callbacks = []
        
        # Setup scheduling
        self._setup_schedule()
        
    def _setup_schedule(self):
        """Setup the monitoring schedule."""
        if self.config.enable_scheduler:
            schedule.every(self.config.monitoring_interval_hours).hours.do(
                self._run_monitoring_cycle
            )
            logger.info(f"Scheduled monitoring every {self.config.monitoring_interval_hours} hours")
    
    def start(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Retraining scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Retraining scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _run_monitoring_cycle(self):
        """Run a monitoring cycle."""
        if not self._is_allowed_time():
            logger.info("Skipping monitoring cycle during quiet hours")
            return
        
        logger.info("Starting scheduled monitoring cycle")
        
        try:
            results = self.pipeline.run_monitoring_cycle()
            self._process_monitoring_results(results)
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
            self._send_notification(
                "error",
                f"Monitoring cycle failed: {str(e)}"
            )
    
    def _process_monitoring_results(self, results: Dict[str, Any]):
        """Process results from monitoring cycle."""
        # Log results
        logger.info(f"Monitoring cycle completed: {len(results['models_checked'])} models checked")
        
        # Handle retraining triggers
        for retrain_info in results['retraining_triggered']:
            model_type = retrain_info['model_type']
            
            # Check if we can start retraining
            if len(self.active_retraining) >= self.config.max_concurrent_retraining:
                logger.warning(f"Max concurrent retraining limit reached, queuing {model_type}")
                continue
            
            # Start retraining in background thread
            self._start_background_retraining(model_type, retrain_info)
        
        # Send notification summary
        if results['retraining_triggered'] or results['errors']:
            self._send_monitoring_summary(results)
    
    def _start_background_retraining(self, model_type: str, retrain_info: Dict[str, Any]):
        """Start retraining in a background thread."""
        if model_type in self.active_retraining:
            logger.warning(f"Retraining already active for {model_type}")
            return
        
        self.active_retraining.add(model_type)
        
        def retraining_worker():
            try:
                logger.info(f"Starting background retraining for {model_type}")
                
                # Record start time
                start_time = datetime.now()
                
                # Perform retraining (this would call the actual retraining methods)
                success = self._perform_retraining(model_type)
                
                # Record completion
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log results
                if success:
                    logger.info(f"Retraining completed for {model_type} in {duration:.1f}s")
                    self._send_notification(
                        "success",
                        f"Model {model_type} retrained successfully"
                    )
                else:
                    logger.error(f"Retraining failed for {model_type}")
                    self._send_notification(
                        "error",
                        f"Model {model_type} retraining failed"
                    )
                
                # Record in history
                self.retraining_history.append({
                    'model_type': model_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_seconds': duration,
                    'success': success,
                    'trigger_info': retrain_info
                })
                
            except Exception as e:
                logger.error(f"Error in background retraining for {model_type}: {str(e)}")
                self._send_notification(
                    "error",
                    f"Retraining error for {model_type}: {str(e)}"
                )
            finally:
                self.active_retraining.discard(model_type)
        
        # Start worker thread
        worker_thread = threading.Thread(target=retraining_worker, daemon=True)
        worker_thread.start()
    
    def _perform_retraining(self, model_type: str) -> bool:
        """Perform the actual retraining with retry logic."""
        for attempt in range(self.config.max_retrain_attempts):
            try:
                logger.info(f"Retraining attempt {attempt + 1} for {model_type}")
                
                # Load training data (placeholder - would be implemented)
                training_data = self._load_retraining_data(model_type)
                
                if model_type == 'lstm':
                    version_id = self.pipeline.retrain_lstm_model(training_data)
                elif model_type == 'cnn':
                    version_id = self.pipeline.retrain_cnn_model(training_data)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                logger.info(f"Retraining successful, new version: {version_id}")
                return True
                
            except Exception as e:
                logger.error(f"Retraining attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retrain_attempts - 1:
                    time.sleep(300)  # Wait 5 minutes before retry
        
        return False
    
    def _load_retraining_data(self, model_type: str) -> Any:
        """Load data for retraining (placeholder)."""
        # This would be implemented based on your data pipeline
        logger.info(f"Loading retraining data for {model_type}")
        return None
    
    def _is_allowed_time(self) -> bool:
        """Check if current time is outside quiet hours."""
        if not self.config.enable_scheduler:
            return False
        
        current_hour = datetime.now().hour
        
        # Handle quiet hours that span midnight
        if self.config.quiet_hours_start > self.config.quiet_hours_end:
            # e.g., 22:00 to 06:00
            return not (current_hour >= self.config.quiet_hours_start or 
                       current_hour < self.config.quiet_hours_end)
        else:
            # e.g., 02:00 to 06:00
            return not (self.config.quiet_hours_start <= current_hour < self.config.quiet_hours_end)
    
    def _send_notification(self, level: str, message: str):
        """Send notification about retraining events."""
        notification = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'source': 'retraining_scheduler'
        }
        
        # Call registered callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Error in notification callback: {str(e)}")
        
        # Send webhook notification if configured
        if self.config.notification_webhook:
            self._send_webhook_notification(notification)
    
    def _send_webhook_notification(self, notification: Dict[str, Any]):
        """Send webhook notification."""
        try:
            import requests
            response = requests.post(
                self.config.notification_webhook,
                json=notification,
                timeout=10
            )
            response.raise_for_status()
            logger.debug("Webhook notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {str(e)}")
    
    def _send_monitoring_summary(self, results: Dict[str, Any]):
        """Send summary of monitoring results."""
        summary = {
            'models_checked': len(results['models_checked']),
            'retraining_triggered': len(results['retraining_triggered']),
            'errors': len(results['errors']),
            'timestamp': results['timestamp'].isoformat()
        }
        
        message = f"Monitoring summary: {summary['models_checked']} models checked, " \
                 f"{summary['retraining_triggered']} retraining triggered, " \
                 f"{summary['errors']} errors"
        
        self._send_notification("info", message)
    
    def add_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a notification callback function."""
        self.notification_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'is_running': self.is_running,
            'active_retraining': list(self.active_retraining),
            'next_monitoring': schedule.next_run().isoformat() if schedule.jobs else None,
            'retraining_history_count': len(self.retraining_history),
            'config': {
                'monitoring_interval_hours': self.config.monitoring_interval_hours,
                'max_concurrent_retraining': self.config.max_concurrent_retraining,
                'enable_scheduler': self.config.enable_scheduler
            }
        }
    
    def get_retraining_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent retraining history."""
        # Sort by start time (most recent first)
        sorted_history = sorted(
            self.retraining_history,
            key=lambda x: x['start_time'],
            reverse=True
        )
        
        # Convert datetime objects to strings for JSON serialization
        history = []
        for entry in sorted_history[:limit]:
            entry_copy = entry.copy()
            entry_copy['start_time'] = entry['start_time'].isoformat()
            entry_copy['end_time'] = entry['end_time'].isoformat()
            history.append(entry_copy)
        
        return history
    
    def force_monitoring_cycle(self) -> Dict[str, Any]:
        """Force an immediate monitoring cycle."""
        logger.info("Forcing immediate monitoring cycle")
        
        try:
            results = self.pipeline.run_monitoring_cycle()
            self._process_monitoring_results(results)
            return results
        except Exception as e:
            logger.error(f"Error in forced monitoring cycle: {str(e)}")
            raise
    
    def cancel_retraining(self, model_type: str) -> bool:
        """Cancel active retraining for a model type."""
        if model_type in self.active_retraining:
            # Note: This is a simple implementation
            # In a production system, you'd need more sophisticated cancellation
            logger.warning(f"Retraining cancellation requested for {model_type}")
            self._send_notification(
                "warning",
                f"Retraining cancellation requested for {model_type}"
            )
            return True
        else:
            logger.info(f"No active retraining found for {model_type}")
            return False


def create_default_scheduler() -> RetrainingScheduler:
    """Create a scheduler with default configuration."""
    config = ScheduleConfig(
        monitoring_interval_hours=24,
        max_concurrent_retraining=1,
        enable_scheduler=True,
        quiet_hours_start=22,
        quiet_hours_end=6
    )
    
    trigger_config = RetrainingTrigger(
        performance_threshold=0.1,
        data_drift_threshold=0.2,
        time_threshold_days=30,
        min_new_samples=100,
        enable_auto_retrain=True
    )
    
    pipeline = ModelRetrainingPipeline(trigger_config)
    scheduler = RetrainingScheduler(config, pipeline)
    
    return scheduler


# Example notification callback
def log_notification_callback(notification: Dict[str, Any]):
    """Example notification callback that logs to file."""
    log_file = Path("logs/retraining_notifications.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(f"{notification['timestamp']}: [{notification['level']}] {notification['message']}\n")


if __name__ == "__main__":
    # Example usage
    scheduler = create_default_scheduler()
    scheduler.add_notification_callback(log_notification_callback)
    
    try:
        scheduler.start()
        logger.info("Scheduler started, press Ctrl+C to stop")
        
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
        scheduler.stop()