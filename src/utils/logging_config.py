"""
Comprehensive Logging System for AgriFlux

This module provides centralized logging configuration with:
- Detailed API request/response logging (Requirement 7.1)
- Download progress logging (Requirement 7.2)
- Training metrics logging (Requirement 7.3)
- Error logging with stack traces (Requirement 7.4)
- Pipeline summary report generation (Requirement 7.5)

Usage:
    from src.utils.logging_config import setup_logging, get_logger
    
    # Setup logging for a module
    logger = setup_logging('my_module', log_file='logs/my_module.log')
    
    # Or get a logger with default configuration
    logger = get_logger(__name__)
"""

import logging
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from logging.handlers import RotatingFileHandler
import threading


# Global lock for thread-safe logging
_logging_lock = threading.Lock()


class APIRequestLogger:
    """
    Logger for detailed API request/response logging (Requirement 7.1).
    
    Logs:
    - Request URL, method, headers, and payload
    - Response status code, headers, and body
    - Request duration
    - Error details if request fails
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request(self, method: str, url: str, headers: Optional[Dict] = None,
                   payload: Optional[Dict] = None, **kwargs) -> None:
        """
        Log API request details.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            payload: Request payload/body
            **kwargs: Additional request parameters
        """
        self.logger.info("="*70)
        self.logger.info(f"API REQUEST: {method} {url}")
        
        if headers:
            # Sanitize sensitive headers
            safe_headers = self._sanitize_headers(headers)
            self.logger.debug(f"  Headers: {json.dumps(safe_headers, indent=2)}")
        
        if payload:
            self.logger.debug(f"  Payload: {json.dumps(payload, indent=2)}")
        
        if kwargs:
            self.logger.debug(f"  Additional params: {json.dumps(kwargs, indent=2)}")
    
    def log_response(self, status_code: int, headers: Optional[Dict] = None,
                    body: Optional[str] = None, duration: Optional[float] = None,
                    error: Optional[Exception] = None) -> None:
        """
        Log API response details.
        
        Args:
            status_code: HTTP status code
            headers: Response headers
            body: Response body (truncated if too long)
            duration: Request duration in seconds
            error: Exception if request failed
        """
        if error:
            self.logger.error(f"API RESPONSE: ERROR - {error}")
            self.logger.error(f"  Status: {status_code}")
            if body:
                self.logger.error(f"  Body: {self._truncate_body(body)}")
        else:
            self.logger.info(f"API RESPONSE: {status_code}")
            if duration:
                self.logger.info(f"  Duration: {duration:.2f}s")
            if headers:
                self.logger.debug(f"  Headers: {json.dumps(dict(headers), indent=2)}")
            if body:
                self.logger.debug(f"  Body: {self._truncate_body(body)}")
        
        self.logger.info("="*70)
    
    def _sanitize_headers(self, headers: Dict) -> Dict:
        """Remove sensitive information from headers."""
        safe_headers = headers.copy()
        sensitive_keys = ['authorization', 'api-key', 'x-api-key', 'token']
        
        for key in safe_headers:
            if key.lower() in sensitive_keys:
                safe_headers[key] = '***REDACTED***'
        
        return safe_headers
    
    def _truncate_body(self, body: str, max_length: int = 1000) -> str:
        """Truncate response body if too long."""
        if len(body) > max_length:
            return body[:max_length] + f"... (truncated, total length: {len(body)})"
        return body


class DownloadProgressLogger:
    """
    Logger for download progress tracking (Requirement 7.2).
    
    Tracks:
    - Total items to download
    - Current progress
    - Success/failure counts
    - Download statistics (size, duration, etc.)
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.total_items = 0
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = None
        self.download_stats = []
    
    def start_download(self, total_items: int, description: str = "items") -> None:
        """
        Start tracking a download batch.
        
        Args:
            total_items: Total number of items to download
            description: Description of items being downloaded
        """
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = datetime.now()
        self.download_stats = []
        
        self.logger.info("="*70)
        self.logger.info(f"DOWNLOAD STARTED: {total_items} {description}")
        self.logger.info(f"  Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)
    
    def log_item_progress(self, item_name: str, success: bool = True,
                         size_bytes: Optional[int] = None,
                         duration: Optional[float] = None,
                         error: Optional[str] = None) -> None:
        """
        Log progress for a single item.
        
        Args:
            item_name: Name/identifier of the item
            success: Whether download succeeded
            size_bytes: Size of downloaded data in bytes
            duration: Download duration in seconds
            error: Error message if failed
        """
        if success:
            self.completed_items += 1
            status = "✓"
            level = logging.INFO
        else:
            self.failed_items += 1
            status = "✗"
            level = logging.ERROR
        
        progress_pct = (self.completed_items + self.failed_items) / self.total_items * 100
        
        msg = f"[{self.completed_items + self.failed_items}/{self.total_items}] {status} {item_name}"
        
        if size_bytes:
            msg += f" ({self._format_size(size_bytes)})"
        if duration:
            msg += f" in {duration:.1f}s"
        if error:
            msg += f" - ERROR: {error}"
        
        self.logger.log(level, msg)
        self.logger.info(f"  Progress: {progress_pct:.1f}%")
        
        # Store stats
        self.download_stats.append({
            'item': item_name,
            'success': success,
            'size_bytes': size_bytes,
            'duration': duration,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def finish_download(self) -> Dict[str, Any]:
        """
        Finish download tracking and return summary.
        
        Returns:
            Dictionary with download statistics
        """
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        total_size = sum(s['size_bytes'] for s in self.download_stats if s['size_bytes'])
        avg_duration = sum(s['duration'] for s in self.download_stats if s['duration']) / len(self.download_stats) if self.download_stats else 0
        
        summary = {
            'total_items': self.total_items,
            'completed': self.completed_items,
            'failed': self.failed_items,
            'success_rate': self.completed_items / self.total_items if self.total_items > 0 else 0,
            'total_duration': total_duration,
            'total_size_bytes': total_size,
            'avg_item_duration': avg_duration,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': end_time.isoformat()
        }
        
        self.logger.info("="*70)
        self.logger.info("DOWNLOAD COMPLETED")
        self.logger.info(f"  Total items: {self.total_items}")
        self.logger.info(f"  Successful: {self.completed_items}")
        self.logger.info(f"  Failed: {self.failed_items}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        self.logger.info(f"  Total duration: {total_duration:.1f}s")
        if total_size > 0:
            self.logger.info(f"  Total size: {self._format_size(total_size)}")
        self.logger.info("="*70)
        
        return summary
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


class TrainingMetricsLogger:
    """
    Logger for training metrics (Requirement 7.3).
    
    Logs:
    - Epoch-by-epoch metrics
    - Loss values
    - Accuracy scores
    - Validation metrics
    - Best model checkpoints
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.training_history = []
        self.best_metrics = {}
    
    def log_training_start(self, model_name: str, total_epochs: int,
                          config: Dict[str, Any]) -> None:
        """
        Log training start.
        
        Args:
            model_name: Name of the model being trained
            total_epochs: Total number of epochs
            config: Training configuration
        """
        self.logger.info("="*70)
        self.logger.info(f"TRAINING STARTED: {model_name}")
        self.logger.info(f"  Total epochs: {total_epochs}")
        self.logger.info(f"  Configuration:")
        for key, value in config.items():
            self.logger.info(f"    {key}: {value}")
        self.logger.info("="*70)
    
    def log_epoch_metrics(self, epoch: int, total_epochs: int,
                         train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float],
                         is_best: bool = False) -> None:
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            train_metrics: Training metrics (loss, accuracy, etc.)
            val_metrics: Validation metrics
            is_best: Whether this is the best epoch so far
        """
        msg = f"Epoch {epoch}/{total_epochs}"
        
        # Training metrics
        train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        msg += f" - Train: {train_str}"
        
        # Validation metrics
        val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        msg += f" - Val: {val_str}"
        
        if is_best:
            msg += " ✓ BEST"
            self.logger.info(msg)
        else:
            self.logger.info(msg)
        
        # Store history
        self.training_history.append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update best metrics
        if is_best:
            self.best_metrics = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
    
    def log_training_complete(self, final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log training completion and return summary.
        
        Args:
            final_metrics: Final evaluation metrics
            
        Returns:
            Training summary dictionary
        """
        summary = {
            'total_epochs': len(self.training_history),
            'best_metrics': self.best_metrics,
            'final_metrics': final_metrics,
            'training_history': self.training_history
        }
        
        self.logger.info("="*70)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"  Total epochs: {len(self.training_history)}")
        if self.best_metrics:
            self.logger.info(f"  Best epoch: {self.best_metrics['epoch']}")
            self.logger.info(f"  Best metrics:")
            for key, value in self.best_metrics.get('val_metrics', {}).items():
                self.logger.info(f"    {key}: {value:.4f}")
        self.logger.info(f"  Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"    {key}: {value:.4f}")
        self.logger.info("="*70)
        
        return summary


class ErrorLogger:
    """
    Logger for errors with stack traces (Requirement 7.4).
    
    Provides detailed error logging including:
    - Exception type and message
    - Full stack trace
    - Context information
    - Error categorization
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_count = 0
        self.errors = []
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                 fatal: bool = False) -> None:
        """
        Log an error with full details.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            fatal: Whether this is a fatal error
        """
        self.error_count += 1
        
        error_info = {
            'error_number': self.error_count,
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'fatal': fatal,
            'timestamp': datetime.now().isoformat()
        }
        
        self.errors.append(error_info)
        
        level = logging.CRITICAL if fatal else logging.ERROR
        
        self.logger.log(level, "="*70)
        self.logger.log(level, f"ERROR #{self.error_count}: {type(error).__name__}")
        self.logger.log(level, f"  Message: {str(error)}")
        
        if context:
            self.logger.log(level, f"  Context:")
            for key, value in context.items():
                self.logger.log(level, f"    {key}: {value}")
        
        self.logger.log(level, f"  Stack trace:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.logger.log(level, f"    {line}")
        
        if fatal:
            self.logger.critical("  FATAL ERROR - Execution will terminate")
        
        self.logger.log(level, "="*70)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all logged errors.
        
        Returns:
            Dictionary with error statistics
        """
        error_types = {}
        for error in self.errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': self.error_count,
            'error_types': error_types,
            'fatal_errors': sum(1 for e in self.errors if e['fatal']),
            'errors': self.errors
        }


class PipelineSummaryReporter:
    """
    Generate pipeline summary reports (Requirement 7.5).
    
    Collects and reports:
    - Overall pipeline status
    - Step-by-step results
    - Performance metrics
    - Error summary
    - Resource usage
    """
    
    def __init__(self, logger: logging.Logger, pipeline_name: str):
        self.logger = logger
        self.pipeline_name = pipeline_name
        self.start_time = datetime.now()
        self.steps = []
        self.metrics = {}
    
    def log_step_start(self, step_name: str, description: str = "") -> None:
        """
        Log the start of a pipeline step.
        
        Args:
            step_name: Name of the step
            description: Optional description
        """
        step = {
            'name': step_name,
            'description': description,
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None,
            'status': 'running',
            'metrics': {},
            'errors': []
        }
        self.steps.append(step)
        
        self.logger.info("="*70)
        self.logger.info(f"STEP: {step_name}")
        if description:
            self.logger.info(f"  {description}")
        self.logger.info(f"  Started: {step['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)
    
    def log_step_complete(self, step_name: str, status: str = 'success',
                         metrics: Optional[Dict[str, Any]] = None,
                         error: Optional[str] = None) -> None:
        """
        Log the completion of a pipeline step.
        
        Args:
            step_name: Name of the step
            status: Status ('success', 'failed', 'skipped')
            metrics: Step-specific metrics
            error: Error message if failed
        """
        # Find the step
        step = next((s for s in self.steps if s['name'] == step_name), None)
        if not step:
            self.logger.warning(f"Step '{step_name}' not found in pipeline")
            return
        
        step['end_time'] = datetime.now()
        step['duration'] = (step['end_time'] - step['start_time']).total_seconds()
        step['status'] = status
        step['metrics'] = metrics or {}
        if error:
            step['errors'].append(error)
        
        status_symbol = "✓" if status == 'success' else "✗" if status == 'failed' else "⊘"
        
        self.logger.info("="*70)
        self.logger.info(f"STEP COMPLETE: {step_name} {status_symbol}")
        self.logger.info(f"  Status: {status.upper()}")
        self.logger.info(f"  Duration: {step['duration']:.1f}s")
        if metrics:
            self.logger.info(f"  Metrics:")
            for key, value in metrics.items():
                self.logger.info(f"    {key}: {value}")
        if error:
            self.logger.error(f"  Error: {error}")
        self.logger.info("="*70)
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add a pipeline-level metric."""
        self.metrics[key] = value
    
    def generate_summary(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate and log pipeline summary report.
        
        Args:
            output_file: Optional file path to save JSON report
            
        Returns:
            Summary dictionary
        """
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        successful_steps = sum(1 for s in self.steps if s['status'] == 'success')
        failed_steps = sum(1 for s in self.steps if s['status'] == 'failed')
        
        summary = {
            'pipeline_name': self.pipeline_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration': total_duration,
            'total_steps': len(self.steps),
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / len(self.steps) if self.steps else 0,
            'overall_status': 'success' if failed_steps == 0 else 'failed',
            'steps': self.steps,
            'metrics': self.metrics
        }
        
        # Log summary
        self.logger.info("\n" + "="*70)
        self.logger.info(f"PIPELINE SUMMARY: {self.pipeline_name}")
        self.logger.info("="*70)
        self.logger.info(f"  Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        self.logger.info(f"  Total steps: {len(self.steps)}")
        self.logger.info(f"  Successful: {successful_steps}")
        self.logger.info(f"  Failed: {failed_steps}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        self.logger.info(f"  Overall status: {summary['overall_status'].upper()}")
        
        if self.metrics:
            self.logger.info(f"\n  Pipeline Metrics:")
            for key, value in self.metrics.items():
                self.logger.info(f"    {key}: {value}")
        
        self.logger.info(f"\n  Step Details:")
        for step in self.steps:
            status_symbol = "✓" if step['status'] == 'success' else "✗" if step['status'] == 'failed' else "⊘"
            self.logger.info(f"    {status_symbol} {step['name']}: {step['status']} ({step['duration']:.1f}s)")
        
        self.logger.info("="*70 + "\n")
        
        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Summary report saved to: {output_file}")
        
        return summary


def setup_logging(module_name: str,
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 console: bool = True,
                 file_rotation: bool = True,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup comprehensive logging for a module.
    
    Args:
        module_name: Name of the module
        log_file: Path to log file (optional)
        level: Logging level
        console: Whether to log to console
        file_rotation: Whether to use rotating file handler
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    with _logging_lock:
        logger = logging.getLogger(module_name)
        
        # Avoid adding handlers multiple times
        if logger.handlers:
            return logger
        
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_rotation:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)
            
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def get_logger(module_name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with default configuration.
    
    Args:
        module_name: Name of the module
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    return setup_logging(module_name, log_file=log_file)


def create_api_logger(logger: logging.Logger) -> APIRequestLogger:
    """Create an API request logger."""
    return APIRequestLogger(logger)


def create_download_logger(logger: logging.Logger) -> DownloadProgressLogger:
    """Create a download progress logger."""
    return DownloadProgressLogger(logger)


def create_training_logger(logger: logging.Logger) -> TrainingMetricsLogger:
    """Create a training metrics logger."""
    return TrainingMetricsLogger(logger)


def create_error_logger(logger: logging.Logger) -> ErrorLogger:
    """Create an error logger."""
    return ErrorLogger(logger)


def create_pipeline_reporter(logger: logging.Logger, pipeline_name: str) -> PipelineSummaryReporter:
    """Create a pipeline summary reporter."""
    return PipelineSummaryReporter(logger, pipeline_name)
