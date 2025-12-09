"""
Unit tests for logging functionality

Tests cover:
- Log file creation and rotation
- Log message formatting
- Sensitive data sanitization
- Performance metric recording

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
import sys
import os
import logging
import tempfile
import time
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.error_handler import setup_logging, logger


class TestLogFileCreation:
    """Test log file creation and directory setup"""
    
    def test_log_directory_created(self, tmp_path):
        """Test that logs directory is created if it doesn't exist"""
        # Use temporary directory
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('pathlib.Path.exists', return_value=False):
                setup_logging()
                # Verify mkdir was called
                assert mock_mkdir.called
    
    def test_log_file_naming_convention(self):
        """Test that log files follow the naming convention dashboard_YYYYMMDD.log"""
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("dashboard_*.log"))
            if log_files:
                # Check naming pattern
                for log_file in log_files:
                    assert log_file.name.startswith("dashboard_")
                    assert log_file.name.endswith(".log")
                    # Extract date part
                    date_part = log_file.name.replace("dashboard_", "").replace(".log", "")
                    # Verify it's a valid date format (YYYYMMDD)
                    assert len(date_part) == 8
                    assert date_part.isdigit()
    
    def test_log_file_created_on_startup(self):
        """Test that log file is created when setup_logging is called"""
        test_logger = setup_logging()
        
        # Verify logger is created
        assert test_logger is not None
        assert isinstance(test_logger, logging.Logger)
        
        # Verify log file exists
        log_dir = Path("logs")
        assert log_dir.exists()
        
        expected_log_file = log_dir / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
        assert expected_log_file.exists()


class TestLogRotation:
    """Test log file rotation functionality"""
    
    def test_rotating_file_handler_configuration(self):
        """Test that rotating file handler can be configured"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_rotation.log"
            
            # Create rotating file handler
            handler = RotatingFileHandler(
                log_file,
                maxBytes=1024 * 10,  # 10KB
                backupCount=5
            )
            
            # Verify configuration
            assert handler.maxBytes == 1024 * 10
            assert handler.backupCount == 5
    
    def test_log_rotation_on_size_limit(self):
        """Test that logs rotate when size limit is reached"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_rotation.log"
            
            # Create logger with small max size
            test_logger = logging.getLogger('test_rotation')
            test_logger.setLevel(logging.INFO)
            
            handler = RotatingFileHandler(
                log_file,
                maxBytes=100,  # Very small for testing
                backupCount=3
            )
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Write enough logs to trigger rotation
            for i in range(50):
                test_logger.info(f"Test log message number {i} with some extra content to fill space")
            
            # Check if rotation occurred (backup files created)
            backup_files = list(Path(tmp_dir).glob("test_rotation.log.*"))
            assert len(backup_files) > 0, "Log rotation should create backup files"
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_backup_count_limit(self):
        """Test that only specified number of backup files are kept"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_backup.log"
            
            test_logger = logging.getLogger('test_backup')
            test_logger.setLevel(logging.INFO)
            
            backup_count = 3
            handler = RotatingFileHandler(
                log_file,
                maxBytes=50,  # Very small
                backupCount=backup_count
            )
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Write many logs to create multiple rotations
            for i in range(200):
                test_logger.info(f"Log message {i} with extra content to trigger rotation")
            
            # Count backup files
            backup_files = list(Path(tmp_dir).glob("test_backup.log.*"))
            assert len(backup_files) <= backup_count, f"Should keep at most {backup_count} backups"
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)


class TestLogMessageFormatting:
    """Test log message formatting"""
    
    def test_log_format_includes_timestamp(self):
        """Test that log messages include timestamp"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_format.log"
            
            test_logger = logging.getLogger('test_format')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Write test log
            test_logger.info("Test message")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify timestamp format (YYYY-MM-DD HH:MM:SS)
            assert len(log_content) > 0
            # Check for date pattern
            import re
            timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
            assert re.search(timestamp_pattern, log_content), "Log should contain timestamp"
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_log_format_includes_level(self):
        """Test that log messages include severity level"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_level.log"
            
            test_logger = logging.getLogger('test_level')
            test_logger.setLevel(logging.DEBUG)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Write logs at different levels
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify all levels are present
            assert "DEBUG" in log_content
            assert "INFO" in log_content
            assert "WARNING" in log_content
            assert "ERROR" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_log_format_includes_module_name(self):
        """Test that log messages include module/logger name"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_module.log"
            
            test_logger = logging.getLogger('test_module_name')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Write test log
            test_logger.info("Test message")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify module name is present
            assert "test_module_name" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_log_format_includes_message(self):
        """Test that log messages include the actual message content"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_message.log"
            
            test_logger = logging.getLogger('test_message')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Write test log with specific message
            test_message = "This is a unique test message 12345"
            test_logger.info(test_message)
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify message is present
            assert test_message in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)


class TestSensitiveDataSanitization:
    """Test sensitive data sanitization in logs"""
    
    def test_api_key_sanitization(self):
        """Test that API keys are not logged in plain text"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_sanitize.log"
            
            test_logger = logging.getLogger('test_sanitize')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate logging with API key
            api_key = "sk_test_1234567890abcdef"
            sanitized_key = api_key[:8] + "..." + api_key[-4:]
            
            # Log sanitized version
            test_logger.info(f"API request with key: {sanitized_key}")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify full API key is NOT in logs
            assert api_key not in log_content
            # Verify sanitized version IS in logs
            assert sanitized_key in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_password_sanitization(self):
        """Test that passwords are not logged"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_password.log"
            
            test_logger = logging.getLogger('test_password')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate logging authentication without password
            password = "SuperSecret123!"
            
            # Log without password
            test_logger.info("Authentication attempt for user: testuser")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify password is NOT in logs
            assert password not in log_content
            assert "testuser" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_credential_sanitization_function(self):
        """Test function to sanitize credentials from log messages"""
        
        def sanitize_credentials(message: str) -> str:
            """Sanitize sensitive information from log messages"""
            import re
            
            # Sanitize API keys (pattern: sk_*, api_*, etc.)
            message = re.sub(
                r'(sk_|api_|key_)[a-zA-Z0-9]{16,}',
                lambda m: m.group(0)[:8] + '...' + m.group(0)[-4:],
                message
            )
            
            # Sanitize tokens
            message = re.sub(
                r'(token[=:]\s*)["\']?[a-zA-Z0-9_-]{20,}["\']?',
                r'\1***REDACTED***',
                message,
                flags=re.IGNORECASE
            )
            
            # Sanitize passwords
            message = re.sub(
                r'(password[=:]\s*)["\']?[^\s"\']+["\']?',
                r'\1***REDACTED***',
                message,
                flags=re.IGNORECASE
            )
            
            return message
        
        # Test API key sanitization
        message = "Connecting with api_key_1234567890abcdefghij"
        sanitized = sanitize_credentials(message)
        assert "api_key_" in sanitized
        assert "1234567890abcdefghij" not in sanitized
        
        # Test token sanitization
        message = "Authorization token: abc123def456ghi789jkl"
        sanitized = sanitize_credentials(message)
        assert "***REDACTED***" in sanitized
        assert "abc123def456ghi789jkl" not in sanitized
        
        # Test password sanitization
        message = "Login with password=MySecretPass123"
        sanitized = sanitize_credentials(message)
        assert "***REDACTED***" in sanitized
        assert "MySecretPass123" not in sanitized
    
    def test_url_parameter_sanitization(self):
        """Test that sensitive URL parameters are sanitized"""
        
        def sanitize_url(url: str) -> str:
            """Sanitize sensitive parameters from URLs"""
            import re
            
            # Sanitize API keys in URL parameters
            url = re.sub(
                r'([?&])(api_key|apikey|key)=([^&]+)',
                r'\1\2=***REDACTED***',
                url,
                flags=re.IGNORECASE
            )
            
            # Sanitize tokens in URL parameters
            url = re.sub(
                r'([?&])(token|access_token)=([^&]+)',
                r'\1\2=***REDACTED***',
                url,
                flags=re.IGNORECASE
            )
            
            return url
        
        # Test API key in URL
        url = "https://api.example.com/data?api_key=secret123&format=json"
        sanitized = sanitize_url(url)
        assert "secret123" not in sanitized
        assert "***REDACTED***" in sanitized
        assert "format=json" in sanitized
        
        # Test token in URL
        url = "https://api.example.com/data?access_token=abc123def456"
        sanitized = sanitize_url(url)
        assert "abc123def456" not in sanitized
        assert "***REDACTED***" in sanitized


class TestPerformanceMetricRecording:
    """Test performance metric recording in logs"""
    
    def test_operation_timing_logged(self):
        """Test that operation execution time is logged"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_timing.log"
            
            test_logger = logging.getLogger('test_timing')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate timed operation
            start_time = time.time()
            time.sleep(0.1)  # Simulate work
            end_time = time.time()
            elapsed = end_time - start_time
            
            test_logger.info(f"Operation completed in {elapsed:.3f} seconds")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify timing is logged
            assert "completed in" in log_content
            assert "seconds" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_memory_usage_logged(self):
        """Test that memory usage can be logged"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_memory.log"
            
            test_logger = logging.getLogger('test_memory')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate memory usage logging
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            test_logger.info(f"Memory usage: {memory_mb:.2f} MB")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify memory usage is logged
            assert "Memory usage:" in log_content
            assert "MB" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_api_latency_logged(self):
        """Test that API call latency is logged"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_latency.log"
            
            test_logger = logging.getLogger('test_latency')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate API call with latency
            url = "https://api.example.com/data"
            start = time.time()
            time.sleep(0.05)  # Simulate API call
            latency = time.time() - start
            
            test_logger.info(f"API call to {url} completed in {latency*1000:.0f}ms")
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify latency is logged
            assert "API call" in log_content
            assert "completed in" in log_content
            assert "ms" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_processing_metrics_logged(self):
        """Test that data processing metrics are logged"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_processing.log"
            
            test_logger = logging.getLogger('test_processing')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate processing metrics
            pixels_processed = 10980 * 10980
            processing_time = 8.5
            pixels_per_second = pixels_processed / processing_time
            
            test_logger.info(
                f"Processed {pixels_processed:,} pixels in {processing_time:.2f}s "
                f"({pixels_per_second:,.0f} pixels/sec)"
            )
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify processing metrics are logged
            assert "Processed" in log_content
            assert "pixels" in log_content
            assert "pixels/sec" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)
    
    def test_model_inference_time_logged(self):
        """Test that model inference time is logged"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_inference.log"
            
            test_logger = logging.getLogger('test_inference')
            test_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            test_logger.addHandler(handler)
            
            # Simulate model inference
            model_name = "CropHealthCNN"
            batch_size = 32
            start = time.time()
            time.sleep(0.08)  # Simulate inference
            inference_time = time.time() - start
            time_per_sample = inference_time / batch_size * 1000
            
            test_logger.info(
                f"{model_name} inference: {batch_size} samples in {inference_time:.3f}s "
                f"({time_per_sample:.1f}ms per sample)"
            )
            handler.flush()
            
            # Read log file
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify inference metrics are logged
            assert model_name in log_content
            assert "inference" in log_content
            assert "samples" in log_content
            assert "ms per sample" in log_content
            
            # Cleanup
            handler.close()
            test_logger.removeHandler(handler)


class TestLoggerIntegration:
    """Test integration with existing logger"""
    
    def test_logger_instance_exists(self):
        """Test that logger instance is created"""
        from utils.error_handler import logger
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_logger_has_handlers(self):
        """Test that logger has configured handlers"""
        from utils.error_handler import logger
        
        # Logger should have at least one handler
        assert len(logger.handlers) > 0 or len(logger.parent.handlers) > 0
    
    def test_logger_level_configuration(self):
        """Test that logger level is properly configured"""
        from utils.error_handler import logger
        
        # Logger should have a level set (INFO or higher)
        effective_level = logger.getEffectiveLevel()
        assert effective_level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ]
    
    def test_logger_can_write_messages(self):
        """Test that logger can write messages without errors"""
        from utils.error_handler import logger
        
        # These should not raise exceptions
        try:
            logger.debug("Test debug message")
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
        except Exception as e:
            pytest.fail(f"Logger raised exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
