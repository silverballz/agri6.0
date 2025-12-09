"""
Unit tests for the comprehensive logging system.

Tests all logging components:
- API request/response logging
- Download progress logging
- Training metrics logging
- Error logging with stack traces
- Pipeline summary reporting
"""

import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import (
    setup_logging,
    create_api_logger,
    create_download_logger,
    create_training_logger,
    create_error_logger,
    create_pipeline_reporter
)


class TestLoggingSystem(unittest.TestCase):
    """Test suite for comprehensive logging system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / 'test.log'
    
    def test_setup_logging(self):
        """Test basic logging setup."""
        logger = setup_logging('test_module', log_file=str(self.log_file))
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_module')
        
        # Test logging
        logger.info("Test message")
        
        # Verify log file created
        self.assertTrue(self.log_file.exists())
    
    def test_api_logger(self):
        """Test API request/response logging."""
        logger = setup_logging('test_api', log_file=str(self.log_file))
        api_logger = create_api_logger(logger)
        
        # Log request
        api_logger.log_request(
            method='POST',
            url='https://api.example.com/test',
            headers={'Authorization': 'Bearer secret'},
            payload={'key': 'value'}
        )
        
        # Log response
        api_logger.log_response(
            status_code=200,
            body='{"result": "success"}',
            duration=1.5
        )
        
        # Verify log file contains expected content
        log_content = self.log_file.read_text()
        self.assertIn('API REQUEST', log_content)
        self.assertIn('POST', log_content)
        self.assertIn('API RESPONSE', log_content)
        self.assertIn('200', log_content)
    
    def test_download_logger(self):
        """Test download progress logging."""
        logger = setup_logging('test_download', log_file=str(self.log_file))
        download_logger = create_download_logger(logger)
        
        # Start download
        download_logger.start_download(total_items=3, description="test items")
        
        # Log items
        download_logger.log_item_progress('item1', success=True, size_bytes=1000, duration=1.0)
        download_logger.log_item_progress('item2', success=True, size_bytes=2000, duration=1.5)
        download_logger.log_item_progress('item3', success=False, error="Test error")
        
        # Finish download
        summary = download_logger.finish_download()
        
        # Verify summary
        self.assertEqual(summary['total_items'], 3)
        self.assertEqual(summary['completed'], 2)
        self.assertEqual(summary['failed'], 1)
        self.assertAlmostEqual(summary['success_rate'], 2/3, places=2)
        
        # Verify log content
        log_content = self.log_file.read_text()
        self.assertIn('DOWNLOAD STARTED', log_content)
        self.assertIn('DOWNLOAD COMPLETED', log_content)
    
    def test_training_logger(self):
        """Test training metrics logging."""
        logger = setup_logging('test_training', log_file=str(self.log_file))
        training_logger = create_training_logger(logger)
        
        # Start training
        training_logger.log_training_start(
            model_name='TestModel',
            total_epochs=5,
            config={'batch_size': 32, 'lr': 0.001}
        )
        
        # Log epochs
        for epoch in range(1, 6):
            training_logger.log_epoch_metrics(
                epoch=epoch,
                total_epochs=5,
                train_metrics={'loss': 1.0 - epoch*0.1, 'accuracy': 0.5 + epoch*0.05},
                val_metrics={'loss': 1.1 - epoch*0.1, 'accuracy': 0.45 + epoch*0.05},
                is_best=(epoch == 5)
            )
        
        # Complete training
        final_metrics = {'test_accuracy': 0.85}
        summary = training_logger.log_training_complete(final_metrics)
        
        # Verify summary
        self.assertEqual(summary['total_epochs'], 5)
        self.assertIn('best_metrics', summary)
        self.assertEqual(summary['best_metrics']['epoch'], 5)
        
        # Verify log content
        log_content = self.log_file.read_text()
        self.assertIn('TRAINING STARTED', log_content)
        self.assertIn('TRAINING COMPLETED', log_content)
        self.assertIn('BEST', log_content)
    
    def test_error_logger(self):
        """Test error logging with stack traces."""
        logger = setup_logging('test_error', log_file=str(self.log_file))
        error_logger = create_error_logger(logger)
        
        # Log an error
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            error_logger.log_error(
                error=e,
                context={'operation': 'test_op', 'step': 'validation'},
                fatal=False
            )
        
        # Get summary
        summary = error_logger.get_error_summary()
        
        # Verify summary
        self.assertEqual(summary['total_errors'], 1)
        self.assertIn('ValueError', summary['error_types'])
        self.assertEqual(summary['error_types']['ValueError'], 1)
        
        # Verify log content
        log_content = self.log_file.read_text()
        self.assertIn('ERROR #1', log_content)
        self.assertIn('ValueError', log_content)
        self.assertIn('Test error message', log_content)
        self.assertIn('Stack trace', log_content)
    
    def test_pipeline_reporter(self):
        """Test pipeline summary reporting."""
        logger = setup_logging('test_pipeline', log_file=str(self.log_file))
        reporter = create_pipeline_reporter(logger, 'Test Pipeline')
        
        # Log steps
        reporter.log_step_start('step1', 'First step')
        reporter.log_step_complete('step1', status='success', metrics={'count': 10})
        
        reporter.log_step_start('step2', 'Second step')
        reporter.log_step_complete('step2', status='failed', error='Test error')
        
        # Add metrics
        reporter.add_metric('total_items', 100)
        reporter.add_metric('processed', 95)
        
        # Generate summary
        summary_file = Path(self.temp_dir) / 'summary.json'
        summary = reporter.generate_summary(output_file=summary_file)
        
        # Verify summary
        self.assertEqual(summary['pipeline_name'], 'Test Pipeline')
        self.assertEqual(summary['total_steps'], 2)
        self.assertEqual(summary['successful_steps'], 1)
        self.assertEqual(summary['failed_steps'], 1)
        self.assertEqual(summary['overall_status'], 'failed')
        self.assertEqual(summary['metrics']['total_items'], 100)
        
        # Verify summary file created
        self.assertTrue(summary_file.exists())
        
        # Verify JSON content
        with open(summary_file) as f:
            saved_summary = json.load(f)
        self.assertEqual(saved_summary['pipeline_name'], 'Test Pipeline')
        
        # Verify log content
        log_content = self.log_file.read_text()
        self.assertIn('PIPELINE SUMMARY', log_content)
        self.assertIn('Test Pipeline', log_content)
    
    def test_header_sanitization(self):
        """Test that sensitive headers are sanitized."""
        import logging
        logger = setup_logging('test_sanitize', log_file=str(self.log_file), level=logging.DEBUG)
        api_logger = create_api_logger(logger)
        
        # Log request with sensitive headers
        api_logger.log_request(
            method='GET',
            url='https://api.example.com/test',
            headers={
                'Authorization': 'Bearer secret_token_12345',
                'X-API-Key': 'api_key_67890',
                'Content-Type': 'application/json'
            }
        )
        
        # Verify log content
        log_content = self.log_file.read_text()
        
        # Sensitive values should be redacted
        self.assertNotIn('secret_token_12345', log_content)
        self.assertNotIn('api_key_67890', log_content)
        
        # Non-sensitive headers should be present (logged at DEBUG level)
        self.assertIn('application/json', log_content)
    
    def test_size_formatting(self):
        """Test human-readable size formatting."""
        logger = setup_logging('test_size', log_file=str(self.log_file))
        download_logger = create_download_logger(logger)
        
        # Test various sizes
        test_cases = [
            (500, 'B'),
            (1500, 'KB'),
            (1500000, 'MB'),
            (1500000000, 'GB'),
        ]
        
        for size_bytes, expected_unit in test_cases:
            formatted = download_logger._format_size(size_bytes)
            self.assertIn(expected_unit, formatted)


class TestLoggingIntegration(unittest.TestCase):
    """Integration tests for logging system."""
    
    def test_complete_pipeline_logging(self):
        """Test complete pipeline with all logging components."""
        temp_dir = tempfile.mkdtemp()
        log_file = Path(temp_dir) / 'integration.log'
        
        # Setup logger
        logger = setup_logging('integration_test', log_file=str(log_file))
        
        # Create all loggers
        api_logger = create_api_logger(logger)
        download_logger = create_download_logger(logger)
        training_logger = create_training_logger(logger)
        error_logger = create_error_logger(logger)
        reporter = create_pipeline_reporter(logger, 'Integration Test Pipeline')
        
        # Simulate pipeline execution
        reporter.log_step_start('api_call', 'Make API request')
        api_logger.log_request('GET', 'https://api.example.com/data')
        api_logger.log_response(200, body='{"data": []}', duration=0.5)
        reporter.log_step_complete('api_call', status='success')
        
        reporter.log_step_start('download', 'Download data')
        download_logger.start_download(total_items=2, description="files")
        download_logger.log_item_progress('file1', success=True, size_bytes=1000)
        download_logger.log_item_progress('file2', success=True, size_bytes=2000)
        summary = download_logger.finish_download()
        reporter.log_step_complete('download', status='success', metrics=summary)
        
        reporter.log_step_start('training', 'Train model')
        training_logger.log_training_start('TestModel', 2, {'lr': 0.001})
        training_logger.log_epoch_metrics(1, 2, {'loss': 0.5}, {'loss': 0.6}, False)
        training_logger.log_epoch_metrics(2, 2, {'loss': 0.3}, {'loss': 0.4}, True)
        train_summary = training_logger.log_training_complete({'accuracy': 0.9})
        reporter.log_step_complete('training', status='success', metrics=train_summary)
        
        # Generate final summary
        summary_file = Path(temp_dir) / 'summary.json'
        pipeline_summary = reporter.generate_summary(output_file=summary_file)
        
        # Verify everything worked
        self.assertTrue(log_file.exists())
        self.assertTrue(summary_file.exists())
        self.assertEqual(pipeline_summary['total_steps'], 3)
        self.assertEqual(pipeline_summary['successful_steps'], 3)
        self.assertEqual(pipeline_summary['overall_status'], 'success')
        
        # Verify log contains all components
        log_content = log_file.read_text()
        self.assertIn('API REQUEST', log_content)
        self.assertIn('DOWNLOAD STARTED', log_content)
        self.assertIn('TRAINING STARTED', log_content)
        self.assertIn('PIPELINE SUMMARY', log_content)


if __name__ == '__main__':
    unittest.main()
