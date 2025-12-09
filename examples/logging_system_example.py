#!/usr/bin/env python3
"""
Example: Comprehensive Logging System Usage

This example demonstrates how to use the comprehensive logging system
for various scenarios in the AgriFlux pipeline.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import (
    setup_logging,
    create_api_logger,
    create_download_logger,
    create_training_logger,
    create_error_logger,
    create_pipeline_reporter
)


def example_api_logging():
    """Example: API request/response logging (Requirement 7.1)"""
    print("\n" + "="*70)
    print("EXAMPLE 1: API Request/Response Logging")
    print("="*70)
    
    # Setup logger
    logger = setup_logging('api_example', log_file='logs/api_example.log')
    api_logger = create_api_logger(logger)
    
    # Log a successful API request
    api_logger.log_request(
        method='POST',
        url='https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search',
        headers={
            'Authorization': 'Bearer secret_token_12345',
            'Content-Type': 'application/json',
            'Accept': 'application/geo+json'
        },
        payload={
            'bbox': [75.8, 30.9, 75.9, 31.0],
            'datetime': '2024-01-01T00:00:00Z/2024-12-31T23:59:59Z',
            'collections': ['sentinel-2-l2a']
        }
    )
    
    time.sleep(0.5)  # Simulate request
    
    api_logger.log_response(
        status_code=200,
        headers={'Content-Type': 'application/geo+json'},
        body='{"type": "FeatureCollection", "features": [...]}',
        duration=0.5
    )
    
    # Log a failed API request
    api_logger.log_request(
        method='GET',
        url='https://services.sentinel-hub.com/api/v1/invalid',
        headers={'Authorization': 'Bearer secret_token_12345'}
    )
    
    api_logger.log_response(
        status_code=404,
        body='{"error": "Not Found"}',
        duration=0.2,
        error=Exception("Resource not found")
    )
    
    print("✓ API logging examples completed. Check logs/api_example.log")


def example_download_logging():
    """Example: Download progress logging (Requirement 7.2)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Download Progress Logging")
    print("="*70)
    
    # Setup logger
    logger = setup_logging('download_example', log_file='logs/download_example.log')
    download_logger = create_download_logger(logger)
    
    # Start download batch
    download_logger.start_download(total_items=5, description="satellite imagery dates")
    
    # Simulate downloading items
    items = [
        ('2024-01-15', True, 15_000_000, 2.5, None),
        ('2024-02-20', True, 14_500_000, 2.3, None),
        ('2024-03-10', False, None, 1.0, "Cloud coverage too high"),
        ('2024-04-05', True, 16_200_000, 2.8, None),
        ('2024-05-12', True, 15_800_000, 2.6, None),
    ]
    
    for item_name, success, size, duration, error in items:
        time.sleep(0.2)  # Simulate processing
        download_logger.log_item_progress(
            item_name=item_name,
            success=success,
            size_bytes=size,
            duration=duration,
            error=error
        )
    
    # Finish and get summary
    summary = download_logger.finish_download()
    
    print(f"✓ Download logging completed. Success rate: {summary['success_rate']:.1%}")
    print(f"  Check logs/download_example.log")


def example_training_logging():
    """Example: Training metrics logging (Requirement 7.3)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Training Metrics Logging")
    print("="*70)
    
    # Setup logger
    logger = setup_logging('training_example', log_file='logs/training_example.log')
    training_logger = create_training_logger(logger)
    
    # Start training
    training_logger.log_training_start(
        model_name='CropHealthCNN',
        total_epochs=10,
        config={
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'early_stopping_patience': 5
        }
    )
    
    # Simulate training epochs
    best_val_acc = 0.0
    for epoch in range(1, 11):
        time.sleep(0.1)  # Simulate training
        
        # Simulate improving metrics
        train_loss = 1.0 - (epoch * 0.08)
        train_acc = 0.5 + (epoch * 0.04)
        val_loss = 1.1 - (epoch * 0.07)
        val_acc = 0.45 + (epoch * 0.045)
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        training_logger.log_epoch_metrics(
            epoch=epoch,
            total_epochs=10,
            train_metrics={'loss': train_loss, 'accuracy': train_acc},
            val_metrics={'loss': val_loss, 'accuracy': val_acc},
            is_best=is_best
        )
    
    # Complete training
    final_metrics = {
        'test_accuracy': 0.87,
        'test_precision': 0.86,
        'test_recall': 0.88,
        'test_f1': 0.87
    }
    
    summary = training_logger.log_training_complete(final_metrics)
    
    print(f"✓ Training logging completed. Best accuracy: {best_val_acc:.4f}")
    print(f"  Check logs/training_example.log")


def example_error_logging():
    """Example: Error logging with stack traces (Requirement 7.4)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Error Logging with Stack Traces")
    print("="*70)
    
    # Setup logger
    logger = setup_logging('error_example', log_file='logs/error_example.log')
    error_logger = create_error_logger(logger)
    
    # Log a non-fatal error
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        error_logger.log_error(
            error=e,
            context={
                'operation': 'calculate_average',
                'input_data': 'empty_list',
                'step': 'data_processing'
            },
            fatal=False
        )
    
    # Log another error
    try:
        # Simulate file not found
        with open('nonexistent_file.txt', 'r') as f:
            data = f.read()
    except FileNotFoundError as e:
        error_logger.log_error(
            error=e,
            context={
                'operation': 'load_configuration',
                'file_path': 'nonexistent_file.txt',
                'step': 'initialization'
            },
            fatal=False
        )
    
    # Get error summary
    summary = error_logger.get_error_summary()
    
    print(f"✓ Error logging completed. Total errors: {summary['total_errors']}")
    print(f"  Error types: {summary['error_types']}")
    print(f"  Check logs/error_example.log")


def example_pipeline_summary():
    """Example: Pipeline summary report (Requirement 7.5)"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Pipeline Summary Report")
    print("="*70)
    
    # Setup logger
    logger = setup_logging('pipeline_example', log_file='logs/pipeline_example.log')
    reporter = create_pipeline_reporter(logger, 'Real Data Download Pipeline')
    
    # Step 1: Authentication
    reporter.log_step_start('authentication', 'Authenticate with Sentinel Hub API')
    time.sleep(0.3)
    reporter.log_step_complete(
        'authentication',
        status='success',
        metrics={'token_expiry': '3600s'}
    )
    
    # Step 2: Query imagery
    reporter.log_step_start('query_imagery', 'Query available imagery dates')
    time.sleep(0.5)
    reporter.log_step_complete(
        'query_imagery',
        status='success',
        metrics={'imagery_found': 20, 'cloud_threshold': '20%'}
    )
    
    # Step 3: Download bands
    reporter.log_step_start('download_bands', 'Download multispectral bands')
    time.sleep(1.0)
    reporter.log_step_complete(
        'download_bands',
        status='success',
        metrics={'bands_downloaded': 80, 'total_size_mb': 450}
    )
    
    # Step 4: Calculate indices
    reporter.log_step_start('calculate_indices', 'Calculate vegetation indices')
    time.sleep(0.4)
    reporter.log_step_complete(
        'calculate_indices',
        status='success',
        metrics={'indices_calculated': ['NDVI', 'SAVI', 'EVI', 'NDWI']}
    )
    
    # Step 5: Save to database (simulate failure)
    reporter.log_step_start('save_database', 'Save processed imagery to database')
    time.sleep(0.2)
    reporter.log_step_complete(
        'save_database',
        status='failed',
        error='Database connection timeout'
    )
    
    # Add pipeline-level metrics
    reporter.add_metric('total_imagery_dates', 20)
    reporter.add_metric('successful_downloads', 18)
    reporter.add_metric('failed_downloads', 2)
    reporter.add_metric('total_data_size_gb', 0.45)
    
    # Generate summary
    summary = reporter.generate_summary(
        output_file=Path('logs/pipeline_summary_example.json')
    )
    
    print(f"✓ Pipeline summary completed. Status: {summary['overall_status']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Check logs/pipeline_example.log and logs/pipeline_summary_example.json")


def main():
    """Run all logging examples."""
    print("\n" + "="*70)
    print("COMPREHENSIVE LOGGING SYSTEM EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates all logging capabilities:")
    print("  1. API request/response logging")
    print("  2. Download progress tracking")
    print("  3. Training metrics logging")
    print("  4. Error logging with stack traces")
    print("  5. Pipeline summary reports")
    print("\nAll logs will be saved to the logs/ directory.")
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    # Run examples
    example_api_logging()
    example_download_logging()
    example_training_logging()
    example_error_logging()
    example_pipeline_summary()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nCheck the following log files:")
    print("  - logs/api_example.log")
    print("  - logs/download_example.log")
    print("  - logs/training_example.log")
    print("  - logs/error_example.log")
    print("  - logs/pipeline_example.log")
    print("  - logs/pipeline_summary_example.json")
    print("\n")


if __name__ == '__main__':
    main()
