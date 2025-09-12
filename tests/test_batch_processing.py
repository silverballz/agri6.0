"""
Tests for batch processing functionality.
Tests batch processing with progress tracking and memory optimization.
"""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.batch_processor import (
    BatchConfig, BatchProgress, ProgressTracker, MemoryOptimizer,
    BatchProcessor, SatelliteImageBatchProcessor, BatchExecutor,
    create_satellite_batch_processor
)


class TestBatchConfig:
    """Test batch configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()
        
        assert config.batch_size == 10
        assert config.max_workers is None
        assert config.memory_limit_gb == 8.0
        assert config.chunk_size_mb == 100
        assert config.enable_progress_tracking is True
        assert config.save_intermediate is True
        assert config.max_retries == 3
        assert config.timeout_seconds == 3600
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchConfig(
            batch_size=5,
            max_workers=2,
            memory_limit_gb=4.0,
            chunk_size_mb=50,
            enable_progress_tracking=False,
            save_intermediate=False,
            max_retries=1,
            timeout_seconds=1800
        )
        
        assert config.batch_size == 5
        assert config.max_workers == 2
        assert config.memory_limit_gb == 4.0
        assert config.chunk_size_mb == 50
        assert config.enable_progress_tracking is False
        assert config.save_intermediate is False
        assert config.max_retries == 1
        assert config.timeout_seconds == 1800


class TestBatchProgress:
    """Test batch progress tracking."""
    
    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = BatchProgress(total_items=100)
        
        assert progress.total_items == 100
        assert progress.completed_items == 0
        assert progress.failed_items == 0
        assert progress.current_batch == 0
        assert progress.total_batches == 0
        assert isinstance(progress.start_time, datetime)
        assert progress.estimated_completion is None
        assert len(progress.errors) == 0
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = BatchProgress(total_items=100)
        
        # Initially 0%
        assert progress.completion_percentage == 0.0
        
        # 50% complete
        progress.completed_items = 50
        assert progress.completion_percentage == 50.0
        
        # 100% complete
        progress.completed_items = 100
        assert progress.completion_percentage == 100.0
        
        # Handle edge case of 0 total items
        progress_empty = BatchProgress(total_items=0)
        assert progress_empty.completion_percentage == 0.0
    
    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start_time = datetime.now() - timedelta(seconds=30)
        progress = BatchProgress(total_items=100, start_time=start_time)
        
        elapsed = progress.elapsed_time
        assert isinstance(elapsed, timedelta)
        assert 25 <= elapsed.total_seconds() <= 35  # Allow some variance
    
    def test_items_per_second(self):
        """Test processing rate calculation."""
        start_time = datetime.now() - timedelta(seconds=10)
        progress = BatchProgress(
            total_items=100,
            completed_items=50,
            start_time=start_time
        )
        
        rate = progress.items_per_second
        assert 4.0 <= rate <= 6.0  # Should be around 5 items/sec
        
        # Test edge case with no elapsed time
        progress_new = BatchProgress(total_items=100)
        assert progress_new.items_per_second == 0.0
    
    def test_estimate_completion_time(self):
        """Test completion time estimation."""
        start_time = datetime.now() - timedelta(seconds=10)
        progress = BatchProgress(
            total_items=100,
            completed_items=25,
            start_time=start_time
        )
        
        progress.estimate_completion_time()
        
        assert progress.estimated_completion is not None
        assert progress.estimated_completion > datetime.now()
        
        # Should estimate about 30 more seconds (75 items at 2.5 items/sec)
        estimated_remaining = progress.estimated_completion - datetime.now()
        assert 20 <= estimated_remaining.total_seconds() <= 40


class TestProgressTracker:
    """Test progress tracker."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ProgressTracker(100)
        
        assert tracker.progress.total_items == 100
        assert len(tracker.callbacks) == 0
    
    def test_progress_update(self):
        """Test progress updates."""
        tracker = ProgressTracker(100)
        
        # Update progress
        tracker.update(completed=10, failed=2, current_batch=1)
        
        progress = tracker.get_progress()
        assert progress.completed_items == 10
        assert progress.failed_items == 2
        assert progress.current_batch == 1
    
    def test_progress_callbacks(self):
        """Test progress callbacks."""
        tracker = ProgressTracker(100)
        
        # Add mock callback
        callback_mock = Mock()
        tracker.add_callback(callback_mock)
        
        # Update progress
        tracker.update(completed=10)
        
        # Callback should be called
        callback_mock.assert_called_once()
        
        # Check callback argument
        call_args = callback_mock.call_args[0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], BatchProgress)
        assert call_args[0].completed_items == 10
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        
        tracker = ProgressTracker(1000)
        
        def update_progress():
            for _ in range(10):
                tracker.update(completed=1)
                time.sleep(0.001)  # Small delay
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_progress)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 50 completed items (5 threads * 10 updates each)
        progress = tracker.get_progress()
        assert progress.completed_items == 50
    
    def test_error_tracking(self):
        """Test error tracking."""
        tracker = ProgressTracker(100)
        
        errors = ["Error 1", "Error 2"]
        tracker.update(failed=2, errors=errors)
        
        progress = tracker.get_progress()
        assert progress.failed_items == 2
        assert len(progress.errors) == 2
        assert "Error 1" in progress.errors
        assert "Error 2" in progress.errors


class TestMemoryOptimizer:
    """Test memory optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = MemoryOptimizer(memory_limit_gb=4.0)
        
        assert optimizer.memory_limit_gb == 4.0
        assert optimizer.memory_limit_bytes == 4.0 * (1024**3)
    
    def test_memory_usage_check(self):
        """Test memory usage checking."""
        optimizer = MemoryOptimizer(memory_limit_gb=1.0)  # Very low limit
        
        usage_gb, is_over_limit = optimizer.check_memory_usage()
        
        assert isinstance(usage_gb, float)
        assert usage_gb > 0
        assert isinstance(is_over_limit, bool)
        # With 1GB limit, we're likely over the limit
        assert is_over_limit is True
    
    def test_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        optimizer = MemoryOptimizer()
        
        # Test with different array shapes
        test_cases = [
            ((1000, 1000, 3), np.float32, 100),  # RGB image
            ((5000, 5000, 1), np.uint16, 50),    # Large single band
            ((100, 100, 12), np.float32, 200),   # Multi-spectral
        ]
        
        for shape, dtype, target_mb in test_cases:
            chunk_height, chunk_width = optimizer.calculate_optimal_chunk_size(
                shape, dtype, target_mb
            )
            
            assert chunk_height > 0
            assert chunk_width > 0
            assert chunk_height <= shape[0]
            assert chunk_width <= shape[1]
            
            # Verify chunk size is reasonable
            bytes_per_pixel = np.dtype(dtype).itemsize * shape[-1]
            chunk_bytes = chunk_height * chunk_width * bytes_per_pixel
            chunk_mb = chunk_bytes / (1024**2)
            
            # Should be close to target (within 50% tolerance)
            assert chunk_mb <= target_mb * 1.5
    
    def test_chunked_processing(self):
        """Test chunked array processing."""
        optimizer = MemoryOptimizer()
        
        # Create test array
        test_array = np.random.rand(100, 100, 3).astype(np.float32)
        
        # Simple processing function (add 1 to all values)
        def add_one(chunk):
            return chunk + 1.0
        
        # Process in chunks
        result = optimizer.process_in_chunks(
            test_array, 
            add_one, 
            chunk_size=(50, 50)
        )
        
        assert result.shape == test_array.shape
        assert result.dtype == test_array.dtype
        
        # Verify processing was applied
        np.testing.assert_allclose(result, test_array + 1.0, rtol=1e-6)
    
    def test_memory_optimization(self):
        """Test memory optimization function."""
        optimizer = MemoryOptimizer()
        
        # Should run without errors
        optimizer.optimize_memory()
        
        # Memory usage should be reported
        usage_gb, _ = optimizer.check_memory_usage()
        assert usage_gb > 0


class MockBatchProcessor(BatchProcessor):
    """Mock batch processor for testing."""
    
    def __init__(self, processing_time=0.1, failure_rate=0.0):
        self.processing_time = processing_time
        self.failure_rate = failure_rate
        self.processed_items = []
    
    def process_item(self, item):
        """Mock processing that may fail based on failure rate."""
        time.sleep(self.processing_time)
        
        # Simulate random failures
        import random
        if random.random() < self.failure_rate:
            raise ValueError(f"Mock processing failed for item: {item}")
        
        # Record processed item
        result = f"processed_{item}"
        self.processed_items.append(result)
        return result
    
    def get_item_id(self, item):
        """Get item identifier."""
        return str(item)


class TestBatchExecutor:
    """Test batch executor."""
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        processor = MockBatchProcessor()
        config = BatchConfig(batch_size=5, max_workers=2)
        
        executor = BatchExecutor(processor, config)
        
        assert executor.processor == processor
        assert executor.config.batch_size == 5
        assert executor.config.max_workers == 2
    
    def test_sequential_execution(self):
        """Test sequential batch execution."""
        processor = MockBatchProcessor(processing_time=0.01)
        config = BatchConfig(batch_size=3, max_workers=1)
        executor = BatchExecutor(processor, config)
        
        items = [1, 2, 3, 4, 5]
        results, progress = executor.execute_batch(items, use_multiprocessing=False)
        
        assert len(results) == 5
        assert progress.completed_items == 5
        assert progress.failed_items == 0
        assert progress.completion_percentage == 100.0
        
        # Check results
        expected_results = ["processed_1", "processed_2", "processed_3", "processed_4", "processed_5"]
        assert results == expected_results
    
    def test_parallel_execution(self):
        """Test parallel batch execution."""
        processor = MockBatchProcessor(processing_time=0.01)
        config = BatchConfig(batch_size=2, max_workers=2)
        executor = BatchExecutor(processor, config)
        
        items = [1, 2, 3, 4]
        results, progress = executor.execute_batch(items, use_multiprocessing=True)
        
        assert len(results) == 4
        assert progress.completed_items == 4
        assert progress.failed_items == 0
        
        # Results should contain all processed items (order may vary due to parallel processing)
        result_set = set(results)
        expected_set = {"processed_1", "processed_2", "processed_3", "processed_4"}
        assert result_set == expected_set
    
    def test_batch_creation(self):
        """Test batch creation logic."""
        processor = MockBatchProcessor()
        config = BatchConfig(batch_size=3)
        executor = BatchExecutor(processor, config)
        
        items = [1, 2, 3, 4, 5, 6, 7]
        batches = executor._create_batches(items)
        
        assert len(batches) == 3  # 3 batches: [1,2,3], [4,5,6], [7]
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
        assert batches[2] == [7]
    
    def test_failure_handling(self):
        """Test handling of processing failures."""
        processor = MockBatchProcessor(processing_time=0.01, failure_rate=0.5)
        config = BatchConfig(batch_size=2, max_retries=1)
        executor = BatchExecutor(processor, config)
        
        items = [1, 2, 3, 4]
        results, progress = executor.execute_batch(items, use_multiprocessing=False)
        
        assert len(results) == 4
        assert progress.total_items == 4
        
        # Some items should have failed
        failed_count = len([r for r in results if r is None])
        assert failed_count > 0
        assert progress.failed_items == failed_count
        assert progress.completed_items + progress.failed_items == 4
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        processor = MockBatchProcessor(processing_time=0.01)
        config = BatchConfig(batch_size=2)
        executor = BatchExecutor(processor, config)
        
        # Add progress callback
        callback_calls = []
        def progress_callback(progress):
            callback_calls.append(progress.completed_items)
        
        executor.add_progress_callback(progress_callback)
        
        items = [1, 2, 3, 4]
        results, final_progress = executor.execute_batch(items, use_multiprocessing=False)
        
        # Callback should have been called multiple times
        assert len(callback_calls) > 0
        assert max(callback_calls) == 4  # Final callback should show 4 completed
    
    def test_timeout_handling(self):
        """Test timeout handling for long-running processes."""
        processor = MockBatchProcessor(processing_time=2.0)  # Long processing time
        config = BatchConfig(batch_size=1, timeout_seconds=1, max_workers=1)
        executor = BatchExecutor(processor, config)
        
        items = [1]
        
        # This should timeout in parallel mode
        start_time = time.time()
        results, progress = executor.execute_batch(items, use_multiprocessing=True)
        elapsed_time = time.time() - start_time
        
        # Should complete quickly due to timeout (not wait for full processing time)
        assert elapsed_time < 5.0  # Much less than the 2 second processing time
        
        # Item should have failed due to timeout
        assert progress.failed_items > 0


class TestSatelliteImageBatchProcessor:
    """Test satellite image batch processor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        config = BatchConfig(memory_limit_gb=2.0)
        processor = SatelliteImageBatchProcessor(config)
        
        assert processor.config.memory_limit_gb == 2.0
        assert processor.memory_optimizer.memory_limit_gb == 2.0
    
    def test_get_item_id(self):
        """Test item ID extraction."""
        processor = SatelliteImageBatchProcessor()
        
        safe_path = "/path/to/S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE"
        item_id = processor.get_item_id(safe_path)
        
        expected_id = "S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE"
        assert item_id == expected_id
    
    @patch('src.data_processing.batch_processor.Sentinel2Parser')
    def test_process_item_mock(self, mock_parser_class):
        """Test item processing with mocked dependencies."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_safe_data = Mock()
        mock_safe_data.product_id = "test_product"
        mock_safe_data.acquisition_date = datetime.now()
        mock_safe_data.tile_id = "43REQ"
        mock_safe_data.cloud_coverage = 10.0
        mock_safe_data.geometry = Mock()
        mock_safe_data.band_files = {
            'B04': 'path/to/B04.jp2',
            'B08': 'path/to/B08.jp2'
        }
        
        mock_parser.parse_safe_directory.return_value = mock_safe_data
        mock_parser.read_band_file.return_value = np.random.rand(100, 100).astype(np.float32)
        
        processor = SatelliteImageBatchProcessor()
        
        # Process mock item
        result = processor.process_item("mock_safe_path")
        
        # Should return SatelliteImage object
        assert result is not None
        assert hasattr(result, 'id')
        assert hasattr(result, 'bands')
        assert hasattr(result, 'indices')


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_satellite_batch_processor(self):
        """Test satellite batch processor creation."""
        config = BatchConfig(batch_size=5, memory_limit_gb=4.0)
        executor = create_satellite_batch_processor(config)
        
        assert isinstance(executor, BatchExecutor)
        assert isinstance(executor.processor, SatelliteImageBatchProcessor)
        assert executor.config.batch_size == 5
        assert executor.config.memory_limit_gb == 4.0
    
    def test_create_satellite_batch_processor_default_config(self):
        """Test satellite batch processor creation with default config."""
        executor = create_satellite_batch_processor()
        
        assert isinstance(executor, BatchExecutor)
        assert isinstance(executor.processor, SatelliteImageBatchProcessor)
        assert executor.config.batch_size == 10  # Default value


class TestBatchProcessingIntegration:
    """Integration tests for batch processing."""
    
    def test_end_to_end_batch_processing(self):
        """Test complete batch processing workflow."""
        # Create mock processor with realistic behavior
        processor = MockBatchProcessor(processing_time=0.05, failure_rate=0.1)
        
        config = BatchConfig(
            batch_size=3,
            max_workers=2,
            enable_progress_tracking=True,
            save_intermediate=False,  # Disable for testing
            max_retries=2
        )
        
        executor = BatchExecutor(processor, config)
        
        # Track progress
        progress_updates = []
        def track_progress(progress):
            progress_updates.append({
                'completed': progress.completed_items,
                'failed': progress.failed_items,
                'percentage': progress.completion_percentage
            })
        
        executor.add_progress_callback(track_progress)
        
        # Process items
        items = list(range(10))  # 10 items
        results, final_progress = executor.execute_batch(items)
        
        # Verify results
        assert len(results) == 10
        assert final_progress.total_items == 10
        assert final_progress.completed_items + final_progress.failed_items == 10
        assert final_progress.completion_percentage == 100.0
        
        # Verify progress tracking
        assert len(progress_updates) > 0
        assert progress_updates[-1]['percentage'] == 100.0
        
        # Verify some items were processed successfully
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) > 0
    
    def test_memory_constrained_processing(self):
        """Test processing under memory constraints."""
        # Create processor with very low memory limit
        config = BatchConfig(
            batch_size=2,
            memory_limit_gb=0.1,  # Very low limit
            chunk_size_mb=10
        )
        
        processor = MockBatchProcessor(processing_time=0.01)
        executor = BatchExecutor(processor, config)
        
        items = [1, 2, 3, 4]
        results, progress = executor.execute_batch(items, use_multiprocessing=False)
        
        # Should still complete successfully despite memory constraints
        assert len(results) == 4
        assert progress.completed_items == 4
        assert progress.failed_items == 0
    
    def test_large_batch_processing(self):
        """Test processing of large batches."""
        processor = MockBatchProcessor(processing_time=0.001)  # Very fast processing
        
        config = BatchConfig(
            batch_size=10,
            max_workers=2,
            enable_progress_tracking=True
        )
        
        executor = BatchExecutor(processor, config)
        
        # Process large number of items
        items = list(range(100))
        
        start_time = time.time()
        results, progress = executor.execute_batch(items)
        elapsed_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 100
        assert progress.completed_items == 100
        assert progress.failed_items == 0
        
        # Should complete in reasonable time (parallel processing should help)
        assert elapsed_time < 5.0  # Should be much faster than 100 * 0.001 = 0.1s sequential
        
        # Verify processing rate
        assert progress.items_per_second > 0
        print(f"Processing rate: {progress.items_per_second:.2f} items/sec")