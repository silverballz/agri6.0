"""
Batch processing system for large-scale satellite imagery and sensor data.

This module provides scalable batch processing capabilities with progress tracking,
memory optimization, and parallel processing for handling large datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Iterator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
import gc
import psutil
from pathlib import Path
import json
import queue
from abc import ABC, abstractmethod

from .sentinel2_parser import Sentinel2SafeParser
from .vegetation_indices import VegetationIndexCalculator
from .cloud_masking import CloudMaskProcessor
from src.models.satellite_image import SatelliteImage
from src.database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10                    # Number of items per batch
    max_workers: int = None                 # Max parallel workers (None = auto)
    memory_limit_gb: float = 8.0           # Memory limit in GB
    chunk_size_mb: int = 100               # Chunk size for raster processing
    enable_progress_tracking: bool = True   # Enable progress tracking
    save_intermediate: bool = True          # Save intermediate results
    intermediate_dir: str = "temp/batch"    # Directory for intermediate files
    max_retries: int = 3                   # Max retries for failed items
    timeout_seconds: int = 3600            # Timeout per batch item


@dataclass
class BatchProgress:
    """Progress tracking for batch operations."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    current_batch: int = 0
    total_batches: int = 0
    estimated_completion: Optional[datetime] = None
    current_memory_usage_gb: float = 0.0
    peak_memory_usage_gb: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time."""
        return datetime.now() - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_time.total_seconds()
        if elapsed == 0:
            return 0.0
        return self.completed_items / elapsed
    
    def estimate_completion_time(self):
        """Estimate completion time based on current progress."""
        if self.completed_items == 0:
            return
        
        rate = self.items_per_second
        if rate > 0:
            remaining_items = self.total_items - self.completed_items
            remaining_seconds = remaining_items / rate
            self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)


class ProgressTracker:
    """Thread-safe progress tracker for batch operations."""
    
    def __init__(self, total_items: int):
        self.progress = BatchProgress(total_items=total_items)
        self.lock = threading.Lock()
        self.callbacks = []
        
    def update(self, 
               completed: int = 0, 
               failed: int = 0, 
               current_batch: int = None,
               errors: List[str] = None):
        """Update progress in a thread-safe manner."""
        with self.lock:
            self.progress.completed_items += completed
            self.progress.failed_items += failed
            
            if current_batch is not None:
                self.progress.current_batch = current_batch
            
            if errors:
                self.progress.errors.extend(errors)
            
            # Update memory usage
            self.progress.current_memory_usage_gb = psutil.virtual_memory().used / (1024**3)
            self.progress.peak_memory_usage_gb = max(
                self.progress.peak_memory_usage_gb,
                self.progress.current_memory_usage_gb
            )
            
            # Estimate completion time
            self.progress.estimate_completion_time()
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(self.progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")
    
    def add_callback(self, callback: Callable[[BatchProgress], None]):
        """Add a progress callback function."""
        self.callbacks.append(callback)
    
    def get_progress(self) -> BatchProgress:
        """Get current progress (thread-safe copy)."""
        with self.lock:
            # Create a copy to avoid race conditions
            return BatchProgress(
                total_items=self.progress.total_items,
                completed_items=self.progress.completed_items,
                failed_items=self.progress.failed_items,
                start_time=self.progress.start_time,
                current_batch=self.progress.current_batch,
                total_batches=self.progress.total_batches,
                estimated_completion=self.progress.estimated_completion,
                current_memory_usage_gb=self.progress.current_memory_usage_gb,
                peak_memory_usage_gb=self.progress.peak_memory_usage_gb,
                errors=self.progress.errors.copy()
            )


class MemoryOptimizer:
    """Memory optimization utilities for large-scale processing."""
    
    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * (1024**3)
        
    def check_memory_usage(self) -> Tuple[float, bool]:
        """Check current memory usage."""
        memory_info = psutil.virtual_memory()
        current_usage_gb = memory_info.used / (1024**3)
        is_over_limit = current_usage_gb > self.memory_limit_gb
        
        return current_usage_gb, is_over_limit
    
    def optimize_memory(self):
        """Perform memory optimization."""
        # Force garbage collection
        gc.collect()
        
        # Log memory usage
        usage_gb, over_limit = self.check_memory_usage()
        logger.info(f"Memory usage: {usage_gb:.2f} GB (limit: {self.memory_limit_gb} GB)")
        
        if over_limit:
            logger.warning("Memory usage exceeds limit, forcing aggressive cleanup")
            # Additional cleanup could be implemented here
    
    def calculate_optimal_chunk_size(self, 
                                   array_shape: Tuple[int, ...], 
                                   dtype: np.dtype,
                                   target_memory_mb: int = 100) -> Tuple[int, int]:
        """Calculate optimal chunk size for raster processing."""
        # Calculate bytes per pixel
        bytes_per_pixel = np.dtype(dtype).itemsize * array_shape[-1]  # Assume last dim is channels
        
        # Calculate target pixels per chunk
        target_bytes = target_memory_mb * (1024**2)
        pixels_per_chunk = target_bytes // bytes_per_pixel
        
        # Calculate chunk dimensions (assuming square chunks)
        chunk_side = int(np.sqrt(pixels_per_chunk))
        
        # Ensure chunk size doesn't exceed image dimensions
        max_height, max_width = array_shape[0], array_shape[1]
        chunk_height = min(chunk_side, max_height)
        chunk_width = min(chunk_side, max_width)
        
        return chunk_height, chunk_width
    
    def process_in_chunks(self,
                         array: np.ndarray,
                         processing_func: Callable,
                         chunk_size: Tuple[int, int] = None,
                         overlap: int = 0) -> np.ndarray:
        """Process large array in memory-efficient chunks."""
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(array.shape, array.dtype)
        
        chunk_height, chunk_width = chunk_size
        height, width = array.shape[:2]
        
        # Calculate output shape (assuming processing doesn't change spatial dimensions)
        output_shape = array.shape
        result = np.zeros(output_shape, dtype=array.dtype)
        
        # Process chunks
        for y in range(0, height, chunk_height - overlap):
            for x in range(0, width, chunk_width - overlap):
                # Calculate chunk boundaries
                y_end = min(y + chunk_height, height)
                x_end = min(x + chunk_width, width)
                
                # Extract chunk
                chunk = array[y:y_end, x:x_end]
                
                # Process chunk
                processed_chunk = processing_func(chunk)
                
                # Handle overlap blending (simple average for now)
                if overlap > 0:
                    # This is a simplified overlap handling
                    # In practice, you might want more sophisticated blending
                    result[y:y_end, x:x_end] = processed_chunk
                else:
                    result[y:y_end, x:x_end] = processed_chunk
                
                # Memory cleanup
                del chunk, processed_chunk
                
                # Check memory usage periodically
                if (y + x) % (chunk_height * chunk_width * 10) == 0:
                    self.optimize_memory()
        
        return result


class BatchProcessor(ABC):
    """Abstract base class for batch processors."""
    
    @abstractmethod
    def process_item(self, item: Any) -> Any:
        """Process a single item."""
        pass
    
    @abstractmethod
    def get_item_id(self, item: Any) -> str:
        """Get unique identifier for an item."""
        pass


class SatelliteImageBatchProcessor(BatchProcessor):
    """Batch processor for satellite imagery."""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.index_calculator = VegetationIndexCalculator()
        self.cloud_processor = CloudMaskProcessor()
        self.memory_optimizer = MemoryOptimizer(self.config.memory_limit_gb)
        
    def process_item(self, safe_path: str) -> Optional[SatelliteImage]:
        """Process a single Sentinel-2A SAFE directory."""
        try:
            logger.info(f"Processing SAFE directory: {safe_path}")
            
            # Create parser for this SAFE directory
            parser = Sentinel2SafeParser(Path(safe_path))
            
            # Parse SAFE directory
            safe_data = parser.parse_safe_directory()
            
            # Load bands with memory optimization
            bands = {}
            for band_name in ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']:
                if band_name in safe_data.band_files:
                    band_path = safe_data.band_files[band_name]
                    
                    # Load band in chunks if it's large
                    band_data = parser.read_band_file(band_path)
                    
                    # Optimize memory usage
                    if band_data.nbytes > self.config.chunk_size_mb * (1024**2):
                        logger.info(f"Processing {band_name} in chunks due to size")
                        # Process in chunks (placeholder - would implement actual chunked processing)
                        pass
                    
                    bands[band_name] = band_data
                    
                    # Memory cleanup
                    self.memory_optimizer.optimize_memory()
            
            # Calculate vegetation indices
            indices = {}
            if all(band in bands for band in ['B04', 'B08']):
                indices['ndvi'] = self.index_calculator.calculate_ndvi(
                    bands['B08'], bands['B04']
                )
            
            if all(band in bands for band in ['B02', 'B03', 'B04', 'B08']):
                indices['evi'] = self.index_calculator.calculate_evi(
                    bands['B08'], bands['B04'], bands['B02']
                )
            
            # Apply cloud masking if SCL is available
            if 'SCL' in safe_data.band_files:
                scl_data = parser.read_band_file(safe_data.band_files['SCL'])
                cloud_mask = self.cloud_processor.create_cloud_mask(scl_data)
                
                # Apply mask to indices
                for index_name, index_data in indices.items():
                    indices[index_name] = np.where(cloud_mask, np.nan, index_data)
            
            # Create SatelliteImage object
            satellite_image = SatelliteImage(
                id=safe_data.product_id,
                acquisition_date=safe_data.acquisition_date,
                tile_id=safe_data.tile_id,
                cloud_coverage=safe_data.cloud_coverage,
                bands=bands,
                indices=indices,
                geometry=safe_data.geometry,
                quality_flags={'processed': True}
            )
            
            logger.info(f"Successfully processed: {safe_data.product_id}")
            return satellite_image
            
        except Exception as e:
            logger.error(f"Error processing {safe_path}: {str(e)}")
            return None
    
    def get_item_id(self, safe_path: str) -> str:
        """Get unique identifier for SAFE directory."""
        return Path(safe_path).name


class BatchExecutor:
    """Main batch execution engine with parallel processing and progress tracking."""
    
    def __init__(self, 
                 processor: BatchProcessor,
                 config: BatchConfig = None):
        self.processor = processor
        self.config = config or BatchConfig()
        self.progress_tracker = None
        
        # Set default max_workers if not specified
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 4)
        
        # Create intermediate directory
        Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
    
    def execute_batch(self, 
                     items: List[Any],
                     use_multiprocessing: bool = True) -> Tuple[List[Any], BatchProgress]:
        """Execute batch processing on a list of items."""
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(len(items))
        
        # Calculate batches
        batches = self._create_batches(items)
        self.progress_tracker.progress.total_batches = len(batches)
        
        logger.info(f"Starting batch processing: {len(items)} items in {len(batches)} batches")
        
        results = []
        
        if use_multiprocessing and self.config.max_workers > 1:
            results = self._execute_parallel(batches)
        else:
            results = self._execute_sequential(batches)
        
        final_progress = self.progress_tracker.get_progress()
        logger.info(f"Batch processing completed: {final_progress.completed_items} successful, "
                   f"{final_progress.failed_items} failed")
        
        return results, final_progress
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Split items into batches."""
        batches = []
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    def _execute_sequential(self, batches: List[List[Any]]) -> List[Any]:
        """Execute batches sequentially."""
        results = []
        
        for batch_idx, batch in enumerate(batches):
            batch_results = self._process_batch(batch, batch_idx)
            results.extend(batch_results)
            
            # Update progress
            completed = len([r for r in batch_results if r is not None])
            failed = len([r for r in batch_results if r is None])
            self.progress_tracker.update(
                completed=completed,
                failed=failed,
                current_batch=batch_idx + 1
            )
        
        return results
    
    def _execute_parallel(self, batches: List[List[Any]]) -> List[Any]:
        """Execute batches in parallel using ProcessPoolExecutor."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch_wrapper, batch, batch_idx): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_results = future.result(timeout=self.config.timeout_seconds)
                    results.extend(batch_results)
                    
                    # Update progress
                    completed = len([r for r in batch_results if r is not None])
                    failed = len([r for r in batch_results if r is None])
                    self.progress_tracker.update(
                        completed=completed,
                        failed=failed,
                        current_batch=batch_idx + 1
                    )
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    batch_size = len(batches[batch_idx])
                    self.progress_tracker.update(
                        failed=batch_size,
                        errors=[f"Batch {batch_idx}: {str(e)}"]
                    )
        
        return results
    
    def _process_batch_wrapper(self, batch: List[Any], batch_idx: int) -> List[Any]:
        """Wrapper for batch processing (needed for multiprocessing)."""
        return self._process_batch(batch, batch_idx)
    
    def _process_batch(self, batch: List[Any], batch_idx: int) -> List[Any]:
        """Process a single batch of items."""
        results = []
        
        for item in batch:
            try:
                # Process item with retry logic
                result = self._process_item_with_retry(item)
                results.append(result)
                
                # Save intermediate result if enabled
                if self.config.save_intermediate and result is not None:
                    self._save_intermediate_result(item, result, batch_idx)
                
            except Exception as e:
                logger.error(f"Failed to process item {self.processor.get_item_id(item)}: {str(e)}")
                results.append(None)
        
        return results
    
    def _process_item_with_retry(self, item: Any) -> Any:
        """Process item with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self.processor.process_item(item)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for {self.processor.get_item_id(item)}, "
                                 f"retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
        
        # All retries failed
        raise last_exception
    
    def _save_intermediate_result(self, item: Any, result: Any, batch_idx: int):
        """Save intermediate result to disk."""
        try:
            item_id = self.processor.get_item_id(item)
            filename = f"batch_{batch_idx}_{item_id}.json"
            filepath = Path(self.config.intermediate_dir) / filename
            
            # Convert result to serializable format (this would need to be customized)
            # For now, just save basic info
            result_data = {
                'item_id': item_id,
                'batch_idx': batch_idx,
                'processed_at': datetime.now().isoformat(),
                'success': result is not None
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save intermediate result: {str(e)}")
    
    def add_progress_callback(self, callback: Callable[[BatchProgress], None]):
        """Add a progress callback function."""
        if self.progress_tracker:
            self.progress_tracker.add_callback(callback)


def create_satellite_batch_processor(config: BatchConfig = None) -> BatchExecutor:
    """Create a batch processor for satellite imagery."""
    processor = SatelliteImageBatchProcessor(config)
    executor = BatchExecutor(processor, config)
    return executor


# Example progress callback functions
def log_progress_callback(progress: BatchProgress):
    """Log progress to console."""
    logger.info(f"Progress: {progress.completion_percentage:.1f}% "
               f"({progress.completed_items}/{progress.total_items}) "
               f"Rate: {progress.items_per_second:.2f} items/sec "
               f"Memory: {progress.current_memory_usage_gb:.2f} GB")


def save_progress_callback(progress: BatchProgress, filepath: str = "batch_progress.json"):
    """Save progress to JSON file."""
    try:
        progress_data = {
            'completion_percentage': progress.completion_percentage,
            'completed_items': progress.completed_items,
            'total_items': progress.total_items,
            'failed_items': progress.failed_items,
            'elapsed_time_seconds': progress.elapsed_time.total_seconds(),
            'items_per_second': progress.items_per_second,
            'memory_usage_gb': progress.current_memory_usage_gb,
            'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save progress: {str(e)}")


if __name__ == "__main__":
    # Example usage
    config = BatchConfig(
        batch_size=5,
        max_workers=2,
        memory_limit_gb=4.0,
        enable_progress_tracking=True
    )
    
    # Create batch processor
    executor = create_satellite_batch_processor(config)
    
    # Add progress callbacks
    executor.add_progress_callback(log_progress_callback)
    executor.add_progress_callback(lambda p: save_progress_callback(p, "progress.json"))
    
    # Example items (SAFE directory paths)
    items = [
        "path/to/safe1.SAFE",
        "path/to/safe2.SAFE",
        "path/to/safe3.SAFE"
    ]
    
    # Execute batch processing
    results, final_progress = executor.execute_batch(items)
    
    print(f"Processing completed: {len(results)} results")
    print(f"Success rate: {final_progress.completion_percentage:.1f}%")