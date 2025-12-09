"""
Property-based tests for export file size accuracy.
Tests universal properties that should hold across all valid inputs.

Feature: production-enhancements
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from typing import List, Dict, Any


# Import the export functionality
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dashboard.data_exporter import DataExporter, generate_mock_vegetation_indices_data


# Strategies for generating test data
zone_count_strategy = st.integers(min_value=1, max_value=5)
days_count_strategy = st.integers(min_value=5, max_value=30)
record_count_strategy = st.integers(min_value=10, max_value=200)


def generate_time_series_data(num_records: int, num_zones: int, seed: int) -> List[Dict[str, Any]]:
    """Generate synthetic time series data for testing."""
    np.random.seed(seed)
    
    data = []
    zone_ids = [f"zone_{i}" for i in range(num_zones)]
    index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']
    base_date = datetime.now()
    
    for i in range(num_records):
        # Generate timestamp
        days_back = np.random.randint(0, 90)
        timestamp = base_date - timedelta(days=days_back, hours=np.random.randint(0, 24))
        
        # Select random zone and index type
        zone_id = np.random.choice(zone_ids)
        index_type = np.random.choice(index_types)
        
        # Generate realistic index value
        mean_value = np.random.uniform(-1.0, 1.0)
        
        record = {
            'zone_id': zone_id,
            'index_type': index_type,
            'timestamp': timestamp,
            'mean_value': round(mean_value, 4),
            'std_deviation': round(np.random.uniform(0.01, 0.1), 4),
            'pixel_count': np.random.randint(500, 2000),
            'quality_score': round(np.random.uniform(0.7, 1.0), 3)
        }
        
        data.append(record)
    
    return data


class TestExportFileSizeAccuracyProperties:
    """Property-based tests for export file size accuracy.
    
    **Feature: production-enhancements, Property 21: Export file size accuracy**
    **Validates: Requirements 5.5**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_reported_size_matches_actual_size(self, num_records, num_zones, seed):
        """
        Property 21: Export file size accuracy (CSV)
        
        For any completed CSV export, reported file size should match 
        actual file size on disk within 1 byte.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get actual file size from disk
            actual_size = os.path.getsize(csv_path)
            
            # Get reported file size from export history
            history = exporter.get_export_history(limit=1)
            
            assert len(history) > 0, "Export history is empty"
            
            reported_size = history[0]['size_bytes']
            
            # Verify sizes match within 1 byte
            size_difference = abs(actual_size - reported_size)
            
            assert size_difference <= 1, \
                f"File size mismatch: actual={actual_size} bytes, " \
                f"reported={reported_size} bytes, difference={size_difference} bytes"
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_size_consistency(self, num_records, num_zones, seed):
        """
        Property: Multiple reads of the same file should report the same size.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read file size multiple times
            sizes = []
            for _ in range(5):
                size = os.path.getsize(csv_path)
                sizes.append(size)
            
            # All sizes should be identical
            assert len(set(sizes)) == 1, \
                f"File size inconsistent across reads: {sizes}"
            
            # Verify reported size matches
            history = exporter.get_export_history(limit=1)
            reported_size = history[0]['size_bytes']
            
            assert reported_size == sizes[0], \
                f"Reported size {reported_size} doesn't match actual size {sizes[0]}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_size_non_zero(self, num_records, num_zones, seed):
        """
        Property: Exported files with data should have non-zero size.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Ensure we have data
        assume(len(time_series_data) > 0)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get file size
            actual_size = os.path.getsize(csv_path)
            
            # Verify size is non-zero
            assert actual_size > 0, \
                f"Exported file has zero size despite containing {len(time_series_data)} records"
            
            # Get reported size
            history = exporter.get_export_history(limit=1)
            reported_size = history[0]['size_bytes']
            
            # Verify reported size is also non-zero
            assert reported_size > 0, \
                f"Reported file size is zero despite actual size being {actual_size} bytes"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_size_increases_with_data(self, num_records, num_zones, seed):
        """
        Property: File size should increase monotonically with more data.
        """
        np.random.seed(seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Generate and export data with increasing record counts
            # Ensure counts are strictly increasing
            base_count = max(10, num_records // 4)
            record_counts = [base_count, base_count * 2, base_count * 3]
            
            sizes = []
            
            for count in record_counts:
                # Generate data
                data = generate_time_series_data(count, num_zones, seed + count)
                
                # Export to CSV
                csv_path = exporter.export_vegetation_indices_csv(data)
                
                # Get file size
                size = os.path.getsize(csv_path)
                sizes.append(size)
                
                # Clean up for next iteration
                os.remove(csv_path)
            
            # Verify sizes increase monotonically
            for i in range(len(sizes) - 1):
                assert sizes[i] < sizes[i + 1], \
                    f"File size did not increase: {sizes[i]} >= {sizes[i + 1]} " \
                    f"for record counts {record_counts[i]} vs {record_counts[i + 1]}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_export_history_size_mb_calculation(self, num_records, num_zones, seed):
        """
        Property: Reported size in MB should be correctly calculated from bytes.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get export history
            history = exporter.get_export_history(limit=1)
            
            assert len(history) > 0, "Export history is empty"
            
            size_bytes = history[0]['size_bytes']
            size_mb = history[0]['size_mb']
            
            # Calculate expected MB value
            expected_mb = round(size_bytes / (1024 * 1024), 2)
            
            # Verify MB calculation is correct
            assert size_mb == expected_mb, \
                f"Size in MB incorrectly calculated: expected {expected_mb}, got {size_mb} " \
                f"(from {size_bytes} bytes)"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_file_stat_size_matches_getsize(self, num_records, num_zones, seed):
        """
        Property: os.stat().st_size should match os.path.getsize().
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get size using os.path.getsize()
            size_getsize = os.path.getsize(csv_path)
            
            # Get size using os.stat()
            size_stat = os.stat(csv_path).st_size
            
            # Verify they match exactly
            assert size_getsize == size_stat, \
                f"Size mismatch: os.path.getsize()={size_getsize}, " \
                f"os.stat().st_size={size_stat}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_export_size_after_file_operations(self, num_records, num_zones, seed):
        """
        Property: File size should remain constant after read operations.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get initial size
            initial_size = os.path.getsize(csv_path)
            
            # Perform read operations
            with open(csv_path, 'r') as f:
                _ = f.read()
            
            with open(csv_path, 'rb') as f:
                _ = f.read()
            
            # Get size after reads
            final_size = os.path.getsize(csv_path)
            
            # Verify size hasn't changed
            assert initial_size == final_size, \
                f"File size changed after read operations: " \
                f"initial={initial_size}, final={final_size}"
    
    @settings(max_examples=30, deadline=None)
    @given(
        num_zones=zone_count_strategy,
        days=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_batch_export_manifest_size_accuracy(self, num_zones, days, seed):
        """
        Property: Batch export manifest should report accurate file sizes.
        """
        np.random.seed(seed)
        
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        time_series_data = generate_mock_vegetation_indices_data(zone_ids, days)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Create batch export configuration
            export_configs = [
                {
                    'type': 'vegetation_indices',
                    'data': time_series_data
                }
            ]
            
            # Create batch export (uncompressed to check manifest)
            batch_dir = exporter.create_batch_export(export_configs, compress=False)
            
            # Read manifest
            import json
            manifest_path = os.path.join(batch_dir, 'export_manifest.json')
            
            assert os.path.exists(manifest_path), "Manifest file not found"
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Verify each file size in manifest
            for file_info in manifest['files']:
                filename = file_info['filename']
                reported_size = file_info['size_bytes']
                
                # Skip manifest itself
                if filename == 'export_manifest.json':
                    continue
                
                # Get actual file size
                file_path = os.path.join(batch_dir, filename)
                
                if os.path.exists(file_path):
                    actual_size = os.path.getsize(file_path)
                    
                    # Verify sizes match within 1 byte
                    size_difference = abs(actual_size - reported_size)
                    
                    assert size_difference <= 1, \
                        f"Manifest size mismatch for {filename}: " \
                        f"actual={actual_size}, reported={reported_size}, " \
                        f"difference={size_difference}"
    
    @settings(max_examples=30, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_export_size_precision(self, num_records, num_zones, seed):
        """
        Property: File size should be reported as an exact integer, not rounded.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get actual file size
            actual_size = os.path.getsize(csv_path)
            
            # Get reported size
            history = exporter.get_export_history(limit=1)
            reported_size = history[0]['size_bytes']
            
            # Verify both are integers
            assert isinstance(actual_size, int), \
                f"Actual size is not an integer: {type(actual_size)}"
            assert isinstance(reported_size, int), \
                f"Reported size is not an integer: {type(reported_size)}"
            
            # Verify exact match (no rounding)
            assert actual_size == reported_size, \
                f"Size mismatch (no rounding allowed): " \
                f"actual={actual_size}, reported={reported_size}"
    
    @settings(max_examples=30, deadline=None)
    @given(
        num_records=record_count_strategy,
        num_zones=zone_count_strategy,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_export_size_immediately_after_creation(self, num_records, num_zones, seed):
        """
        Property: File size should be accurate immediately after file creation.
        """
        # Generate test data
        time_series_data = generate_time_series_data(num_records, num_zones, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Get size immediately after creation
            immediate_size = os.path.getsize(csv_path)
            
            # Get reported size from history (which is populated during export)
            history = exporter.get_export_history(limit=1)
            reported_size = history[0]['size_bytes']
            
            # Verify sizes match within 1 byte
            size_difference = abs(immediate_size - reported_size)
            
            assert size_difference <= 1, \
                f"Size mismatch immediately after creation: " \
                f"immediate={immediate_size}, reported={reported_size}, " \
                f"difference={size_difference}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
