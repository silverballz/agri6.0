"""
Property-based tests for CSV export functionality.
Tests universal properties that should hold across all valid inputs.

Feature: production-enhancements
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any


# Import the export function
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dashboard.data_exporter import DataExporter


# Strategy for generating zone IDs
zone_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'),
    min_size=3,
    max_size=20
).filter(lambda x: x and not x.startswith('_') and not x.startswith('-'))

# Strategy for generating index types
index_type_strategy = st.sampled_from(['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'])

# Strategy for generating vegetation index values (realistic ranges)
index_value_strategy = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for generating timestamps
def generate_timestamp_strategy(days_back: int = 90):
    """Generate timestamps within the last N days"""
    base_date = datetime.now()
    return st.datetimes(
        min_value=base_date - timedelta(days=days_back),
        max_value=base_date
    )


def generate_time_series_data(
    num_records: int,
    zone_ids: List[str],
    index_types: List[str],
    seed: int
) -> List[Dict[str, Any]]:
    """
    Generate synthetic time series data for testing.
    
    Args:
        num_records: Number of records to generate
        zone_ids: List of zone IDs to use
        index_types: List of index types to use
        seed: Random seed for reproducibility
        
    Returns:
        List of time series data dictionaries
    """
    np.random.seed(seed)
    
    data = []
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


class TestCSVExportCompletenessProperties:
    """Property-based tests for CSV export completeness.
    
    **Feature: production-enhancements, Property 19: CSV export completeness**
    **Validates: Requirements 5.2**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_preserves_all_timestamps(self, num_records, num_zones, num_indices, seed):
        """
        Property 19: CSV export completeness (Timestamps)
        
        For any time series data, exported CSV should contain all timestamps without data loss.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Extract original timestamps
        original_timestamps = sorted([record['timestamp'] for record in time_series_data])
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            
            # Convert timestamp column back to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            exported_timestamps = sorted(df['timestamp'].tolist())
            
            # Verify all timestamps are preserved
            assert len(exported_timestamps) == len(original_timestamps), \
                f"Timestamp count mismatch: expected {len(original_timestamps)}, got {len(exported_timestamps)}"
            
            # Verify timestamps match (within 1 second tolerance for datetime serialization)
            for orig, exp in zip(original_timestamps, exported_timestamps):
                time_diff = abs((orig - exp).total_seconds())
                assert time_diff < 1.0, \
                    f"Timestamp mismatch: expected {orig}, got {exp} (diff: {time_diff}s)"
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_preserves_all_index_values(self, num_records, num_zones, num_indices, seed):
        """
        Property 19: CSV export completeness (Index Values)
        
        For any time series data, exported CSV should contain all index values without data loss.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Extract original values
        original_values = sorted([record['mean_value'] for record in time_series_data])
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            
            # Verify mean_value column exists
            assert 'mean_value' in df.columns, "mean_value column missing from exported CSV"
            
            exported_values = sorted(df['mean_value'].tolist())
            
            # Verify all values are preserved
            assert len(exported_values) == len(original_values), \
                f"Value count mismatch: expected {len(original_values)}, got {len(exported_values)}"
            
            # Verify values match (within floating point tolerance)
            np.testing.assert_allclose(
                exported_values,
                original_values,
                rtol=1e-4,
                atol=1e-6,
                err_msg="Index values not preserved in CSV export"
            )
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_preserves_all_metadata_columns(self, num_records, num_zones, num_indices, seed):
        """
        Property 19: CSV export completeness (Metadata Columns)
        
        For any time series data, exported CSV should contain all metadata columns without data loss.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Define expected columns (from the data structure)
        expected_data_columns = ['zone_id', 'index_type', 'timestamp', 'mean_value', 
                                'std_deviation', 'pixel_count', 'quality_score']
        
        # Additional derived columns added by the exporter
        expected_derived_columns = ['date', 'year', 'month', 'day_of_year']
        
        all_expected_columns = expected_data_columns + expected_derived_columns
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            
            # Verify all expected columns are present
            for col in all_expected_columns:
                assert col in df.columns, \
                    f"Expected column '{col}' missing from exported CSV. Found columns: {list(df.columns)}"
            
            # Verify no data loss in metadata columns
            assert len(df) == len(time_series_data), \
                f"Row count mismatch: expected {len(time_series_data)}, got {len(df)}"
            
            # Verify zone_id preservation
            original_zone_ids = set(record['zone_id'] for record in time_series_data)
            exported_zone_ids = set(df['zone_id'].unique())
            assert original_zone_ids == exported_zone_ids, \
                f"Zone IDs not preserved: expected {original_zone_ids}, got {exported_zone_ids}"
            
            # Verify index_type preservation
            original_index_types = set(record['index_type'] for record in time_series_data)
            exported_index_types = set(df['index_type'].unique())
            assert original_index_types == exported_index_types, \
                f"Index types not preserved: expected {original_index_types}, got {exported_index_types}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_no_data_corruption(self, num_records, num_zones, num_indices, seed):
        """
        Property 19: CSV export completeness (No Data Corruption)
        
        For any time series data, exported CSV should not corrupt or modify any data values.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            
            # Convert to comparable format
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort both datasets the same way for comparison
            df_sorted = df.sort_values(['zone_id', 'index_type', 'timestamp']).reset_index(drop=True)
            
            original_df = pd.DataFrame(time_series_data)
            original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
            original_df_sorted = original_df.sort_values(['zone_id', 'index_type', 'timestamp']).reset_index(drop=True)
            
            # Verify each field matches
            for col in ['zone_id', 'index_type', 'mean_value', 'std_deviation', 'pixel_count', 'quality_score']:
                if col in ['mean_value', 'std_deviation', 'quality_score']:
                    # Numeric columns - check with tolerance
                    np.testing.assert_allclose(
                        df_sorted[col].values,
                        original_df_sorted[col].values,
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"Data corruption detected in column '{col}'"
                    )
                elif col == 'pixel_count':
                    # Integer column - exact match
                    assert (df_sorted[col].values == original_df_sorted[col].values).all(), \
                        f"Data corruption detected in column '{col}'"
                else:
                    # String columns - exact match
                    assert (df_sorted[col].values == original_df_sorted[col].values).all(), \
                        f"Data corruption detected in column '{col}'"
    
    @settings(max_examples=100, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_derived_columns_correctness(self, num_records, num_zones, num_indices, seed):
        """
        Property 19: CSV export completeness (Derived Columns)
        
        For any time series data, exported CSV should correctly compute derived columns
        (date, year, month, day_of_year) from timestamps.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Verify derived columns are computed correctly
            for idx, row in df.iterrows():
                timestamp = row['timestamp']
                
                # Check date
                expected_date = timestamp.date()
                actual_date = pd.to_datetime(row['date']).date()
                assert actual_date == expected_date, \
                    f"Derived 'date' column incorrect: expected {expected_date}, got {actual_date}"
                
                # Check year
                assert row['year'] == timestamp.year, \
                    f"Derived 'year' column incorrect: expected {timestamp.year}, got {row['year']}"
                
                # Check month
                assert row['month'] == timestamp.month, \
                    f"Derived 'month' column incorrect: expected {timestamp.month}, got {row['month']}"
                
                # Check day_of_year
                expected_doy = timestamp.timetuple().tm_yday
                assert row['day_of_year'] == expected_doy, \
                    f"Derived 'day_of_year' column incorrect: expected {expected_doy}, got {row['day_of_year']}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_sorting_consistency(self, num_records, num_zones, num_indices, seed):
        """
        Property: CSV export should sort data consistently by zone_id, index_type, and timestamp.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Verify sorting
            # Check if data is sorted by zone_id, index_type, timestamp
            is_sorted = True
            for i in range(len(df) - 1):
                curr_row = df.iloc[i]
                next_row = df.iloc[i + 1]
                
                # Compare zone_id first
                if curr_row['zone_id'] > next_row['zone_id']:
                    is_sorted = False
                    break
                elif curr_row['zone_id'] == next_row['zone_id']:
                    # If zone_id is same, compare index_type
                    if curr_row['index_type'] > next_row['index_type']:
                        is_sorted = False
                        break
                    elif curr_row['index_type'] == next_row['index_type']:
                        # If index_type is same, compare timestamp
                        if curr_row['timestamp'] > next_row['timestamp']:
                            is_sorted = False
                            break
            
            assert is_sorted, "CSV data is not sorted by zone_id, index_type, timestamp"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_file_readability(self, num_records, num_zones, num_indices, seed):
        """
        Property: Exported CSV should be readable and parseable by standard CSV readers.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Verify file exists
            assert os.path.exists(csv_path), f"CSV file not created at {csv_path}"
            
            # Verify file is not empty
            file_size = os.path.getsize(csv_path)
            assert file_size > 0, "CSV file is empty"
            
            # Verify file can be read by pandas
            try:
                df = pd.read_csv(csv_path)
                assert len(df) > 0, "CSV file contains no data rows"
            except Exception as e:
                pytest.fail(f"CSV file is not readable: {e}")
            
            # Verify file can be read line by line
            try:
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) > 1, "CSV file has no data rows (only header)"
            except Exception as e:
                pytest.fail(f"CSV file cannot be read as text: {e}")
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_records=st.integers(min_value=10, max_value=200),
        num_zones=st.integers(min_value=1, max_value=5),
        num_indices=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_csv_export_no_missing_values(self, num_records, num_zones, num_indices, seed):
        """
        Property: Exported CSV should not introduce missing values (NaN) in any column.
        """
        # Generate test data
        zone_ids = [f"zone_{i}" for i in range(num_zones)]
        index_types = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI'][:num_indices]
        
        time_series_data = generate_time_series_data(num_records, zone_ids, index_types, seed)
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DataExporter(output_dir=temp_dir)
            
            # Export to CSV
            csv_path = exporter.export_vegetation_indices_csv(time_series_data)
            
            # Read back CSV
            df = pd.read_csv(csv_path)
            
            # Check for missing values in each column
            for col in df.columns:
                missing_count = df[col].isna().sum()
                assert missing_count == 0, \
                    f"Column '{col}' has {missing_count} missing values in exported CSV"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
