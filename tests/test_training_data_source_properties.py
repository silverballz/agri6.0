"""
Property-based tests for training data source validation.

**Feature: real-satellite-data-integration, Property 3: Training data contains only real imagery**
**Validates: Requirements 4.1**
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_real_training_data import RealDatasetPreparator


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_mock_imagery_dir(
    parent_dir: Path,
    tile_id: str,
    acquisition_date: str,
    is_synthetic: bool
) -> Path:
    """
    Create a mock imagery directory with metadata and band files.
    
    Args:
        parent_dir: Parent directory for imagery
        tile_id: Tile identifier
        acquisition_date: Acquisition date string
        is_synthetic: Whether this is synthetic data
        
    Returns:
        Path to created imagery directory
    """
    # Create directory name
    dir_name = f"{tile_id}_{acquisition_date}"
    img_dir = parent_dir / dir_name
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        'tile_id': tile_id,
        'acquisition_date': acquisition_date,
        'synthetic': is_synthetic,
        'data_source': 'Synthetic Generator' if is_synthetic else 'Sentinel Hub API',
        'bands': ['B02', 'B03', 'B04', 'B08'],
        'indices': ['NDVI', 'SAVI', 'EVI', 'NDWI']
    }
    
    with open(img_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Create mock band files (small arrays for testing)
    for band in ['B02', 'B03', 'B04', 'B08']:
        band_data = np.random.uniform(0, 10000, size=(100, 100)).astype(np.uint16)
        np.save(img_dir / f"{band}.npy", band_data)
    
    # Create mock NDVI file
    ndvi_data = np.random.uniform(-1, 1, size=(100, 100)).astype(np.float32)
    np.save(img_dir / 'NDVI.npy', ndvi_data)
    
    return img_dir


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(
    num_real_imagery=st.integers(min_value=1, max_value=10),
    num_synthetic_imagery=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_training_data_contains_only_real_imagery_property(
    num_real_imagery,
    num_synthetic_imagery
):
    """
    Property 3: Training data contains only real imagery
    
    For any dataset prepared by RealDatasetPreparator, all source imagery
    directories used should have synthetic=false in their metadata.
    
    This property validates Requirements 4.1.
    """
    # Create temporary directories for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create mock imagery directories
        real_dirs = []
        synthetic_dirs = []
        
        # Create real imagery directories
        for i in range(num_real_imagery):
            img_dir = create_mock_imagery_dir(
                temp_processed,
                tile_id='43REQ',
                acquisition_date=f'2024-09-{i+1:02d}',
                is_synthetic=False
            )
            real_dirs.append(img_dir)
        
        # Create synthetic imagery directories
        for i in range(num_synthetic_imagery):
            img_dir = create_mock_imagery_dir(
                temp_processed,
                tile_id='43REQ',
                acquisition_date=f'2024-10-{i+1:02d}',
                is_synthetic=True
            )
            synthetic_dirs.append(img_dir)
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        # Property 1: Only real imagery directories should be found
        assert len(found_dirs) == num_real_imagery, \
            f"Expected {num_real_imagery} real directories, found {len(found_dirs)}"
        
        # Property 2: All found directories should have synthetic=false
        for img_dir in found_dirs:
            metadata_file = img_dir / 'metadata.json'
            assert metadata_file.exists(), f"Metadata file missing for {img_dir}"
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            is_synthetic = metadata.get('synthetic', True)
            assert is_synthetic is False, \
                f"Found directory {img_dir.name} has synthetic={is_synthetic}, expected False"
        
        # Property 3: No synthetic directories should be included
        found_dir_names = {d.name for d in found_dirs}
        synthetic_dir_names = {d.name for d in synthetic_dirs}
        
        overlap = found_dir_names & synthetic_dir_names
        assert len(overlap) == 0, \
            f"Synthetic directories found in real data: {overlap}"


@given(
    num_imagery=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_all_real_imagery_property(num_imagery):
    """
    Property: When all imagery is real, all should be found.
    
    For any collection of imagery where all have synthetic=false,
    the preparator should find all of them.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create only real imagery
        created_dirs = []
        for i in range(num_imagery):
            img_dir = create_mock_imagery_dir(
                temp_processed,
                tile_id='43REQ',
                acquisition_date=f'2024-09-{i+1:02d}',
                is_synthetic=False
            )
            created_dirs.append(img_dir)
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        # Property: All created directories should be found
        assert len(found_dirs) == num_imagery, \
            f"Expected to find all {num_imagery} real directories, found {len(found_dirs)}"
        
        # Verify all are real
        for img_dir in found_dirs:
            metadata_file = img_dir / 'metadata.json'
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert metadata.get('synthetic', True) is False, \
                f"Directory {img_dir.name} should be marked as real"


@given(
    num_imagery=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_all_synthetic_imagery_property(num_imagery):
    """
    Property: When all imagery is synthetic, none should be found.
    
    For any collection of imagery where all have synthetic=true,
    the preparator should find none of them.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create only synthetic imagery
        for i in range(num_imagery):
            create_mock_imagery_dir(
                temp_processed,
                tile_id='43REQ',
                acquisition_date=f'2024-10-{i+1:02d}',
                is_synthetic=True
            )
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        # Property: No directories should be found
        assert len(found_dirs) == 0, \
            f"Expected to find 0 real directories, found {len(found_dirs)}"


def test_missing_metadata_excluded():
    """
    Test that directories without metadata.json are excluded.
    
    This is a concrete example test to ensure robustness.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create directory without metadata
        no_meta_dir = temp_processed / '43REQ_2024-09-01'
        no_meta_dir.mkdir()
        
        # Create directory with metadata (real)
        with_meta_dir = create_mock_imagery_dir(
            temp_processed,
            tile_id='43REQ',
            acquisition_date='2024-09-02',
            is_synthetic=False
        )
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        # Only the directory with metadata should be found
        assert len(found_dirs) == 1
        assert found_dirs[0] == with_meta_dir


def test_malformed_metadata_excluded():
    """
    Test that directories with malformed metadata.json are excluded.
    
    This is a concrete example test to ensure robustness.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create directory with malformed metadata
        bad_meta_dir = temp_processed / '43REQ_2024-09-01'
        bad_meta_dir.mkdir()
        with open(bad_meta_dir / 'metadata.json', 'w') as f:
            f.write("{ invalid json }")
        
        # Create directory with valid metadata (real)
        good_meta_dir = create_mock_imagery_dir(
            temp_processed,
            tile_id='43REQ',
            acquisition_date='2024-09-02',
            is_synthetic=False
        )
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        # Only the directory with valid metadata should be found
        assert len(found_dirs) == 1
        assert found_dirs[0] == good_meta_dir


def test_default_synthetic_true():
    """
    Test that missing synthetic flag defaults to True (synthetic).
    
    This ensures that data is only included if explicitly marked as real.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create directory with metadata missing synthetic flag
        no_flag_dir = temp_processed / '43REQ_2024-09-01'
        no_flag_dir.mkdir()
        
        metadata = {
            'tile_id': '43REQ',
            'acquisition_date': '2024-09-01',
            # synthetic flag intentionally missing
            'data_source': 'Unknown'
        }
        
        with open(no_flag_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Create directory with explicit synthetic=false
        real_dir = create_mock_imagery_dir(
            temp_processed,
            tile_id='43REQ',
            acquisition_date='2024-09-02',
            is_synthetic=False
        )
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        # Only the directory with explicit synthetic=false should be found
        assert len(found_dirs) == 1
        assert found_dirs[0] == real_dir


def test_data_source_field_consistency():
    """
    Test that real data has appropriate data_source field.
    
    This validates that the data provenance is properly tracked.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_processed = Path(temp_dir) / 'processed'
        temp_output = Path(temp_dir) / 'output'
        temp_processed.mkdir()
        temp_output.mkdir()
        
        # Create real imagery with Sentinel Hub source
        real_dir = create_mock_imagery_dir(
            temp_processed,
            tile_id='43REQ',
            acquisition_date='2024-09-01',
            is_synthetic=False
        )
        
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=temp_processed,
            output_dir=temp_output
        )
        
        # Find real imagery directories
        found_dirs = preparator._find_real_imagery_dirs()
        
        assert len(found_dirs) == 1
        
        # Check data source
        with open(found_dirs[0] / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        assert metadata.get('synthetic') is False
        assert 'Sentinel Hub' in metadata.get('data_source', '') or \
               metadata.get('data_source') != 'Synthetic Generator', \
               "Real data should not have 'Synthetic Generator' as data source"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
