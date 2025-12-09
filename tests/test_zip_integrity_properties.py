"""
Property-based tests for ZIP archive integrity.
Tests universal properties that should hold across all valid inputs.

Feature: production-enhancements
"""

import pytest
import os
import tempfile
import shutil
import hashlib
import zipfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import numpy as np


def calculate_file_checksum(filepath: str, algorithm: str = 'md5') -> str:
    """Calculate checksum of a file."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def create_test_file(filepath: str, content: bytes):
    """Create a test file with given content."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(content)


def create_zip_archive(file_paths: list, output_path: str) -> str:
    """
    Package files into ZIP archive.
    
    This is the function being tested - it should preserve file integrity.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                # Get relative path for archive
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)
    
    return output_path


def extract_zip_archive(zip_path: str, extract_dir: str) -> list:
    """Extract ZIP archive and return list of extracted file paths."""
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_dir)
        return [os.path.join(extract_dir, name) for name in zipf.namelist()]


# Strategies for generating test data
file_content_strategy = st.binary(min_size=0, max_size=10000)
file_count_strategy = st.integers(min_value=1, max_value=10)
filename_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=65, max_codepoint=122),
    min_size=5,
    max_size=20
).map(lambda s: s + '.txt')


class TestZIPIntegrityProperties:
    """Property-based tests for ZIP archive integrity.
    
    **Feature: production-enhancements, Property 20: ZIP archive integrity**
    **Validates: Requirements 5.4**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_checksum_preservation(self, file_contents, seed):
        """
        Property 20: ZIP archive integrity
        
        For any set of files packaged into ZIP, extracting and verifying 
        checksums should match original files.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files with generated content
            original_files = []
            original_checksums = {}
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                
                # Calculate original checksum
                original_checksums[filename] = calculate_file_checksum(filepath)
            
            # Create ZIP archive
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            
            # Verify ZIP was created
            assert os.path.exists(zip_path), "ZIP archive was not created"
            assert os.path.getsize(zip_path) > 0, "ZIP archive is empty"
            
            # Extract ZIP archive
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            # Verify all files were extracted
            assert len(extracted_files) == len(original_files), \
                f"File count mismatch: expected {len(original_files)}, got {len(extracted_files)}"
            
            # Verify checksums match
            for extracted_file in extracted_files:
                filename = os.path.basename(extracted_file)
                
                assert filename in original_checksums, \
                    f"Unexpected file in archive: {filename}"
                
                extracted_checksum = calculate_file_checksum(extracted_file)
                original_checksum = original_checksums[filename]
                
                assert extracted_checksum == original_checksum, \
                    f"Checksum mismatch for {filename}: " \
                    f"original={original_checksum}, extracted={extracted_checksum}"
        
        finally:
            # Clean up temporary directories
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=100, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_file_size_preservation(self, file_contents, seed):
        """
        Property: ZIP extraction should preserve file sizes exactly.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            original_files = []
            original_sizes = {}
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                
                # Record original size
                original_sizes[filename] = os.path.getsize(filepath)
            
            # Create and extract ZIP
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            # Verify file sizes match
            for extracted_file in extracted_files:
                filename = os.path.basename(extracted_file)
                extracted_size = os.path.getsize(extracted_file)
                original_size = original_sizes[filename]
                
                assert extracted_size == original_size, \
                    f"File size mismatch for {filename}: " \
                    f"original={original_size}, extracted={extracted_size}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=100, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_content_preservation(self, file_contents, seed):
        """
        Property: ZIP extraction should preserve file content byte-for-byte.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            original_files = []
            original_contents = {}
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                
                # Store original content
                original_contents[filename] = content
            
            # Create and extract ZIP
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            # Verify content matches byte-for-byte
            for extracted_file in extracted_files:
                filename = os.path.basename(extracted_file)
                
                with open(extracted_file, 'rb') as f:
                    extracted_content = f.read()
                
                original_content = original_contents[filename]
                
                assert extracted_content == original_content, \
                    f"Content mismatch for {filename}: " \
                    f"lengths original={len(original_content)}, extracted={len(extracted_content)}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=50, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_testzip_integrity(self, file_contents, seed):
        """
        Property: ZIP archive should pass built-in integrity test.
        """
        np.random.seed(seed)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            original_files = []
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
            
            # Create ZIP archive
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            
            # Test ZIP integrity using built-in method
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                test_result = zipf.testzip()
            
            # testzip() returns None if all files are OK, or the name of the first bad file
            assert test_result is None, \
                f"ZIP integrity test failed: {test_result}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @settings(max_examples=50, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000),
        algorithm=st.sampled_from(['md5', 'sha1', 'sha256'])
    )
    def test_zip_multiple_hash_algorithms(self, file_contents, seed, algorithm):
        """
        Property: ZIP integrity should be verifiable with multiple hash algorithms.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            original_files = []
            original_checksums = {}
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                
                # Calculate checksum with specified algorithm
                original_checksums[filename] = calculate_file_checksum(filepath, algorithm)
            
            # Create and extract ZIP
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            # Verify checksums with same algorithm
            for extracted_file in extracted_files:
                filename = os.path.basename(extracted_file)
                extracted_checksum = calculate_file_checksum(extracted_file, algorithm)
                original_checksum = original_checksums[filename]
                
                assert extracted_checksum == original_checksum, \
                    f"Checksum mismatch for {filename} using {algorithm}: " \
                    f"original={original_checksum}, extracted={extracted_checksum}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=50, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_file_count_preservation(self, file_contents, seed):
        """
        Property: Number of files in ZIP should equal number of files extracted.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            original_files = []
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
            
            # Create ZIP archive
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            
            # Count files in ZIP
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zip_file_count = len(zipf.namelist())
            
            # Extract and count files
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            extracted_file_count = len(extracted_files)
            
            # Verify counts match
            assert zip_file_count == len(original_files), \
                f"ZIP file count mismatch: expected {len(original_files)}, got {zip_file_count}"
            assert extracted_file_count == len(original_files), \
                f"Extracted file count mismatch: expected {len(original_files)}, got {extracted_file_count}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=50, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_filename_preservation(self, file_contents, seed):
        """
        Property: ZIP should preserve original filenames exactly.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files with specific names
            original_files = []
            original_filenames = set()
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                original_filenames.add(filename)
            
            # Create and extract ZIP
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            # Extract filenames
            extracted_filenames = {os.path.basename(f) for f in extracted_files}
            
            # Verify filenames match exactly
            assert extracted_filenames == original_filenames, \
                f"Filename mismatch: original={original_filenames}, extracted={extracted_filenames}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=50, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_compression_does_not_affect_integrity(self, file_contents, seed):
        """
        Property: Compression should not affect file integrity after extraction.
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            original_files = []
            original_checksums = {}
            
            for i, content in enumerate(file_contents):
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                original_checksums[filename] = calculate_file_checksum(filepath)
            
            # Create ZIP with compression
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            
            # Verify compression is applied
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                for info in zipf.infolist():
                    # compress_type should be ZIP_DEFLATED (8)
                    assert info.compress_type == zipfile.ZIP_DEFLATED, \
                        f"Compression not applied to {info.filename}"
            
            # Extract and verify integrity
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            for extracted_file in extracted_files:
                filename = os.path.basename(extracted_file)
                extracted_checksum = calculate_file_checksum(extracted_file)
                original_checksum = original_checksums[filename]
                
                assert extracted_checksum == original_checksum, \
                    f"Compression affected integrity of {filename}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    @settings(max_examples=30, deadline=None)
    @given(
        file_contents=st.lists(file_content_strategy, min_size=1, max_size=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_zip_empty_files_preservation(self, file_contents, seed):
        """
        Property: ZIP should correctly handle empty files (0 bytes).
        """
        np.random.seed(seed)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        extract_dir = tempfile.mkdtemp()
        
        try:
            # Create test files including at least one empty file
            original_files = []
            original_sizes = {}
            
            # Add an empty file
            empty_filename = "empty_file.txt"
            empty_filepath = os.path.join(temp_dir, empty_filename)
            create_test_file(empty_filepath, b'')
            original_files.append(empty_filepath)
            original_sizes[empty_filename] = 0
            
            # Add other files
            for i, content in enumerate(file_contents[:5]):  # Limit to 5 for performance
                filename = f"test_file_{i}.txt"
                filepath = os.path.join(temp_dir, filename)
                create_test_file(filepath, content)
                original_files.append(filepath)
                original_sizes[filename] = len(content)
            
            # Create and extract ZIP
            zip_path = os.path.join(temp_dir, 'test_archive.zip')
            create_zip_archive(original_files, zip_path)
            extracted_files = extract_zip_archive(zip_path, extract_dir)
            
            # Verify empty file was preserved
            empty_extracted = None
            for extracted_file in extracted_files:
                if os.path.basename(extracted_file) == empty_filename:
                    empty_extracted = extracted_file
                    break
            
            assert empty_extracted is not None, "Empty file not found in extracted files"
            assert os.path.getsize(empty_extracted) == 0, \
                f"Empty file has non-zero size after extraction: {os.path.getsize(empty_extracted)}"
            
            # Verify all file sizes
            for extracted_file in extracted_files:
                filename = os.path.basename(extracted_file)
                extracted_size = os.path.getsize(extracted_file)
                original_size = original_sizes[filename]
                
                assert extracted_size == original_size, \
                    f"Size mismatch for {filename}: original={original_size}, extracted={extracted_size}"
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
