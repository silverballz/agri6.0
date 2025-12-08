"""
Tests for error handling framework
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.error_handler import (
    safe_page,
    safe_operation,
    handle_data_loading,
    ErrorMessages,
    check_critical_paths
)


def test_safe_page_decorator_success():
    """Test that safe_page decorator allows successful function execution"""
    
    @safe_page
    def test_function():
        return "success"
    
    result = test_function()
    assert result == "success"


def test_safe_page_decorator_file_not_found():
    """Test that safe_page decorator handles FileNotFoundError"""
    
    @safe_page
    def test_function():
        raise FileNotFoundError("test.txt")
    
    # Should return None and not raise exception
    result = test_function()
    assert result is None


def test_safe_page_decorator_import_error():
    """Test that safe_page decorator handles ImportError"""
    
    @safe_page
    def test_function():
        raise ImportError("No module named 'test_module'")
    
    # Should return None and not raise exception
    result = test_function()
    assert result is None


def test_safe_operation_decorator():
    """Test safe_operation decorator"""
    
    @safe_operation("Test operation")
    def test_function():
        return "success"
    
    result = test_function()
    assert result == "success"


def test_safe_operation_decorator_with_error():
    """Test safe_operation decorator handles errors"""
    
    @safe_operation("Test operation")
    def test_function():
        raise ValueError("Test error")
    
    # Should return None and not raise exception
    result = test_function()
    assert result is None


def test_error_messages():
    """Test ErrorMessages class"""
    
    # Test missing dependency message
    msg = ErrorMessages.missing_dependency("test_package")
    assert "test_package" in msg
    assert "pip install" in msg
    
    # Test missing data message
    msg = ErrorMessages.missing_data("test_data")
    assert "test_data" in msg
    
    # Test processing error message
    msg = ErrorMessages.processing_error("test_operation")
    assert "test_operation" in msg
    
    # Test database error message
    msg = ErrorMessages.database_error()
    assert "database" in msg.lower()


def test_check_critical_paths():
    """Test check_critical_paths function"""
    
    status = check_critical_paths()
    
    # Should return a dictionary
    assert isinstance(status, dict)
    
    # Should have keys for critical paths
    assert 'data' in status or 'logs' in status
    
    # Values should be boolean
    for key, value in status.items():
        assert isinstance(value, bool)


def test_handle_data_loading_decorator():
    """Test handle_data_loading decorator"""
    
    @handle_data_loading
    def test_function():
        return "data loaded"
    
    result = test_function()
    assert result == "data loaded"


def test_handle_data_loading_with_error():
    """Test handle_data_loading decorator with error"""
    
    @handle_data_loading
    def test_function():
        raise FileNotFoundError("data.csv")
    
    # Should return None and not raise exception
    result = test_function()
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
