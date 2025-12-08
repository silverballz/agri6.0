"""
Tests for dependency checker
"""

import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dependency_checker import DependencyChecker


def test_dependency_checker_initialization():
    """Test DependencyChecker initialization"""
    
    checker = DependencyChecker()
    
    assert checker.results is not None
    assert 'required_packages' in checker.results
    assert 'optional_packages' in checker.results
    assert 'critical_paths' in checker.results
    assert 'optional_paths' in checker.results
    assert 'overall_status' in checker.results


def test_check_required_packages():
    """Test checking required packages"""
    
    checker = DependencyChecker()
    checker.check_required_packages()
    
    # Should have checked all required packages
    assert len(checker.results['required_packages']) > 0
    
    # Each package should have status
    for pkg_key, info in checker.results['required_packages'].items():
        assert 'status' in info
        assert info['status'] in ['installed', 'missing']
        assert 'display_name' in info
        assert 'import_name' in info


def test_check_optional_packages():
    """Test checking optional packages"""
    
    checker = DependencyChecker()
    checker.check_optional_packages()
    
    # Should have checked optional packages
    assert 'optional_packages' in checker.results
    
    # Each package should have status
    for pkg_key, info in checker.results['optional_packages'].items():
        assert 'status' in info
        assert info['status'] in ['installed', 'missing']


def test_check_critical_paths():
    """Test checking critical paths"""
    
    checker = DependencyChecker()
    checker.check_critical_paths()
    
    # Should have checked critical paths
    assert len(checker.results['critical_paths']) > 0
    
    # Each path should have status
    for path_key, info in checker.results['critical_paths'].items():
        assert 'status' in info
        assert info['status'] in ['exists', 'missing']
        assert 'path' in info


def test_check_optional_paths():
    """Test checking optional paths"""
    
    checker = DependencyChecker()
    checker.check_optional_paths()
    
    # Should have checked optional paths
    assert 'optional_paths' in checker.results


def test_check_all():
    """Test running all checks"""
    
    checker = DependencyChecker()
    results = checker.check_all()
    
    # Should return results dictionary
    assert isinstance(results, dict)
    
    # Should have all sections
    assert 'required_packages' in results
    assert 'optional_packages' in results
    assert 'critical_paths' in results
    assert 'optional_paths' in results
    assert 'overall_status' in results
    
    # Overall status should be one of the valid values
    assert results['overall_status'] in ['excellent', 'good', 'warning', 'critical', 'unknown']


def test_determine_overall_status():
    """Test overall status determination"""
    
    checker = DependencyChecker()
    
    # Mock some results
    checker.results['required_packages'] = {
        'test_pkg': {'status': 'installed', 'display_name': 'Test', 'import_name': 'test'}
    }
    checker.results['optional_packages'] = {}
    checker.results['critical_paths'] = {
        'test_path': {'status': 'exists', 'path': 'test'}
    }
    checker.results['optional_paths'] = {}
    
    checker.determine_overall_status()
    
    # Should have determined a status
    assert checker.results['overall_status'] in ['excellent', 'good', 'warning', 'critical']
    assert 'status_message' in checker.results


def test_create_missing_paths():
    """Test creating missing paths"""
    
    checker = DependencyChecker()
    
    # Mock a missing path
    checker.results['critical_paths'] = {
        'test': {'status': 'missing', 'path': 'test_temp_dir'}
    }
    
    created, failed = checker.create_missing_paths()
    
    # Should return lists
    assert isinstance(created, list)
    assert isinstance(failed, list)
    
    # Clean up if created
    import shutil
    from pathlib import Path
    test_path = Path('test_temp_dir')
    if test_path.exists():
        shutil.rmtree(test_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
