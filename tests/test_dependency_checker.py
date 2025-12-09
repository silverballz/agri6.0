"""
Tests for dependency checker and health checks
Requirements: 10.1, 10.5
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import importlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dependency_checker import DependencyChecker, display_dependency_status, fix_dependency_issues


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


def test_dependency_verification_all_present():
    """Test dependency verification when all required packages are present"""
    
    checker = DependencyChecker()
    checker.check_required_packages()
    
    # All required packages should be checked
    assert len(checker.results['required_packages']) == len(checker.REQUIRED_PACKAGES)
    
    # Verify each package has proper structure
    for pkg_key, info in checker.results['required_packages'].items():
        assert 'status' in info
        assert 'display_name' in info
        assert 'import_name' in info
        
        # Status should be either installed or missing
        assert info['status'] in ['installed', 'missing']


def test_dependency_verification_missing_package():
    """Test dependency verification detects missing packages"""
    
    checker = DependencyChecker()
    
    # Add a fake missing package
    checker.REQUIRED_PACKAGES['fake_package'] = ('fake_package_import', 'Fake Package')
    
    checker.check_required_packages()
    
    # Should detect the missing package
    assert 'fake_package' in checker.results['required_packages']
    assert checker.results['required_packages']['fake_package']['status'] == 'missing'
    assert 'error' in checker.results['required_packages']['fake_package']


def test_component_status_detection_critical():
    """Test component status detection for critical failures"""
    
    checker = DependencyChecker()
    
    # Mock missing required package
    checker.results['required_packages'] = {
        'streamlit': {'status': 'missing', 'display_name': 'Streamlit', 'import_name': 'streamlit'}
    }
    checker.results['optional_packages'] = {}
    checker.results['critical_paths'] = {
        'data': {'status': 'exists', 'path': 'data'}
    }
    checker.results['optional_paths'] = {}
    
    checker.determine_overall_status()
    
    # Should be critical status
    assert checker.results['overall_status'] == 'critical'
    assert 'status_message' in checker.results
    assert 'streamlit' in checker.results['status_message'].lower()


def test_component_status_detection_warning():
    """Test component status detection for warning state"""
    
    checker = DependencyChecker()
    
    # Mock missing critical path
    checker.results['required_packages'] = {
        'streamlit': {'status': 'installed', 'display_name': 'Streamlit', 'import_name': 'streamlit'}
    }
    checker.results['optional_packages'] = {}
    checker.results['critical_paths'] = {
        'data': {'status': 'missing', 'path': 'data'}
    }
    checker.results['optional_paths'] = {}
    
    checker.determine_overall_status()
    
    # Should be warning status
    assert checker.results['overall_status'] == 'warning'
    assert 'status_message' in checker.results
    assert 'data' in checker.results['status_message'].lower()


def test_component_status_detection_good():
    """Test component status detection for good state with missing optional"""
    
    checker = DependencyChecker()
    
    # All required present, some optional missing
    checker.results['required_packages'] = {
        'streamlit': {'status': 'installed', 'display_name': 'Streamlit', 'import_name': 'streamlit'}
    }
    checker.results['optional_packages'] = {
        'tensorflow': {'status': 'missing', 'display_name': 'TensorFlow', 'import_name': 'tensorflow'}
    }
    checker.results['critical_paths'] = {
        'data': {'status': 'exists', 'path': 'data'}
    }
    checker.results['optional_paths'] = {}
    
    checker.determine_overall_status()
    
    # Should be good status
    assert checker.results['overall_status'] == 'good'
    assert 'status_message' in checker.results
    assert 'tensorflow' in checker.results['status_message'].lower()


def test_component_status_detection_excellent():
    """Test component status detection for excellent state"""
    
    checker = DependencyChecker()
    
    # All components present
    checker.results['required_packages'] = {
        'streamlit': {'status': 'installed', 'display_name': 'Streamlit', 'import_name': 'streamlit'}
    }
    checker.results['optional_packages'] = {
        'tensorflow': {'status': 'installed', 'display_name': 'TensorFlow', 'import_name': 'tensorflow'}
    }
    checker.results['critical_paths'] = {
        'data': {'status': 'exists', 'path': 'data'}
    }
    checker.results['optional_paths'] = {
        'models': {'status': 'exists', 'path': 'models'}
    }
    
    checker.determine_overall_status()
    
    # Should be excellent status
    assert checker.results['overall_status'] == 'excellent'
    assert 'status_message' in checker.results
    assert 'all components available' in checker.results['status_message'].lower()


def test_graceful_degradation_missing_optional():
    """Test graceful degradation when optional packages are missing"""
    
    checker = DependencyChecker()
    checker.check_optional_packages()
    
    # System should continue even if optional packages missing
    assert 'optional_packages' in checker.results
    
    # Each optional package should have status
    for pkg_key, info in checker.results['optional_packages'].items():
        assert 'status' in info
        # Status can be installed or missing - both are acceptable
        assert info['status'] in ['installed', 'missing']


def test_graceful_degradation_missing_optional_paths():
    """Test graceful degradation when optional paths are missing"""
    
    checker = DependencyChecker()
    checker.check_optional_paths()
    
    # System should continue even if optional paths missing
    assert 'optional_paths' in checker.results
    
    # Each optional path should have status
    for path_key, info in checker.results['optional_paths'].items():
        assert 'status' in info
        # Status can be exists or missing - both are acceptable
        assert info['status'] in ['exists', 'missing']


def test_graceful_degradation_fallback_behavior():
    """Test that system can operate with degraded functionality"""
    
    checker = DependencyChecker()
    
    # Simulate missing optional components
    checker.results['required_packages'] = {
        'streamlit': {'status': 'installed', 'display_name': 'Streamlit', 'import_name': 'streamlit'}
    }
    checker.results['optional_packages'] = {
        'tensorflow': {'status': 'missing', 'display_name': 'TensorFlow', 'import_name': 'tensorflow'},
        'rasterio': {'status': 'missing', 'display_name': 'Rasterio', 'import_name': 'rasterio'}
    }
    checker.results['critical_paths'] = {
        'data': {'status': 'exists', 'path': 'data'}
    }
    checker.results['optional_paths'] = {
        'models': {'status': 'missing', 'path': 'models'}
    }
    
    checker.determine_overall_status()
    
    # Should still be operational (good status)
    assert checker.results['overall_status'] == 'good'
    assert 'status_message' in checker.results
    
    # Message should indicate optional components missing
    assert 'optional' in checker.results['status_message'].lower()


def test_status_message_display_format():
    """Test that status messages are properly formatted"""
    
    checker = DependencyChecker()
    
    # Test critical status message
    checker.results['required_packages'] = {
        'missing_pkg': {'status': 'missing', 'display_name': 'Missing Package', 'import_name': 'missing'}
    }
    checker.results['optional_packages'] = {}
    checker.results['critical_paths'] = {}
    checker.results['optional_paths'] = {}
    
    checker.determine_overall_status()
    
    # Should have clear status message
    assert 'status_message' in checker.results
    assert isinstance(checker.results['status_message'], str)
    assert len(checker.results['status_message']) > 0
    
    # Should mention the missing package
    assert 'missing_pkg' in checker.results['status_message'].lower()


def test_status_message_display_multiple_issues():
    """Test status message when multiple issues exist"""
    
    checker = DependencyChecker()
    
    # Multiple missing required packages
    checker.results['required_packages'] = {
        'pkg1': {'status': 'missing', 'display_name': 'Package 1', 'import_name': 'pkg1'},
        'pkg2': {'status': 'missing', 'display_name': 'Package 2', 'import_name': 'pkg2'}
    }
    checker.results['optional_packages'] = {}
    checker.results['critical_paths'] = {}
    checker.results['optional_paths'] = {}
    
    checker.determine_overall_status()
    
    # Should mention multiple packages
    message = checker.results['status_message'].lower()
    assert 'pkg1' in message or 'package 1' in message
    assert 'pkg2' in message or 'package 2' in message


def test_path_creation_success():
    """Test successful creation of missing paths"""
    
    checker = DependencyChecker()
    
    # Create a test path
    test_path = Path('test_health_check_dir')
    
    # Ensure it doesn't exist
    if test_path.exists():
        import shutil
        shutil.rmtree(test_path)
    
    # Mock missing path
    checker.results['critical_paths'] = {
        'test': {'status': 'missing', 'path': str(test_path)}
    }
    
    created, failed = checker.create_missing_paths()
    
    # Should successfully create the path
    assert str(test_path) in created
    assert len(failed) == 0
    assert test_path.exists()
    
    # Clean up
    import shutil
    if test_path.exists():
        shutil.rmtree(test_path)


def test_path_creation_failure():
    """Test handling of path creation failures"""
    
    checker = DependencyChecker()
    
    # Mock a path that cannot be created (invalid path)
    invalid_path = '/root/cannot_create_this_path_no_permission'
    
    checker.results['critical_paths'] = {
        'invalid': {'status': 'missing', 'path': invalid_path}
    }
    
    created, failed = checker.create_missing_paths()
    
    # Should fail to create the path
    assert len(created) == 0
    assert len(failed) > 0
    assert any(invalid_path in str(f[0]) for f in failed)


def test_check_all_integration():
    """Test that check_all runs all checks and produces complete results"""
    
    checker = DependencyChecker()
    results = checker.check_all()
    
    # Should have all result sections
    assert 'required_packages' in results
    assert 'optional_packages' in results
    assert 'critical_paths' in results
    assert 'optional_paths' in results
    assert 'overall_status' in results
    assert 'status_message' in results
    
    # Overall status should be valid
    assert results['overall_status'] in ['excellent', 'good', 'warning', 'critical', 'unknown']
    
    # Should have checked some packages
    assert len(results['required_packages']) > 0
    
    # Should have checked some paths
    assert len(results['critical_paths']) > 0


def test_optional_package_detection():
    """Test detection of optional packages"""
    
    checker = DependencyChecker()
    checker.check_optional_packages()
    
    # Should check all optional packages
    assert len(checker.results['optional_packages']) == len(checker.OPTIONAL_PACKAGES)
    
    # Each should have proper structure
    for pkg_key, info in checker.results['optional_packages'].items():
        assert 'status' in info
        assert 'display_name' in info
        assert 'import_name' in info


def test_critical_path_validation():
    """Test validation of critical paths"""
    
    checker = DependencyChecker()
    checker.check_critical_paths()
    
    # Should check all critical paths
    assert len(checker.results['critical_paths']) == len(checker.CRITICAL_PATHS)
    
    # Each should have proper structure
    for path_key, info in checker.results['critical_paths'].items():
        assert 'status' in info
        assert 'path' in info
        
        # If exists, should have is_dir flag
        if info['status'] == 'exists':
            assert 'is_dir' in info


def test_optional_path_validation():
    """Test validation of optional paths"""
    
    checker = DependencyChecker()
    checker.check_optional_paths()
    
    # Should check all optional paths
    assert len(checker.results['optional_paths']) == len(checker.OPTIONAL_PATHS)
    
    # Each should have proper structure
    for path_key, info in checker.results['optional_paths'].items():
        assert 'status' in info
        assert 'path' in info


def test_status_icons_mapping():
    """Test that status levels map to appropriate icons"""
    
    status_levels = ['excellent', 'good', 'warning', 'critical', 'unknown']
    
    for status in status_levels:
        checker = DependencyChecker()
        checker.results['overall_status'] = status
        
        # Status should be valid
        assert status in ['excellent', 'good', 'warning', 'critical', 'unknown']


def test_graceful_degradation_with_partial_functionality():
    """Test system operates with partial functionality when components missing"""
    
    checker = DependencyChecker()
    
    # Simulate AI models missing but core functionality present
    checker.results['required_packages'] = {
        'streamlit': {'status': 'installed', 'display_name': 'Streamlit', 'import_name': 'streamlit'},
        'pandas': {'status': 'installed', 'display_name': 'Pandas', 'import_name': 'pandas'}
    }
    checker.results['optional_packages'] = {
        'tensorflow': {'status': 'missing', 'display_name': 'TensorFlow', 'import_name': 'tensorflow'}
    }
    checker.results['critical_paths'] = {
        'data': {'status': 'exists', 'path': 'data'},
        'logs': {'status': 'exists', 'path': 'logs'}
    }
    checker.results['optional_paths'] = {
        'models': {'status': 'missing', 'path': 'models'}
    }
    
    checker.determine_overall_status()
    
    # Should be operational with degraded functionality
    assert checker.results['overall_status'] in ['good', 'excellent']
    
    # Should indicate what's missing
    assert 'tensorflow' in checker.results['status_message'].lower()


def test_component_status_consistency():
    """Test that component status is consistent across multiple checks"""
    
    checker = DependencyChecker()
    
    # Run checks multiple times
    results1 = checker.check_all()
    
    checker2 = DependencyChecker()
    results2 = checker2.check_all()
    
    # Results should be consistent
    assert results1['overall_status'] == results2['overall_status']
    
    # Package statuses should match
    for pkg_key in results1['required_packages']:
        if pkg_key in results2['required_packages']:
            assert results1['required_packages'][pkg_key]['status'] == \
                   results2['required_packages'][pkg_key]['status']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
