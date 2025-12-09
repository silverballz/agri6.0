#!/usr/bin/env python3
"""
Quick test to verify the pipeline orchestration script is working correctly.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        import run_complete_pipeline
        print("  ✓ run_complete_pipeline module imported successfully")
        
        # Test classes
        assert hasattr(run_complete_pipeline, 'StepStatus')
        print("  ✓ StepStatus enum found")
        
        assert hasattr(run_complete_pipeline, 'PipelineStep')
        print("  ✓ PipelineStep dataclass found")
        
        assert hasattr(run_complete_pipeline, 'PipelineReport')
        print("  ✓ PipelineReport dataclass found")
        
        assert hasattr(run_complete_pipeline, 'PipelineOrchestrator')
        print("  ✓ PipelineOrchestrator class found")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_orchestrator_creation():
    """Test that orchestrator can be created."""
    print("\nTesting orchestrator creation...")
    try:
        from run_complete_pipeline import PipelineOrchestrator
        
        # Create orchestrator with skip flags
        orchestrator = PipelineOrchestrator(
            skip_download=True,
            skip_validation=True
        )
        
        print(f"  ✓ Orchestrator created successfully")
        print(f"  ✓ Total steps defined: {len(orchestrator.steps)}")
        
        # Check that steps are defined
        assert len(orchestrator.steps) > 0
        print(f"  ✓ Pipeline has {len(orchestrator.steps)} steps")
        
        # Check step structure
        first_step = orchestrator.steps[0]
        assert hasattr(first_step, 'step_number')
        assert hasattr(first_step, 'name')
        assert hasattr(first_step, 'description')
        print(f"  ✓ Step structure is correct")
        
        return True
    except Exception as e:
        print(f"  ✗ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_definitions():
    """Test that all pipeline steps are properly defined."""
    print("\nTesting step definitions...")
    try:
        from run_complete_pipeline import PipelineOrchestrator, StepStatus
        
        orchestrator = PipelineOrchestrator(skip_download=False, skip_validation=False)
        
        expected_steps = [
            "Download Real Satellite Data",
            "Validate Data Quality",
            "Prepare CNN Training Data",
            "Prepare LSTM Training Data",
            "Train CNN Model",
            "Train LSTM Model",
            "Compare Model Performance",
            "Update Configuration"
        ]
        
        actual_steps = [step.name for step in orchestrator.steps]
        
        for expected in expected_steps:
            if expected in actual_steps:
                print(f"  ✓ Step found: {expected}")
            else:
                print(f"  ✗ Step missing: {expected}")
                return False
        
        print(f"  ✓ All {len(expected_steps)} steps are properly defined")
        return True
        
    except Exception as e:
        print(f"  ✗ Step definition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Pipeline Orchestration Script Verification")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Orchestrator Creation Test", test_orchestrator_creation()))
    results.append(("Step Definitions Test", test_step_definitions()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✅ All tests passed! Pipeline script is ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
