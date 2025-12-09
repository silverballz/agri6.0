#!/usr/bin/env python3
"""
Test script to verify deployed models load correctly.

This script tests that:
1. Production models exist at expected locations
2. Models can be loaded with PyTorch
3. Model metadata indicates real data training
4. Model registry is correct
"""

import json
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_model_files_exist():
    """Test that production model files exist."""
    print("=" * 80)
    print("TEST 1: Verify model files exist")
    print("=" * 80)
    
    models_dir = Path("models")
    
    required_files = [
        "crop_health_cnn.pth",
        "cnn_model_metrics.json",
        "crop_health_lstm.pth",
        "lstm_model_metrics.json",
        "model_registry.json"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = models_dir / filename
        exists = filepath.exists()
        symbol = "✓" if exists else "✗"
        print(f"{symbol} {filename}: {'Found' if exists else 'NOT FOUND'}")
        all_exist = all_exist and exists
    
    print(f"\nResult: {'PASS' if all_exist else 'FAIL'}")
    return all_exist


def test_models_load():
    """Test that models can be loaded with PyTorch."""
    print("\n" + "=" * 80)
    print("TEST 2: Verify models load correctly")
    print("=" * 80)
    
    models_dir = Path("models")
    
    models_to_test = {
        'CNN': 'crop_health_cnn.pth',
        'LSTM': 'crop_health_lstm.pth'
    }
    
    all_loaded = True
    for model_name, filename in models_to_test.items():
        print(f"\nLoading {model_name} model...")
        
        try:
            model_path = models_dir / filename
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            num_params = sum(p.numel() for p in state_dict.values())
            
            print(f"  ✓ {model_name} loaded successfully")
            print(f"  ✓ Parameters: {num_params:,}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            all_loaded = False
    
    print(f"\nResult: {'PASS' if all_loaded else 'FAIL'}")
    return all_loaded


def test_metadata_indicates_real_data():
    """Test that model metadata indicates training on real data."""
    print("\n" + "=" * 80)
    print("TEST 3: Verify metadata indicates real data training")
    print("=" * 80)
    
    models_dir = Path("models")
    
    metadata_files = {
        'CNN': 'cnn_model_metrics.json',
        'LSTM': 'lstm_model_metrics.json'
    }
    
    all_correct = True
    for model_name, filename in metadata_files.items():
        print(f"\nChecking {model_name} metadata...")
        
        try:
            metadata_path = models_dir / filename
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check trained_on field
            trained_on = metadata.get('trained_on', '')
            is_real = 'real' in trained_on.lower()
            symbol = "✓" if is_real else "✗"
            print(f"  {symbol} trained_on: {trained_on}")
            
            # Check data_source field
            data_source = metadata.get('data_source', '')
            is_sentinel = 'sentinel' in data_source.lower()
            symbol = "✓" if is_sentinel else "✗"
            print(f"  {symbol} data_source: {data_source}")
            
            # Check data_type field
            data_type = metadata.get('data_type', '')
            is_real_type = data_type == 'real'
            symbol = "✓" if is_real_type else "✗"
            print(f"  {symbol} data_type: {data_type}")
            
            model_correct = is_real and is_sentinel and is_real_type
            all_correct = all_correct and model_correct
            
        except Exception as e:
            print(f"  ✗ Failed to check {model_name} metadata: {e}")
            all_correct = False
    
    print(f"\nResult: {'PASS' if all_correct else 'FAIL'}")
    return all_correct


def test_model_registry():
    """Test that model registry is correct."""
    print("\n" + "=" * 80)
    print("TEST 4: Verify model registry")
    print("=" * 80)
    
    registry_path = Path("models/model_registry.json")
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        print(f"✓ Registry loaded successfully")
        print(f"  - Last updated: {registry.get('last_updated', 'N/A')}")
        print(f"  - Deployment type: {registry.get('deployment_type', 'N/A')}")
        
        models = registry.get('models', {})
        print(f"\n  Models in registry: {len(models)}")
        
        all_correct = True
        for model_type, model_info in models.items():
            print(f"\n  {model_type.upper()}:")
            print(f"    - Model file: {model_info.get('model_file', 'N/A')}")
            print(f"    - Trained on: {model_info.get('trained_on', 'N/A')}")
            print(f"    - Data source: {model_info.get('data_source', 'N/A')}")
            print(f"    - Accuracy: {model_info.get('accuracy', 'N/A')}")
            print(f"    - Status: {model_info.get('status', 'N/A')}")
            
            # Verify it's marked as real data
            trained_on = model_info.get('trained_on', '')
            is_real = 'real' in trained_on.lower()
            if not is_real:
                print(f"    ✗ WARNING: Not marked as real data training")
                all_correct = False
        
        print(f"\nResult: {'PASS' if all_correct else 'FAIL'}")
        return all_correct
        
    except Exception as e:
        print(f"✗ Failed to load registry: {e}")
        print(f"\nResult: FAIL")
        return False


def test_backup_exists():
    """Test that backup was created."""
    print("\n" + "=" * 80)
    print("TEST 5: Verify backup exists")
    print("=" * 80)
    
    backups_dir = Path("models/backups")
    
    if not backups_dir.exists():
        print("✗ Backups directory does not exist")
        print("\nResult: FAIL")
        return False
    
    # Find most recent backup
    backup_dirs = sorted(backups_dir.iterdir(), reverse=True)
    
    if not backup_dirs:
        print("✗ No backup directories found")
        print("\nResult: FAIL")
        return False
    
    latest_backup = backup_dirs[0]
    print(f"✓ Latest backup: {latest_backup.name}")
    
    # Check backup contents
    backup_files = list(latest_backup.iterdir())
    print(f"✓ Backup contains {len(backup_files)} files:")
    for f in backup_files:
        print(f"  - {f.name}")
    
    print("\nResult: PASS")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("MODEL DEPLOYMENT VERIFICATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Model files exist", test_model_files_exist),
        ("Models load correctly", test_models_load),
        ("Metadata indicates real data", test_metadata_indicates_real_data),
        ("Model registry correct", test_model_registry),
        ("Backup exists", test_backup_exists)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("  Real-trained models are deployed and ready for use!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Please review the failures above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
