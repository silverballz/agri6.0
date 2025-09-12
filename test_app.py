#!/usr/bin/env python3
"""
Test script to verify AgriFlux app works before deployment
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ Numpy imported successfully")
    except ImportError as e:
        print(f"❌ Numpy import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_app_structure():
    """Test app file structure"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'streamlit_app.py',
        'src/dashboard/main.py',
        'src/dashboard/pages/__init__.py',
        'src/dashboard/pages/overview.py',
        'requirements-render.txt',
        'render.yaml'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_dashboard_imports():
    """Test dashboard module imports"""
    print("\n🎛️ Testing dashboard imports...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from dashboard.pages import overview
        print("✅ Overview page imported successfully")
    except ImportError as e:
        print(f"❌ Overview page import failed: {e}")
        return False
    
    try:
        from dashboard import main
        print("✅ Main dashboard imported successfully")
    except ImportError as e:
        print(f"❌ Main dashboard import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🌱 AgriFlux Deployment Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_app_structure),
        ("Dashboard Import Test", test_dashboard_imports)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if not test_func():
            all_passed = False
            print(f"❌ {test_name} FAILED")
        else:
            print(f"✅ {test_name} PASSED")
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests PASSED! Ready for deployment!")
        print("\n📋 Next steps:")
        print("1. git add . && git commit -m 'Ready for deployment'")
        print("2. git push origin main")
        print("3. Deploy on Render.com using render.yaml")
    else:
        print("❌ Some tests FAILED. Fix issues before deployment.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())