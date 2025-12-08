#!/usr/bin/env python3
"""
🌱 AgriFlux - Local Development Runner
Simple script to run the AgriFlux dashboard locally
"""

import sys
import os
import subprocess

def main():
    """Run the AgriFlux dashboard locally"""
    
    print("🌱 Starting AgriFlux - Smart Agricultural Intelligence Platform")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed")
    
    # Check other dependencies
    dependencies = ['pandas', 'numpy', 'plotly']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} found")
        except ImportError:
            print(f"❌ {dep} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed")
    
    print("\n🚀 Starting AgriFlux Dashboard...")
    print("📍 Local URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Run the dashboard
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/dashboard/main.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 AgriFlux stopped by user")
    except Exception as e:
        print(f"\n❌ Error running AgriFlux: {e}")

if __name__ == "__main__":
    main()