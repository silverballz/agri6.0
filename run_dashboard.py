#!/usr/bin/env python3
"""
AgriFlux - Smart Agricultural Intelligence Platform
Dashboard Launcher - Run this script to start the Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    
    # Change to the dashboard directory
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard')
    
    # Run streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        os.path.join(dashboard_path, 'main.py'),
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--theme.base', 'light'
    ]
    
    print("🌱 Starting AgriFlux Dashboard...")
    print(f"📍 Dashboard will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, cwd=dashboard_path)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()