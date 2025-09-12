#!/usr/bin/env python3
"""
🌱 AgriFlux - Smart Agricultural Intelligence Platform
Streamlit Community Cloud Deployment Entry Point

Free deployment optimized for Streamlit Community Cloud
"""

import sys
import os
import streamlit as st

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="🌱 AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/agriflux',
        'Report a bug': 'https://github.com/your-username/agriflux/issues',
        'About': """
        # AgriFlux 🌱
        **Smart Agricultural Intelligence Platform**
        
        Monitor crop health using satellite imagery and AI-powered analytics.
        
        Built with ❤️ for farmers and agricultural professionals.
        """
    }
)

# Import and run the main dashboard
try:
    from dashboard.main import main
    main()
except ImportError as e:
    st.error(f"""
    🚨 **Import Error**: {e}
    
    This might be a deployment issue. Please check:
    1. All required files are in the repository
    2. The `src/` directory structure is correct
    3. All dependencies are in requirements.txt
    """)
    st.stop()
except Exception as e:
    st.error(f"""
    🚨 **Application Error**: {e}
    
    Something went wrong while starting AgriFlux. 
    Please check the logs or contact support.
    """)
    st.stop()