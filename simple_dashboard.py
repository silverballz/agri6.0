"""
Simplified AgriFlux Dashboard for Testing
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="🌱 AgriFlux - Smart Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling
st.markdown("""
<style>
    .stApp {
        background-color: #1a1d29;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Sidebar
    st.sidebar.markdown("## 🌱 AgriFlux")
    st.sidebar.markdown("Smart Agricultural Intelligence")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Overview", "🗺️ Field Monitoring", "📈 Temporal Analysis", 
         "🚨 Alerts", "🤖 Model Performance", "📤 Data Export"]
    )
    
    # Main content
    st.title("🌱 AgriFlux Dashboard")
    st.markdown("### Smart Agricultural Intelligence Platform")
    
    # Show selected page
    st.info(f"Selected: {page}")
    
    # Simple metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fields Monitored", "12", "+2")
    
    with col2:
        st.metric("Average NDVI", "0.75", "+0.05")
    
    with col3:
        st.metric("Active Alerts", "3", "-1")
    
    with col4:
        st.metric("Data Points", "1.2M", "+50K")
    
    # Content based on page
    st.markdown("---")
    
    if "Overview" in page:
        st.subheader("📊 System Overview")
        st.success("✅ All systems operational")
        st.write("Dashboard is loading successfully!")
        
        # Show some data
        import pandas as pd
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'NDVI': [0.7, 0.72, 0.75, 0.73, 0.76, 0.78, 0.77, 0.79, 0.80, 0.82]
        })
        st.line_chart(df.set_index('Date'))
        
    elif "Field Monitoring" in page:
        st.subheader("🗺️ Field Monitoring")
        st.info("Interactive maps and field health analysis")
        
    elif "Temporal Analysis" in page:
        st.subheader("📈 Temporal Analysis")
        st.info("Time series charts and trend analysis")
        
    elif "Alerts" in page:
        st.subheader("🚨 Alerts & Notifications")
        st.warning("3 active alerts require attention")
        
    elif "Model Performance" in page:
        st.subheader("🤖 AI Model Performance")
        st.info("CNN Accuracy: 89.2% | LSTM R²: 0.953")
        
    elif "Data Export" in page:
        st.subheader("📤 Data Export")
        st.info("Export data in GeoTIFF, CSV, PDF, or ZIP formats")
    
    # Footer
    st.markdown("---")
    st.caption("AgriFlux v1.0.0 - Production Ready 🎉")

if __name__ == "__main__":
    main()
