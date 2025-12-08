"""
Temporal Analysis page - Time series visualization and trend analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
from scipy import stats

# Optional import for rasterio (not available on all platforms)
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.error_handler import safe_page, handle_data_loading, logger
from database.db_manager import DatabaseManager

@safe_page
def show_page():
    """Display the temporal analysis page"""
    
    st.title("📈 Temporal Analysis")
    st.markdown("Time series visualization and trend analysis for vegetation indices")
    
    # Check if demo mode is active
    if st.session_state.get('demo_mode', False) and st.session_state.get('demo_data'):
        show_demo_temporal_analysis()
        return
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Load imagery data
    try:
        imagery_list = db_manager.list_processed_imagery(limit=50)
        
        if not imagery_list:
            st.warning("No processed imagery available for temporal analysis. Please process satellite data first.")
            st.info("Enable **Demo Mode** from the sidebar to explore with sample data.")
            return
        
        # Analysis controls
        display_analysis_controls(imagery_list)
        
        # Load time series data
        time_series_data = load_time_series_data(imagery_list, db_manager)
        
        if time_series_data.empty:
            st.warning("Unable to load time series data from imagery files.")
            return
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            display_time_series_chart(time_series_data)
            display_comparison_chart(time_series_data)
        
        with col2:
            display_statistics_panel(time_series_data)
            display_trend_analysis(time_series_data)
    
    except Exception as e:
        logger.error(f"Error in temporal analysis page: {e}")
        st.error(f"Error loading temporal analysis: {str(e)}")

def load_time_series_data(imagery_list, db_manager):
    """Load time series data from imagery records"""
    
    time_series_data = []
    
    for imagery in imagery_list:
        try:
            acq_date = datetime.fromisoformat(imagery['acquisition_date'])
            
            # Load each vegetation index
            for index_name in ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']:
                index_path_key = f"{index_name.lower()}_path"
                index_path = imagery.get(index_path_key)
                
                if index_path and Path(index_path).exists():
                    try:
                        with rasterio.open(index_path) as src:
                            index_data = src.read(1)
                            valid_data = index_data[index_data != src.nodata]
                            
                            if len(valid_data) > 0:
                                time_series_data.append({
                                    'Date': acq_date,
                                    'Index': index_name,
                                    'Mean': np.mean(valid_data),
                                    'Median': np.median(valid_data),
                                    'Std': np.std(valid_data),
                                    'Min': np.min(valid_data),
                                    'Max': np.max(valid_data),
                                    'P25': np.percentile(valid_data, 25),
                                    'P75': np.percentile(valid_data, 75)
                                })
                    except Exception as e:
                        logger.warning(f"Error reading {index_name} from {index_path}: {e}")
        except Exception as e:
            logger.warning(f"Error processing imagery {imagery.get('id')}: {e}")
    
    return pd.DataFrame(time_series_data)

def display_analysis_controls(imagery_list):
    """Display analysis control panel"""
    
    st.subheader("🎛️ Analysis Controls")
    
    # Get date range from imagery
    dates = [datetime.fromisoformat(img['acquisition_date']) for img in imagery_list]
    min_date = min(dates) if dates else datetime.now() - timedelta(days=90)
    max_date = max(dates) if dates else datetime.now()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_indices = st.multiselect(
            "Vegetation Indices",
            ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"],
            default=["NDVI"],
            key="temporal_indices"
        )
    
    with col2:
        show_confidence = st.checkbox(
            "Show Confidence Intervals",
            value=True,
            key="show_confidence"
        )
    
    with col3:
        show_anomalies = st.checkbox(
            "Highlight Anomalies",
            value=True,
            key="show_anomalies"
        )
    
    # Display data range info
    st.info(f"📅 Available data: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({len(imagery_list)} observations)")

def display_time_series_chart(df):
    """Display main time series chart with real data"""
    
    st.subheader("📊 Vegetation Index Time Series")
    
    selected_indices = st.session_state.get('temporal_indices', ['NDVI'])
    show_confidence = st.session_state.get('show_confidence', True)
    show_anomalies = st.session_state.get('show_anomalies', True)
    
    if not selected_indices:
        st.warning("Please select at least one vegetation index.")
        return
    
    # Filter data
    df_filtered = df[df['Index'].isin(selected_indices)].copy()
    
    if df_filtered.empty:
        st.warning("No data available for selected indices.")
        return
    
    # Create subplot for each index
    fig = make_subplots(
        rows=len(selected_indices),
        cols=1,
        subplot_titles=selected_indices,
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, index in enumerate(selected_indices):
        index_data = df_filtered[df_filtered['Index'] == index].sort_values('Date')
        
        if not index_data.empty:
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=index_data['Date'],
                    y=index_data['Mean'],
                    mode='lines+markers',
                    name=index,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    showlegend=(i == 0),
                    hovertemplate=f"<b>{index}</b><br>" +
                                "Mean: %{y:.3f}<br>" +
                                "Date: %{x}<br>" +
                                "<extra></extra>"
                ),
                row=i+1, col=1
            )
            
            # Add confidence intervals (using P25 and P75 as bounds)
            if show_confidence and 'P25' in index_data.columns and 'P75' in index_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=index_data['Date'].tolist() + index_data['Date'].tolist()[::-1],
                        y=index_data['P75'].tolist() + index_data['P25'].tolist()[::-1],
                        fill='toself',
                        fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{index} CI"
                    ),
                    row=i+1, col=1
                )
            
            # Detect and highlight anomalies
            if show_anomalies and len(index_data) > 3:
                # Use z-score to detect anomalies
                z_scores = np.abs(stats.zscore(index_data['Mean']))
                anomalies = index_data[z_scores > 2]
                
                if not anomalies.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=anomalies['Date'],
                            y=anomalies['Mean'],
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='x'),
                            name='Anomalies',
                            showlegend=(i == 0),
                            hovertemplate="<b>Anomaly Detected</b><br>" +
                                        f"{index}: %{{y:.3f}}<br>" +
                                        "Date: %{x}<br>" +
                                        "<extra></extra>"
                        ),
                        row=i+1, col=1
                    )
            
            # Add threshold lines for NDVI
            if index == 'NDVI':
                fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                            annotation_text="Healthy", row=i+1, col=1)
                fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                            annotation_text="Moderate", row=i+1, col=1)
    
    fig.update_layout(
        height=300 * len(selected_indices),
        title="Vegetation Index Trends Over Time",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=len(selected_indices), col=1)
    
    for i in range(len(selected_indices)):
        fig.update_yaxes(title_text="Index Value", row=i+1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_comparison_chart(df):
    """Display multi-index comparison chart"""
    
    st.subheader("🔄 Multi-Index Comparison")
    
    selected_indices = st.session_state.get('temporal_indices', ['NDVI'])
    
    if len(selected_indices) < 2:
        st.info("Select multiple indices to compare them.")
        return
    
    # Filter data
    df_filtered = df[df['Index'].isin(selected_indices)].copy()
    
    if df_filtered.empty:
        st.warning("No data available for comparison.")
        return
    
    # Create comparison chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, index in enumerate(selected_indices):
        index_data = df_filtered[df_filtered['Index'] == index].sort_values('Date')
        
        if not index_data.empty:
            fig.add_trace(go.Scatter(
                x=index_data['Date'],
                y=index_data['Mean'],
                mode='lines+markers',
                name=index,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Multi-Index Comparison",
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_statistics_panel(df):
    """Display statistics panel with real data"""
    
    st.subheader("📊 Statistics")
    
    selected_indices = st.session_state.get('temporal_indices', ['NDVI'])
    
    for index in selected_indices:
        index_data = df[df['Index'] == index]
        
        if not index_data.empty:
            st.markdown(f"**{index}:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Mean",
                    f"{index_data['Mean'].mean():.3f}"
                )
                
                st.metric(
                    "Std Dev",
                    f"{index_data['Mean'].std():.3f}"
                )
            
            with col2:
                # Calculate trend
                if len(index_data) >= 2:
                    first_val = index_data.iloc[0]['Mean']
                    last_val = index_data.iloc[-1]['Mean']
                    change = last_val - first_val
                    
                    if change > 0.05:
                        trend = "Increasing"
                    elif change < -0.05:
                        trend = "Decreasing"
                    else:
                        trend = "Stable"
                    
                    st.metric(
                        "Trend",
                        trend,
                        delta=f"{change:+.3f}"
                    )
                    
                    # Calculate R² for linear trend
                    if len(index_data) >= 3:
                        x = np.arange(len(index_data))
                        y = index_data['Mean'].values
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        st.metric(
                            "R²",
                            f"{r_value**2:.3f}"
                        )
            
            st.markdown("---")

def display_trend_analysis(df):
    """Display trend analysis results with real data"""
    
    st.subheader("📈 Trend Analysis")
    
    selected_indices = st.session_state.get('temporal_indices', ['NDVI'])
    
    for index in selected_indices:
        index_data = df[df['Index'] == index].sort_values('Date')
        
        if len(index_data) >= 3:
            st.markdown(f"**{index}:**")
            
            # Perform linear regression
            x = np.arange(len(index_data))
            y = index_data['Mean'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine direction
            if slope > 0.001:
                direction = "Increasing"
                direction_icon = "📈"
            elif slope < -0.001:
                direction = "Decreasing"
                direction_icon = "📉"
            else:
                direction = "Stable"
                direction_icon = "➡️"
            
            # Determine strength based on R²
            r_squared = r_value ** 2
            if r_squared > 0.7:
                strength = "Strong"
            elif r_squared > 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            # Significance
            if p_value < 0.01:
                significance = "p < 0.01 (Highly significant)"
            elif p_value < 0.05:
                significance = "p < 0.05 (Significant)"
            else:
                significance = "p > 0.05 (Not significant)"
            
            st.markdown(f"{direction_icon} {direction} ({strength})")
            st.markdown(f"📊 Slope: {slope:.6f} per observation")
            st.markdown(f"🔬 {significance}")
            st.markdown(f"📉 R² = {r_squared:.3f}")
            
            # Detect significant changes
            if len(index_data) >= 2:
                first_val = index_data.iloc[0]['Mean']
                last_val = index_data.iloc[-1]['Mean']
                total_change = last_val - first_val
                pct_change = (total_change / first_val * 100) if first_val != 0 else 0
                
                st.markdown(f"📊 Total Change: {total_change:+.3f} ({pct_change:+.1f}%)")
            
            st.markdown("---")
    
    # Export trend analysis
    if st.button("📊 Export Trend Analysis", key="export_trends"):
        # Create export data
        export_data = []
        for index in selected_indices:
            index_data = df[df['Index'] == index].sort_values('Date')
            if len(index_data) >= 3:
                x = np.arange(len(index_data))
                y = index_data['Mean'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                export_data.append({
                    'Index': index,
                    'Slope': slope,
                    'R_squared': r_value**2,
                    'P_value': p_value,
                    'Observations': len(index_data)
                })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def generate_time_series_data():
    """Generate mock time series data"""
    
    zones = ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"]
    indices = ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"]
    
    # Generate 90 days of data
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
    
    data = []
    
    for zone in zones:
        for index in indices:
            # Base values for different indices
            base_values = {
                "NDVI": 0.7,
                "SAVI": 0.65,
                "EVI": 0.6,
                "NDWI": 0.3,
                "NDSI": 0.4
            }
            
            base_value = base_values[index]
            
            for i, date in enumerate(dates):
                # Add seasonal trend
                seasonal = 0.1 * np.sin(2 * np.pi * i / 365)
                
                # Add zone-specific variation
                zone_factor = {
                    "North Field A": 1.0,
                    "South Field B": 0.95,
                    "East Pasture C": 1.1,
                    "West Orchard D": 1.05,
                    "Central Plot E": 0.85
                }[zone]
                
                # Add noise
                noise = np.random.normal(0, 0.02)
                
                # Calculate value
                value = base_value * zone_factor + seasonal + noise
                value = max(0, min(1, value))  # Clamp to valid range
                
                # Add confidence intervals
                ci_width = 0.05
                upper_ci = min(1, value + ci_width)
                lower_ci = max(0, value - ci_width)
                
                data.append({
                    'Date': date,
                    'Zone': zone,
                    'Index': index,
                    'Value': value,
                    'Upper_CI': upper_ci,
                    'Lower_CI': lower_ci
                })
    
    return pd.DataFrame(data)

def generate_comparison_data():
    """Generate mock comparison data for box plots"""
    
    zones = ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"]
    indices = ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"]
    
    data = []
    
    for zone in zones:
        for index in indices:
            # Generate 30 random samples for each zone/index combination
            base_values = {
                "NDVI": 0.7,
                "SAVI": 0.65, 
                "EVI": 0.6,
                "NDWI": 0.3,
                "NDSI": 0.4
            }
            
            base_value = base_values[index]
            
            zone_factor = {
                "North Field A": 1.0,
                "South Field B": 0.95,
                "East Pasture C": 1.1,
                "West Orchard D": 1.05,
                "Central Plot E": 0.85
            }[zone]
            
            for _ in range(30):
                noise = np.random.normal(0, 0.05)
                value = base_value * zone_factor + noise
                value = max(0, min(1, value))
                
                data.append({
                    'Zone': zone,
                    'Index': index,
                    'Value': value
                })
    
    return pd.DataFrame(data)

def calculate_statistics():
    """Calculate statistics for each zone"""
    
    # Mock statistics
    stats = {
        "North Field A": {
            "ndvi_mean": 0.752,
            "ndvi_std": 0.045,
            "ndvi_change": 0.023,
            "trend": "Increasing",
            "trend_strength": 65.4,
            "r_squared": 0.734
        },
        "South Field B": {
            "ndvi_mean": 0.684,
            "ndvi_std": 0.038,
            "ndvi_change": -0.012,
            "trend": "Stable",
            "trend_strength": 23.1,
            "r_squared": 0.234
        },
        "East Pasture C": {
            "ndvi_mean": 0.823,
            "ndvi_std": 0.029,
            "ndvi_change": 0.045,
            "trend": "Increasing",
            "trend_strength": 87.2,
            "r_squared": 0.856
        },
        "West Orchard D": {
            "ndvi_mean": 0.715,
            "ndvi_std": 0.041,
            "ndvi_change": -0.008,
            "trend": "Stable",
            "trend_strength": 18.7,
            "r_squared": 0.187
        },
        "Central Plot E": {
            "ndvi_mean": 0.592,
            "ndvi_std": 0.052,
            "ndvi_change": -0.034,
            "trend": "Decreasing",
            "trend_strength": 78.9,
            "r_squared": 0.789
        }
    }
    
    return stats


def show_demo_temporal_analysis():
    """Display temporal analysis page with demo data"""
    
    demo_manager = st.session_state.demo_data
    scenario_name = st.session_state.get('demo_scenario', 'healthy_field')
    
    # Get demo data
    time_series = demo_manager.get_time_series(scenario_name)
    
    if not time_series:
        st.error("Failed to load demo time series data")
        return
    
    st.info(f"**Demo Scenario:** {demo_manager.get_scenario_description(scenario_name)}")
    
    # Time series chart
    st.subheader("📊 NDVI Time Series")
    
    dates = [point['date'] for point in time_series]
    ndvi_values = []
    for point in time_series:
        ndvi = point['ndvi']
        if isinstance(ndvi, np.ndarray):
            ndvi_values.append(float(np.mean(ndvi)))
        else:
            ndvi_values.append(float(ndvi))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=ndvi_values,
        mode='lines+markers',
        name='NDVI',
        line=dict(color='#66bb6a', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Healthy")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Stressed")
    
    fig.update_layout(
        title='NDVI Trend Over Time',
        xaxis_title="Date",
        yaxis_title="NDVI Value",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Trend Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean NDVI", f"{np.mean(ndvi_values):.3f}")
    
    with col2:
        st.metric("Std Dev", f"{np.std(ndvi_values):.3f}")
    
    with col3:
        trend = "Improving" if ndvi_values[-1] > ndvi_values[0] else "Declining"
        st.metric("Trend", trend)
    
    with col4:
        change = ndvi_values[-1] - ndvi_values[0]
        st.metric("Total Change", f"{change:+.3f}")
    
    # Multi-index comparison
    st.subheader("🌿 Multi-Index Comparison")
    
    # Create comparison chart with multiple indices
    fig2 = go.Figure()
    
    # NDVI
    fig2.add_trace(go.Scatter(
        x=dates,
        y=ndvi_values,
        mode='lines+markers',
        name='NDVI',
        line=dict(color='#66bb6a', width=2)
    ))
    
    # SAVI (simulated from NDVI)
    savi_values = [v * 0.9 for v in ndvi_values]
    fig2.add_trace(go.Scatter(
        x=dates,
        y=savi_values,
        mode='lines+markers',
        name='SAVI',
        line=dict(color='#42a5f5', width=2)
    ))
    
    # EVI (simulated from NDVI)
    evi_values = [v * 1.1 if v < 0.8 else v * 0.95 for v in ndvi_values]
    fig2.add_trace(go.Scatter(
        x=dates,
        y=evi_values,
        mode='lines+markers',
        name='EVI',
        line=dict(color='#ffa726', width=2)
    ))
    
    fig2.update_layout(
        title='Vegetation Indices Comparison',
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Insights
    st.subheader("💡 Key Insights")
    
    if ndvi_values[-1] > ndvi_values[0]:
        st.success("✅ Vegetation health is improving over time")
    else:
        st.warning("⚠️ Vegetation health is declining - intervention may be needed")
    
    if np.std(ndvi_values) < 0.1:
        st.info("📊 Stable vegetation conditions with low variability")
    else:
        st.warning("📊 High variability detected - monitor closely")
