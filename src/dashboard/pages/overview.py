"""
Overview page - Main dashboard with summary metrics and key insights
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import rasterio
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.error_handler import safe_page, handle_data_loading, logger
from database.db_manager import DatabaseManager
from utils.roi_calculator import ROICalculator, FarmParameters, format_currency, format_percentage, format_quantity

@safe_page
def show_page():
    """Display the overview page"""
    
    st.title("🌱 AgriFlux Overview")
    st.markdown("Real-time insights and AI-powered analytics across all monitoring zones")
    
    # Check if demo mode is active
    if st.session_state.get('demo_mode', False) and st.session_state.get('demo_data'):
        show_demo_overview()
        return
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Load real data from database
    try:
        latest_imagery = db_manager.get_latest_imagery()
        active_alerts = db_manager.get_active_alerts()
        db_stats = db_manager.get_database_stats()
        
        # Check if we have any data
        if not latest_imagery and db_stats.get('imagery_count', 0) == 0:
            st.warning("⚠️ No satellite data has been processed yet.")
            st.info("""
            **To get started:**
            1. Place Sentinel-2 satellite data in the `data/` directory
            2. Run the data processing script: `python scripts/process_satellite_data.py`
            3. Refresh this dashboard to see your data
            
            Or enable **Demo Mode** from the sidebar to explore with sample data.
            """)
            return
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        st.error("⚠️ Database connection error. Please check your database configuration.")
        st.info("Enable **Demo Mode** from the sidebar to explore the dashboard with sample data.")
        return
    
    # Quick stats row
    display_quick_stats(latest_imagery, active_alerts, db_stats)
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_health_summary_chart(latest_imagery, db_manager)
        display_recent_trends(db_manager)
    
    with col2:
        display_alert_summary(active_alerts)
        display_system_status(db_stats, latest_imagery)
    
    # Bottom section - ROI and impact metrics
    st.markdown("---")
    display_roi_metrics(latest_imagery, db_manager)
    
    # Bottom section - zone comparison
    display_zone_comparison()

def display_quick_stats(latest_imagery, active_alerts, db_stats):
    """Display quick statistics cards with real data"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Calculate health index from latest NDVI
        health_index = "N/A"
        if latest_imagery and latest_imagery.get('ndvi_path'):
            try:
                ndvi_path = latest_imagery['ndvi_path']
                if Path(ndvi_path).exists():
                    with rasterio.open(ndvi_path) as src:
                        ndvi_data = src.read(1)
                        # Calculate mean NDVI (excluding nodata values)
                        valid_ndvi = ndvi_data[ndvi_data != src.nodata]
                        if len(valid_ndvi) > 0:
                            mean_ndvi = np.mean(valid_ndvi)
                            health_index = f"{mean_ndvi:.2f}"
            except Exception as e:
                logger.warning(f"Error reading NDVI data: {e}")
        
        st.metric(
            "Health Index (NDVI)",
            health_index,
            help="Average NDVI from latest imagery"
        )
    
    with col2:
        # Real alert count from database
        alert_count = len(active_alerts)
        critical_count = sum(1 for a in active_alerts if a['severity'] == 'critical')
        
        st.metric(
            "Active Alerts",
            str(alert_count),
            delta=f"{critical_count} critical" if critical_count > 0 else "No critical",
            delta_color="inverse" if critical_count > 0 else "normal",
            help="Unacknowledged alerts from database"
        )
    
    with col3:
        # Data quality from processing metadata
        data_quality = "N/A"
        if latest_imagery:
            cloud_coverage = latest_imagery.get('cloud_coverage', 0)
            if cloud_coverage is not None:
                if cloud_coverage < 10:
                    data_quality = "Excellent"
                elif cloud_coverage < 30:
                    data_quality = "Good"
                elif cloud_coverage < 50:
                    data_quality = "Fair"
                else:
                    data_quality = "Poor"
        
        st.metric(
            "Data Quality",
            data_quality,
            delta=f"{latest_imagery.get('cloud_coverage', 0):.1f}% clouds" if latest_imagery else "",
            help="Based on cloud coverage in latest imagery"
        )
    
    with col4:
        # Total imagery records
        imagery_count = db_stats.get('imagery_count', 0)
        
        st.metric(
            "Imagery Records",
            str(imagery_count),
            help="Total processed imagery in database"
        )
    
    with col5:
        # Latest acquisition date
        latest_date = "N/A"
        if latest_imagery:
            try:
                acq_date = datetime.fromisoformat(latest_imagery['acquisition_date'])
                latest_date = acq_date.strftime("%Y-%m-%d")
            except:
                latest_date = latest_imagery.get('acquisition_date', 'N/A')
        
        st.metric(
            "Latest Data",
            latest_date,
            help="Most recent imagery acquisition date"
        )

def display_health_summary_chart(latest_imagery, db_manager):
    """Display vegetation health summary chart with real data"""
    
    st.subheader("🌱 Vegetation Health Summary")
    
    if not latest_imagery:
        st.info("No imagery data available. Please process satellite data first.")
        return
    
    # Load real NDVI and SAVI data
    try:
        ndvi_path = latest_imagery.get('ndvi_path')
        savi_path = latest_imagery.get('savi_path')
        
        if not ndvi_path or not Path(ndvi_path).exists():
            st.warning("NDVI data file not found. Please check data processing.")
            return
        
        # Read NDVI data
        with rasterio.open(ndvi_path) as src:
            ndvi_data = src.read(1)
            valid_ndvi = ndvi_data[ndvi_data != src.nodata]
            
            if len(valid_ndvi) == 0:
                st.warning("No valid NDVI data found in imagery.")
                return
            
            # Calculate statistics for different zones (simplified - using percentiles as zones)
            zones = ['Zone 1 (Best)', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5 (Worst)']
            percentiles = [90, 70, 50, 30, 10]
            ndvi_values = [np.percentile(valid_ndvi, p) for p in percentiles]
        
        # Read SAVI data if available
        savi_values = []
        if savi_path and Path(savi_path).exists():
            try:
                with rasterio.open(savi_path) as src:
                    savi_data = src.read(1)
                    valid_savi = savi_data[savi_data != src.nodata]
                    if len(valid_savi) > 0:
                        savi_values = [np.percentile(valid_savi, p) for p in percentiles]
            except Exception as e:
                logger.warning(f"Error reading SAVI data: {e}")
        
        # Create health status based on NDVI
        colors = []
        for ndvi in ndvi_values:
            if ndvi >= 0.7:
                colors.append('#4CAF50')  # Green - Healthy
            elif ndvi >= 0.5:
                colors.append('#FF9800')  # Orange - Moderate
            else:
                colors.append('#F44336')  # Red - Stressed
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='NDVI',
            x=zones,
            y=ndvi_values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in ndvi_values],
            textposition='auto',
        ))
        
        if savi_values:
            fig.add_trace(go.Scatter(
                name='SAVI',
                x=zones,
                y=savi_values,
                mode='markers+lines',
                marker=dict(size=8, color='blue'),
                line=dict(color='blue', dash='dash')
            ))
        
        fig.update_layout(
            title=f"Vegetation Indices Distribution (Acquired: {latest_imagery['acquisition_date']})",
            xaxis_title="Health Zones (Percentile-based)",
            yaxis_title="Index Value",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean NDVI", f"{np.mean(valid_ndvi):.3f}")
        with col2:
            st.metric("Std Dev", f"{np.std(valid_ndvi):.3f}")
        with col3:
            healthy_pct = (valid_ndvi >= 0.7).sum() / len(valid_ndvi) * 100
            st.metric("Healthy Area", f"{healthy_pct:.1f}%")
    
    except Exception as e:
        logger.error(f"Error displaying health summary: {e}")
        st.error(f"Error loading vegetation data: {str(e)}")

def display_recent_trends(db_manager):
    """Display recent trend analysis with real data"""
    
    st.subheader("📈 Recent Trends")
    
    try:
        # Get all imagery records
        imagery_list = db_manager.list_processed_imagery(limit=50)
        
        if not imagery_list:
            st.info("No historical data available for trend analysis.")
            return
        
        # Extract NDVI values from each imagery record
        trend_data = []
        
        for imagery in imagery_list:
            try:
                ndvi_path = imagery.get('ndvi_path')
                if ndvi_path and Path(ndvi_path).exists():
                    with rasterio.open(ndvi_path) as src:
                        ndvi_data = src.read(1)
                        valid_ndvi = ndvi_data[ndvi_data != src.nodata]
                        
                        if len(valid_ndvi) > 0:
                            mean_ndvi = np.mean(valid_ndvi)
                            acq_date = datetime.fromisoformat(imagery['acquisition_date'])
                            
                            trend_data.append({
                                'Date': acq_date,
                                'NDVI': mean_ndvi,
                                'Cloud_Coverage': imagery.get('cloud_coverage', 0)
                            })
            except Exception as e:
                logger.warning(f"Error reading imagery {imagery.get('id')}: {e}")
                continue
        
        if not trend_data:
            st.warning("Unable to extract NDVI values from imagery files.")
            return
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(trend_data)
        df = df.sort_values('Date')
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['NDVI'],
            mode='lines+markers',
            name='Mean NDVI',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>NDVI:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                     annotation_text="Healthy (0.7)", annotation_position="right")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Moderate (0.5)", annotation_position="right")
        
        fig.update_layout(
            title=f'NDVI Trend ({len(df)} observations)',
            xaxis_title="Acquisition Date",
            yaxis_title="Mean NDVI",
            height=300,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show trend statistics
        if len(df) >= 2:
            first_ndvi = df.iloc[0]['NDVI']
            last_ndvi = df.iloc[-1]['NDVI']
            change = last_ndvi - first_ndvi
            change_pct = (change / first_ndvi) * 100 if first_ndvi != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("First Observation", f"{first_ndvi:.3f}")
            with col2:
                st.metric("Latest Observation", f"{last_ndvi:.3f}")
            with col3:
                st.metric("Change", f"{change:+.3f}", delta=f"{change_pct:+.1f}%")
    
    except Exception as e:
        logger.error(f"Error displaying trends: {e}")
        st.error(f"Error loading trend data: {str(e)}")

def display_alert_summary(active_alerts):
    """Display alert summary widget with real data"""
    
    st.subheader("🚨 Active Alerts")
    
    if not active_alerts:
        st.success("✅ No active alerts - all systems normal")
        return
    
    # Display up to 5 most recent alerts
    display_alerts = active_alerts[:5]
    
    severity_icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢"
    }
    
    for alert in display_alerts:
        severity = alert.get('severity', 'medium')
        alert_type = alert.get('alert_type', 'Unknown')
        message = alert.get('message', 'No message')
        created_at = alert.get('created_at', '')
        
        # Calculate time ago
        try:
            created_time = datetime.fromisoformat(created_at)
            time_diff = datetime.now() - created_time
            if time_diff.days > 0:
                time_ago = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
            elif time_diff.seconds >= 3600:
                hours = time_diff.seconds // 3600
                time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
            else:
                minutes = time_diff.seconds // 60
                time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        except:
            time_ago = "Recently"
        
        with st.container():
            st.markdown(f"""
            {severity_icons.get(severity, '⚪')} **{alert_type.replace('_', ' ').title()}** ({severity.upper()})  
            {message}  
            🕒 {time_ago}
            """)
            st.markdown("---")
    
    if len(active_alerts) > 5:
        st.info(f"Showing 5 of {len(active_alerts)} active alerts")
    
    if st.button("View All Alerts", key="view_all_alerts"):
        st.session_state.page_selector = "🚨 Alerts & Notifications"
        st.rerun()

def display_system_status(db_stats, latest_imagery):
    """Display system status and data quality indicators"""
    
    st.subheader("📊 System Status")
    
    # Database statistics
    st.markdown("**Database:**")
    st.markdown(f"- Imagery Records: {db_stats.get('imagery_count', 0)}")
    st.markdown(f"- Total Alerts: {db_stats.get('total_alerts', 0)}")
    st.markdown(f"- Active Alerts: {db_stats.get('active_alerts', 0)}")
    st.markdown(f"- Predictions: {db_stats.get('predictions_count', 0)}")
    
    # Data range
    date_range = db_stats.get('date_range', {})
    if date_range.get('earliest') and date_range.get('latest'):
        st.markdown(f"- Date Range: {date_range['earliest']} to {date_range['latest']}")
    
    st.markdown("---")
    
    # Data quality indicators
    st.markdown("**Data Quality:**")
    
    if latest_imagery:
        cloud_coverage = latest_imagery.get('cloud_coverage', 0)
        
        # Cloud coverage indicator
        if cloud_coverage < 10:
            st.success(f"☀️ Excellent ({cloud_coverage:.1f}% clouds)")
        elif cloud_coverage < 30:
            st.info(f"🌤️ Good ({cloud_coverage:.1f}% clouds)")
        elif cloud_coverage < 50:
            st.warning(f"⛅ Fair ({cloud_coverage:.1f}% clouds)")
        else:
            st.error(f"☁️ Poor ({cloud_coverage:.1f}% clouds)")
        
        # Check if all indices are available
        indices_available = []
        for idx in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
            if latest_imagery.get(idx):
                indices_available.append(idx.replace('_path', '').upper())
        
        if indices_available:
            st.markdown(f"- Available Indices: {', '.join(indices_available)}")
    else:
        st.warning("No imagery data available")
    
    st.markdown("---")
    
    # System health
    st.markdown("**System Health:**")
    
    # Check if database is accessible
    if db_stats.get('imagery_count', 0) > 0:
        st.success("✅ Database: Online")
    else:
        st.warning("⚠️ Database: No data")
    
    # Check if latest data is recent
    if latest_imagery:
        try:
            acq_date = datetime.fromisoformat(latest_imagery['acquisition_date'])
            days_old = (datetime.now() - acq_date).days
            
            if days_old <= 7:
                st.success(f"✅ Data Freshness: {days_old} days old")
            elif days_old <= 30:
                st.info(f"ℹ️ Data Freshness: {days_old} days old")
            else:
                st.warning(f"⚠️ Data Freshness: {days_old} days old")
        except:
            pass

def display_roi_metrics(latest_imagery, db_manager):
    """Display ROI and impact metrics with cost savings calculator"""
    
    st.subheader("💰 ROI & Impact Metrics")
    
    # Initialize ROI calculator with default parameters
    if 'roi_params' not in st.session_state:
        st.session_state.roi_params = FarmParameters()
    
    # Calculate current health index
    health_index = 0.7  # Default
    mean_ndvi = 0.7  # Default
    
    if latest_imagery and latest_imagery.get('ndvi_path'):
        try:
            ndvi_path = latest_imagery['ndvi_path']
            if Path(ndvi_path).exists():
                with rasterio.open(ndvi_path) as src:
                    ndvi_data = src.read(1)
                    valid_ndvi = ndvi_data[ndvi_data != src.nodata]
                    if len(valid_ndvi) > 0:
                        mean_ndvi = np.mean(valid_ndvi)
                        health_index = mean_ndvi
        except Exception as e:
            logger.warning(f"Error reading NDVI for ROI calculation: {e}")
    
    # Get alert response rate (simplified - based on acknowledged alerts)
    try:
        total_alerts = db_manager.get_database_stats().get('total_alerts', 0)
        active_alerts = len(db_manager.get_active_alerts())
        
        if total_alerts > 0:
            acknowledged_alerts = total_alerts - active_alerts
            alert_response_rate = acknowledged_alerts / total_alerts
        else:
            alert_response_rate = 0.9  # Default high response rate
    except:
        alert_response_rate = 0.9
    
    # Calculate ROI metrics
    roi_calculator = ROICalculator(st.session_state.roi_params)
    metrics = roi_calculator.calculate_full_roi(
        health_index=health_index,
        alert_response_rate=alert_response_rate,
        irrigation_zones_used=True,
        precision_application=True,
        mean_ndvi=mean_ndvi
    )
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💵 Annual Savings",
            format_currency(metrics.total_annual_savings),
            help="Total estimated annual cost savings from all sources"
        )
    
    with col2:
        st.metric(
            "📈 ROI",
            format_percentage(metrics.roi_pct),
            help="Return on investment percentage"
        )
    
    with col3:
        st.metric(
            "⏱️ Payback Period",
            f"{metrics.payback_period_years:.1f} years" if metrics.payback_period_years < 10 else ">10 years",
            help="Time to recover AgriFlux investment"
        )
    
    with col4:
        st.metric(
            "🌱 Net Benefit",
            format_currency(metrics.net_benefit),
            help="Annual savings minus AgriFlux cost"
        )
    
    # Expandable sections for detailed breakdowns
    with st.expander("💰 Cost Savings Breakdown", expanded=False):
        display_cost_savings_detail(metrics, roi_calculator)
    
    with st.expander("♻️ Resource Efficiency Metrics", expanded=False):
        display_resource_efficiency_detail(metrics)
    
    with st.expander("🧮 ROI Calculator (Customize)", expanded=False):
        display_roi_calculator_widget(roi_calculator, health_index, alert_response_rate, mean_ndvi)
    
    with st.expander("📋 Assumptions & Methodology", expanded=False):
        display_assumptions(roi_calculator)


def display_cost_savings_detail(metrics, roi_calculator):
    """Display detailed cost savings breakdown"""
    
    st.markdown("### Yield Improvement from Early Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Yield Improvement",
            format_percentage(metrics.yield_improvement_pct),
            help="Percentage increase in crop yield"
        )
        st.metric(
            "Additional Yield",
            format_quantity(metrics.yield_improvement_kg, "kg"),
            help="Additional crop production in kg"
        )
    
    with col2:
        st.metric(
            "Revenue Increase",
            format_currency(metrics.revenue_increase),
            help="Additional revenue from improved yield"
        )
        st.metric(
            "Crop Price",
            format_currency(roi_calculator.params.crop_price_per_kg) + "/kg",
            help="Price per kilogram of crop"
        )
    
    st.markdown("---")
    st.markdown("**How it works:** Early detection of crop stress through satellite monitoring allows for timely intervention, preventing yield losses. Industry studies show 5-15% yield improvements are achievable with precision agriculture.")


def display_resource_efficiency_detail(metrics):
    """Display detailed resource efficiency metrics"""
    
    st.markdown("### Resource Savings")
    
    # Water savings
    st.markdown("#### 💧 Water Efficiency")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Water Saved",
            format_percentage(metrics.water_savings_pct)
        )
    
    with col2:
        st.metric(
            "Volume Saved",
            format_quantity(metrics.water_savings_m3, "m³")
        )
    
    with col3:
        st.metric(
            "Cost Savings",
            format_currency(metrics.water_cost_savings)
        )
    
    st.markdown("Precision irrigation zones ensure water is applied only where needed, reducing waste.")
    
    st.markdown("---")
    
    # Fertilizer reduction
    st.markdown("#### 🌾 Fertilizer Efficiency")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Reduction",
            format_percentage(metrics.fertilizer_reduction_pct)
        )
    
    with col2:
        st.metric(
            "Cost Savings",
            format_currency(metrics.fertilizer_cost_savings)
        )
    
    st.markdown("Targeted fertilizer application based on vegetation indices reduces over-application.")
    
    st.markdown("---")
    
    # Pesticide reduction
    st.markdown("#### 🐛 Pesticide Efficiency")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Reduction",
            format_percentage(metrics.pesticide_reduction_pct)
        )
    
    with col2:
        st.metric(
            "Cost Savings",
            format_currency(metrics.pesticide_cost_savings)
        )
    
    st.markdown("Precision application targets only affected areas, reducing chemical usage.")
    
    st.markdown("---")
    
    # Environmental impact
    st.markdown("#### 🌍 Environmental Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Carbon Sequestered",
            format_quantity(metrics.carbon_sequestration_tons, "tons CO₂")
        )
    
    with col2:
        st.metric(
            "Carbon Credit Value",
            format_currency(metrics.carbon_value)
        )
    
    st.markdown("Improved crop health increases biomass and carbon sequestration.")


def display_roi_calculator_widget(roi_calculator, current_health_index, current_alert_response, current_ndvi):
    """Display interactive ROI calculator widget"""
    
    st.markdown("### Customize Your ROI Calculation")
    st.markdown("Adjust parameters to see how AgriFlux impacts your specific farm:")
    
    # Farm parameters
    st.markdown("#### 🚜 Farm Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        farm_size = st.number_input(
            "Farm Size (hectares)",
            min_value=1.0,
            max_value=10000.0,
            value=st.session_state.roi_params.farm_size_ha,
            step=10.0,
            help="Total farm area in hectares"
        )
        
        crop_type = st.selectbox(
            "Crop Type",
            options=["wheat", "corn", "rice", "soybeans", "cotton", "vegetables"],
            index=0,
            help="Primary crop type"
        )
        
        baseline_yield = st.number_input(
            "Baseline Yield (kg/ha)",
            min_value=500.0,
            max_value=15000.0,
            value=st.session_state.roi_params.baseline_yield_kg_ha,
            step=100.0,
            help="Current average yield per hectare"
        )
        
        crop_price = st.number_input(
            "Crop Price ($/kg)",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.roi_params.crop_price_per_kg,
            step=0.05,
            help="Market price per kilogram"
        )
    
    with col2:
        water_cost = st.number_input(
            "Water Cost ($/m³)",
            min_value=0.01,
            max_value=5.0,
            value=st.session_state.roi_params.water_cost_per_m3,
            step=0.10,
            help="Cost per cubic meter of water"
        )
        
        fertilizer_cost = st.number_input(
            "Fertilizer Cost ($/ha)",
            min_value=10.0,
            max_value=1000.0,
            value=st.session_state.roi_params.fertilizer_cost_per_ha,
            step=10.0,
            help="Annual fertilizer cost per hectare"
        )
        
        pesticide_cost = st.number_input(
            "Pesticide Cost ($/ha)",
            min_value=10.0,
            max_value=500.0,
            value=st.session_state.roi_params.pesticide_cost_per_ha,
            step=10.0,
            help="Annual pesticide cost per hectare"
        )
        
        agriflux_cost = st.number_input(
            "AgriFlux Annual Cost ($)",
            min_value=1000.0,
            max_value=50000.0,
            value=st.session_state.roi_params.agriflux_annual_cost,
            step=500.0,
            help="Annual subscription cost for AgriFlux"
        )
    
    # Update parameters
    if st.button("Calculate Custom ROI", type="primary"):
        st.session_state.roi_params = FarmParameters(
            farm_size_ha=farm_size,
            crop_type=crop_type,
            baseline_yield_kg_ha=baseline_yield,
            crop_price_per_kg=crop_price,
            water_cost_per_m3=water_cost,
            fertilizer_cost_per_ha=fertilizer_cost,
            pesticide_cost_per_ha=pesticide_cost,
            agriflux_annual_cost=agriflux_cost
        )
        
        # Recalculate with new parameters
        custom_calculator = ROICalculator(st.session_state.roi_params)
        custom_metrics = custom_calculator.calculate_full_roi(
            health_index=current_health_index,
            alert_response_rate=current_alert_response,
            irrigation_zones_used=True,
            precision_application=True,
            mean_ndvi=current_ndvi
        )
        
        st.success("✅ ROI Calculated!")
        
        # Display custom results
        st.markdown("---")
        st.markdown("### Your Custom ROI Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Annual Savings",
                format_currency(custom_metrics.total_annual_savings)
            )
        
        with col2:
            st.metric(
                "Net Benefit",
                format_currency(custom_metrics.net_benefit)
            )
        
        with col3:
            st.metric(
                "ROI",
                format_percentage(custom_metrics.roi_pct)
            )
        
        # Break-even analysis
        st.markdown("---")
        st.markdown("### Break-Even Analysis")
        
        if custom_metrics.payback_period_years < 10:
            st.success(f"✅ **Payback Period:** {custom_metrics.payback_period_years:.1f} years")
            st.markdown(f"Your investment in AgriFlux will pay for itself in **{custom_metrics.payback_period_years:.1f} years**.")
            
            # Show cumulative savings over 5 years
            years = list(range(1, 6))
            cumulative_savings = [custom_metrics.net_benefit * year for year in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative_savings,
                mode='lines+markers',
                name='Cumulative Net Benefit',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Break-even"
            )
            
            fig.update_layout(
                title="5-Year Cumulative Net Benefit",
                xaxis_title="Year",
                yaxis_title="Cumulative Savings ($)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**5-Year Total Benefit:** {format_currency(custom_metrics.net_benefit * 5)}")
        else:
            st.warning("⚠️ Payback period exceeds 10 years. Consider adjusting parameters or farm size.")


def display_assumptions(roi_calculator):
    """Display transparent assumptions used in calculations"""
    
    st.markdown("### Calculation Assumptions")
    st.markdown("All ROI calculations are based on the following assumptions:")
    
    assumptions = roi_calculator.get_assumptions()
    
    for category, description in assumptions.items():
        st.markdown(f"**{category}:** {description}")
    
    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    - **Yield Improvements:** Based on peer-reviewed studies on precision agriculture (Gebbers & Adamchuk, 2010; Mulla, 2013)
    - **Water Savings:** USDA and FAO reports on precision irrigation efficiency
    - **Input Reduction:** Industry benchmarks from precision agriculture equipment manufacturers
    - **Carbon Pricing:** Voluntary carbon market averages (World Bank Carbon Pricing Dashboard)
    - **Conservative Approach:** All estimates use the lower end of reported ranges to provide realistic expectations
    """)
    
    st.markdown("---")
    st.info("💡 **Note:** Actual results may vary based on specific farm conditions, management practices, and environmental factors. These calculations provide estimated potential benefits.")


def display_zone_comparison():
    """Display zone comparison table"""
    
    st.subheader("🗺️ Zone Comparison")
    
    # Mock comparison data
    comparison_data = {
        'Zone': ['North Field A', 'South Field B', 'East Pasture C', 'West Orchard D', 'Central Plot E'],
        'Area (ha)': [245, 312, 189, 278, 223],
        'NDVI': [0.75, 0.68, 0.82, 0.71, 0.59],
        'SAVI': [0.71, 0.64, 0.78, 0.67, 0.55],
        'Soil Moisture (%)': [72, 65, 78, 69, 58],
        'Health Status': ['Healthy', 'Healthy', 'Excellent', 'Healthy', 'Stressed'],
        'Active Alerts': [1, 1, 0, 0, 2]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    def style_health_status(val):
        if val == 'Excellent':
            return 'background-color: #c8e6c9'
        elif val == 'Healthy':
            return 'background-color: #e8f5e8'
        elif val == 'Stressed':
            return 'background-color: #ffcdd2'
        return ''
    
    def style_alerts(val):
        if val > 1:
            return 'background-color: #ffcdd2; font-weight: bold'
        elif val == 1:
            return 'background-color: #fff3e0'
        return 'background-color: #e8f5e8'
    
    styled_df = df.style.applymap(style_health_status, subset=['Health Status']) \
                       .applymap(style_alerts, subset=['Active Alerts']) \
                       .format({
                           'NDVI': '{:.2f}',
                           'SAVI': '{:.2f}',
                           'Area (ha)': '{:.0f}'
                       })
    
    st.dataframe(styled_df, use_container_width=True)


def show_demo_overview():
    """Display overview page with demo data"""
    
    demo_manager = st.session_state.demo_data
    scenario_name = st.session_state.get('demo_scenario', 'healthy_field')
    
    # Get demo data
    scenario = demo_manager.get_scenario(scenario_name)
    time_series = demo_manager.get_time_series(scenario_name)
    alerts = demo_manager.get_active_alerts(scenario_name)
    predictions = demo_manager.get_predictions(scenario_name)
    
    if not scenario:
        st.error("Failed to load demo scenario data")
        return
    
    # Quick stats row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Calculate mean NDVI from scenario data
        ndvi_data = scenario.get('ndvi', [])
        if isinstance(ndvi_data, np.ndarray) and len(ndvi_data) > 0:
            current_ndvi = np.mean(ndvi_data)
        elif time_series and len(time_series) > 0:
            current_ndvi = time_series[-1].get('ndvi', 0.7)
        else:
            current_ndvi = 0.7
        
        st.metric(
            "Health Index (NDVI)",
            f"{current_ndvi:.2f}",
            help="Average NDVI from demo scenario"
        )
    
    with col2:
        alert_count = len(alerts)
        critical_count = sum(1 for a in alerts if a.get('severity') == 'critical')
        
        st.metric(
            "Active Alerts",
            str(alert_count),
            delta=f"{critical_count} critical" if critical_count > 0 else "No critical",
            delta_color="inverse" if critical_count > 0 else "normal"
        )
    
    with col3:
        health_status = scenario.get('health_status', 'moderate')
        quality_map = {
            'excellent': 'Excellent',
            'healthy': 'Good',
            'moderate': 'Fair',
            'stressed': 'Poor'
        }
        st.metric("Data Quality", quality_map.get(health_status, 'Good'))
    
    with col4:
        st.metric("Imagery Records", "5", help="Demo time series points")
    
    with col5:
        latest_date = time_series[-1]['date'] if time_series else "2024-09-23"
        st.metric("Latest Data", latest_date)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Health summary chart
        st.subheader("🌱 Vegetation Health Summary")
        
        if time_series:
            dates = [point['date'] for point in time_series]
            # Handle both scalar and array NDVI values
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
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                         annotation_text="Healthy (0.7)")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                         annotation_text="Moderate (0.5)")
            
            fig.update_layout(
                title='NDVI Trend Over Time',
                xaxis_title="Date",
                yaxis_title="NDVI Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent trends
        st.subheader("📈 Recent Trends")
        if time_series and len(time_series) >= 2:
            # Extract NDVI values (handle both scalar and array)
            first_ndvi_raw = time_series[0]['ndvi']
            last_ndvi_raw = time_series[-1]['ndvi']
            
            # Convert to scalar if array
            if isinstance(first_ndvi_raw, np.ndarray):
                first_ndvi = float(np.mean(first_ndvi_raw))
            else:
                first_ndvi = float(first_ndvi_raw)
            
            if isinstance(last_ndvi_raw, np.ndarray):
                last_ndvi = float(np.mean(last_ndvi_raw))
            else:
                last_ndvi = float(last_ndvi_raw)
            
            change = last_ndvi - first_ndvi
            change_pct = (change / first_ndvi) * 100 if first_ndvi != 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("First Observation", f"{first_ndvi:.3f}")
            with col_b:
                st.metric("Latest Observation", f"{last_ndvi:.3f}")
            with col_c:
                st.metric("Change", f"{change:+.3f}", delta=f"{change_pct:+.1f}%")
    
    with col2:
        # Alert summary
        st.subheader("🚨 Active Alerts")
        
        if alerts:
            severity_icons = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }
            
            for alert in alerts[:5]:
                severity = alert.get('severity', 'medium')
                alert_type = alert.get('alert_type', 'Unknown')
                message = alert.get('message', 'No message')
                
                st.markdown(f"""
                {severity_icons.get(severity, '⚪')} **{alert_type.replace('_', ' ').title()}** ({severity.upper()})  
                {message}
                """)
                st.markdown("---")
        else:
            st.success("✅ No active alerts - all systems normal")
        
        # System status
        st.subheader("📊 System Status")
        st.markdown("**Demo Mode Active**")
        st.markdown(f"- Scenario: {scenario.get('name', scenario_name)}")
        st.markdown(f"- Health Status: {health_status.title()}")
        st.markdown(f"- Time Points: {len(time_series)}")
        st.markdown(f"- Total Alerts: {len(alerts)}")
    
    # ROI Metrics
    st.markdown("---")
    st.subheader("💰 ROI & Impact Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💵 Annual Savings", "$45,000", help="Estimated from demo scenario")
    
    with col2:
        st.metric("📈 ROI", "320%", help="Return on investment")
    
    with col3:
        st.metric("⏱️ Payback Period", "0.8 years", help="Time to recover investment")
    
    with col4:
        st.metric("🌱 Net Benefit", "$40,000", help="Annual savings minus cost")
