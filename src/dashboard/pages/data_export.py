"""
Data Export page - Export functionality for analysis results and reports
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64
import json
import os
from typing import Dict, List, Any

# Import report generation and data export components
try:
    from ..report_generator import ReportGenerator, prepare_report_data
    from ..report_scheduler import ReportScheduler, ScheduledReport, create_daily_report_schedule
    from ..data_exporter import DataExporter, generate_mock_vegetation_indices_data, generate_mock_sensor_data, generate_mock_monitoring_zones
except ImportError:
    # Fallback for development
    ReportGenerator = None
    ReportScheduler = None
    DataExporter = None

def show_page():
    """Display the data export page"""
    
    st.title("📤 Data Export & Reports")
    st.markdown("Export analysis results, generate reports, and manage automated reporting")
    
    # Create tabs for different export functions
    tab1, tab2, tab3 = st.tabs(["📊 Data Export", "📋 Report Generation", "⏰ Scheduled Reports"])
    
    with tab1:
        display_data_export_tab()
    
    with tab2:
        display_report_generation_tab()
    
    with tab3:
        display_scheduled_reports_tab()


def display_data_export_tab():
    """Display the data export functionality"""
    
    st.subheader("Data Export")
    st.markdown("Export raw data and analysis results for further processing")
    
    # Export options
    display_export_options()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_data_preview()
        display_export_formats()
    
    with col2:
        display_export_history()
        display_quick_exports()

def display_export_options():
    """Display export configuration options"""
    
    st.subheader("🎛️ Export Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        export_type = st.selectbox(
            "Data Type",
            ["Vegetation Indices", "Satellite Images", "Sensor Data", "Alert Reports", "Zone Boundaries"],
            key="export_data_type"
        )
    
    with col2:
        selected_zones = st.multiselect(
            "Zones",
            ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"],
            default=["North Field A"],
            key="export_zones"
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="export_date_range"
        )
    
    with col4:
        export_format = st.selectbox(
            "Format",
            get_available_formats(export_type),
            key="export_format"
        )

def get_available_formats(data_type):
    """Get available export formats for data type"""
    
    format_map = {
        "Vegetation Indices": ["CSV", "Excel", "JSON", "Parquet"],
        "Satellite Images": ["GeoTIFF", "PNG", "JPEG", "NetCDF"],
        "Sensor Data": ["CSV", "Excel", "JSON", "HDF5"],
        "Alert Reports": ["PDF", "Excel", "CSV", "HTML"],
        "Zone Boundaries": ["GeoJSON", "Shapefile", "KML", "CSV"]
    }
    
    return format_map.get(data_type, ["CSV"])

def display_data_preview():
    """Display preview of data to be exported"""
    
    st.subheader("👁️ Data Preview")
    
    export_type = st.session_state.get('export_data_type', 'Vegetation Indices')
    selected_zones = st.session_state.get('export_zones', ['North Field A'])
    
    if not selected_zones:
        st.warning("Please select at least one zone to preview data.")
        return
    
    # Generate preview data based on export type
    preview_data = generate_preview_data(export_type, selected_zones)
    
    if preview_data is not None:
        st.markdown(f"**Preview: {export_type}**")
        st.markdown(f"Zones: {', '.join(selected_zones)}")
        st.markdown(f"Records: {len(preview_data):,}")
        
        # Show data preview
        if isinstance(preview_data, pd.DataFrame):
            st.dataframe(preview_data.head(10), use_container_width=True)
            
            if len(preview_data) > 10:
                st.caption(f"Showing first 10 of {len(preview_data):,} records")
        else:
            st.text(str(preview_data)[:500] + "..." if len(str(preview_data)) > 500 else str(preview_data))
    
    else:
        st.info("No data available for the selected criteria.")

def display_export_formats():
    """Display export format options and controls"""
    
    st.subheader("📋 Export Options")
    
    export_type = st.session_state.get('export_data_type', 'Vegetation Indices')
    export_format = st.session_state.get('export_format', 'CSV')
    
    # Format-specific options
    if export_format in ['CSV', 'Excel']:
        col1, col2 = st.columns(2)
        
        with col1:
            include_headers = st.checkbox(
                "Include Headers",
                value=True,
                key="export_include_headers"
            )
            
            include_metadata = st.checkbox(
                "Include Metadata",
                value=True,
                key="export_include_metadata"
            )
        
        with col2:
            if export_format == 'CSV':
                delimiter = st.selectbox(
                    "Delimiter",
                    [",", ";", "\t", "|"],
                    key="export_delimiter"
                )
            
            date_format = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY", "ISO 8601"],
                key="export_date_format"
            )
    
    elif export_format in ['GeoTIFF', 'PNG', 'JPEG']:
        col1, col2 = st.columns(2)
        
        with col1:
            image_resolution = st.selectbox(
                "Resolution",
                ["10m", "20m", "60m", "Original"],
                key="export_resolution"
            )
        
        with col2:
            color_scale = st.selectbox(
                "Color Scale",
                ["RdYlGn", "Viridis", "Spectral", "Grayscale"],
                key="export_color_scale"
            )
    
    elif export_format == 'PDF':
        col1, col2 = st.columns(2)
        
        with col1:
            include_maps = st.checkbox(
                "Include Maps",
                value=True,
                key="export_include_maps"
            )
            
            include_charts = st.checkbox(
                "Include Charts",
                value=True,
                key="export_include_charts"
            )
        
        with col2:
            report_template = st.selectbox(
                "Report Template",
                ["Standard", "Executive Summary", "Technical Report", "Field Report"],
                key="export_report_template"
            )
    
    # Compression options
    if export_format in ['CSV', 'JSON', 'GeoJSON']:
        compress_output = st.checkbox(
            "Compress Output (ZIP)",
            value=False,
            key="export_compress"
        )
    
    # Export button
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📤 Export Data", key="export_data_button", type="primary"):
            perform_export()
    
    with col2:
        if st.button("👁️ Preview Export", key="preview_export_button"):
            preview_export()
    
    with col3:
        if st.button("📧 Email Export", key="email_export_button"):
            email_export()
    
    with col4:
        if st.button("📦 Batch Export", key="batch_export_button"):
            show_batch_export_dialog()

def display_export_history():
    """Display export history"""
    
    st.subheader("📜 Export History")
    
    # Mock export history
    history_data = [
        {
            "Date": "2024-12-09 14:30",
            "Type": "Vegetation Indices",
            "Format": "CSV",
            "Zones": "North Field A, South Field B",
            "Size": "2.3 MB",
            "Status": "Completed"
        },
        {
            "Date": "2024-12-09 10:15",
            "Type": "Alert Reports",
            "Format": "PDF",
            "Zones": "All Zones",
            "Size": "5.7 MB",
            "Status": "Completed"
        },
        {
            "Date": "2024-12-08 16:45",
            "Type": "Satellite Images",
            "Format": "GeoTIFF",
            "Zones": "Central Plot E",
            "Size": "45.2 MB",
            "Status": "Completed"
        },
        {
            "Date": "2024-12-08 09:20",
            "Type": "Sensor Data",
            "Format": "Excel",
            "Zones": "East Pasture C",
            "Size": "1.8 MB",
            "Status": "Failed"
        }
    ]
    
    for item in history_data:
        status_color = {
            "Completed": "🟢",
            "In Progress": "🟡",
            "Failed": "🔴"
        }
        
        with st.expander(f"{status_color[item['Status']]} {item['Date']} - {item['Type']}"):
            st.markdown(f"**Format:** {item['Format']}")
            st.markdown(f"**Zones:** {item['Zones']}")
            st.markdown(f"**Size:** {item['Size']}")
            st.markdown(f"**Status:** {item['Status']}")
            
            if item['Status'] == 'Completed':
                col1, col2 = st.columns(2)
                with col1:
                    st.button("📥 Download", key=f"download_{item['Date']}")
                with col2:
                    st.button("🔄 Re-export", key=f"reexport_{item['Date']}")

def display_quick_exports():
    """Display quick export options"""
    
    st.subheader("⚡ Quick Exports")
    
    quick_options = [
        {
            "name": "Current Week NDVI",
            "description": "NDVI data for all zones, last 7 days",
            "format": "CSV"
        },
        {
            "name": "Alert Summary Report",
            "description": "PDF report of all active alerts",
            "format": "PDF"
        },
        {
            "name": "Zone Boundaries",
            "description": "GeoJSON file with all monitoring zones",
            "format": "GeoJSON"
        },
        {
            "name": "Sensor Status Report",
            "description": "Excel report of sensor network status",
            "format": "Excel"
        }
    ]
    
    for option in quick_options:
        with st.container():
            st.markdown(f"**{option['name']}**")
            st.caption(option['description'])
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Export", key=f"quick_{option['name']}"):
                    st.success(f"Exporting {option['name']}...")
            
            st.markdown("---")

def generate_preview_data(export_type, selected_zones):
    """Generate preview data based on export type"""
    
    if export_type == "Vegetation Indices":
        return generate_vegetation_indices_data(selected_zones)
    elif export_type == "Sensor Data":
        return generate_sensor_data(selected_zones)
    elif export_type == "Alert Reports":
        return generate_alert_data(selected_zones)
    elif export_type == "Zone Boundaries":
        return generate_zone_boundaries(selected_zones)
    elif export_type == "Satellite Images":
        return f"Satellite image data for {len(selected_zones)} zones"
    
    return None

def generate_vegetation_indices_data(zones):
    """Generate mock vegetation indices data"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']
    
    data = []
    
    for zone in zones:
        for date in dates:
            for index in indices:
                # Generate mock values
                base_values = {'NDVI': 0.7, 'SAVI': 0.65, 'EVI': 0.6, 'NDWI': 0.3, 'NDSI': 0.4}
                base_value = base_values[index]
                
                # Add some variation
                value = base_value + np.random.normal(0, 0.05)
                value = max(0, min(1, value))
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Zone': zone,
                    'Index': index,
                    'Value': round(value, 4),
                    'Quality_Flag': np.random.choice(['Good', 'Fair', 'Poor'], p=[0.8, 0.15, 0.05]),
                    'Cloud_Coverage': round(np.random.uniform(0, 30), 1)
                })
    
    return pd.DataFrame(data)

def generate_sensor_data(zones):
    """Generate mock sensor data"""
    
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='H'
    )
    
    data = []
    
    for zone in zones:
        for timestamp in timestamps:
            data.append({
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Zone': zone,
                'Soil_Moisture': round(np.random.uniform(40, 80), 1),
                'Air_Temperature': round(np.random.uniform(15, 35), 1),
                'Humidity': round(np.random.uniform(30, 90), 1),
                'Leaf_Wetness': round(np.random.uniform(0, 100), 1),
                'Solar_Radiation': round(np.random.uniform(0, 1200), 1),
                'Wind_Speed': round(np.random.uniform(0, 25), 1)
            })
    
    return pd.DataFrame(data)

def generate_alert_data(zones):
    """Generate mock alert data"""
    
    alert_types = ['Vegetation Stress', 'Pest Risk', 'Disease Risk', 'Environmental']
    severities = ['High', 'Medium', 'Low']
    
    data = []
    
    for i in range(20):  # Generate 20 mock alerts
        data.append({
            'Alert_ID': f"ALT_{i+1:04d}",
            'Timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S'),
            'Zone': np.random.choice(zones),
            'Type': np.random.choice(alert_types),
            'Severity': np.random.choice(severities),
            'Description': f"Mock alert description {i+1}",
            'Status': np.random.choice(['Active', 'Acknowledged', 'Resolved']),
            'Response_Time_Hours': round(np.random.uniform(0.5, 12), 1)
        })
    
    return pd.DataFrame(data)

def generate_zone_boundaries(zones):
    """Generate mock zone boundary data"""
    
    data = []
    
    for zone in zones:
        # Mock coordinates (would be real GeoJSON in production)
        data.append({
            'Zone_Name': zone,
            'Area_Hectares': round(np.random.uniform(150, 350), 1),
            'Crop_Type': np.random.choice(['Corn', 'Soybeans', 'Wheat', 'Barley', 'Apple Trees']),
            'Planting_Date': '2024-04-15',
            'Coordinates': f"Mock coordinates for {zone}"
        })
    
    return pd.DataFrame(data)

def perform_export():
    """Perform the actual export operation using the DataExporter"""
    
    export_type = st.session_state.get('export_data_type', 'Vegetation Indices')
    export_format = st.session_state.get('export_format', 'CSV')
    selected_zones = st.session_state.get('export_zones', [])
    date_range = st.session_state.get('export_date_range', (datetime.now() - timedelta(days=30), datetime.now()))
    
    if not selected_zones:
        st.error("Please select at least one zone to export.")
        return
    
    if not DataExporter:
        st.error("Data export system not available. Please check installation.")
        return
    
    try:
        with st.spinner(f"Exporting {export_type} in {export_format} format..."):
            exporter = DataExporter()
            
            # Convert zone names to IDs (mock mapping)
            zone_ids = [f"zone_{i+1}" for i, zone in enumerate(["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"]) if zone in selected_zones]
            
            start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range[0], date_range[1])
            
            if export_type == "Vegetation Indices":
                if export_format in ['CSV', 'Excel', 'JSON']:
                    # Generate mock data
                    mock_data = generate_mock_vegetation_indices_data(zone_ids, days=30)
                    
                    if export_format == 'CSV':
                        filepath = exporter.export_vegetation_indices_csv(
                            mock_data, zones=zone_ids, start_date=start_date, end_date=end_date
                        )
                    else:
                        # For Excel/JSON, export as CSV first then convert
                        filepath = exporter.export_vegetation_indices_csv(
                            mock_data, zones=zone_ids, start_date=start_date, end_date=end_date
                        )
                    
                    # Read file and create download
                    with open(filepath, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=file_data,
                        file_name=os.path.basename(filepath),
                        mime="text/csv" if export_format == 'CSV' else "application/octet-stream"
                    )
            
            elif export_type == "Satellite Images":
                if export_format == 'GeoTIFF':
                    mock_images = [{"acquisition_date": datetime.now().isoformat()}]
                    filepath = exporter.export_satellite_imagery_geotiff(
                        mock_images, index_type='NDVI', zones=zone_ids
                    )
                    
                    with open(filepath, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="📥 Download GeoTIFF",
                        data=file_data,
                        file_name=os.path.basename(filepath),
                        mime="image/tiff"
                    )
                else:
                    st.error(f"Format {export_format} not supported for satellite images")
                    return
            
            elif export_type == "Zone Boundaries":
                if export_format == 'GeoJSON':
                    mock_zones = generate_mock_monitoring_zones(len(zone_ids))
                    filepath = exporter.export_monitoring_zones_geojson(mock_zones)
                    
                    with open(filepath, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="📥 Download GeoJSON",
                        data=file_data,
                        file_name=os.path.basename(filepath),
                        mime="application/geo+json"
                    )
                else:
                    st.error(f"Format {export_format} not supported for zone boundaries")
                    return
            
            elif export_type == "Sensor Data":
                if export_format in ['CSV', 'Excel']:
                    mock_data = generate_mock_sensor_data(zone_ids, hours=168)  # 1 week
                    filepath = exporter.export_sensor_data_csv(
                        mock_data, zones=zone_ids, start_date=start_date, end_date=end_date
                    )
                    
                    with open(filepath, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=file_data,
                        file_name=os.path.basename(filepath),
                        mime="text/csv"
                    )
                else:
                    st.error(f"Format {export_format} not supported for sensor data")
                    return
            
            elif export_type == "Alert Reports":
                if export_format in ['CSV', 'Excel']:
                    # Generate mock alert data
                    mock_alerts = []
                    for zone_id in zone_ids:
                        for i in range(np.random.randint(1, 5)):
                            mock_alerts.append({
                                'id': f"alert_{zone_id}_{i}",
                                'zone_id': zone_id,
                                'severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                                'message': f"Mock alert {i+1} for {zone_id}",
                                'status': np.random.choice(['active', 'resolved']),
                                'created_at': datetime.now() - timedelta(days=np.random.randint(0, 30))
                            })
                    
                    filepath = exporter.export_alerts_csv(mock_alerts, zones=zone_ids)
                    
                    with open(filepath, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=file_data,
                        file_name=os.path.basename(filepath),
                        mime="text/csv"
                    )
                else:
                    st.error(f"Format {export_format} not supported for alert reports")
                    return
            
            st.success(f"✅ {export_type} exported successfully in {export_format} format!")
            
    except Exception as e:
        st.error(f"Error during export: {str(e)}")
        st.exception(e)

def preview_export():
    """Preview the export without downloading"""
    
    st.success("Export preview generated! Check the Data Preview section above.")

def email_export():
    """Email the export to specified recipients"""
    
    with st.form("email_export_form"):
        st.subheader("📧 Email Export")
        
        recipients = st.text_input(
            "Recipients (comma-separated emails)",
            placeholder="user@example.com, manager@example.com"
        )
        
        subject = st.text_input(
            "Subject",
            value=f"Agricultural Monitoring Export - {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        message = st.text_area(
            "Message",
            value="Please find the attached agricultural monitoring data export.",
            height=100
        )
        
        if st.form_submit_button("📧 Send Email"):
            if recipients:
                st.success(f"Export emailed to: {recipients}")
            else:
                st.error("Please enter at least one recipient email address.")


def display_report_generation_tab():
    """Display the report generation functionality"""
    
    st.subheader("PDF Report Generation")
    st.markdown("Generate comprehensive PDF reports with field condition summaries")
    
    if not ReportGenerator:
        st.error("Report generation system not available. Please check installation.")
        return
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Report Configuration**")
        
        report_template = st.selectbox(
            "Report Template",
            ["standard", "executive", "technical", "field"],
            format_func=lambda x: {
                "standard": "Standard Report - Comprehensive analysis",
                "executive": "Executive Summary - High-level overview", 
                "technical": "Technical Report - Detailed technical analysis",
                "field": "Field Report - Operational field summary"
            }[x],
            key="report_template"
        )
        
        selected_zones = st.multiselect(
            "Zones to Include",
            ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"],
            default=["North Field A", "South Field B"],
            key="report_zones"
        )
        
        report_period = st.selectbox(
            "Report Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Custom range"],
            key="report_period"
        )
        
        if report_period == "Custom range":
            date_range = st.date_input(
                "Custom Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key="report_date_range"
            )
        else:
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            days = days_map[report_period]
            date_range = (datetime.now() - timedelta(days=days), datetime.now())
    
    with col2:
        st.markdown("**Report Options**")
        
        include_maps = st.checkbox("Include Maps", value=True, key="report_include_maps")
        include_charts = st.checkbox("Include Charts", value=True, key="report_include_charts")
        include_recommendations = st.checkbox("Include Recommendations", value=True, key="report_include_recommendations")
        
        st.markdown("**Delivery Options**")
        
        email_recipients = st.text_input(
            "Email Recipients (optional)",
            placeholder="manager@farm.com, agronomist@farm.com",
            key="report_email_recipients"
        )
    
    # Generate report button
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Generate Report", type="primary", key="generate_report_btn"):
            generate_pdf_report(report_template, selected_zones, date_range, 
                              include_maps, include_charts, include_recommendations)
    
    with col2:
        if st.button("👁️ Preview Report", key="preview_report_btn"):
            preview_pdf_report(report_template, selected_zones, date_range)
    
    with col3:
        if st.button("📧 Generate & Email", key="generate_email_report_btn"):
            if email_recipients:
                generate_and_email_report(report_template, selected_zones, date_range, 
                                        email_recipients, include_maps, include_charts, include_recommendations)
            else:
                st.error("Please enter email recipients")
    
    # Recent reports
    st.markdown("---")
    display_recent_reports()


def display_scheduled_reports_tab():
    """Display the scheduled reports functionality"""
    
    st.subheader("Automated Report Scheduling")
    st.markdown("Set up automated report generation and delivery")
    
    if not ReportScheduler:
        st.error("Report scheduling system not available. Please check installation.")
        return
    
    # Initialize scheduler in session state
    if 'report_scheduler' not in st.session_state:
        st.session_state.report_scheduler = ReportScheduler()
        # Set mock data provider
        st.session_state.report_scheduler.set_data_provider(get_mock_report_data)
    
    scheduler = st.session_state.report_scheduler
    
    # Scheduler status
    status = scheduler.get_schedule_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Scheduler Status", "Running" if status['running'] else "Stopped")
    
    with col2:
        st.metric("Total Reports", status['total_reports'])
    
    with col3:
        st.metric("Enabled Reports", status['enabled_reports'])
    
    with col4:
        if st.button("▶️ Start Scheduler" if not status['running'] else "⏸️ Stop Scheduler"):
            if status['running']:
                scheduler.stop_scheduler()
                st.success("Scheduler stopped")
            else:
                scheduler.start_scheduler()
                st.success("Scheduler started")
            st.rerun()
    
    st.markdown("---")
    
    # Create new scheduled report
    with st.expander("➕ Create New Scheduled Report", expanded=False):
        create_scheduled_report_form(scheduler)
    
    # List existing scheduled reports
    st.subheader("Existing Scheduled Reports")
    
    scheduled_reports = scheduler.get_scheduled_reports()
    
    if not scheduled_reports:
        st.info("No scheduled reports configured. Create one using the form above.")
    else:
        for report in scheduled_reports:
            display_scheduled_report_card(scheduler, report)


def create_scheduled_report_form(scheduler: 'ReportScheduler'):
    """Create form for new scheduled report"""
    
    with st.form("new_scheduled_report"):
        col1, col2 = st.columns(2)
        
        with col1:
            report_name = st.text_input("Report Name", placeholder="Daily Field Summary")
            
            template = st.selectbox(
                "Template",
                ["standard", "executive", "technical", "field"],
                format_func=lambda x: x.title()
            )
            
            frequency = st.selectbox(
                "Frequency",
                ["daily", "weekly", "monthly"]
            )
            
            time = st.time_input("Generation Time", value=datetime.strptime("08:00", "%H:%M").time())
        
        with col2:
            zones = st.multiselect(
                "Zones to Include",
                ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5"],
                default=["zone_1"]
            )
            
            recipients = st.text_area(
                "Email Recipients (one per line)",
                placeholder="manager@farm.com\nagronomist@farm.com"
            )
            
            enabled = st.checkbox("Enable Report", value=True)
        
        if st.form_submit_button("Create Scheduled Report"):
            if report_name and recipients:
                recipient_list = [email.strip() for email in recipients.split('\n') if email.strip()]
                
                scheduled_report = ScheduledReport(
                    id=f"{frequency}_{report_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    name=report_name,
                    template=template,
                    frequency=frequency,
                    time=time.strftime("%H:%M"),
                    recipients=recipient_list,
                    zones=zones,
                    enabled=enabled
                )
                
                scheduler.add_scheduled_report(scheduled_report)
                st.success(f"Scheduled report '{report_name}' created successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")


def display_scheduled_report_card(scheduler: 'ReportScheduler', report: 'ScheduledReport'):
    """Display a card for a scheduled report"""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            status_icon = "🟢" if report.enabled else "🔴"
            st.markdown(f"**{status_icon} {report.name}**")
            st.caption(f"Template: {report.template.title()} | Frequency: {report.frequency.title()}")
        
        with col2:
            st.markdown(f"**Next Run:** {report.next_generation.strftime('%Y-%m-%d %H:%M') if report.next_generation else 'Not scheduled'}")
            st.caption(f"Recipients: {len(report.recipients)}")
        
        with col3:
            st.markdown(f"**Last Generated:** {report.last_generated.strftime('%Y-%m-%d %H:%M') if report.last_generated else 'Never'}")
            st.caption(f"Zones: {len(report.zones)}")
        
        with col4:
            col4a, col4b = st.columns(2)
            
            with col4a:
                if st.button("▶️", key=f"run_{report.id}", help="Run now"):
                    scheduler.generate_report_now(report.id)
                    st.success("Report generated!")
            
            with col4b:
                if st.button("🗑️", key=f"delete_{report.id}", help="Delete"):
                    scheduler.remove_scheduled_report(report.id)
                    st.success("Report deleted!")
                    st.rerun()
        
        st.markdown("---")


def generate_pdf_report(template: str, zones: List[str], date_range: tuple, 
                       include_maps: bool, include_charts: bool, include_recommendations: bool):
    """Generate a PDF report"""
    
    try:
        with st.spinner("Generating PDF report..."):
            # Get mock data
            mock_zones, mock_alerts, mock_time_series = get_mock_report_data()
            
            # Filter data for selected zones
            if zones:
                mock_zones = [z for z in mock_zones if z.get('name') in zones]
            
            # Prepare report data
            start_date, end_date = date_range
            report_data = prepare_report_data(mock_zones, mock_alerts, mock_time_series, start_date, end_date)
            
            # Generate PDF
            generator = ReportGenerator()
            pdf_content = generator.generate_report(template, report_data)
            
            # Create download button
            filename = f"agricultural_report_{template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_content,
                file_name=filename,
                mime="application/pdf"
            )
            
            st.success("Report generated successfully!")
    
    except Exception as e:
        st.error(f"Error generating report: {e}")


def preview_pdf_report(template: str, zones: List[str], date_range: tuple):
    """Preview report content"""
    
    st.info("Report preview functionality would show a summary of the report content here.")
    
    # Mock preview content
    st.markdown("**Report Preview:**")
    st.markdown(f"- Template: {template.title()}")
    st.markdown(f"- Zones: {', '.join(zones) if zones else 'All zones'}")
    st.markdown(f"- Period: {date_range[0]} to {date_range[1]}")
    st.markdown("- Sections: Executive Summary, Zone Analysis, Vegetation Trends, Alerts, Recommendations")


def generate_and_email_report(template: str, zones: List[str], date_range: tuple, 
                            recipients: str, include_maps: bool, include_charts: bool, include_recommendations: bool):
    """Generate and email a PDF report"""
    
    try:
        with st.spinner("Generating and emailing report..."):
            # Generate report (same as above)
            mock_zones, mock_alerts, mock_time_series = get_mock_report_data()
            
            if zones:
                mock_zones = [z for z in mock_zones if z.get('name') in zones]
            
            start_date, end_date = date_range
            report_data = prepare_report_data(mock_zones, mock_alerts, mock_time_series, start_date, end_date)
            
            generator = ReportGenerator()
            pdf_content = generator.generate_report(template, report_data)
            
            # Mock email sending
            recipient_list = [email.strip() for email in recipients.split(',') if email.strip()]
            
            st.success(f"Report generated and emailed to: {', '.join(recipient_list)}")
            st.info("Note: In production, this would send actual emails with the PDF attachment.")
    
    except Exception as e:
        st.error(f"Error generating and emailing report: {e}")


def display_recent_reports():
    """Display recent report generation history"""
    
    st.subheader("Recent Reports")
    
    # Mock recent reports data
    recent_reports = [
        {
            "name": "Weekly Executive Summary",
            "template": "executive",
            "generated": "2024-12-09 08:00:00",
            "zones": ["North Field A", "South Field B"],
            "status": "Completed",
            "size": "2.1 MB"
        },
        {
            "name": "Monthly Technical Report",
            "template": "technical", 
            "generated": "2024-12-08 10:30:00",
            "zones": ["All Zones"],
            "status": "Completed",
            "size": "5.7 MB"
        },
        {
            "name": "Field Operations Report",
            "template": "field",
            "generated": "2024-12-07 14:15:00",
            "zones": ["East Pasture C"],
            "status": "Failed",
            "size": "N/A"
        }
    ]
    
    for report in recent_reports:
        status_color = {"Completed": "🟢", "Failed": "🔴", "In Progress": "🟡"}
        
        with st.expander(f"{status_color[report['status']]} {report['name']} - {report['generated']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Template:** {report['template'].title()}")
                st.markdown(f"**Zones:** {', '.join(report['zones']) if isinstance(report['zones'], list) else report['zones']}")
            
            with col2:
                st.markdown(f"**Status:** {report['status']}")
                st.markdown(f"**Size:** {report['size']}")
            
            if report['status'] == 'Completed':
                col1, col2 = st.columns(2)
                with col1:
                    st.button("📥 Download", key=f"download_recent_{report['name']}")
                with col2:
                    st.button("📧 Email", key=f"email_recent_{report['name']}")


def get_mock_report_data():
    """Get mock data for report generation"""
    
    # Mock zones data
    zones = [
        {
            "id": "zone_1",
            "name": "North Field A",
            "crop_type": "Corn",
            "area": 125.5,
            "planting_date": "2024-04-15",
            "days_since_planting": 238,
            "recommendations": ["Monitor for pest activity", "Consider irrigation adjustment"]
        },
        {
            "id": "zone_2", 
            "name": "South Field B",
            "crop_type": "Soybeans",
            "area": 98.2,
            "planting_date": "2024-05-01",
            "days_since_planting": 222,
            "recommendations": ["Maintain current management practices"]
        }
    ]
    
    # Mock alerts data
    alerts = [
        {
            "id": "alert_1",
            "zone_id": "zone_1",
            "zone": "North Field A",
            "severity": "high",
            "message": "Vegetation stress detected in northern section",
            "status": "active"
        },
        {
            "id": "alert_2",
            "zone_id": "zone_2", 
            "zone": "South Field B",
            "severity": "medium",
            "message": "Soil moisture levels below optimal range",
            "status": "active"
        }
    ]
    
    # Mock time series data
    time_series = []
    for zone_id in ["zone_1", "zone_2"]:
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            time_series.append({
                "zone_id": zone_id,
                "index_type": "NDVI",
                "timestamp": date,
                "mean_value": 0.7 + np.random.normal(0, 0.05)
            })
    
    return zones, alerts, time_series


def show_batch_export_dialog():
    """Show batch export configuration dialog"""
    
    if not DataExporter:
        st.error("Data export system not available.")
        return
    
    st.subheader("📦 Batch Export Configuration")
    st.markdown("Configure multiple datasets for batch export")
    
    # Initialize batch export configuration in session state
    if 'batch_export_configs' not in st.session_state:
        st.session_state.batch_export_configs = []
    
    # Add new export configuration
    with st.expander("➕ Add Export Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_export_type = st.selectbox(
                "Data Type",
                ["vegetation_indices", "satellite_imagery", "monitoring_zones", "sensor_data", "alerts"],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="batch_export_type"
            )
            
            batch_zones = st.multiselect(
                "Zones",
                ["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"],
                key="batch_zones"
            )
        
        with col2:
            if batch_export_type == "vegetation_indices":
                batch_indices = st.multiselect(
                    "Indices",
                    ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"],
                    default=["NDVI"],
                    key="batch_indices"
                )
            elif batch_export_type == "satellite_imagery":
                batch_index_type = st.selectbox(
                    "Index Type",
                    ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"],
                    key="batch_index_type"
                )
                batch_resolution = st.selectbox(
                    "Resolution",
                    ["10m", "20m", "60m"],
                    key="batch_resolution"
                )
            elif batch_export_type == "sensor_data":
                batch_sensor_types = st.multiselect(
                    "Sensor Types",
                    ["soil_moisture", "air_temperature", "humidity", "leaf_wetness"],
                    key="batch_sensor_types"
                )
            elif batch_export_type == "alerts":
                batch_severities = st.multiselect(
                    "Severities",
                    ["low", "medium", "high", "critical"],
                    key="batch_severities"
                )
        
        batch_date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="batch_date_range"
        )
        
        if st.button("➕ Add to Batch", key="add_to_batch"):
            config = {
                "type": batch_export_type,
                "zones": [f"zone_{i+1}" for i, zone in enumerate(["North Field A", "South Field B", "East Pasture C", "West Orchard D", "Central Plot E"]) if zone in batch_zones],
                "start_date": batch_date_range[0] if isinstance(batch_date_range, tuple) else batch_date_range[0],
                "end_date": batch_date_range[1] if isinstance(batch_date_range, tuple) else batch_date_range[1]
            }
            
            # Add type-specific configurations
            if batch_export_type == "vegetation_indices":
                config["indices"] = st.session_state.get("batch_indices", ["NDVI"])
            elif batch_export_type == "satellite_imagery":
                config["index_type"] = st.session_state.get("batch_index_type", "NDVI")
                config["resolution"] = st.session_state.get("batch_resolution", "10m")
            elif batch_export_type == "sensor_data":
                config["sensor_types"] = st.session_state.get("batch_sensor_types", [])
            elif batch_export_type == "alerts":
                config["severities"] = st.session_state.get("batch_severities", [])
            
            # Generate mock data for the configuration
            if batch_export_type == "vegetation_indices":
                config["data"] = generate_mock_vegetation_indices_data(config["zones"], days=30)
            elif batch_export_type == "satellite_imagery":
                config["data"] = [{"acquisition_date": datetime.now().isoformat()}]
            elif batch_export_type == "monitoring_zones":
                config["data"] = generate_mock_monitoring_zones(len(config["zones"]))
            elif batch_export_type == "sensor_data":
                config["data"] = generate_mock_sensor_data(config["zones"], hours=168)
            elif batch_export_type == "alerts":
                mock_alerts = []
                for zone_id in config["zones"]:
                    for i in range(np.random.randint(1, 5)):
                        mock_alerts.append({
                            'id': f"alert_{zone_id}_{i}",
                            'zone_id': zone_id,
                            'severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                            'message': f"Mock alert {i+1} for {zone_id}",
                            'status': np.random.choice(['active', 'resolved']),
                            'created_at': datetime.now() - timedelta(days=np.random.randint(0, 30))
                        })
                config["data"] = mock_alerts
            
            st.session_state.batch_export_configs.append(config)
            st.success(f"Added {batch_export_type.replace('_', ' ').title()} to batch export")
            st.rerun()
    
    # Display current batch configuration
    if st.session_state.batch_export_configs:
        st.subheader("Current Batch Configuration")
        
        for i, config in enumerate(st.session_state.batch_export_configs):
            with st.expander(f"{config['type'].replace('_', ' ').title()} - {len(config.get('zones', []))} zones"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Type:** {config['type'].replace('_', ' ').title()}")
                    st.markdown(f"**Zones:** {len(config.get('zones', []))}")
                    st.markdown(f"**Date Range:** {config.get('start_date')} to {config.get('end_date')}")
                    
                    # Show type-specific info
                    if config['type'] == 'vegetation_indices':
                        st.markdown(f"**Indices:** {', '.join(config.get('indices', []))}")
                    elif config['type'] == 'satellite_imagery':
                        st.markdown(f"**Index:** {config.get('index_type', 'N/A')}")
                        st.markdown(f"**Resolution:** {config.get('resolution', 'N/A')}")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_batch_{i}"):
                        st.session_state.batch_export_configs.pop(i)
                        st.rerun()
        
        # Batch export options
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            compress_batch = st.checkbox("Compress as ZIP", value=True, key="compress_batch")
        
        with col2:
            if st.button("📦 Execute Batch Export", type="primary", key="execute_batch"):
                execute_batch_export(st.session_state.batch_export_configs, compress_batch)
    
    else:
        st.info("No export configurations added yet. Use the form above to add datasets to the batch.")


def execute_batch_export(configs: List[Dict[str, Any]], compress: bool = True):
    """Execute the batch export"""
    
    try:
        with st.spinner("Executing batch export..."):
            exporter = DataExporter()
            
            # Execute batch export
            result_path = exporter.create_batch_export(configs, compress=compress)
            
            # Read the result file
            with open(result_path, 'rb') as f:
                file_data = f.read()
            
            filename = os.path.basename(result_path)
            mime_type = "application/zip" if compress else "application/octet-stream"
            
            st.download_button(
                label="📥 Download Batch Export",
                data=file_data,
                file_name=filename,
                mime=mime_type
            )
            
            st.success(f"✅ Batch export completed! {len(configs)} datasets exported.")
            
            # Clear the batch configuration
            st.session_state.batch_export_configs = []
            
    except Exception as e:
        st.error(f"Error during batch export: {str(e)}")
        st.exception(e)