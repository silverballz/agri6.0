"""
Field Monitoring page - Interactive maps and spatial analysis
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path
import json

# Optional import for rasterio (not available on all platforms)
try:
    import rasterio
    from rasterio.plot import reshape_as_image
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.error_handler import safe_page, handle_data_loading, logger
from ai_models.crop_health_predictor import CropHealthPredictor
from database.db_manager import DatabaseManager
from data_processing.day_wise_map_viewer import DayWiseMapViewer

@safe_page
def show_page():
    """Display the field monitoring page"""
    
    st.title("🗺️ Field Monitoring")
    st.markdown("Interactive maps showing spectral health zones and monitoring data")
    
    # Check if demo mode is active
    if st.session_state.get('demo_mode', False) and st.session_state.get('demo_data'):
        show_demo_field_monitoring()
        return
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Initialize AI predictor if not already done
    if 'crop_health_predictor' not in st.session_state:
        st.session_state.crop_health_predictor = CropHealthPredictor()
        logger.info("Initialized crop health predictor")
    
    # Load imagery data
    try:
        imagery_list = db_manager.list_processed_imagery(limit=10)
        if not imagery_list:
            st.warning("No processed imagery available. Please run data processing first.")
            st.info("Enable **Demo Mode** from the sidebar to explore with sample data.")
            return
        
        # Display imagery selector
        selected_imagery = display_imagery_selector(imagery_list)
        
        if not selected_imagery:
            st.info("Please select an imagery date to display.")
            return
        
        # Display prediction mode indicator
        display_prediction_mode_indicator()
        
        # Page controls
        display_map_controls()
        
        # Main map display
        display_interactive_map(selected_imagery, db_manager)
        
        # Display metadata
        display_imagery_metadata(selected_imagery)
        
        # Add day-wise map viewer section
        st.markdown("---")
        display_day_wise_map_section(imagery_list)
        
        # Zone details panel
        if 'selected_zone' in st.session_state and st.session_state.selected_zone:
            display_zone_details()
    
    except Exception as e:
        logger.error(f"Error loading field monitoring page: {e}")
        st.error(f"Error loading data: {str(e)}")

def display_imagery_selector(imagery_list):
    """Display selector for choosing imagery date"""
    
    st.subheader("📅 Select Imagery Date")
    
    # Create options for selector
    options = []
    for img in imagery_list:
        acq_date = img['acquisition_date']
        tile_id = img['tile_id']
        cloud_cov = img.get('cloud_coverage', 0)
        options.append(f"{acq_date} | Tile: {tile_id} | Clouds: {cloud_cov:.1f}%")
    
    if not options:
        return None
    
    selected_option = st.selectbox(
        "Available Imagery",
        options,
        key="selected_imagery_option"
    )
    
    # Get the corresponding imagery record
    selected_idx = options.index(selected_option)
    return imagery_list[selected_idx]

def display_prediction_mode_indicator():
    """Display indicator showing which prediction mode is active"""
    
    predictor = st.session_state.crop_health_predictor
    mode_info = predictor.get_model_info()
    
    if mode_info['mode'] == 'ai':
        st.success(f"🤖 AI Prediction Mode Active - Model: {mode_info['model_version']}")
    else:
        st.info(f"📊 Rule-Based Prediction Mode - Using NDVI thresholds")
    
    with st.expander("ℹ️ About Prediction Modes"):
        st.markdown("""
        **AI Mode**: Uses trained CNN model for crop health classification
        - Higher accuracy with complex patterns
        - Requires model weights file
        
        **Rule-Based Mode**: Uses NDVI threshold classification
        - Reliable fallback when AI model unavailable
        - Based on established vegetation health thresholds
        
        **Classification Categories:**
        - 🟢 Healthy: NDVI > 0.7
        - 🟡 Moderate: NDVI 0.5-0.7
        - 🟠 Stressed: NDVI 0.3-0.5
        - 🔴 Critical: NDVI < 0.3
        """)

def display_map_controls():
    """Display map control panel"""
    
    st.subheader("🎛️ Map Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        map_layer = st.selectbox(
            "Base Layer",
            ["Satellite", "Terrain", "OpenStreetMap"],
            key="map_base_layer"
        )
    
    with col2:
        vegetation_index = st.selectbox(
            "Vegetation Index",
            ["NDVI", "SAVI", "EVI", "NDWI", "NDSI"],
            key="map_vegetation_index"
        )
    
    with col3:
        show_alerts = st.checkbox(
            "Show Alerts",
            value=True,
            key="map_show_alerts"
        )
    
    with col4:
        show_predictions = st.checkbox(
            "Show AI Predictions",
            value=True,
            key="map_show_predictions",
            help="Display crop health predictions overlay"
        )
    
    # Second row of controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_sensors = st.checkbox(
            "Show Sensors",
            value=True,
            key="map_show_sensors"
        )
    
    with col2:
        color_scale = st.selectbox(
            "Color Scale",
            ["RdYlGn", "Viridis", "Spectral", "RdBu"],
            key="map_color_scale"
        )
    
    with col3:
        opacity = st.slider(
            "Layer Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="map_opacity"
        )

def display_imagery_metadata(imagery):
    """Display metadata for selected imagery"""
    
    st.subheader("📋 Imagery Metadata")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Acquisition Date", imagery['acquisition_date'])
    
    with col2:
        st.metric("Tile ID", imagery['tile_id'])
    
    with col3:
        st.metric("Cloud Coverage", f"{imagery.get('cloud_coverage', 0):.1f}%")
    
    with col4:
        try:
            processed_at = datetime.fromisoformat(imagery['processed_at'])
            st.metric("Processed", processed_at.strftime("%Y-%m-%d"))
        except:
            st.metric("Processed", imagery.get('processed_at', 'N/A'))
    
    # Show available indices
    available_indices = []
    for idx in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path', 'ndsi_path']:
        if imagery.get(idx) and Path(imagery[idx]).exists():
            available_indices.append(idx.replace('_path', '').upper())
    
    if available_indices:
        st.info(f"Available Indices: {', '.join(available_indices)}")

def display_interactive_map(imagery, db_manager):
    """Display the main interactive map with real data"""
    
    selected_index = st.session_state.get('map_vegetation_index', 'NDVI')
    st.subheader(f"📍 {selected_index} Health Map")
    
    # Get the path for selected index
    index_path_key = f"{selected_index.lower()}_path"
    index_path = imagery.get(index_path_key)
    
    if not index_path or not Path(index_path).exists():
        st.warning(f"{selected_index} data not available for this imagery.")
        return
    
    # Read the raster data to get bounds
    try:
        with rasterio.open(index_path) as src:
            bounds = src.bounds
            # Calculate center
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2
            
            # Read data for statistics
            index_data = src.read(1)
            valid_data = index_data[index_data != src.nodata]
            
            if len(valid_data) == 0:
                st.warning("No valid data in selected imagery.")
                return
    except Exception as e:
        logger.error(f"Error reading raster data: {e}")
        st.error(f"Error reading {selected_index} data: {str(e)}")
        return
    
    # Create base map
    center_lat = st.session_state.get('map_center', [center_lat, center_lon])[0]
    center_lon = st.session_state.get('map_center', [center_lat, center_lon])[1]
    
    # Create base map with proper tile configuration
    base_layer = st.session_state.get('map_base_layer', 'OpenStreetMap')
    
    if base_layer == 'OpenStreetMap':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles='OpenStreetMap'
        )
    elif base_layer == 'Satellite':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles=None
        )
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
    elif base_layer == 'Terrain':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles=None
        )
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(m)
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=st.session_state.get('map_zoom', 12),
            tiles='OpenStreetMap'
        )
    
    # Add monitoring zones with vegetation index coloring
    add_monitoring_zones(m, imagery, index_path)
    
    # Add AI prediction overlay if enabled
    if st.session_state.get('map_show_predictions', True):
        try:
            from dashboard.components.model_overlay import (
                create_prediction_overlay,
                add_prediction_legend,
                create_prediction_summary_card,
                display_prediction_statistics
            )
            
            predictor = st.session_state.crop_health_predictor
            opacity = st.session_state.get('map_opacity', 0.7)
            
            # Create prediction overlay
            prediction_overlay = create_prediction_overlay(
                index_data,
                (bounds.left, bounds.bottom, bounds.right, bounds.top),
                predictor,
                opacity=opacity
            )
            
            if prediction_overlay:
                prediction_overlay.add_to(m)
                add_prediction_legend(m)
                
                # Store predictions for statistics display
                result = predictor.predict(index_data)
                st.session_state.current_predictions = result
                st.session_state.current_index_data = index_data
        except Exception as e:
            logger.error(f"Error adding prediction overlay: {e}")
            st.warning("AI predictions unavailable for this view")
    
    # Add alerts if enabled
    if st.session_state.get('map_show_alerts', True):
        alerts = db_manager.get_active_alerts(limit=10)
        add_alert_markers(m, alerts, bounds)
    
    # Add legend
    add_map_legend(m)
    
    # Display map and capture interactions
    map_data = st_folium(
        m,
        width=None,
        height=600,
        returned_objects=["last_object_clicked_popup", "last_clicked", "bounds"]
    )
    
    # Handle map interactions
    handle_map_interactions(map_data)
    
    # Display statistics for the selected index
    display_index_statistics(valid_data, selected_index)
    
    # Display AI prediction statistics if available
    if st.session_state.get('map_show_predictions', True) and 'current_predictions' in st.session_state:
        st.markdown("---")
        try:
            from dashboard.components.model_overlay import (
                display_prediction_statistics,
                create_prediction_summary_card
            )
            
            result = st.session_state.current_predictions
            
            # Display summary card
            summary_html = create_prediction_summary_card(
                result.predictions,
                result.confidence_scores
            )
            st.markdown(summary_html, unsafe_allow_html=True)
            
            # Display detailed statistics
            display_prediction_statistics(
                result.predictions,
                result.confidence_scores
            )
        except Exception as e:
            logger.error(f"Error displaying prediction statistics: {e}")



def add_monitoring_zones(map_obj, imagery, index_path):
    """Add monitoring zones with vegetation index coloring and AI predictions using real data"""
    
    # Read the raster data
    try:
        with rasterio.open(index_path) as src:
            index_data = src.read(1)
            bounds = src.bounds
            valid_data = index_data[index_data != src.nodata]
            
            if len(valid_data) == 0:
                logger.warning("No valid data in raster")
                return
            
            # Create zones based on spatial divisions of the raster
            # Divide the area into a grid
            lat_min, lat_max = bounds.bottom, bounds.top
            lon_min, lon_max = bounds.left, bounds.right
            
            # Create 5 zones by dividing the area
            zones = []
            zone_names = ["North Zone", "South Zone", "East Zone", "West Zone", "Central Zone"]
            
            # Calculate zone boundaries
            lat_mid = (lat_min + lat_max) / 2
            lon_mid = (lon_min + lon_max) / 2
            
            zone_coords = [
                # North Zone
                [[lat_mid, lon_min], [lat_max, lon_min], [lat_max, lon_max], [lat_mid, lon_max]],
                # South Zone
                [[lat_min, lon_min], [lat_mid, lon_min], [lat_mid, lon_max], [lat_min, lon_max]],
                # East Zone
                [[lat_min, lon_mid], [lat_max, lon_mid], [lat_max, lon_max], [lat_min, lon_max]],
                # West Zone
                [[lat_min, lon_min], [lat_max, lon_min], [lat_max, lon_mid], [lat_min, lon_mid]],
                # Central Zone (smaller area in center)
                [[lat_mid - (lat_max-lat_min)*0.15, lon_mid - (lon_max-lon_min)*0.15],
                 [lat_mid + (lat_max-lat_min)*0.15, lon_mid - (lon_max-lon_min)*0.15],
                 [lat_mid + (lat_max-lat_min)*0.15, lon_mid + (lon_max-lon_min)*0.15],
                 [lat_mid - (lat_max-lat_min)*0.15, lon_mid + (lon_max-lon_min)*0.15]]
            ]
            
            # Calculate statistics for each zone
            for i, (name, coords) in enumerate(zip(zone_names, zone_coords)):
                # Calculate mean value for this zone (simplified - using percentiles)
                percentile = [90, 10, 75, 25, 50][i]
                zone_value = np.percentile(valid_data, percentile)
                
                zones.append({
                    "name": name,
                    "coordinates": coords,
                    "ndvi": float(zone_value),
                    "area": int((lat_max - lat_min) * (lon_max - lon_min) * 111 * 111 / 5),  # Rough area in hectares
                    "crop": ["Wheat", "Rice", "Cotton", "Maize", "Sugarcane"][i]
                })
    
    except Exception as e:
        logger.error(f"Error creating zones from raster: {e}")
        # Fallback to default zones
        zones = [
            {
                "name": "Zone 1",
                "coordinates": [[31.12, 75.78], [31.13, 75.78], [31.13, 75.80], [31.12, 75.80]],
                "ndvi": 0.78,
                "area": 450,
                "crop": "Wheat"
            }
        ]
    
    # Get AI predictions for zones if enabled
    show_predictions = st.session_state.get('map_show_predictions', True)
    if show_predictions and 'crop_health_predictor' in st.session_state:
        zones = add_predictions_to_zones(zones)
    
    # Add Ludhiana AOI boundary (hardcoded coordinates for free deployment)
    ludhiana_boundary = [
        [31.055, 75.765], [31.055, 75.855], [31.145, 75.855], [31.145, 75.765]
    ]
    
    folium.Polygon(
        locations=ludhiana_boundary,
        color='blue',
        weight=3,
        fillColor='lightblue',
        fillOpacity=0.1,
        popup=folium.Popup("Ludhiana 10km x 10km AOI", max_width=200),
        tooltip="Ludhiana Area of Interest"
    ).add_to(map_obj)
    
    # Color mapping based on NDVI values
    def get_zone_color(ndvi):
        if ndvi >= 0.8:
            return '#2E8B57'  # Dark green - excellent
        elif ndvi >= 0.7:
            return '#32CD32'  # Lime green - healthy
        elif ndvi >= 0.6:
            return '#FFD700'  # Gold - moderate
        elif ndvi >= 0.5:
            return '#FF8C00'  # Dark orange - stressed
        else:
            return '#DC143C'  # Crimson - critical
    
    # Add each zone as a polygon
    for zone in zones:
        color = get_zone_color(zone['ndvi'])
        
        # Create popup content with predictions if available
        popup_content = f"""
        <div style="width: 220px;">
            <h4>{zone['name']}</h4>
            <p><strong>Crop:</strong> {zone['crop']}</p>
            <p><strong>Area:</strong> {zone['area']} ha</p>
            <p><strong>NDVI:</strong> {zone['ndvi']:.2f}</p>
            <p><strong>Status:</strong> {get_health_status(zone['ndvi'])}</p>
        """
        
        # Add prediction info if available
        if 'prediction' in zone:
            pred_class = zone['prediction']['class_name']
            pred_confidence = zone['prediction']['confidence']
            pred_method = zone['prediction']['method']
            
            method_icon = "🤖" if pred_method == 'ai' else "📊"
            popup_content += f"""
            <hr>
            <p><strong>{method_icon} Prediction:</strong> {pred_class}</p>
            <p><strong>Confidence:</strong> {pred_confidence:.1%}</p>
            """
        
        popup_content += """
            <button onclick="selectZone('{zone['name']}')">View Details</button>
        </div>
        """
        
        # Create tooltip with prediction info
        tooltip_text = f"{zone['name']} - NDVI: {zone['ndvi']:.2f}"
        if 'prediction' in zone:
            tooltip_text += f" | {zone['prediction']['class_name']} ({zone['prediction']['confidence']:.0%})"
        
        folium.Polygon(
            locations=zone['coordinates'],
            color=color,
            weight=2,
            fillColor=color,
            fillOpacity=st.session_state.get('map_opacity', 0.7),
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=tooltip_text
        ).add_to(map_obj)

def add_predictions_to_zones(zones):
    """Add AI predictions to zone data"""
    
    predictor = st.session_state.crop_health_predictor
    
    for zone in zones:
        # Create NDVI array for the zone (simplified - using zone average)
        ndvi_array = np.array([zone['ndvi']])
        
        try:
            # Get prediction
            result = predictor.predict(ndvi_array)
            
            # Add prediction to zone data
            zone['prediction'] = {
                'class_idx': int(result.predictions[0]),
                'class_name': result.class_names[result.predictions[0]],
                'confidence': float(result.confidence_scores[0]),
                'method': result.method
            }
            
        except Exception as e:
            logger.warning(f"Failed to predict for zone {zone['name']}: {e}")
            # Continue without prediction for this zone
    
    return zones

def get_health_status(ndvi):
    """Get health status text based on NDVI value"""
    if ndvi >= 0.8:
        return "Excellent"
    elif ndvi >= 0.7:
        return "Healthy"
    elif ndvi >= 0.6:
        return "Moderate"
    elif ndvi >= 0.5:
        return "Stressed"
    else:
        return "Critical"

def display_index_statistics(data, index_name):
    """Display statistics for the selected vegetation index"""
    
    st.subheader(f"📊 {index_name} Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean", f"{np.mean(data):.3f}")
    
    with col2:
        st.metric("Median", f"{np.median(data):.3f}")
    
    with col3:
        st.metric("Std Dev", f"{np.std(data):.3f}")
    
    with col4:
        st.metric("Min", f"{np.min(data):.3f}")
    
    with col5:
        st.metric("Max", f"{np.max(data):.3f}")
    
    # Health distribution
    if index_name == "NDVI":
        healthy = (data >= 0.7).sum() / len(data) * 100
        moderate = ((data >= 0.5) & (data < 0.7)).sum() / len(data) * 100
        stressed = (data < 0.5).sum() / len(data) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Healthy", f"{healthy:.1f}%", help="NDVI >= 0.7")
        with col2:
            st.metric("🟡 Moderate", f"{moderate:.1f}%", help="0.5 <= NDVI < 0.7")
        with col3:
            st.metric("🔴 Stressed", f"{stressed:.1f}%", help="NDVI < 0.5")

def add_alert_markers(map_obj, alerts, bounds):
    """Add alert markers to the map using real alert data"""
    
    if not alerts:
        return
    
    # Place alerts within the map bounds
    lat_min, lat_max = bounds.bottom, bounds.top
    lon_min, lon_max = bounds.left, bounds.right
    
    # Severity colors
    severity_colors = {
        "critical": "red",
        "high": "red",
        "medium": "orange",
        "low": "yellow"
    }
    
    # Severity icons
    severity_icons = {
        "critical": "exclamation-triangle",
        "high": "exclamation-triangle",
        "medium": "exclamation-circle",
        "low": "info-circle"
    }
    
    for i, alert in enumerate(alerts[:5]):  # Limit to 5 alerts on map
        # Place alerts randomly within bounds
        alert_lat = lat_min + (lat_max - lat_min) * (0.2 + 0.6 * (i / 5))
        alert_lon = lon_min + (lon_max - lon_min) * (0.2 + 0.6 * ((i + 2) % 5 / 5))
        
        severity = alert.get('severity', 'medium')
        alert_type = alert.get('alert_type', 'Unknown').replace('_', ' ').title()
        message = alert.get('message', 'No message')
        
        # Calculate time ago
        try:
            created_at = datetime.fromisoformat(alert['created_at'])
            time_diff = datetime.now() - created_at
            if time_diff.days > 0:
                time_ago = f"{time_diff.days} day(s) ago"
            elif time_diff.seconds >= 3600:
                time_ago = f"{time_diff.seconds // 3600} hour(s) ago"
            else:
                time_ago = f"{time_diff.seconds // 60} minute(s) ago"
        except:
            time_ago = "Recently"
        
        popup_content = f"""
        <div style="width: 200px;">
            <h4>🚨 {alert_type}</h4>
            <p><strong>Severity:</strong> {severity.upper()}</p>
            <p><strong>Message:</strong> {message[:100]}</p>
            <p><strong>Time:</strong> {time_ago}</p>
        </div>
        """
        
        folium.Marker(
            location=[alert_lat, alert_lon],
            popup=folium.Popup(popup_content, max_width=220),
            tooltip=f"{alert_type} - {severity.upper()}",
            icon=folium.Icon(
                color=severity_colors.get(severity, 'blue'),
                icon=severity_icons.get(severity, 'info-circle'),
                prefix='fa'
            )
        ).add_to(map_obj)

def add_sensor_markers(map_obj):
    """Add sensor location markers to the map"""
    
    # Mock sensor locations in Ludhiana area
    sensors = [
        {
            "lat": 31.11,
            "lon": 75.81,
            "type": "Weather Station",
            "status": "Online",
            "last_reading": "5 min ago"
        },
        {
            "lat": 31.09,
            "lon": 75.83,
            "type": "Soil Moisture",
            "status": "Online", 
            "last_reading": "2 min ago"
        },
        {
            "lat": 31.12,
            "lon": 75.79,
            "type": "Leaf Wetness",
            "status": "Online",
            "last_reading": "10 min ago"
        },
        {
            "lat": 31.085,
            "lon": 75.81,
            "type": "Temperature Sensor",
            "status": "Offline",
            "last_reading": "2 hours ago"
        }
    ]
    
    # Status colors
    status_colors = {
        "Online": "green",
        "Offline": "red",
        "Warning": "orange"
    }
    
    for sensor in sensors:
        popup_content = f"""
        <div style="width: 160px;">
            <h4>📡 {sensor['type']}</h4>
            <p><strong>Status:</strong> {sensor['status']}</p>
            <p><strong>Last Reading:</strong> {sensor['last_reading']}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[sensor['lat'], sensor['lon']],
            radius=8,
            popup=folium.Popup(popup_content, max_width=180),
            tooltip=f"{sensor['type']} - {sensor['status']}",
            color=status_colors[sensor['status']],
            fillColor=status_colors[sensor['status']],
            fillOpacity=0.8
        ).add_to(map_obj)

def add_map_legend(map_obj):
    """Add legend to the map"""
    
    # Check if predictions are enabled
    show_predictions = st.session_state.get('map_show_predictions', True)
    predictor_mode = st.session_state.crop_health_predictor.get_mode() if 'crop_health_predictor' in st.session_state else 'rule_based'
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:13px; padding: 10px">
    <h4 style="margin-top: 0;">NDVI Health Scale</h4>
    <p style="margin: 3px 0;"><i class="fa fa-square" style="color:#2E8B57"></i> Excellent (0.8+)</p>
    <p style="margin: 3px 0;"><i class="fa fa-square" style="color:#32CD32"></i> Healthy (0.7-0.8)</p>
    <p style="margin: 3px 0;"><i class="fa fa-square" style="color:#FFD700"></i> Moderate (0.6-0.7)</p>
    <p style="margin: 3px 0;"><i class="fa fa-square" style="color:#FF8C00"></i> Stressed (0.5-0.6)</p>
    <p style="margin: 3px 0;"><i class="fa fa-square" style="color:#DC143C"></i> Critical (&lt;0.5)</p>
    '''
    
    if show_predictions:
        mode_icon = "🤖" if predictor_mode == 'ai' else "📊"
        mode_text = "AI Model" if predictor_mode == 'ai' else "Rule-Based"
        legend_html += f'''
        <hr style="margin: 8px 0;">
        <p style="margin: 3px 0; font-size: 12px;"><strong>Predictions:</strong> {mode_icon} {mode_text}</p>
        '''
    
    legend_html += '''
    </div>
    '''
    
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def handle_map_interactions(map_data):
    """Handle map click interactions"""
    
    if map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        
        # Display clicked coordinates
        st.sidebar.markdown(f"**Last Clicked:**")
        st.sidebar.markdown(f"Lat: {clicked_lat:.4f}")
        st.sidebar.markdown(f"Lng: {clicked_lng:.4f}")
        
        # Check if click was on a zone (simplified logic)
        # In production, this would use proper spatial queries
        zone_name = get_zone_from_coordinates(clicked_lat, clicked_lng)
        if zone_name:
            st.session_state.selected_zone = zone_name

def get_zone_from_coordinates(lat, lng):
    """Get zone name from coordinates (simplified)"""
    
    # Mock zone detection based on Ludhiana coordinate ranges
    if 31.12 <= lat <= 31.13 and 75.78 <= lng <= 75.80:
        return "Ludhiana North Farm"
    elif 31.10 <= lat <= 31.11 and 75.82 <= lng <= 75.84:
        return "Pakhowal Road Fields"
    elif 31.08 <= lat <= 31.09 and 75.80 <= lng <= 75.82:
        return "Sidhwan Bet Area"
    elif 31.11 <= lat <= 31.12 and 75.76 <= lng <= 75.78:
        return "Raikot Agricultural Zone"
    elif 31.09 <= lat <= 31.10 and 75.84 <= lng <= 75.86:
        return "Khanna District Fields"
    
    return None

def display_day_wise_map_section(imagery_list):
    """Display day-wise map viewer section"""
    
    if len(imagery_list) < 2:
        st.info("Day-wise map comparison requires at least 2 imagery dates. More satellite data will enable this feature.")
        return
    
    # Add expandable section for day-wise map viewer
    with st.expander("🗺️ Day-Wise Map View", expanded=False):
        try:
            # Initialize day-wise map viewer
            map_viewer = DayWiseMapViewer(imagery_list)
            
            # Render the viewer
            map_viewer.render_temporal_map_viewer()
        
        except Exception as e:
            logger.error(f"Error rendering day-wise map viewer: {e}")
            st.error(f"Error loading day-wise map viewer: {str(e)}")
            st.info("This feature requires folium and rasterio libraries.")

def display_zone_details():
    """Display detailed information for selected zone"""
    
    st.subheader(f"📋 Zone Details: {st.session_state.selected_zone}")
    
    # Mock detailed zone data
    zone_details = get_zone_details(st.session_state.selected_zone)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.markdown(f"- **Area:** {zone_details['area']} hectares")
        st.markdown(f"- **Crop Type:** {zone_details['crop']}")
        st.markdown(f"- **Planting Date:** {zone_details['planting_date']}")
        st.markdown(f"- **Expected Harvest:** {zone_details['harvest_date']}")
        
        st.markdown("**Current Conditions:**")
        st.markdown(f"- **NDVI:** {zone_details['ndvi']:.2f}")
        st.markdown(f"- **SAVI:** {zone_details['savi']:.2f}")
        st.markdown(f"- **Soil Moisture:** {zone_details['soil_moisture']}%")
        st.markdown(f"- **Temperature:** {zone_details['temperature']}°C")
    
    with col2:
        st.markdown("**Health Assessment:**")
        health_status = get_health_status(zone_details['ndvi'])
        status_color = {
            "Excellent": "🟢",
            "Healthy": "🟢", 
            "Moderate": "🟡",
            "Stressed": "🟠",
            "Critical": "🔴"
        }
        st.markdown(f"- **Status:** {status_color.get(health_status, '⚪')} {health_status}")
        
        # Add AI prediction if available
        if 'crop_health_predictor' in st.session_state:
            predictor = st.session_state.crop_health_predictor
            try:
                ndvi_array = np.array([zone_details['ndvi']])
                result = predictor.predict(ndvi_array)
                
                pred_class = result.class_names[result.predictions[0]]
                pred_confidence = result.confidence_scores[0]
                method_icon = "🤖" if result.method == 'ai' else "📊"
                
                st.markdown(f"- **{method_icon} Prediction:** {pred_class} ({pred_confidence:.1%} confidence)")
            except Exception as e:
                logger.warning(f"Failed to generate prediction: {e}")
        
        st.markdown("**Active Alerts:**")
        for alert in zone_details['alerts']:
            st.markdown(f"- 🚨 {alert}")
        
        if not zone_details['alerts']:
            st.markdown("- ✅ No active alerts")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Trends", key=f"trends_{st.session_state.selected_zone}"):
            st.session_state.page_selector = "📈 Temporal Analysis"
            st.rerun()
    
    with col2:
        if st.button("📤 Export Data", key=f"export_{st.session_state.selected_zone}"):
            st.session_state.page_selector = "📤 Data Export"
            st.rerun()
    
    with col3:
        if st.button("❌ Clear Selection", key=f"clear_{st.session_state.selected_zone}"):
            st.session_state.selected_zone = None
            st.rerun()

def get_zone_details(zone_name):
    """Get detailed information for a specific zone"""
    
    # Mock detailed data for Ludhiana agricultural zones
    zone_data = {
        "Ludhiana North Farm": {
            "area": 450,
            "crop": "Wheat",
            "planting_date": "2024-11-15",
            "harvest_date": "2025-04-20",
            "ndvi": 0.78,
            "savi": 0.74,
            "soil_moisture": 68,
            "temperature": 22.3,
            "alerts": []
        },
        "Pakhowal Road Fields": {
            "area": 380,
            "crop": "Rice", 
            "planting_date": "2024-06-20",
            "harvest_date": "2024-11-10",
            "ndvi": 0.72,
            "savi": 0.68,
            "soil_moisture": 75,
            "temperature": 24.1,
            "alerts": ["Pest risk elevated"]
        },
        "Sidhwan Bet Area": {
            "area": 320,
            "crop": "Sugarcane",
            "planting_date": "2024-03-01",
            "harvest_date": "2025-02-15",
            "ndvi": 0.85,
            "savi": 0.81,
            "soil_moisture": 82,
            "temperature": 25.8,
            "alerts": []
        },
        "Raikot Agricultural Zone": {
            "area": 520,
            "crop": "Cotton",
            "planting_date": "2024-05-15",
            "harvest_date": "2024-12-01",
            "ndvi": 0.69,
            "savi": 0.65,
            "soil_moisture": 62,
            "temperature": 26.2,
            "alerts": ["Irrigation needed in sector 2"]
        },
        "Khanna District Fields": {
            "area": 290,
            "crop": "Maize",
            "planting_date": "2024-07-01",
            "harvest_date": "2024-11-30",
            "ndvi": 0.63,
            "savi": 0.59,
            "soil_moisture": 58,
            "temperature": 25.5,
            "alerts": ["Vegetation stress detected", "Low soil moisture"]
        }
    }
    
    return zone_data.get(zone_name, {})


def show_demo_field_monitoring():
    """Display field monitoring page with demo data"""
    
    demo_manager = st.session_state.demo_data
    scenario_name = st.session_state.get('demo_scenario', 'healthy_field')
    
    # Get demo data
    scenario = demo_manager.get_scenario(scenario_name)
    
    if not scenario:
        st.error("Failed to load demo scenario data")
        return
    
    # Display scenario info
    st.info(f"**Demo Scenario:** {scenario.get('name', scenario_name)} - {scenario.get('description', '')}")
    
    # Create map centered on demo location
    st.subheader("📍 Field Location Map")
    
    # Create a folium map
    m = folium.Map(
        location=[31.1, 75.81],  # Ludhiana, Punjab
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add a marker for the demo field
    folium.Marker(
        [31.1, 75.81],
        popup=f"Demo Field: {scenario.get('name', 'Field')}",
        tooltip="Click for details",
        icon=folium.Icon(color='green', icon='leaf', prefix='fa')
    ).add_to(m)
    
    # Add a circle to show field area
    folium.Circle(
        location=[31.1, 75.81],
        radius=500,  # 500 meters
        color='green',
        fill=True,
        fillColor='green',
        fillOpacity=0.2,
        popup=f"Health Status: {scenario.get('health_status', 'healthy').title()}"
    ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)
    
    # Health metrics
    st.subheader("🌱 Vegetation Health Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate stats from NDVI data
    ndvi_data = scenario.get('ndvi', [])
    if isinstance(ndvi_data, np.ndarray) and len(ndvi_data) > 0:
        mean_ndvi = np.mean(ndvi_data)
        std_ndvi = np.std(ndvi_data)
        healthy_pct = (ndvi_data >= 0.6).sum() / len(ndvi_data) * 100
    else:
        mean_ndvi = 0.7
        std_ndvi = 0.1
        healthy_pct = 75
    
    with col1:
        st.metric("Mean NDVI", f"{mean_ndvi:.3f}")
    
    with col2:
        st.metric("Std Dev", f"{std_ndvi:.3f}")
    
    with col3:
        st.metric("Healthy Area", f"{healthy_pct:.1f}%")
    
    with col4:
        st.metric("Field Size", "50 hectares")
    
    # Health zones visualization
    st.subheader("🗺️ Health Zones Distribution")
    
    # Create a simple bar chart showing health distribution
    import plotly.graph_objects as go
    
    zones = ['Excellent\n(0.8-1.0)', 'Healthy\n(0.6-0.8)', 'Moderate\n(0.4-0.6)', 'Stressed\n(<0.4)']
    
    # Generate values based on scenario health status
    health_status = scenario.get('health_status', 'healthy')
    if health_status == 'excellent':
        values = [60, 30, 8, 2]
        colors = ['#2e7d32', '#66bb6a', '#ffa726', '#ef5350']
    elif health_status == 'healthy':
        values = [40, 45, 12, 3]
        colors = ['#2e7d32', '#66bb6a', '#ffa726', '#ef5350']
    elif health_status == 'moderate':
        values = [15, 40, 35, 10]
        colors = ['#2e7d32', '#66bb6a', '#ffa726', '#ef5350']
    else:  # stressed
        values = [5, 20, 35, 40]
        colors = ['#2e7d32', '#66bb6a', '#ffa726', '#ef5350']
    
    fig = go.Figure(data=[
        go.Bar(
            x=zones,
            y=values,
            marker_color=colors,
            text=[f'{v}%' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Field Area by Health Zone',
        xaxis_title='Health Zone',
        yaxis_title='Percentage of Field (%)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Predictions
    st.subheader("🤖 AI Health Predictions")
    
    predictions = demo_manager.get_predictions(scenario_name)
    
    if predictions:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Assessment:**")
            st.markdown(f"- Health Class: **{predictions.get('health_class', 'Healthy').title()}**")
            st.markdown(f"- Confidence: **{predictions.get('confidence', 0.85):.1%}**")
            st.markdown(f"- Risk Level: **{predictions.get('risk_level', 'Low').title()}**")
        
        with col2:
            st.markdown("**7-Day Forecast:**")
            forecast = predictions.get('forecast', {})
            st.markdown(f"- Predicted NDVI: **{forecast.get('ndvi', 0.75):.3f}**")
            st.markdown(f"- Trend: **{forecast.get('trend', 'Stable').title()}**")
            st.markdown(f"- Action Needed: **{forecast.get('action', 'Monitor').title()}**")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = scenario.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.success("✅ No immediate actions required. Continue regular monitoring.")
