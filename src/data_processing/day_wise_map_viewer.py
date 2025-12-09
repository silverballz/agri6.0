"""
Day-Wise Map Viewer - Interactive temporal map visualization with multiple view modes
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Optional imports
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class DayWiseMapViewer:
    """Interactive day-wise map viewer with temporal navigation"""
    
    def __init__(self, imagery_list: List[Dict]):
        """
        Initialize the DayWiseMapViewer
        
        Args:
            imagery_list: List of imagery dictionaries with acquisition dates and file paths
        """
        self.imagery_list = sorted(imagery_list, key=lambda x: x['acquisition_date'])
        self.dates = [datetime.fromisoformat(img['acquisition_date']) for img in self.imagery_list]
        
        if not FOLIUM_AVAILABLE:
            logger.warning("folium not available - map visualization will be limited")
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available - raster processing will be limited")
    
    def render_temporal_map_viewer(self):
        """Render interactive temporal map viewer with date slider"""
        
        if not self.imagery_list:
            st.warning("No imagery available for map visualization.")
            return
        
        st.subheader("🗺️ Day-Wise Map View")
        
        # Add explanation
        with st.expander("📖 How to use the map viewer", expanded=False):
            st.markdown("""
            **Map View Modes:**
            
            - **Single Date**: Navigate through dates with a slider to see how vegetation changes over time
            - **Side-by-Side Comparison**: Compare two dates side-by-side to see differences
            - **Difference Map**: Visualize pixel-level changes between two dates (red=decline, green=improvement)
            - **Animation**: Watch a time-lapse of vegetation changes (coming soon)
            
            **Tips:**
            - Use the layer selector to switch between different vegetation indices
            - Hover over the map to see values at specific locations
            - Use the date slider for quick navigation
            """)
        
        # View mode selector
        view_mode = st.radio(
            "View Mode:",
            ["Single Date", "Side-by-Side Comparison", "Difference Map", "Animation"],
            horizontal=True,
            key="map_view_mode"
        )
        
        if view_mode == "Single Date":
            self._render_single_date_view()
        elif view_mode == "Side-by-Side Comparison":
            self._render_side_by_side_view()
        elif view_mode == "Difference Map":
            self._render_difference_map()
        else:
            self._render_animation_view()
    
    def _render_single_date_view(self):
        """Render single date view with slider"""
        
        st.markdown("### 📅 Single Date View")
        
        # Date slider
        selected_idx = st.slider(
            "Select Date:",
            min_value=0,
            max_value=len(self.dates) - 1,
            value=len(self.dates) - 1,
            format="Date %d",
            key="single_date_slider"
        )
        
        selected_date = self.dates[selected_idx]
        imagery = self.imagery_list[selected_idx]
        
        # Display date info
        st.info(f"📅 Viewing: **{selected_date.strftime('%B %d, %Y')}** ({selected_idx + 1} of {len(self.dates)})")
        
        # Layer selector
        layer_type = st.selectbox(
            "Map Layer:",
            ["NDVI", "SAVI", "EVI", "NDWI", "True Color RGB"],
            key="single_layer"
        )
        
        # Display map
        self._display_map(imagery, layer_type)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(selected_idx == 0), key="prev_btn"):
                st.session_state.single_date_slider = selected_idx - 1
                st.rerun()
        
        with col3:
            if st.button("Next ➡️", disabled=(selected_idx == len(self.dates) - 1), key="next_btn"):
                st.session_state.single_date_slider = selected_idx + 1
                st.rerun()
    
    def _render_side_by_side_view(self):
        """Render side-by-side comparison of two dates"""
        
        st.markdown("### 🔄 Side-by-Side Comparison")
        
        st.markdown("Compare vegetation indices across two dates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date1_idx = st.selectbox(
                "Date 1:",
                range(len(self.dates)),
                index=0,
                format_func=lambda i: self.dates[i].strftime('%b %d, %Y'),
                key="date1_idx"
            )
            
            layer1 = st.selectbox(
                "Layer 1:",
                ["NDVI", "SAVI", "EVI", "NDWI", "True Color RGB"],
                key="layer1"
            )
            
            st.markdown(f"### 📅 {self.dates[date1_idx].strftime('%B %d, %Y')}")
            self._display_map(self.imagery_list[date1_idx], layer1, map_key="map1")
        
        with col2:
            date2_idx = st.selectbox(
                "Date 2:",
                range(len(self.dates)),
                index=len(self.dates) - 1,
                format_func=lambda i: self.dates[i].strftime('%b %d, %Y'),
                key="date2_idx"
            )
            
            layer2 = st.selectbox(
                "Layer 2:",
                ["NDVI", "SAVI", "EVI", "NDWI", "True Color RGB"],
                key="layer2"
            )
            
            st.markdown(f"### 📅 {self.dates[date2_idx].strftime('%B %d, %Y')}")
            self._display_map(self.imagery_list[date2_idx], layer2, map_key="map2")
        
        # Calculate and display change statistics
        if date1_idx != date2_idx:
            self._display_change_statistics(
                self.imagery_list[date1_idx],
                self.imagery_list[date2_idx],
                self.dates[date1_idx],
                self.dates[date2_idx]
            )
    
    def _render_difference_map(self):
        """Render difference map visualization"""
        
        st.markdown("### 📊 Difference Map")
        
        st.markdown("Visualize pixel-level changes between two dates")
        
        # Date selectors
        col1, col2 = st.columns(2)
        
        with col1:
            date1_idx = st.selectbox(
                "From Date:",
                range(len(self.dates)),
                index=0,
                format_func=lambda i: self.dates[i].strftime('%b %d, %Y'),
                key="diff_date1_idx"
            )
        
        with col2:
            date2_idx = st.selectbox(
                "To Date:",
                range(len(self.dates)),
                index=len(self.dates) - 1,
                format_func=lambda i: self.dates[i].strftime('%b %d, %Y'),
                key="diff_date2_idx"
            )
        
        # Index selector
        index_type = st.selectbox(
            "Vegetation Index:",
            ["NDVI", "SAVI", "EVI", "NDWI"],
            key="diff_index"
        )
        
        if date1_idx == date2_idx:
            st.warning("Please select two different dates for comparison.")
            return
        
        # Calculate and display difference map
        diff_map, stats = self._calculate_difference_map(
            self.imagery_list[date1_idx],
            self.imagery_list[date2_idx],
            index_type
        )
        
        if diff_map is not None:
            # Display difference map
            st.markdown(f"**Change from {self.dates[date1_idx].strftime('%b %d, %Y')} to {self.dates[date2_idx].strftime('%b %d, %Y')}**")
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "% Improved",
                    f"{stats['pct_improved']:.1f}%",
                    help="Percentage of area showing improvement"
                )
            
            with col2:
                st.metric(
                    "% Stable",
                    f"{stats['pct_stable']:.1f}%",
                    help="Percentage of area with minimal change"
                )
            
            with col3:
                st.metric(
                    "% Declined",
                    f"{stats['pct_declined']:.1f}%",
                    help="Percentage of area showing decline"
                )
            
            with col4:
                st.metric(
                    "Mean Change",
                    f"{stats['mean_change']:+.3f}",
                    help="Average change in index value"
                )
            
            # Interpretation
            interpretation = self._interpret_difference_map(stats)
            if "improvement" in interpretation.lower():
                st.success(f"✅ {interpretation}")
            elif "decline" in interpretation.lower():
                st.warning(f"⚠️ {interpretation}")
            else:
                st.info(f"ℹ️ {interpretation}")
            
            # Note about visualization
            st.info("📊 Difference map visualization: Green = improvement, Yellow = stable, Red = decline")
        else:
            st.error("Unable to calculate difference map. Check that index files exist for both dates.")
    
    def _render_animation_view(self):
        """Render animation/time-lapse view"""
        
        st.markdown("### 🎬 Animation View")
        
        st.info("🚧 Animation view coming soon! This will show a time-lapse of vegetation changes.")
        
        # Placeholder for animation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            frame_delay = st.slider(
                "Frame Delay (ms):",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                key="frame_delay",
                disabled=True
            )
        
        with col2:
            loop_animation = st.checkbox(
                "Loop Animation",
                value=True,
                key="loop_animation",
                disabled=True
            )
        
        with col3:
            st.button("▶️ Play", disabled=True, key="play_btn")
            st.button("⏸️ Pause", disabled=True, key="pause_btn")
            st.button("⏹️ Stop", disabled=True, key="stop_btn")
    
    def _display_map(self, imagery: Dict, layer_type: str, map_key: str = "map"):
        """Display map for given imagery and layer type"""
        
        if not FOLIUM_AVAILABLE or not RASTERIO_AVAILABLE:
            st.warning("Map visualization requires folium and rasterio libraries.")
            return
        
        # Get the appropriate file path
        layer_path = self._get_layer_path(imagery, layer_type)
        
        if not layer_path:
            st.warning(f"No {layer_type} data available for this date.")
            return
        
        try:
            # Read raster data
            with rasterio.open(layer_path) as src:
                # Get bounds
                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                center_lon = (bounds.left + bounds.right) / 2
                
                # Create folium map
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=13,
                    tiles='OpenStreetMap'
                )
                
                # Add layer info
                folium.Marker(
                    [center_lat, center_lon],
                    popup=f"{layer_type} - {imagery['acquisition_date']}",
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)
                
                # Display map
                st_folium(m, width=700, height=500, key=map_key)
        
        except Exception as e:
            logger.error(f"Error displaying map: {e}")
            st.error(f"Error loading map: {str(e)}")
    
    def _get_layer_path(self, imagery: Dict, layer_type: str) -> Optional[str]:
        """Get file path for specified layer type"""
        
        layer_map = {
            "NDVI": "ndvi_path",
            "SAVI": "savi_path",
            "EVI": "evi_path",
            "NDWI": "ndwi_path",
            "True Color RGB": "rgb_path"
        }
        
        path_key = layer_map.get(layer_type)
        if path_key:
            return imagery.get(path_key)
        
        return None
    
    def _calculate_difference_map(
        self, 
        imagery1: Dict, 
        imagery2: Dict, 
        index_type: str
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Calculate difference map between two dates"""
        
        if not RASTERIO_AVAILABLE:
            return None, {}
        
        # Get paths
        path1 = self._get_layer_path(imagery1, index_type)
        path2 = self._get_layer_path(imagery2, index_type)
        
        if not path1 or not path2:
            return None, {}
        
        try:
            # Read both rasters
            with rasterio.open(path1) as src1:
                data1 = src1.read(1)
                nodata1 = src1.nodata
            
            with rasterio.open(path2) as src2:
                data2 = src2.read(1)
                nodata2 = src2.nodata
            
            # Create masks for valid data
            mask1 = data1 != nodata1 if nodata1 is not None else np.ones_like(data1, dtype=bool)
            mask2 = data2 != nodata2 if nodata2 is not None else np.ones_like(data2, dtype=bool)
            valid_mask = mask1 & mask2
            
            # Calculate difference
            diff_map = np.where(valid_mask, data2 - data1, np.nan)
            
            # Calculate statistics
            valid_diff = diff_map[valid_mask]
            
            if len(valid_diff) > 0:
                # Define thresholds for change
                improvement_threshold = 0.05
                decline_threshold = -0.05
                
                improved = np.sum(valid_diff > improvement_threshold)
                declined = np.sum(valid_diff < decline_threshold)
                stable = np.sum((valid_diff >= decline_threshold) & (valid_diff <= improvement_threshold))
                total = len(valid_diff)
                
                stats = {
                    'pct_improved': (improved / total) * 100,
                    'pct_stable': (stable / total) * 100,
                    'pct_declined': (declined / total) * 100,
                    'mean_change': np.mean(valid_diff),
                    'max_increase': np.max(valid_diff),
                    'max_decrease': np.min(valid_diff)
                }
            else:
                stats = {
                    'pct_improved': 0,
                    'pct_stable': 0,
                    'pct_declined': 0,
                    'mean_change': 0,
                    'max_increase': 0,
                    'max_decrease': 0
                }
            
            return diff_map, stats
        
        except Exception as e:
            logger.error(f"Error calculating difference map: {e}")
            return None, {}
    
    def _interpret_difference_map(self, stats: Dict) -> str:
        """Interpret the meaning of difference map statistics"""
        
        if stats['pct_improved'] > 60:
            return f"Significant overall improvement - {stats['pct_improved']:.1f}% of area showing positive growth"
        elif stats['pct_declined'] > 60:
            return f"Significant overall decline - {stats['pct_declined']:.1f}% of area showing negative change. Investigation recommended."
        elif stats['pct_stable'] > 70:
            return f"Mostly stable conditions - {stats['pct_stable']:.1f}% of area with minimal change"
        elif stats['pct_improved'] > stats['pct_declined']:
            return f"Moderate improvement - {stats['pct_improved']:.1f}% improved vs {stats['pct_declined']:.1f}% declined"
        elif stats['pct_declined'] > stats['pct_improved']:
            return f"Moderate decline - {stats['pct_declined']:.1f}% declined vs {stats['pct_improved']:.1f}% improved. Monitor closely."
        else:
            return "Mixed conditions - roughly equal areas of improvement and decline"
    
    def _display_change_statistics(
        self, 
        imagery1: Dict, 
        imagery2: Dict, 
        date1: datetime, 
        date2: datetime
    ):
        """Display change statistics between two dates"""
        
        st.markdown("### 📊 Change Statistics")
        
        # Calculate statistics for each index
        indices = ["NDVI", "SAVI", "EVI", "NDWI"]
        
        for index in indices:
            diff_map, stats = self._calculate_difference_map(imagery1, imagery2, index)
            
            if diff_map is not None and stats:
                with st.expander(f"{index} Changes", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("% Improved", f"{stats['pct_improved']:.1f}%")
                    
                    with col2:
                        st.metric("% Stable", f"{stats['pct_stable']:.1f}%")
                    
                    with col3:
                        st.metric("% Declined", f"{stats['pct_declined']:.1f}%")
                    
                    st.info(self._interpret_difference_map(stats))
