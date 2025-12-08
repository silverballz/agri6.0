"""
Before/After Comparison Widget

Provides interactive comparison tools for temporal imagery analysis.
Includes side-by-side view and slider-based comparison.

Part of the USP features for AgriFlux dashboard.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional, Dict
import logging

# Optional import for rasterio (not available on all platforms)
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComparisonWidget:
    """
    Widget for comparing two dates of imagery.
    """
    
    def __init__(self):
        """Initialize comparison widget."""
        self.colorscales = {
            'NDVI': 'RdYlGn',
            'SAVI': 'RdYlGn',
            'EVI': 'RdYlGn',
            'NDWI': 'Blues',
            'NDSI': 'YlOrBr'
        }
    
    def load_image_data(self, geotiff_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load image data from GeoTIFF.
        
        Args:
            geotiff_path: Path to GeoTIFF file
            
        Returns:
            Tuple of (data_array, metadata_dict)
        """
        try:
            with rasterio.open(geotiff_path) as src:
                data = src.read(1)
                metadata = {
                    'bounds': src.bounds,
                    'crs': src.crs,
                    'shape': data.shape
                }
                return data, metadata
        except Exception as e:
            logger.error(f"Failed to load {geotiff_path}: {e}")
            raise
    
    def create_side_by_side_view(self,
                                 before_path: str,
                                 after_path: str,
                                 before_date: str,
                                 after_date: str,
                                 index_name: str = "NDVI") -> go.Figure:
        """
        Create side-by-side comparison view.
        
        Args:
            before_path: Path to earlier date GeoTIFF
            after_path: Path to later date GeoTIFF
            before_date: Date label for before image
            after_date: Date label for after image
            index_name: Name of vegetation index
            
        Returns:
            Plotly figure object
        """
        # Load data
        before_data, _ = self.load_image_data(before_path)
        after_data, _ = self.load_image_data(after_path)
        
        # Get colorscale
        colorscale = self.colorscales.get(index_name, 'Viridis')
        
        # Determine value range
        valid_before = before_data[np.isfinite(before_data)]
        valid_after = after_data[np.isfinite(after_data)]
        
        if len(valid_before) > 0 and len(valid_after) > 0:
            vmin = min(np.min(valid_before), np.min(valid_after))
            vmax = max(np.max(valid_before), np.max(valid_after))
        else:
            vmin, vmax = -1, 1
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Before: {before_date}', f'After: {after_date}'),
            horizontal_spacing=0.05
        )
        
        # Add before image
        fig.add_trace(
            go.Heatmap(
                z=before_data,
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                showscale=False,
                hovertemplate=f'{index_name}: %{{z:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add after image
        fig.add_trace(
            go.Heatmap(
                z=after_data,
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                showscale=True,
                colorbar=dict(title=index_name),
                hovertemplate=f'{index_name}: %{{z:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        fig.update_layout(
            title=f'{index_name} Comparison: {before_date} vs {after_date}',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_difference_map(self,
                             before_path: str,
                             after_path: str,
                             before_date: str,
                             after_date: str,
                             index_name: str = "NDVI") -> go.Figure:
        """
        Create difference map showing changes.
        
        Args:
            before_path: Path to earlier date GeoTIFF
            after_path: Path to later date GeoTIFF
            before_date: Date label for before image
            after_date: Date label for after image
            index_name: Name of vegetation index
            
        Returns:
            Plotly figure object
        """
        # Load data
        before_data, _ = self.load_image_data(before_path)
        after_data, _ = self.load_image_data(after_path)
        
        # Calculate difference
        difference = after_data - before_data
        
        # Create figure
        fig = go.Figure()
        
        # Add difference heatmap
        fig.add_trace(
            go.Heatmap(
                z=difference,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Change'),
                hovertemplate='Change: %{z:.3f}<extra></extra>'
            )
        )
        
        # Update layout
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        fig.update_layout(
            title=f'{index_name} Change: {before_date} → {after_date}',
            height=500
        )
        
        return fig
    
    def create_histogram_comparison(self,
                                   before_path: str,
                                   after_path: str,
                                   before_date: str,
                                   after_date: str,
                                   index_name: str = "NDVI") -> go.Figure:
        """
        Create histogram comparison of value distributions.
        
        Args:
            before_path: Path to earlier date GeoTIFF
            after_path: Path to later date GeoTIFF
            before_date: Date label for before image
            after_date: Date label for after image
            index_name: Name of vegetation index
            
        Returns:
            Plotly figure object
        """
        # Load data
        before_data, _ = self.load_image_data(before_path)
        after_data, _ = self.load_image_data(after_path)
        
        # Get valid values
        valid_before = before_data[np.isfinite(before_data)]
        valid_after = after_data[np.isfinite(after_data)]
        
        # Create figure
        fig = go.Figure()
        
        # Add histograms
        fig.add_trace(
            go.Histogram(
                x=valid_before,
                name=f'Before ({before_date})',
                opacity=0.7,
                nbinsx=50
            )
        )
        
        fig.add_trace(
            go.Histogram(
                x=valid_after,
                name=f'After ({after_date})',
                opacity=0.7,
                nbinsx=50
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{index_name} Distribution Comparison',
            xaxis_title=index_name,
            yaxis_title='Pixel Count',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def create_statistics_comparison(self,
                                    before_path: str,
                                    after_path: str,
                                    index_name: str = "NDVI") -> Dict[str, Dict[str, float]]:
        """
        Calculate and compare statistics.
        
        Args:
            before_path: Path to earlier date GeoTIFF
            after_path: Path to later date GeoTIFF
            index_name: Name of vegetation index
            
        Returns:
            Dictionary with before/after statistics
        """
        # Load data
        before_data, _ = self.load_image_data(before_path)
        after_data, _ = self.load_image_data(after_path)
        
        # Get valid values
        valid_before = before_data[np.isfinite(before_data)]
        valid_after = after_data[np.isfinite(after_data)]
        
        # Calculate statistics
        stats = {
            'before': {
                'mean': float(np.mean(valid_before)) if len(valid_before) > 0 else 0,
                'median': float(np.median(valid_before)) if len(valid_before) > 0 else 0,
                'std': float(np.std(valid_before)) if len(valid_before) > 0 else 0,
                'min': float(np.min(valid_before)) if len(valid_before) > 0 else 0,
                'max': float(np.max(valid_before)) if len(valid_before) > 0 else 0
            },
            'after': {
                'mean': float(np.mean(valid_after)) if len(valid_after) > 0 else 0,
                'median': float(np.median(valid_after)) if len(valid_after) > 0 else 0,
                'std': float(np.std(valid_after)) if len(valid_after) > 0 else 0,
                'min': float(np.min(valid_after)) if len(valid_after) > 0 else 0,
                'max': float(np.max(valid_after)) if len(valid_after) > 0 else 0
            }
        }
        
        # Calculate changes
        stats['change'] = {
            'mean': stats['after']['mean'] - stats['before']['mean'],
            'median': stats['after']['median'] - stats['before']['median'],
            'std': stats['after']['std'] - stats['before']['std']
        }
        
        return stats


def render_comparison_widget(before_imagery: Dict,
                            after_imagery: Dict,
                            index_name: str = "NDVI"):
    """
    Render complete comparison widget in Streamlit.
    
    Args:
        before_imagery: Dictionary with before imagery data
        after_imagery: Dictionary with after imagery data
        index_name: Name of vegetation index to compare
    """
    st.subheader(f"📊 {index_name} Temporal Comparison")
    
    # Get paths
    index_key = f"{index_name.lower()}_path"
    before_path = before_imagery.get(index_key)
    after_path = after_imagery.get(index_key)
    
    if not before_path or not after_path:
        st.error(f"{index_name} data not available for comparison")
        return
    
    # Get dates
    before_date = before_imagery.get('acquisition_date', 'Unknown')
    after_date = after_imagery.get('acquisition_date', 'Unknown')
    
    # Create widget
    widget = ComparisonWidget()
    
    # Comparison mode selector
    comparison_mode = st.radio(
        "Comparison Mode",
        ["Side-by-Side", "Difference Map", "Distribution", "Statistics"],
        horizontal=True
    )
    
    try:
        if comparison_mode == "Side-by-Side":
            fig = widget.create_side_by_side_view(
                before_path, after_path,
                before_date, after_date,
                index_name
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif comparison_mode == "Difference Map":
            fig = widget.create_difference_map(
                before_path, after_path,
                before_date, after_date,
                index_name
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation guide
            st.info("""
            **Interpretation:**
            - 🔴 Red areas: Decrease in vegetation health
            - ⚪ White areas: No significant change
            - 🔵 Blue areas: Increase in vegetation health
            """)
            
        elif comparison_mode == "Distribution":
            fig = widget.create_histogram_comparison(
                before_path, after_path,
                before_date, after_date,
                index_name
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif comparison_mode == "Statistics":
            stats = widget.create_statistics_comparison(
                before_path, after_path,
                index_name
            )
            
            # Display statistics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mean (Before)",
                    f"{stats['before']['mean']:.3f}",
                    help="Average value before"
                )
                st.metric(
                    "Std Dev (Before)",
                    f"{stats['before']['std']:.3f}",
                    help="Standard deviation before"
                )
            
            with col2:
                st.metric(
                    "Mean (After)",
                    f"{stats['after']['mean']:.3f}",
                    help="Average value after"
                )
                st.metric(
                    "Std Dev (After)",
                    f"{stats['after']['std']:.3f}",
                    help="Standard deviation after"
                )
            
            with col3:
                change = stats['change']['mean']
                st.metric(
                    "Mean Change",
                    f"{change:.3f}",
                    delta=f"{change:.3f}",
                    help="Change in average value"
                )
                
                change_pct = (change / stats['before']['mean'] * 100) if stats['before']['mean'] != 0 else 0
                st.metric(
                    "Change %",
                    f"{change_pct:.1f}%",
                    help="Percentage change"
                )
            
            # Detailed statistics table
            st.markdown("### Detailed Statistics")
            
            import pandas as pd
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Before': [
                    f"{stats['before']['mean']:.3f}",
                    f"{stats['before']['median']:.3f}",
                    f"{stats['before']['std']:.3f}",
                    f"{stats['before']['min']:.3f}",
                    f"{stats['before']['max']:.3f}"
                ],
                'After': [
                    f"{stats['after']['mean']:.3f}",
                    f"{stats['after']['median']:.3f}",
                    f"{stats['after']['std']:.3f}",
                    f"{stats['after']['min']:.3f}",
                    f"{stats['after']['max']:.3f}"
                ],
                'Change': [
                    f"{stats['change']['mean']:.3f}",
                    f"{stats['change']['median']:.3f}",
                    f"{stats['change']['std']:.3f}",
                    '-',
                    '-'
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating comparison: {str(e)}")
        logger.error(f"Comparison widget error: {e}", exc_info=True)


def render_multi_date_slider(imagery_records: list,
                             index_name: str = "NDVI"):
    """
    Render slider for comparing multiple dates.
    
    Args:
        imagery_records: List of imagery record dictionaries
        index_name: Name of vegetation index
    """
    if len(imagery_records) < 2:
        st.warning("Need at least 2 dates for comparison")
        return
    
    st.subheader("🎚️ Multi-Date Comparison Slider")
    
    # Sort by date
    sorted_records = sorted(
        imagery_records,
        key=lambda x: x.get('acquisition_date', '')
    )
    
    # Create date options
    date_options = [r.get('acquisition_date', 'Unknown') for r in sorted_records]
    
    # Date selectors
    col1, col2 = st.columns(2)
    
    with col1:
        before_idx = st.selectbox(
            "Before Date",
            range(len(date_options)),
            format_func=lambda i: date_options[i],
            key="before_date_selector"
        )
    
    with col2:
        after_idx = st.selectbox(
            "After Date",
            range(len(date_options)),
            index=min(before_idx + 1, len(date_options) - 1),
            format_func=lambda i: date_options[i],
            key="after_date_selector"
        )
    
    # Render comparison
    if before_idx != after_idx:
        render_comparison_widget(
            sorted_records[before_idx],
            sorted_records[after_idx],
            index_name
        )
    else:
        st.info("Please select different dates for comparison")
