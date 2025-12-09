"""
Model prediction overlay components for field monitoring.

This module provides functions to overlay AI model predictions on field maps.
"""

import numpy as np
import folium
from folium import plugins
import streamlit as st
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_prediction_overlay(
    index_data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    predictor,
    opacity: float = 0.6
) -> Optional[folium.raster_layers.ImageOverlay]:
    """
    Create a prediction overlay for the map.
    
    Args:
        index_data: NDVI or other vegetation index data
        bounds: Map bounds (minx, miny, maxx, maxy)
        predictor: CropHealthPredictor instance
        opacity: Overlay opacity (0-1)
    
    Returns:
        Folium ImageOverlay or None if prediction fails
    """
    try:
        # Get predictions
        result = predictor.predict(index_data)
        
        # Create color-coded prediction map
        prediction_rgb = predictions_to_rgb(result.predictions)
        
        # Create image overlay
        overlay = folium.raster_layers.ImageOverlay(
            image=prediction_rgb,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=opacity,
            name='AI Predictions'
        )
        
        return overlay
    
    except Exception as e:
        logger.error(f"Error creating prediction overlay: {e}")
        return None


def predictions_to_rgb(predictions: np.ndarray) -> np.ndarray:
    """
    Convert prediction classes to RGB colors.
    
    Args:
        predictions: Array of class predictions (0=Healthy, 1=Moderate, 2=Stressed, 3=Critical)
    
    Returns:
        RGB image array
    """
    # Define colors for each class
    colors = {
        0: [102, 187, 106],  # Healthy - Green
        1: [255, 235, 59],   # Moderate - Yellow
        2: [255, 152, 0],    # Stressed - Orange
        3: [239, 83, 80]     # Critical - Red
    }
    
    # Create RGB image
    rgb = np.zeros((*predictions.shape, 3), dtype=np.uint8)
    
    for class_idx, color in colors.items():
        mask = predictions == class_idx
        rgb[mask] = color
    
    return rgb


def create_confidence_heatmap(
    confidence_scores: np.ndarray,
    bounds: Tuple[float, float, float, float],
    opacity: float = 0.5
) -> Optional[folium.raster_layers.ImageOverlay]:
    """
    Create a confidence heatmap overlay.
    
    Args:
        confidence_scores: Array of confidence values (0-1)
        bounds: Map bounds (minx, miny, maxx, maxy)
        opacity: Overlay opacity (0-1)
    
    Returns:
        Folium ImageOverlay or None if creation fails
    """
    try:
        # Normalize confidence to 0-255
        confidence_normalized = (confidence_scores * 255).astype(np.uint8)
        
        # Create RGB heatmap (blue to red)
        rgb = np.zeros((*confidence_scores.shape, 3), dtype=np.uint8)
        rgb[:, :, 0] = confidence_normalized  # Red channel
        rgb[:, :, 2] = 255 - confidence_normalized  # Blue channel (inverse)
        
        # Create overlay
        overlay = folium.raster_layers.ImageOverlay(
            image=rgb,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=opacity,
            name='Confidence Heatmap'
        )
        
        return overlay
    
    except Exception as e:
        logger.error(f"Error creating confidence heatmap: {e}")
        return None


def add_prediction_legend(m: folium.Map):
    """
    Add a legend for prediction classes to the map.
    
    Args:
        m: Folium map instance
    """
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 0; font-weight: bold; text-align: center;">Crop Health Classes</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 5px 0;"><span style="background-color: #66bb6a; padding: 2px 10px; border-radius: 3px;">■</span> Healthy</p>
        <p style="margin: 5px 0;"><span style="background-color: #ffeb3b; padding: 2px 10px; border-radius: 3px;">■</span> Moderate</p>
        <p style="margin: 5px 0;"><span style="background-color: #ff9800; padding: 2px 10px; border-radius: 3px;">■</span> Stressed</p>
        <p style="margin: 5px 0;"><span style="background-color: #ef5350; padding: 2px 10px; border-radius: 3px;">■</span> Critical</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


def display_prediction_statistics(predictions: np.ndarray, confidence_scores: np.ndarray):
    """
    Display statistics about predictions.
    
    Args:
        predictions: Array of class predictions
        confidence_scores: Array of confidence scores
    """
    st.markdown("### 📊 Prediction Statistics")
    
    # Calculate class distribution
    class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    total_pixels = predictions.size
    
    col1, col2, col3, col4 = st.columns(4)
    
    for i, (col, class_name) in enumerate(zip([col1, col2, col3, col4], class_names)):
        count = np.sum(predictions == i)
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        avg_conf = np.mean(confidence_scores[predictions == i]) if count > 0 else 0
        
        with col:
            st.metric(
                class_name,
                f"{percentage:.1f}%",
                delta=f"Conf: {avg_conf*100:.0f}%",
                help=f"{count:,} pixels classified as {class_name}"
            )
    
    # Overall confidence
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_confidence = np.mean(confidence_scores) * 100
        st.metric(
            "Average Confidence",
            f"{avg_confidence:.1f}%",
            help="Average confidence across all predictions"
        )
    
    with col2:
        low_conf_pct = np.sum(confidence_scores < 0.5) / total_pixels * 100
        st.metric(
            "Low Confidence Areas",
            f"{low_conf_pct:.1f}%",
            delta="< 50% confidence",
            delta_color="inverse",
            help="Percentage of pixels with confidence below 50%"
        )


def display_per_pixel_classification(
    predictions: np.ndarray,
    confidence_scores: np.ndarray,
    index_data: np.ndarray,
    row: int,
    col: int
):
    """
    Display classification details for a specific pixel.
    
    Args:
        predictions: Array of class predictions
        confidence_scores: Array of confidence scores
        index_data: Original vegetation index data
        row: Pixel row
        col: Pixel column
    """
    if row < 0 or row >= predictions.shape[0] or col < 0 or col >= predictions.shape[1]:
        st.warning("Invalid pixel coordinates")
        return
    
    class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    
    predicted_class = predictions[row, col]
    confidence = confidence_scores[row, col]
    index_value = index_data[row, col]
    
    st.markdown(f"### 🔍 Pixel Details ({row}, {col})")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Class",
            class_names[predicted_class],
            help="AI model prediction for this pixel"
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{confidence*100:.1f}%",
            help="Model confidence in this prediction"
        )
    
    with col3:
        st.metric(
            "NDVI Value",
            f"{index_value:.3f}",
            help="Vegetation index value for this pixel"
        )
    
    # Visual confidence indicator
    if confidence > 0.8:
        st.success("✅ High confidence prediction")
    elif confidence > 0.5:
        st.info("ℹ️ Moderate confidence prediction")
    else:
        st.warning("⚠️ Low confidence - verify with ground truth")


def create_prediction_summary_card(predictions: np.ndarray, confidence_scores: np.ndarray) -> str:
    """
    Create an HTML summary card for predictions.
    
    Args:
        predictions: Array of class predictions
        confidence_scores: Array of confidence scores
    
    Returns:
        HTML string for the summary card
    """
    class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']
    colors = ['#66bb6a', '#ffeb3b', '#ff9800', '#ef5350']
    
    total_pixels = predictions.size
    
    # Calculate percentages
    percentages = []
    for i in range(4):
        count = np.sum(predictions == i)
        pct = (count / total_pixels) * 100 if total_pixels > 0 else 0
        percentages.append(pct)
    
    # Determine overall health
    if percentages[0] > 60:
        overall_status = "Excellent"
        status_color = "#66bb6a"
    elif percentages[0] + percentages[1] > 70:
        overall_status = "Good"
        status_color = "#ffeb3b"
    elif percentages[2] > 40:
        overall_status = "Needs Attention"
        status_color = "#ff9800"
    else:
        overall_status = "Critical"
        status_color = "#ef5350"
    
    avg_confidence = np.mean(confidence_scores) * 100
    
    html = f'''
    <div style="background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); 
                padding: 20px; border-radius: 10px; border: 2px solid {status_color}; 
                margin: 10px 0;">
        <h3 style="color: {status_color}; margin: 0 0 15px 0;">
            🌱 Field Health Summary: {overall_status}
        </h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: #a0aec0; font-size: 0.9em;">Healthy</p>
                <p style="margin: 5px 0 0 0; color: {colors[0]}; font-size: 1.5em; font-weight: bold;">
                    {percentages[0]:.1f}%
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: #a0aec0; font-size: 0.9em;">Moderate</p>
                <p style="margin: 5px 0 0 0; color: {colors[1]}; font-size: 1.5em; font-weight: bold;">
                    {percentages[1]:.1f}%
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: #a0aec0; font-size: 0.9em;">Stressed</p>
                <p style="margin: 5px 0 0 0; color: {colors[2]}; font-size: 1.5em; font-weight: bold;">
                    {percentages[2]:.1f}%
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: #a0aec0; font-size: 0.9em;">Critical</p>
                <p style="margin: 5px 0 0 0; color: {colors[3]}; font-size: 1.5em; font-weight: bold;">
                    {percentages[3]:.1f}%
                </p>
            </div>
        </div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
            <p style="margin: 0; color: #a0aec0; font-size: 0.9em;">Average Model Confidence</p>
            <p style="margin: 5px 0 0 0; color: #4caf50; font-size: 1.3em; font-weight: bold;">
                {avg_confidence:.1f}%
            </p>
        </div>
    </div>
    '''
    
    return html
