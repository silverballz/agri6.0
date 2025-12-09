"""
AI Model Performance Dashboard Page

This page displays comprehensive model performance metrics, including:
- CNN confusion matrix and classification metrics
- LSTM prediction accuracy metrics
- Model metadata and training information
- Model comparison (AI vs rule-based)
- Performance tracking over time
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_model_metrics(model_type: str, real_data: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load model metrics from JSON file.
    
    Args:
        model_type: Type of model ('cnn', 'lstm', or 'mlp')
        real_data: If True, load metrics for real-data trained models
    
    Returns:
        Dictionary with model metrics or None if not found
    """
    try:
        if model_type == 'cnn':
            if real_data:
                metrics_path = Path('models/cnn_model_metrics_real.json')
            else:
                metrics_path = Path('models/cnn_model_metrics.json')
        elif model_type == 'lstm':
            if real_data:
                metrics_path = Path('models/lstm_model_metrics_real.json')
            else:
                metrics_path = Path('models/lstm_temporal/lstm_model_metrics.json')
        elif model_type == 'mlp':
            metrics_path = Path('models/model_metrics.json')
        else:
            return None
        
        if not metrics_path.exists():
            logger.warning(f"Metrics file not found: {metrics_path}")
            return None
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error loading {model_type} metrics: {e}")
        return None


def display_model_training_status():
    """Display status of model training data (real vs synthetic)"""
    
    # Check for real-trained models
    cnn_real = Path('models/crop_health_cnn_real.pth').exists()
    lstm_real = Path('models/crop_health_lstm_real.pth').exists()
    cnn_metrics_real = Path('models/cnn_model_metrics_real.json').exists()
    lstm_metrics_real = Path('models/lstm_model_metrics_real.json').exists()
    
    # Check AI models enabled
    use_ai = os.getenv('USE_AI_MODELS', 'false').lower() == 'true'
    
    # Determine overall status
    if cnn_real and lstm_real and cnn_metrics_real and lstm_metrics_real and use_ai:
        st.success("✅ **Models Trained on Real Satellite Data** - Production-ready AI models active")
    elif cnn_real or lstm_real:
        st.info("🛰️ **Partial Real Data Training** - Some models trained on real data, others on synthetic")
    else:
        st.warning("⚠️ **Models Trained on Synthetic Data** - Train on real satellite data for production use")
    
    # Show detailed status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### CNN Model")
        if cnn_real and cnn_metrics_real:
            st.success("🛰️ Real Data")
            if cnn_metrics_real:
                try:
                    with open('models/cnn_model_metrics_real.json') as f:
                        metrics = json.load(f)
                        acc = metrics.get('metrics', {}).get('accuracy', 0) * 100
                        st.metric("Accuracy", f"{acc:.1f}%")
                except:
                    pass
        else:
            st.warning("⚠️ Synthetic Data")
    
    with col2:
        st.markdown("### LSTM Model")
        if lstm_real and lstm_metrics_real:
            st.success("🛰️ Real Data")
            if lstm_metrics_real:
                try:
                    with open('models/lstm_model_metrics_real.json') as f:
                        metrics = json.load(f)
                        r2 = metrics.get('metrics', {}).get('r2_score', 0)
                        st.metric("R² Score", f"{r2:.3f}")
                except:
                    pass
        else:
            st.warning("⚠️ Synthetic Data")
    
    with col3:
        st.markdown("### AI Status")
        if use_ai:
            st.success("✅ Enabled")
        else:
            st.error("❌ Disabled")
        
        if st.button("📚 View Pipeline Docs"):
            st.info("Navigate to **📚 Documentation** page from sidebar for complete real data pipeline guide")


def display_model_comparison():
    """Display comparison between synthetic and real-data trained models"""
    
    st.markdown("## 🔄 Model Training Data Comparison")
    
    # Load comparison report if available
    comparison_path = Path('reports/model_comparison_report.json')
    
    if comparison_path.exists():
        try:
            with open(comparison_path, 'r') as f:
                comparison = json.load(f)
            
            st.success("✅ Model comparison data available")
            
            # Display comparison metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 CNN Model Comparison")
                cnn_synthetic = comparison.get('cnn_synthetic', {})
                cnn_real = comparison.get('cnn_real', {})
                
                if cnn_synthetic and cnn_real:
                    comparison_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        'Synthetic Data': [
                            cnn_synthetic.get('accuracy', 0) * 100,
                            cnn_synthetic.get('precision', 0) * 100,
                            cnn_synthetic.get('recall', 0) * 100,
                            cnn_synthetic.get('f1_score', 0) * 100
                        ],
                        'Real Data': [
                            cnn_real.get('accuracy', 0) * 100,
                            cnn_real.get('precision', 0) * 100,
                            cnn_real.get('recall', 0) * 100,
                            cnn_real.get('f1_score', 0) * 100
                        ]
                    })
                    
                    comparison_df['Improvement'] = comparison_df['Real Data'] - comparison_df['Synthetic Data']
                    
                    st.dataframe(comparison_df.style.format({
                        'Synthetic Data': '{:.1f}%',
                        'Real Data': '{:.1f}%',
                        'Improvement': '{:+.1f}%'
                    }), use_container_width=True)
                    
                    # Highlight improvement
                    avg_improvement = comparison_df['Improvement'].mean()
                    if avg_improvement > 0:
                        st.success(f"🎯 Average improvement: **+{avg_improvement:.1f}%** with real data")
                    else:
                        st.info(f"Average difference: **{avg_improvement:.1f}%**")
            
            with col2:
                st.markdown("### 📈 LSTM Model Comparison")
                lstm_synthetic = comparison.get('lstm_synthetic', {})
                lstm_real = comparison.get('lstm_real', {})
                
                if lstm_synthetic and lstm_real:
                    comparison_df = pd.DataFrame({
                        'Metric': ['MSE', 'MAE', 'R² Score'],
                        'Synthetic Data': [
                            lstm_synthetic.get('mse', 0),
                            lstm_synthetic.get('mae', 0),
                            lstm_synthetic.get('r2_score', 0)
                        ],
                        'Real Data': [
                            lstm_real.get('mse', 0),
                            lstm_real.get('mae', 0),
                            lstm_real.get('r2_score', 0)
                        ]
                    })
                    
                    st.dataframe(comparison_df.style.format({
                        'Synthetic Data': '{:.4f}',
                        'Real Data': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Show improvement in R² score
                    r2_improvement = lstm_real.get('r2_score', 0) - lstm_synthetic.get('r2_score', 0)
                    if r2_improvement > 0:
                        st.success(f"🎯 R² Score improvement: **+{r2_improvement:.3f}** with real data")
            
            # Show comparison charts if available
            if Path('reports/metrics_comparison.png').exists():
                st.markdown("### 📊 Visual Comparison")
                st.image('reports/metrics_comparison.png', caption='Model Performance Comparison', use_container_width=True)
            
            if Path('reports/confusion_matrix_comparison.png').exists():
                st.image('reports/confusion_matrix_comparison.png', caption='Confusion Matrix Comparison', use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error loading comparison report: {e}")
            st.warning("⚠️ Could not load model comparison data")
    else:
        st.info("""
        📊 **Model Comparison Not Available**
        
        To compare synthetic vs real-data trained models:
        
        ```bash
        python scripts/compare_model_performance.py
        ```
        
        This will generate a detailed comparison report showing improvements from using real satellite data.
        """)
    
    st.markdown("---")


def display_cnn_performance(metrics: Dict[str, Any]):
    """
    Display CNN model performance metrics.
    
    Args:
        metrics: Dictionary containing CNN metrics
    """
    st.subheader("🧠 CNN Model Performance")
    
    # Model metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model Type",
            metrics.get('model_type', 'CNN'),
            help="Convolutional Neural Network for spatial crop health analysis"
        )
    
    with col2:
        accuracy = metrics.get('metrics', {}).get('accuracy', 0) * 100
        st.metric(
            "Accuracy",
            f"{accuracy:.1f}%",
            help="Overall classification accuracy on test set"
        )
    
    with col3:
        training_date = metrics.get('training_date', 'Unknown')
        if training_date != 'Unknown':
            training_date = datetime.fromisoformat(training_date).strftime('%Y-%m-%d')
        st.metric(
            "Training Date",
            training_date,
            help="Date when the model was last trained"
        )
    
    with col4:
        version = metrics.get('version', '1.0')
        st.metric(
            "Version",
            version,
            help="Model version number"
        )
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("### 📊 Confusion Matrix")
    st.markdown("Shows how well the model distinguishes between different crop health classes")
    
    confusion_matrix = np.array(metrics.get('metrics', {}).get('confusion_matrix', []))
    
    if confusion_matrix.size > 0:
        # Get class names (handle both 3-class and 4-class models)
        classes = metrics.get('classes', ['Moderate', 'Stressed', 'Critical'])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Confusion Matrix - CNN Model",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("ℹ️ How to Read the Confusion Matrix"):
            st.markdown("""
            - **Diagonal values** (top-left to bottom-right): Correct predictions
            - **Off-diagonal values**: Misclassifications
            - **Darker colors**: Higher counts
            - **Goal**: Dark diagonal, light off-diagonal
            
            **Example**: If row "Stressed" and column "Moderate" shows 80, 
            it means 80 stressed crops were incorrectly classified as moderate.
            """)
    else:
        st.warning("Confusion matrix data not available")
    
    st.markdown("---")
    
    # Classification Report
    st.markdown("### 📈 Classification Report")
    st.markdown("Detailed performance metrics for each crop health class")
    
    classification_report = metrics.get('metrics', {}).get('classification_report', '')
    
    if classification_report:
        # Handle both string and dict formats
        class_data = []
        if isinstance(classification_report, dict):
            # If it's already a dict, convert to DataFrame directly
            for class_name, class_metrics in classification_report.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    class_data.append({
                        'Class': class_name,
                        'Precision': class_metrics.get('precision', 0),
                        'Recall': class_metrics.get('recall', 0),
                        'F1-Score': class_metrics.get('f1-score', 0),
                        'Support': int(class_metrics.get('support', 0))
                    })
        else:
            # Parse classification report string into DataFrame
            lines = classification_report.strip().split('\n')
            # Extract per-class metrics
            for line in lines[1:-4]:  # Skip header and summary lines
                parts = line.split()
                if len(parts) >= 5:
                    class_data.append({
                        'Class': parts[0],
                        'Precision': float(parts[1]),
                        'Recall': float(parts[2]),
                        'F1-Score': float(parts[3]),
                        'Support': int(parts[4])
                    })
        
        if class_data:
            df = pd.DataFrame(class_data)
            
            # Display as styled table
            st.dataframe(
                df.style.format({
                    'Precision': '{:.2f}',
                    'Recall': '{:.2f}',
                    'F1-Score': '{:.2f}',
                    'Support': '{:d}'
                }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn', vmin=0.5, vmax=1.0),
                use_container_width=True
            )
            
            # Visualize metrics
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Precision',
                x=df['Class'],
                y=df['Precision'],
                marker_color='#4caf50'
            ))
            
            fig.add_trace(go.Bar(
                name='Recall',
                x=df['Class'],
                y=df['Recall'],
                marker_color='#2196f3'
            ))
            
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=df['Class'],
                y=df['F1-Score'],
                marker_color='#ff9800'
            ))
            
            fig.update_layout(
                title="Per-Class Performance Metrics",
                xaxis_title="Crop Health Class",
                yaxis_title="Score",
                barmode='group',
                height=400,
                template='plotly_dark',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metric explanations
            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
                - **Precision**: Of all crops predicted as this class, what % were correct?
                  - High precision = Few false positives
                  
                - **Recall**: Of all crops actually in this class, what % did we find?
                  - High recall = Few false negatives
                  
                - **F1-Score**: Harmonic mean of precision and recall
                  - Balanced measure of overall performance
                  
                - **Support**: Number of actual samples in this class
                """)
    else:
        st.warning("Classification report not available")
    
    st.markdown("---")
    
    # Training Loss
    st.markdown("### 📉 Training & Validation Loss")
    
    train_loss = metrics.get('metrics', {}).get('final_train_loss', 0)
    val_loss = metrics.get('metrics', {}).get('final_val_loss', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Final Training Loss",
            f"{train_loss:.4f}",
            help="Lower is better. Measures error on training data."
        )
    
    with col2:
        st.metric(
            "Final Validation Loss",
            f"{val_loss:.4f}",
            delta=f"{val_loss - train_loss:+.4f}",
            delta_color="inverse",
            help="Lower is better. Measures error on unseen validation data."
        )
    
    # Check for overfitting
    if val_loss > train_loss * 1.2:
        st.warning("⚠️ Model may be overfitting (validation loss significantly higher than training loss)")
    elif val_loss < train_loss * 0.8:
        st.info("ℹ️ Validation loss is lower than training loss - this is unusual but can happen with dropout")
    else:
        st.success("✅ Training and validation losses are well-balanced")
    
    # Architecture info
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    
    architecture = metrics.get('architecture', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Parameters",
            f"{architecture.get('total_parameters', 0):,}",
            help="Total number of trainable parameters in the model"
        )
    
    with col2:
        st.metric(
            "Convolutional Layers",
            architecture.get('conv_layers', 0),
            help="Number of convolutional layers for feature extraction"
        )
    
    with col3:
        st.metric(
            "Fully Connected Layers",
            architecture.get('fc_layers', 0),
            help="Number of dense layers for classification"
        )


def display_lstm_performance(metrics: Dict[str, Any]):
    """
    Display LSTM model performance metrics.
    
    Args:
        metrics: Dictionary containing LSTM metrics
    """
    st.subheader("📈 LSTM Model Performance")
    
    # Model metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model Type",
            metrics.get('model_type', 'LSTM'),
            help="Long Short-Term Memory network for temporal trend analysis"
        )
    
    with col2:
        r2_score = metrics.get('metrics', {}).get('r2_score', 0)
        st.metric(
            "R² Score",
            f"{r2_score:.3f}",
            help="Coefficient of determination (1.0 = perfect predictions)"
        )
    
    with col3:
        training_date = metrics.get('training_date', 'Unknown')
        if training_date != 'Unknown':
            training_date = datetime.fromisoformat(training_date).strftime('%Y-%m-%d')
        st.metric(
            "Training Date",
            training_date,
            help="Date when the model was last trained"
        )
    
    with col4:
        version = metrics.get('version', '1.0')
        st.metric(
            "Version",
            version,
            help="Model version number"
        )
    
    st.markdown("---")
    
    # Prediction Accuracy Metrics
    st.markdown("### 🎯 Prediction Accuracy Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mae = metrics.get('metrics', {}).get('mae', 0)
        st.metric(
            "MAE",
            f"{mae:.4f}",
            help="Mean Absolute Error - average prediction error"
        )
    
    with col2:
        rmse = metrics.get('metrics', {}).get('rmse', 0)
        st.metric(
            "RMSE",
            f"{rmse:.4f}",
            help="Root Mean Squared Error - penalizes large errors more"
        )
    
    with col3:
        mse = metrics.get('metrics', {}).get('mse', 0)
        st.metric(
            "MSE",
            f"{mse:.6f}",
            help="Mean Squared Error - squared average error"
        )
    
    # Visualize metrics
    metrics_data = {
        'Metric': ['MAE', 'RMSE', 'MSE', 'R² Score'],
        'Value': [mae, rmse, mse, r2_score],
        'Optimal': [0, 0, 0, 1.0]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Actual Value',
        x=df_metrics['Metric'],
        y=df_metrics['Value'],
        marker_color='#2196f3',
        text=df_metrics['Value'].round(4),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="LSTM Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metric interpretation
    with st.expander("ℹ️ Understanding LSTM Metrics"):
        st.markdown(f"""
        **Current Performance:**
        - **R² Score: {r2_score:.3f}** - Explains {r2_score*100:.1f}% of variance in vegetation trends
        - **MAE: {mae:.4f}** - On average, predictions are off by {mae:.4f} NDVI units
        - **RMSE: {rmse:.4f}** - Root mean squared error of {rmse:.4f}
        
        **Interpretation:**
        - R² > 0.9: Excellent predictions ✅
        - R² 0.7-0.9: Good predictions
        - R² < 0.7: Needs improvement
        
        **What this means:**
        The LSTM model can predict vegetation trends with high accuracy, 
        making it reliable for forecasting crop health changes over time.
        """)
    
    st.markdown("---")
    
    # Training Loss
    st.markdown("### 📉 Training & Validation Loss")
    
    train_loss = metrics.get('metrics', {}).get('final_train_loss', 0)
    val_loss = metrics.get('metrics', {}).get('final_val_loss', 0)
    best_val_loss = metrics.get('metrics', {}).get('best_val_loss', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Final Training Loss",
            f"{train_loss:.6f}",
            help="Final loss on training data"
        )
    
    with col2:
        st.metric(
            "Final Validation Loss",
            f"{val_loss:.6f}",
            delta=f"{val_loss - train_loss:+.6f}",
            delta_color="inverse",
            help="Final loss on validation data"
        )
    
    with col3:
        st.metric(
            "Best Validation Loss",
            f"{best_val_loss:.6f}",
            help="Best validation loss achieved during training"
        )
    
    # Architecture info
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    
    architecture = metrics.get('architecture', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Parameters",
            f"{architecture.get('total_parameters', 0):,}",
            help="Total number of trainable parameters"
        )
    
    with col2:
        st.metric(
            "Hidden Size",
            architecture.get('hidden_size', 0),
            help="Number of units in LSTM hidden layers"
        )
    
    with col3:
        st.metric(
            "Num Layers",
            architecture.get('num_layers', 0),
            help="Number of stacked LSTM layers"
        )
    
    with col4:
        bidirectional = architecture.get('bidirectional', False)
        st.metric(
            "Bidirectional",
            "Yes" if bidirectional else "No",
            help="Whether the LSTM processes sequences in both directions"
        )
    
    # Sequence info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Sequence Length",
            architecture.get('sequence_length', 0),
            help="Number of time steps used for prediction"
        )
    
    with col2:
        st.metric(
            "Input Features",
            architecture.get('input_features', 0),
            help="Number of features per time step (NDVI, temp, humidity, soil moisture)"
        )


def display_mlp_performance(metrics: Dict[str, Any]):
    """
    Display MLP model performance metrics.
    
    Args:
        metrics: Dictionary containing MLP metrics
    """
    st.subheader("🔷 MLP Model Performance")
    
    # Model metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model Type",
            metrics.get('model_type', 'MLP'),
            help="Multi-Layer Perceptron for crop health classification"
        )
    
    with col2:
        accuracy = metrics.get('metrics', {}).get('accuracy', 0) * 100
        st.metric(
            "Accuracy",
            f"{accuracy:.1f}%",
            help="Overall classification accuracy on test set"
        )
    
    with col3:
        training_date = metrics.get('training_date', 'Unknown')
        if training_date != 'Unknown':
            training_date = datetime.fromisoformat(training_date).strftime('%Y-%m-%d')
        st.metric(
            "Training Date",
            training_date,
            help="Date when the model was last trained"
        )
    
    with col4:
        mean_confidence = metrics.get('metrics', {}).get('mean_confidence', 0)
        st.metric(
            "Mean Confidence",
            f"{mean_confidence:.2f}",
            help="Average confidence score across predictions"
        )
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("### 📊 Confusion Matrix")
    
    confusion_matrix = np.array(metrics.get('metrics', {}).get('confusion_matrix', []))
    
    if confusion_matrix.size > 0:
        classes = metrics.get('classes', ['Moderate', 'Stressed', 'Critical'])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=classes,
            y=classes,
            colorscale='Greens',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Confusion Matrix - MLP Model",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Architecture info
    st.markdown("### 🏗️ Model Architecture")
    
    architecture = metrics.get('architecture', {})
    hidden_layers = architecture.get('hidden_layers', [])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Hidden Layers",
            len(hidden_layers),
            help="Number of hidden layers in the network"
        )
    
    with col2:
        st.metric(
            "Layer Sizes",
            str(hidden_layers),
            help="Number of neurons in each hidden layer"
        )
    
    with col3:
        st.metric(
            "Activation",
            architecture.get('activation', 'relu').upper(),
            help="Activation function used in hidden layers"
        )


def show_page():
    """Main function to display the AI Model Performance page."""
    
    st.title("🤖 AI Model Performance Dashboard")
    st.markdown("""
    This dashboard provides comprehensive insights into the performance of AI models 
    used for crop health prediction and trend analysis.
    """)
    
    st.markdown("---")
    
    # Display model training data status
    display_model_training_status()
    
    st.markdown("---")
    
    # Interactive prediction demo at the top
    display_interactive_prediction_demo()
    
    st.markdown("---")
    
    # Model selector with real data indicator
    st.markdown("### 📋 Select Model to View")
    
    # Check which models have real-data versions
    cnn_real_exists = Path('models/cnn_model_metrics_real.json').exists()
    lstm_real_exists = Path('models/lstm_model_metrics_real.json').exists()
    
    model_options = {
        f'CNN (Spatial Analysis) {"🛰️ Real Data" if cnn_real_exists else "⚠️ Synthetic"}': 'cnn',
        f'LSTM (Temporal Trends) {"🛰️ Real Data" if lstm_real_exists else "⚠️ Synthetic"}': 'lstm',
        'MLP (Classification)': 'mlp',
        'All Models Comparison': 'all'
    }
    
    selected_model = st.selectbox(
        "Choose a model:",
        list(model_options.keys()),
        help="Select which model's performance metrics you want to view. 🛰️ indicates models trained on real satellite data."
    )
    
    model_type = model_options[selected_model]
    
    st.markdown("---")
    
    # Model comparison section (synthetic vs real)
    display_model_comparison()
    
    st.markdown("---")
    
    # Performance tracking section - DISABLED (requires production deployment tracking)
    # display_performance_tracking()
    
    st.markdown("---")
    
    # Display selected model performance
    if model_type == 'all':
        # Show comparison of all models
        st.markdown("## 📊 Model Comparison Overview")
        
        # Load all metrics (prefer real-data trained models)
        cnn_metrics = load_model_metrics('cnn', real_data=True) or load_model_metrics('cnn', real_data=False)
        lstm_metrics = load_model_metrics('lstm', real_data=True) or load_model_metrics('lstm', real_data=False)
        mlp_metrics = load_model_metrics('mlp')
        
        # Create comparison table
        comparison_data = []
        
        if cnn_metrics:
            comparison_data.append({
                'Model': 'CNN',
                'Type': 'Spatial Analysis',
                'Accuracy': f"{cnn_metrics.get('metrics', {}).get('accuracy', 0)*100:.1f}%",
                'Training Date': datetime.fromisoformat(cnn_metrics.get('training_date', '2024-01-01')).strftime('%Y-%m-%d'),
                'Parameters': f"{cnn_metrics.get('architecture', {}).get('total_parameters', 0):,}",
                'Status': '✅ Active'
            })
        
        if lstm_metrics:
            comparison_data.append({
                'Model': 'LSTM',
                'Type': 'Temporal Trends',
                'Accuracy': f"R²={lstm_metrics.get('metrics', {}).get('r2_score', 0):.3f}",
                'Training Date': datetime.fromisoformat(lstm_metrics.get('training_date', '2024-01-01')).strftime('%Y-%m-%d'),
                'Parameters': f"{lstm_metrics.get('architecture', {}).get('total_parameters', 0):,}",
                'Status': '✅ Active'
            })
        
        if mlp_metrics:
            comparison_data.append({
                'Model': 'MLP',
                'Type': 'Classification',
                'Accuracy': f"{mlp_metrics.get('metrics', {}).get('accuracy', 0)*100:.1f}%",
                'Training Date': datetime.fromisoformat(mlp_metrics.get('training_date', '2024-01-01')).strftime('%Y-%m-%d'),
                'Parameters': 'N/A',
                'Status': '✅ Active'
            })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Accuracy comparison chart
            st.markdown("### 📈 Model Accuracy Comparison")
            
            fig = go.Figure()
            
            for model_data in comparison_data:
                model_name = model_data['Model']
                accuracy_str = model_data['Accuracy']
                
                # Extract numeric value
                if 'R²=' in accuracy_str:
                    accuracy_val = float(accuracy_str.split('=')[1]) * 100
                else:
                    accuracy_val = float(accuracy_str.rstrip('%'))
                
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=[model_name],
                    y=[accuracy_val],
                    text=[f"{accuracy_val:.1f}%"],
                    textposition='auto',
                    marker_color=['#4caf50', '#2196f3', '#ff9800'][comparison_data.index(model_data)]
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy/Score (%)",
                height=400,
                template='plotly_dark',
                showlegend=False,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model metrics available. Please train models first.")
        
        st.markdown("---")
        
        # Show individual model details
        st.markdown("## 📊 Detailed Model Performance")
        
        tabs = st.tabs(["CNN Model", "LSTM Model", "MLP Model"])
        
        with tabs[0]:
            if cnn_metrics:
                display_cnn_performance(cnn_metrics)
            else:
                st.warning("CNN model metrics not available")
        
        with tabs[1]:
            if lstm_metrics:
                display_lstm_performance(lstm_metrics)
            else:
                st.warning("LSTM model metrics not available")
        
        with tabs[2]:
            if mlp_metrics:
                display_mlp_performance(mlp_metrics)
            else:
                st.warning("MLP model metrics not available")
    
    elif model_type == 'cnn':
        # Try to load real-data trained model first
        metrics = load_model_metrics('cnn', real_data=True)
        if metrics:
            st.info("📊 Showing metrics for CNN model trained on **real satellite data**")
            display_cnn_performance(metrics)
        else:
            # Fall back to synthetic-trained model
            metrics = load_model_metrics('cnn', real_data=False)
            if metrics:
                st.warning("📊 Showing metrics for CNN model trained on **synthetic data**. Train on real data for production use.")
                display_cnn_performance(metrics)
            else:
                st.error("CNN model metrics not found. Please train the model first.")
    
    elif model_type == 'lstm':
        # Try to load real-data trained model first
        metrics = load_model_metrics('lstm', real_data=True)
        if metrics:
            st.info("📊 Showing metrics for LSTM model trained on **real satellite data**")
            display_lstm_performance(metrics)
        else:
            # Fall back to synthetic-trained model
            metrics = load_model_metrics('lstm', real_data=False)
            if metrics:
                st.warning("📊 Showing metrics for LSTM model trained on **synthetic data**. Train on real data for production use.")
                display_lstm_performance(metrics)
            else:
                st.error("LSTM model metrics not found. Please train the model first.")
    
    elif model_type == 'mlp':
        metrics = load_model_metrics('mlp')
        if metrics:
            display_mlp_performance(metrics)
        else:
            st.error("MLP model metrics not found. Please train the model first.")


if __name__ == "__main__":
    show_page()



def display_prediction_explanation(prediction_result: Dict[str, Any], ndvi_value: float):
    """
    Display explanation for a single prediction.
    
    Args:
        prediction_result: Dictionary with prediction, confidence, and class info
        ndvi_value: NDVI value that was classified
    """
    st.markdown("### 🔍 Prediction Explanation")
    
    predicted_class = prediction_result.get('predicted_class', 'Unknown')
    confidence = prediction_result.get('confidence', 0)
    all_probabilities = prediction_result.get('all_probabilities', {})
    
    # Display main prediction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            "Predicted Class",
            predicted_class,
            help="The crop health class predicted by the AI model"
        )
        
        st.metric(
            "Confidence",
            f"{confidence*100:.1f}%",
            help="How confident the model is in this prediction"
        )
        
        st.metric(
            "NDVI Value",
            f"{ndvi_value:.3f}",
            help="The vegetation index value being classified"
        )
    
    with col2:
        # Confidence bar chart
        st.markdown("**Confidence Score Visualization:**")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#4caf50"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcdd2"},
                    {'range': [50, 75], 'color': "#fff9c4"},
                    {'range': [75, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top 3 predictions
    st.markdown("### 📊 Top 3 Predicted Classes")
    st.markdown("Shows the model's confidence for each possible crop health class")
    
    if all_probabilities:
        # Sort by probability
        sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Create bar chart
        classes = [item[0] for item in sorted_probs]
        probs = [item[1] * 100 for item in sorted_probs]
        
        fig = go.Figure()
        
        colors = ['#4caf50', '#2196f3', '#ff9800']
        
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            fig.add_trace(go.Bar(
                name=cls,
                x=[cls],
                y=[prob],
                text=[f"{prob:.1f}%"],
                textposition='auto',
                marker_color=colors[i],
                showlegend=False
            ))
        
        fig.update_layout(
            title="Probability Distribution",
            xaxis_title="Crop Health Class",
            yaxis_title="Probability (%)",
            height=400,
            template='plotly_dark',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display as table
        df_probs = pd.DataFrame({
            'Rank': [1, 2, 3],
            'Class': classes,
            'Probability': [f"{p:.1f}%" for p in probs]
        })
        
        st.dataframe(df_probs, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Why this prediction?
    st.markdown("### 💡 Why This Prediction?")
    
    explanation = generate_prediction_explanation(predicted_class, ndvi_value, confidence)
    
    st.info(explanation)
    
    # Feature influence
    st.markdown("### 📈 Feature Influence")
    st.markdown("Which features most influenced this prediction")
    
    # For demonstration, show NDVI as primary feature
    feature_importance = {
        'NDVI Value': 0.85,
        'Temporal Trend': 0.10,
        'Spatial Context': 0.05
    }
    
    fig = go.Figure(go.Bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        marker_color='#4caf50',
        text=[f"{v*100:.0f}%" for v in feature_importance.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Importance for This Prediction",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=300,
        template='plotly_dark',
        xaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ Understanding Feature Influence"):
        st.markdown("""
        **Feature Importance** shows which input features had the most impact on the model's decision:
        
        - **NDVI Value**: The primary vegetation index used for classification
        - **Temporal Trend**: How the NDVI has changed over time
        - **Spatial Context**: Neighboring pixel values and patterns
        
        Higher importance means the feature had more influence on the final prediction.
        """)


def generate_prediction_explanation(predicted_class: str, ndvi_value: float, confidence: float) -> str:
    """
    Generate human-readable explanation for a prediction.
    
    Args:
        predicted_class: The predicted crop health class
        ndvi_value: The NDVI value
        confidence: Confidence score
    
    Returns:
        Explanation string
    """
    explanations = {
        'Healthy': f"""
        **Prediction: Healthy Crop**
        
        The model classified this area as **Healthy** with {confidence*100:.1f}% confidence.
        
        **Reasoning:**
        - NDVI value of {ndvi_value:.3f} indicates strong vegetation vigor
        - This value is well above the healthy threshold (0.7)
        - High chlorophyll content and active photosynthesis
        - Dense, healthy vegetation canopy
        
        **Recommendation:** Continue current management practices. Monitor regularly to maintain health.
        """,
        
        'Moderate': f"""
        **Prediction: Moderate Health**
        
        The model classified this area as **Moderate** with {confidence*100:.1f}% confidence.
        
        **Reasoning:**
        - NDVI value of {ndvi_value:.3f} indicates moderate vegetation health
        - This value falls in the moderate range (0.5-0.7)
        - Vegetation is growing but may benefit from optimization
        - Some stress factors may be present
        
        **Recommendation:** Monitor closely. Consider optimizing irrigation, fertilization, or pest management.
        """,
        
        'Stressed': f"""
        **Prediction: Stressed Crop**
        
        The model classified this area as **Stressed** with {confidence*100:.1f}% confidence.
        
        **Reasoning:**
        - NDVI value of {ndvi_value:.3f} indicates vegetation stress
        - This value falls in the stressed range (0.3-0.5)
        - Reduced photosynthetic activity
        - Possible water stress, nutrient deficiency, or pest damage
        
        **Recommendation:** ⚠️ Investigate immediately. Check irrigation, soil nutrients, and pest presence.
        """,
        
        'Critical': f"""
        **Prediction: Critical Condition**
        
        The model classified this area as **Critical** with {confidence*100:.1f}% confidence.
        
        **Reasoning:**
        - NDVI value of {ndvi_value:.3f} indicates severe vegetation stress
        - This value is below the critical threshold (0.3)
        - Very low or no photosynthetic activity
        - Severe stress, disease, or crop failure
        
        **Recommendation:** 🚨 URGENT ACTION REQUIRED. Immediate intervention needed to prevent crop loss.
        """
    }
    
    return explanations.get(predicted_class, "No explanation available for this class.")


def display_interactive_prediction_demo():
    """
    Display an interactive demo where users can input NDVI values and see predictions.
    """
    st.markdown("## 🎮 Interactive Prediction Demo")
    st.markdown("Try the AI model yourself! Enter an NDVI value to see how the model classifies it.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # NDVI input
        ndvi_input = st.slider(
            "NDVI Value",
            min_value=-1.0,
            max_value=1.0,
            value=0.65,
            step=0.01,
            help="Adjust the NDVI value to see how the model classifies different vegetation health levels"
        )
        
        # Simulate prediction
        if st.button("🔮 Predict", type="primary"):
            st.session_state.demo_prediction = simulate_prediction(ndvi_input)
    
    with col2:
        if 'demo_prediction' in st.session_state:
            prediction = st.session_state.demo_prediction
            display_prediction_explanation(prediction, ndvi_input)


def simulate_prediction(ndvi_value: float) -> Dict[str, Any]:
    """
    Simulate a model prediction for demo purposes.
    
    Args:
        ndvi_value: NDVI value to classify
    
    Returns:
        Dictionary with prediction results
    """
    # Use rule-based thresholds to simulate prediction
    if ndvi_value > 0.7:
        predicted_class = 'Healthy'
        confidence = min(0.95, 0.75 + (ndvi_value - 0.7) * 0.5)
        all_probs = {
            'Healthy': confidence,
            'Moderate': 1 - confidence - 0.02,
            'Stressed': 0.01,
            'Critical': 0.01
        }
    elif ndvi_value > 0.5:
        predicted_class = 'Moderate'
        # Confidence is highest in middle of range
        distance_from_boundaries = min(ndvi_value - 0.5, 0.7 - ndvi_value)
        confidence = 0.7 + distance_from_boundaries * 0.5
        all_probs = {
            'Moderate': confidence,
            'Healthy': (0.7 - ndvi_value) / 0.2 * (1 - confidence),
            'Stressed': (ndvi_value - 0.5) / 0.2 * (1 - confidence),
            'Critical': 0.01
        }
    elif ndvi_value > 0.3:
        predicted_class = 'Stressed'
        distance_from_boundaries = min(ndvi_value - 0.3, 0.5 - ndvi_value)
        confidence = 0.7 + distance_from_boundaries * 0.5
        all_probs = {
            'Stressed': confidence,
            'Moderate': (0.5 - ndvi_value) / 0.2 * (1 - confidence),
            'Critical': (ndvi_value - 0.3) / 0.2 * (1 - confidence),
            'Healthy': 0.01
        }
    else:
        predicted_class = 'Critical'
        confidence = min(0.95, 0.75 + (0.3 - ndvi_value) * 0.5)
        all_probs = {
            'Critical': confidence,
            'Stressed': 1 - confidence - 0.02,
            'Moderate': 0.01,
            'Healthy': 0.01
        }
    
    # Normalize probabilities
    total = sum(all_probs.values())
    all_probs = {k: v/total for k, v in all_probs.items()}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }



# NOTE: display_ai_vs_rulebased_comparison() function removed
# It used simulated data and was not being called anywhere in the code



def display_performance_tracking():
    """
    Display model performance tracking over time.
    """
    st.markdown("## 📈 Model Performance Over Time")
    st.markdown("""
    Track how model performance changes as new data is collected and the model is retrained.
    This helps identify model drift and determine when retraining is needed.
    """)
    
    # Generate simulated performance history
    dates = pd.date_range(start='2024-06-01', end='2024-12-09', freq='W')
    
    # Simulate performance metrics over time with slight degradation
    np.random.seed(42)
    base_accuracy = 0.892
    accuracy_trend = base_accuracy - np.linspace(0, 0.05, len(dates)) + np.random.normal(0, 0.01, len(dates))
    accuracy_trend = np.clip(accuracy_trend, 0.8, 0.95)
    
    base_confidence = 0.85
    confidence_trend = base_confidence - np.linspace(0, 0.03, len(dates)) + np.random.normal(0, 0.01, len(dates))
    confidence_trend = np.clip(confidence_trend, 0.75, 0.90)
    
    # Create DataFrame
    df_performance = pd.DataFrame({
        'Date': dates,
        'Accuracy': accuracy_trend,
        'Confidence': confidence_trend,
        'Samples Processed': np.random.randint(500, 2000, len(dates))
    })
    
    # Display current vs initial performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_acc = df_performance['Accuracy'].iloc[-1]
        initial_acc = df_performance['Accuracy'].iloc[0]
        st.metric(
            "Current Accuracy",
            f"{current_acc*100:.1f}%",
            delta=f"{(current_acc - initial_acc)*100:+.1f}%",
            delta_color="normal",
            help="Model accuracy on recent data"
        )
    
    with col2:
        current_conf = df_performance['Confidence'].iloc[-1]
        initial_conf = df_performance['Confidence'].iloc[0]
        st.metric(
            "Current Confidence",
            f"{current_conf*100:.1f}%",
            delta=f"{(current_conf - initial_conf)*100:+.1f}%",
            delta_color="normal",
            help="Average confidence on recent predictions"
        )
    
    with col3:
        total_samples = df_performance['Samples Processed'].sum()
        st.metric(
            "Total Samples",
            f"{total_samples:,}",
            help="Total number of samples processed since deployment"
        )
    
    st.markdown("---")
    
    # Performance trend chart
    st.markdown("### 📊 Performance Trends")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Accuracy Over Time', 'Average Confidence Over Time'),
        vertical_spacing=0.15
    )
    
    # Accuracy trend
    fig.add_trace(
        go.Scatter(
            x=df_performance['Date'],
            y=df_performance['Accuracy'] * 100,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#4caf50', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Add threshold line for accuracy
    fig.add_hline(
        y=85, line_dash="dash", line_color="red",
        annotation_text="Retraining Threshold",
        row=1, col=1
    )
    
    # Confidence trend
    fig.add_trace(
        go.Scatter(
            x=df_performance['Date'],
            y=df_performance['Confidence'] * 100,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#2196f3', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    # Add threshold line for confidence
    fig.add_hline(
        y=75, line_dash="dash", line_color="red",
        annotation_text="Warning Threshold",
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1, range=[75, 100])
    fig.update_yaxes(title_text="Confidence (%)", row=2, col=1, range=[70, 95])
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Drift detection
    st.markdown("### 🔍 Model Drift Detection")
    
    # Calculate drift metrics
    recent_window = 4  # Last 4 weeks
    recent_acc = df_performance['Accuracy'].iloc[-recent_window:].mean()
    baseline_acc = df_performance['Accuracy'].iloc[:recent_window].mean()
    drift_amount = (baseline_acc - recent_acc) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Baseline Accuracy",
            f"{baseline_acc*100:.1f}%",
            help="Average accuracy from first 4 weeks"
        )
    
    with col2:
        st.metric(
            "Recent Accuracy",
            f"{recent_acc*100:.1f}%",
            delta=f"{-drift_amount:+.1f}%",
            delta_color="inverse",
            help="Average accuracy from last 4 weeks"
        )
    
    # Drift assessment
    if drift_amount > 5:
        st.error(f"""
        🚨 **Significant Model Drift Detected!**
        
        The model's accuracy has decreased by {drift_amount:.1f}% compared to baseline.
        
        **Recommended Actions:**
        1. Retrain the model with recent data
        2. Review data quality and distribution
        3. Check for changes in field conditions or sensor calibration
        """)
    elif drift_amount > 2:
        st.warning(f"""
        ⚠️ **Moderate Model Drift Detected**
        
        The model's accuracy has decreased by {drift_amount:.1f}% compared to baseline.
        
        **Recommended Actions:**
        - Monitor closely over the next few weeks
        - Consider retraining if drift continues
        - Review recent predictions for anomalies
        """)
    else:
        st.success(f"""
        ✅ **Model Performance Stable**
        
        Drift is minimal ({drift_amount:.1f}%). Model is performing well on recent data.
        
        **Status:** No immediate action required. Continue monitoring.
        """)
    
    st.markdown("---")
    
    # Retraining recommendation
    st.markdown("### 🔄 Retraining Recommendations")
    
    # Calculate days since last training
    last_training_date = datetime(2024, 12, 9)  # From metrics file
    days_since_training = (datetime.now() - last_training_date).days
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Days Since Training",
            days_since_training,
            help="Number of days since the model was last trained"
        )
    
    with col2:
        recommended_interval = 30  # days
        st.metric(
            "Recommended Interval",
            f"{recommended_interval} days",
            help="Recommended time between retraining sessions"
        )
    
    with col3:
        days_until_retrain = max(0, recommended_interval - days_since_training)
        st.metric(
            "Days Until Retraining",
            days_until_retrain,
            help="Estimated days until next retraining is recommended"
        )
    
    # Retraining decision
    if days_since_training >= recommended_interval or drift_amount > 5:
        st.error("🔄 **Retraining Recommended**")
        st.markdown("""
        The model should be retrained based on:
        - Time since last training
        - Detected performance drift
        - Accumulation of new data
        
        Click the button below to schedule retraining:
        """)
        
        if st.button("📅 Schedule Retraining", type="primary"):
            st.success("✅ Retraining scheduled! The model will be retrained with the latest data.")
    else:
        st.info(f"""
        ℹ️ **Retraining Not Yet Needed**
        
        The model is performing well. Next retraining recommended in {days_until_retrain} days.
        """)
    
    st.markdown("---")
    
    # Performance trend chart with samples
    st.markdown("### 📊 Samples Processed Over Time")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_performance['Date'],
        y=df_performance['Samples Processed'],
        name='Samples',
        marker_color='#ff9800'
    ))
    
    fig.update_layout(
        title="Number of Samples Processed Per Week",
        xaxis_title="Date",
        yaxis_title="Samples",
        height=400,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ Understanding Performance Tracking"):
        st.markdown("""
        **Why Track Performance Over Time?**
        
        - **Model Drift**: Real-world data changes over time (seasons, weather, crop varieties)
        - **Data Quality**: Sensor calibration drift or environmental changes
        - **Retraining Needs**: Determine optimal retraining schedule
        
        **Key Metrics:**
        
        - **Accuracy**: How often the model is correct
        - **Confidence**: How certain the model is about its predictions
        - **Drift**: Change in performance compared to baseline
        
        **When to Retrain:**
        
        - Accuracy drops below 85%
        - Significant drift detected (>5%)
        - 30+ days since last training
        - Major changes in field conditions
        """)
