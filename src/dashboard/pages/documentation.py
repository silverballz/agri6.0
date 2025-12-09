"""Documentation page for AgriFlux Dashboard"""

import streamlit as st
from pathlib import Path

def show_page():
    """Display documentation page with links to guides"""
    
    st.title("📚 Documentation & Resources")
    st.markdown("---")
    
    # Quick links section
    st.markdown("## 🚀 Quick Links")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🛰️ Real Data Pipeline
        - [Complete Guide](../docs/REAL_DATA_PIPELINE_GUIDE.md)
        - [Quick Reference](../docs/REAL_DATA_QUICK_REFERENCE.md)
        - [API Troubleshooting](../docs/API_TROUBLESHOOTING_GUIDE.md)
        """)
    
    with col2:
        st.markdown("""
        ### 📖 User Guides
        - [User Guide](../docs/user-guide.md)
        - [FAQ](../docs/faq.md)
        - [Training Materials](../docs/training-materials.md)
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 Technical Docs
        - [Technical Documentation](../docs/technical-documentation.md)
        - [Model Deployment](../docs/MODEL_DEPLOYMENT_GUIDE.md)
        - [Logging System](../docs/LOGGING_SYSTEM.md)
        """)
    
    st.markdown("---")
    
    # Real Data Pipeline Section
    st.markdown("## 🛰️ Real Satellite Data Pipeline")
    
    with st.expander("📥 **Downloading Real Satellite Data**", expanded=False):
        st.markdown("""
        ### Overview
        The AgriFlux platform now supports downloading real Sentinel-2 satellite imagery 
        from the Sentinel Hub API, replacing synthetic data with actual agricultural observations.
        
        ### Quick Start
        ```bash
        # Step 1: Set up credentials
        export SENTINEL_HUB_CLIENT_ID=your_client_id
        export SENTINEL_HUB_CLIENT_SECRET=your_client_secret
        
        # Step 2: Download real data
        python scripts/download_real_satellite_data.py --target-count 20
        
        # Step 3: Validate data quality
        python scripts/validate_data_quality.py
        ```
        
        ### Features
        - ✅ Automatic date validation (prevents future date queries)
        - ✅ STAC API v1 compliant requests
        - ✅ Exponential backoff retry logic
        - ✅ Cloud coverage filtering (<20%)
        - ✅ Multi-temporal data support (15-20 dates)
        - ✅ Comprehensive logging and error handling
        
        ### Documentation
        - 📖 [Complete Pipeline Guide](../docs/REAL_DATA_PIPELINE_GUIDE.md)
        - ⚡ [Quick Reference](../docs/REAL_DATA_QUICK_REFERENCE.md)
        - 🔍 [API Troubleshooting](../docs/API_TROUBLESHOOTING_GUIDE.md)
        """)
    
    with st.expander("🤖 **Training AI Models on Real Data**", expanded=False):
        st.markdown("""
        ### Overview
        Train CNN and LSTM models on real satellite imagery for production-ready accuracy.
        
        ### Training Workflow
        ```bash
        # Step 1: Prepare CNN training data
        python scripts/prepare_real_training_data.py --samples-per-class 2000
        
        # Step 2: Prepare LSTM training data
        python scripts/prepare_lstm_training_data.py --sequence-length 10
        
        # Step 3: Train CNN model
        python scripts/train_cnn_on_real_data.py --epochs 50 --min-accuracy 0.85
        
        # Step 4: Train LSTM model
        python scripts/train_lstm_on_real_data.py --epochs 100 --min-accuracy 0.80
        
        # Step 5: Deploy models
        python scripts/deploy_real_trained_models.py
        ```
        
        ### Expected Results
        - **CNN Model**: ≥85% validation accuracy
        - **LSTM Model**: ≥80% validation accuracy
        - **Improvement**: +5-15% over synthetic data
        - **Training Time**: 15-30 min (GPU), 2-4 hours (CPU)
        
        ### Documentation
        - 📊 [Model Deployment Guide](../docs/MODEL_DEPLOYMENT_GUIDE.md)
        - 📝 [Scripts Documentation](../scripts/README_REAL_DATA_PIPELINE.md)
        """)
    
    with st.expander("🔍 **Data Quality & Validation**", expanded=False):
        st.markdown("""
        ### Overview
        Ensure downloaded imagery meets quality requirements before training.
        
        ### Validation Checks
        - ✅ All required bands present (B02, B03, B04, B08)
        - ✅ Vegetation indices within valid ranges
        - ✅ Minimum 15 imagery dates available
        - ✅ Metadata synthetic flag is false
        - ✅ No corrupted or incomplete downloads
        
        ### Running Validation
        ```bash
        python scripts/validate_data_quality.py
        ```
        
        ### Validation Report
        The script generates a detailed JSON report with:
        - Band completeness check
        - Index range validation
        - Temporal coverage verification
        - Data provenance confirmation
        - Quality score (0-100)
        
        ### Documentation
        - 📋 [Data Quality Guide](../docs/REAL_DATA_PIPELINE_GUIDE.md#data-quality-validation)
        """)
    
    with st.expander("🐛 **Troubleshooting Common Issues**", expanded=False):
        st.markdown("""
        ### Common Issues & Quick Fixes
        
        | Issue | Quick Fix |
        |-------|-----------|
        | `401 Unauthorized` | Check credentials in `.env` |
        | `406 Not Acceptable` | Update to latest API client |
        | `429 Rate Limited` | Wait or reduce `--target-count` |
        | `No imagery found` | Increase `--cloud-threshold` or `--days-back` |
        | `Insufficient training data` | Download more imagery |
        | `Model accuracy below threshold` | Download more data or adjust hyperparameters |
        | `Out of memory` | Reduce `--batch-size` |
        | `Database locked` | Kill conflicting processes |
        
        ### Getting Help
        1. Check logs in `logs/` directory
        2. Review [API Troubleshooting Guide](../docs/API_TROUBLESHOOTING_GUIDE.md)
        3. Run validation scripts
        4. Check [FAQ](../docs/faq.md)
        
        ### Documentation
        - 🔍 [Complete Troubleshooting Guide](../docs/API_TROUBLESHOOTING_GUIDE.md)
        - ⚡ [Quick Reference](../docs/REAL_DATA_QUICK_REFERENCE.md)
        """)
    
    st.markdown("---")
    
    # System Information Section
    st.markdown("## 📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 File Locations
        - **Processed imagery**: `data/processed/`
        - **Training data**: `data/training/`
        - **Models**: `models/`
        - **Logs**: `logs/`
        - **Reports**: `reports/`
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 External Resources
        - [Sentinel Hub API Docs](https://docs.sentinel-hub.com/)
        - [Sentinel-2 Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
        - [STAC API Spec](https://github.com/radiantearth/stac-api-spec)
        - [PyTorch Docs](https://pytorch.org/docs/)
        """)
    
    st.markdown("---")
    
    # Performance Benchmarks
    st.markdown("## ⚡ Performance Benchmarks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📥 Download Performance
        - **1 imagery date**: ~30-60 seconds
        - **20 imagery dates**: ~10-20 minutes
        - **Bottleneck**: API rate limits
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Training Performance (GPU)
        - **CNN (50 epochs)**: ~15-30 minutes
        - **LSTM (100 epochs)**: ~30-60 minutes
        - **Bottleneck**: Dataset size
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Expected Accuracy
        - **CNN**: 85-92% validation
        - **LSTM**: 80-88% validation
        - **Improvement**: +5-15% over synthetic
        """)
    
    st.markdown("---")
    
    # Scripts Reference
    st.markdown("## 📝 Scripts Reference")
    
    scripts_info = {
        "download_real_satellite_data.py": {
            "description": "Download real Sentinel-2 imagery from Sentinel Hub API",
            "usage": "python scripts/download_real_satellite_data.py --target-count 20",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "validate_data_quality.py": {
            "description": "Validate downloaded imagery meets quality requirements",
            "usage": "python scripts/validate_data_quality.py",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "prepare_real_training_data.py": {
            "description": "Prepare CNN training dataset from real imagery",
            "usage": "python scripts/prepare_real_training_data.py --samples-per-class 2000",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "prepare_lstm_training_data.py": {
            "description": "Prepare LSTM training dataset from temporal sequences",
            "usage": "python scripts/prepare_lstm_training_data.py --sequence-length 10",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "train_cnn_on_real_data.py": {
            "description": "Train CNN model on real satellite imagery",
            "usage": "python scripts/train_cnn_on_real_data.py --epochs 50",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "train_lstm_on_real_data.py": {
            "description": "Train LSTM model on real temporal data",
            "usage": "python scripts/train_lstm_on_real_data.py --epochs 100",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "compare_model_performance.py": {
            "description": "Compare synthetic-trained vs real-trained models",
            "usage": "python scripts/compare_model_performance.py",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        },
        "deploy_real_trained_models.py": {
            "description": "Deploy real-trained models to production",
            "usage": "python scripts/deploy_real_trained_models.py",
            "docs": "scripts/README_REAL_DATA_PIPELINE.md"
        }
    }
    
    for script_name, info in scripts_info.items():
        with st.expander(f"📜 **{script_name}**", expanded=False):
            st.markdown(f"""
            **Description**: {info['description']}
            
            **Usage**:
            ```bash
            {info['usage']}
            ```
            
            **Documentation**: [{info['docs']}](../{info['docs']})
            """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #94a3b8;">
        <p>📚 For complete documentation, visit the <code>docs/</code> directory</p>
        <p>💡 Need help? Check the <a href="../docs/API_TROUBLESHOOTING_GUIDE.md">Troubleshooting Guide</a></p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">AgriFlux v1.0.0 - Real Data Pipeline Integration Complete 🎉</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_page()
