"""
Help System Module - Comprehensive inline documentation and FAQ for AgriFlux Dashboard
"""

import streamlit as st

# FAQ Content
FAQ_GETTING_STARTED = [
    {"question": "How do I get started with AgriFlux?", "answer": "1. Select fields\n2. Choose date range\n3. Pick indices\n4. Navigate pages"},
    {"question": "What is Demo Mode?", "answer": "Demo Mode loads sample data for demonstrations."}
]

FAQ_VEGETATION = [
    {"question": "Which vegetation index should I use?", "answer": "NDVI - General\nSAVI - Sparse vegetation\nEVI - Dense vegetation"},
    {"question": "What do NDVI values mean?", "answer": "0.8-1.0: Excellent\n0.6-0.8: Healthy\n0.4-0.6: Moderate\n<0.4: Stressed"}
]

FAQ_ALERTS = [
    {"question": "What do alert severity levels mean?", "answer": "Critical: Act now\nHigh: 24h\nMedium: 48-72h\nLow: Info"}
]

def show_faq_section(category="all"):
    """Display FAQ section"""
    categories = {"Getting Started": FAQ_GETTING_STARTED, "Vegetation Indices": FAQ_VEGETATION, "Alerts": FAQ_ALERTS}
    if category == "all":
        for cat_name, questions in categories.items():
            st.subheader(f"📚 {cat_name}")
            for qa in questions:
                with st.expander(qa["question"]):
                    st.markdown(qa["answer"])
    else:
        questions = categories.get(category, [])
        for qa in questions:
            with st.expander(qa["question"]):
                st.markdown(qa["answer"])

def show_quick_help():
    """Display quick help in sidebar"""
    st.sidebar.markdown("### 📚 Quick Help")
    with st.sidebar.expander("🚀 Getting Started"):
        st.markdown("1. Select fields\n2. Choose date range\n3. Pick indices\n4. Navigate pages")
    with st.sidebar.expander("📊 NDVI Values"):
        st.markdown("• 0.8-1.0: Excellent 🟢\n• 0.6-0.8: Healthy 🟢\n• 0.4-0.6: Moderate 🟡\n• <0.4: Stressed 🔴")
