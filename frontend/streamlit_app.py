import streamlit as st
import requests
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import page modules
try:
    from pages.predictor_page import show_predictor_page
    from pages.performance_page import show_performance_page  
    from pages.analytics_page import show_analytics_page
except ImportError:
    st.error("Unable to import page modules. Please check file structure.")
    st.stop()

# "https://manufacturing-api-z7yd.onrender.com"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Manufacturing Analytics", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "predictor"

# Navigation Sidebar with Buttons
st.sidebar.title("🏭 Manufacturing Analytics")
st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button("🎯 Production Predictor", use_container_width=True):
    st.session_state.current_page = "predictor"

if st.sidebar.button("📊 Model Performance", use_container_width=True):
    st.session_state.current_page = "performance"

if st.sidebar.button("📈 Analytics Dashboard", use_container_width=True):
    st.session_state.current_page = "analytics"

st.sidebar.markdown("---")

# Display current page indicator
page_names = {
    "predictor": "🎯 Production Predictor",
    "performance": "📊 Model Performance", 
    "analytics": "📈 Analytics Dashboard"
}
st.sidebar.info(f"**Current Page:** {page_names[st.session_state.current_page]}")

# Add footer with additional info
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 System Status:**
- ✅ API Connected
- ✅ Model v2.0 Ready
- ✅ Real-time Predictions

**📞 Support:**
- Documentation: Coming Soon
- Bug Reports: Use Analytics page
""")

# Page Content Based on Navigation
if st.session_state.current_page == "predictor":
    show_predictor_page(API_URL)

elif st.session_state.current_page == "performance":
    show_performance_page(API_URL)

elif st.session_state.current_page == "analytics":
    show_analytics_page()
