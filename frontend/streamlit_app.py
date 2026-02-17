import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import os

# Configuration
API_URL = os.getenv("API_URL", "https://manufacturing-api-z7yd.onrender.com")

st.set_page_config(
    page_title="Manufacturing Analytics", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🎯"
)


def _encode_categorical_features(user_data, features, shift_choice, machine_choice, material_choice, day_choice):
    """Encode categorical features into one-hot format"""
    
    # Shift encoding
    if "Shift_Evening" in features:
        user_data["Shift_Evening"] = 1.0 if shift_choice == "Evening" else 0.0
    if "Shift_Night" in features:
        user_data["Shift_Night"] = 1.0 if shift_choice == "Night" else 0.0
    
    # Machine type encoding
    if "Machine_Type_Type_B" in features:
        user_data["Machine_Type_Type_B"] = 1.0 if machine_choice == "Type_B" else 0.0
    if "Machine_Type_Type_C" in features:
        user_data["Machine_Type_Type_C"] = 1.0 if machine_choice == "Type_C" else 0.0
    
    # Material grade encoding
    if "Material_Grade_Premium" in features:
        user_data["Material_Grade_Premium"] = 1.0 if material_choice == "Premium" else 0.0
    if "Material_Grade_Standard" in features:
        user_data["Material_Grade_Standard"] = 1.0 if material_choice == "Standard" else 0.0
    if "Material_Grade_Economy" in features:
        user_data["Material_Grade_Economy"] = 1.0 if material_choice == "Economy" else 0.0
    
    # Day of week encoding
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        col = f"Day_of_Week_{day}"
        if col in features:
            user_data[col] = 1.0 if day_choice == day else 0.0


def _handle_prediction(API_URL, user_data, shift_choice, machine_choice, material_choice, day_choice, numeric_features):
    """Handle the prediction request and display results"""
    
    res = requests.post(f"{API_URL}/predict", json={"data": user_data})
    
    if res.status_code == 200:
        pred = res.json()["prediction"]
        
        # Enhanced prediction display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Production Output (Parts/Hour)", 'font': {'size': 20, 'color': "white"}},
                delta={'reference': 35, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 70], 'tickwidth': 1, 'tickcolor': "red"},
                    'bar': {'color': "darkblue", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': "#ffcccc"},
                        {'range': [25, 40], 'color': "#fff2cc"},
                        {'range': [40, 50], 'color': "#d4edda"},
                        {'range': [50, 70], 'color': "#ccffcc"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': pred
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Performance categorization
            if pred >= 50:
                st.success(" **Excellent Performance**\n\nAbove industry benchmark!")
            elif pred >= 40:
                st.info(" **Good Performance**\n\nRoom for optimization")
            elif pred >= 25:
                st.warning(" **Average Performance**\n\nConsider parameter adjustments")
            else:
                st.error(" **Below Average**\n\nImmediate optimization needed")
            
            # Quick stats
            st.metric("Predicted Output", f"{pred:.1f} parts/hr")
            st.metric("vs Industry Avg", f"{pred-35:.1f}", delta=f"{((pred-35)/35)*100:.1f}%")
            
            # Current settings summary
            with st.expander(" Current Settings"):
                st.write(f"**Shift:** {shift_choice}")
                st.write(f"**Machine:** {machine_choice}")
                st.write(f"**Material:** {material_choice}")
                st.write(f"**Day:** {day_choice}")
                
    else:
        st.error("Prediction failed")
        st.write(res.text)


st.title(" Manufacturing Production Predictor")

# Fetch feature columns from API
try:
    features = requests.get(f"{API_URL}/features").json()["feature_columns"]
except:
    st.error("Backend API not running. Start FastAPI first.")
    st.stop()

st.write("Configure your manufacturing parameters below:")

# Dropdown UI (human-friendly) in columns
col1, col2 = st.columns(2)
with col1:
    shift_choice = st.selectbox("Shift", ["Day", "Evening", "Night"])
    machine_choice = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
with col2:
    material_choice = st.selectbox("Material Grade", ["Standard", "Economy", "Premium"]) 
    day_choice = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

st.divider()

# Create default input dict
user_data = {feat: 0.0 for feat in features}

# Numeric input fields
one_hot_prefixes = ("Shift_", "Machine_Type_", "Material_Grade_", "Day_of_Week_")
numeric_features = [feat for feat in features if not feat.startswith(one_hot_prefixes)]

if len(numeric_features) > 0:
    st.header(" Manufacturing Parameters")
    
    # Organize numeric inputs in columns
    cols = st.columns(min(3, len(numeric_features)))
    for i, feat in enumerate(numeric_features):
        with cols[i % len(cols)]:
            user_data[feat] = st.number_input(
                feat.replace('_', ' ').title(),
                value=0.0,
                help=f"Enter the {feat.lower()} value"
            )

# Apply dropdown values to one-hot columns
_encode_categorical_features(user_data, features, shift_choice, machine_choice, material_choice, day_choice)
    
# Prediction button
if st.button(" Predict Production Output", type="primary", use_container_width=True):
    _handle_prediction(API_URL, user_data, shift_choice, machine_choice, material_choice, day_choice, numeric_features)
