import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://manufacturing-api-z7yd.onrender.com")

st.set_page_config(page_title="Manufacturing App", layout="centered")
st.title("🏭 Manufacturing Prediction App")

# -----------------------------
# Fetch feature columns from API
# -----------------------------
try:
    features = requests.get(f"{API_URL}/features").json()["feature_columns"]
except:
    st.error("Backend API not running. Start FastAPI first.")
    st.stop()

st.write("Fill feature values below:")

# -----------------------------
# Dropdown UI (human-friendly)
# -----------------------------
shift_choice = st.selectbox("Shift", ["Day", "Evening", "Night"])
machine_choice = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
material_choice = st.selectbox("Material Grade", ["Standard", "Premium"])
day_choice = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

st.divider()

# -----------------------------
# Create default input dict (all zeros)
# -----------------------------
user_data = {feat: 0.0 for feat in features}

# -----------------------------
# Numeric input fields (only real numeric ones)
# Ignore one-hot fields because dropdown will handle them
# -----------------------------
one_hot_prefixes = (
    "Shift_",
    "Machine_Type_",
    "Material_Grade_",
    "Day_of_Week_"
)

for feat in features:
    if feat.startswith(one_hot_prefixes):
        continue
    user_data[feat] = st.number_input(feat, value=0.0)

# -----------------------------
# Apply dropdown values into one-hot columns
# -----------------------------

# Shift encoding
# (your dataset uses Shift_Evening and Shift_Night, Day means both 0)
if "Shift_Evening" in features:
    user_data["Shift_Evening"] = 1.0 if shift_choice == "Evening" else 0.0
if "Shift_Night" in features:
    user_data["Shift_Night"] = 1.0 if shift_choice == "Night" else 0.0

# Machine type encoding
# (Type_A is baseline -> no column)
if "Machine_Type_Type_B" in features:
    user_data["Machine_Type_Type_B"] = 1.0 if machine_choice == "Type_B" else 0.0
if "Machine_Type_Type_C" in features:
    user_data["Machine_Type_Type_C"] = 1.0 if machine_choice == "Type_C" else 0.0

# Material grade encoding
# (maybe both columns exist)
if "Material_Grade_Premium" in features:
    user_data["Material_Grade_Premium"] = 1.0 if material_choice == "Premium" else 0.0
if "Material_Grade_Standard" in features:
    user_data["Material_Grade_Standard"] = 1.0 if material_choice == "Standard" else 0.0

# Day of week encoding
for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    col = f"Day_of_Week_{day}"
    if col in features:
        user_data[col] = 1.0 if day_choice == day else 0.0

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):
    res = requests.post(f"{API_URL}/predict", json={"data": user_data})

    if res.status_code == 200:
        pred = res.json()["prediction"]
        st.success(f"Prediction: {pred}")
    else:
        st.error("Prediction failed")
        st.write(res.text)
