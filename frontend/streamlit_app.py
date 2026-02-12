import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://manufacturing-api-z7yd.onrender.com")

st.set_page_config(page_title="Manufacturing App", layout="centered")
st.title("🏭 Manufacturing Prediction App")

# -----------------------------
# Get features from backend
# -----------------------------
try:
    features = requests.get(f"{API_URL}/features").json()["feature_columns"]
except:
    st.error("Backend API not running. Start FastAPI first.")
    st.stop()

st.write("Fill feature values below:")

user_data = {}

# -----------------------------
# Dropdown selections
# -----------------------------
shift_choice = st.selectbox("Shift", ["Evening", "Night"])
machine_choice = st.selectbox("Machine Type", ["Type_B", "Type_C"])
material_choice = st.selectbox("Material Grade", ["Premium", "Standard"])
day_choice = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

st.divider()

# -----------------------------
# Add numeric inputs ONLY for non-onehot columns
# -----------------------------
onehot_prefixes = [
    "Shift_",
    "Machine_Type_",
    "Material_Grade_",
    "Day_of_Week_"
]

for feat in features:
    # skip one-hot columns (handled by dropdowns)
    if any(feat.startswith(prefix) for prefix in onehot_prefixes):
        continue

    user_data[feat] = st.number_input(feat, value=0.0)

# -----------------------------
# Now add one-hot values from dropdowns
# -----------------------------
# Shift
user_data["Shift_Evening"] = 1.0 if shift_choice == "Evening" else 0.0
user_data["Shift_Night"] = 1.0 if shift_choice == "Night" else 0.0

# Machine Type
user_data["Machine_Type_Type_B"] = 1.0 if machine_choice == "Type_B" else 0.0
user_data["Machine_Type_Type_C"] = 1.0 if machine_choice == "Type_C" else 0.0

# Material Grade
user_data["Material_Grade_Premium"] = 1.0 if material_choice == "Premium" else 0.0
user_data["Material_Grade_Standard"] = 1.0 if material_choice == "Standard" else 0.0

# Day of Week
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for d in days:
    col_name = f"Day_of_Week_{d}"
    user_data[col_name] = 1.0 if day_choice == d else 0.0

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
