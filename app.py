import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os

# ==============================
# Load or Generate Dataset
# ==============================
data_path = "data/crop_yield_sample.csv"

if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
        if df.empty or df.shape[1] == 0:
            raise ValueError("CSV file is empty or has no columns")
    except Exception as e:
        st.warning(f"⚠️ Problem with CSV: {e}. Generating synthetic dataset.")
        df = pd.DataFrame({
            "Rainfall_mm": [350, 420, 500, 600, 700, 300, 450, 550],
            "Temperature_C": [18, 20, 22, 25, 27, 30, 24, 26],
            "Fertilizer_kg": [80, 100, 110, 120, 130, 90, 115, 125],
            "Yield_kg_ha": [2800, 3200, 3500, 3800, 3700, 2600, 3300, 3600]
        })
else:
    st.warning("⚠️ CSV not found. Generating synthetic dataset.")
    df = pd.DataFrame({
        "Rainfall_mm": [350, 420, 500, 600, 700, 300, 450, 550],
        "Temperature_C": [18, 20, 22, 25, 27, 30, 24, 26],
        "Fertilizer_kg": [80, 100, 110, 120, 130, 90, 115, 125],
        "Yield_kg_ha": [2800, 3200, 3500, 3800, 3700, 2600, 3300, 3600]
    })

# ==============================
# Train Model
# ==============================
X = df[["Rainfall_mm", "Temperature_C", "Fertilizer_kg"]]
y = df["Yield_kg_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# Streamlit UI
# ==============================
st.title("🌱 Crop Yield Prediction DSS")
st.write("Predict crop yield based on rainfall, temperature, and fertilizer use.")

# --- User Inputs
rainfall = st.slider("Rainfall (mm)", 200, 800, 500)
temperature = st.slider("Temperature (°C)", 15, 35, 25)
fertilizer = st.slider("Fertilizer (kg/hectare)", 50, 150, 100)

# --- Prediction
pred_yield = model.predict([[rainfall, temperature, fertilizer]])[0]
st.subheader(f"Predicted Yield: {pred_yield:.2f} kg/ha")

# ==============================
# Visualizations
# ==============================
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

st.subheader("📈 Fertilizer vs Yield")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="Fertilizer_kg", y="Yield_kg_ha", data=df, ax=ax2)
st.pyplot(fig2)

