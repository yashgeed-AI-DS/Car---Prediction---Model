import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "linear_regression_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file not found. Please make sure "
        "'linear_regression_model.pkl' is in the same folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🚗 Used Car Price Prediction")
st.markdown(
    "### Predict the estimated price of a used car using Linear Regression"
)

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("🚗 Car Prediction")

st.sidebar.info(
    "Enter the encoded feature values used during model training."
)

# ---------------------------------------------------------
# FEATURE INPUTS
# ---------------------------------------------------------
st.subheader("🔧 Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    Brand = st.number_input(
        "Brand",
        value=0.0,
        step=1.0
    )

    Body = st.number_input(
        "Body",
        value=0.0,
        step=1.0
    )

    Mileage = st.number_input(
        "Mileage",
        value=100.0,
        step=1.0
    )

with col2:
    EngineV = st.number_input(
        "EngineV",
        value=2.0,
        min_value=0.1,
        max_value=10.0,
        step=0.1
    )

    Engine_Type = st.number_input(
        "Engine Type",
        value=0.0,
        step=1.0
    )

    Registration = st.number_input(
        "Registration",
        value=1.0,
        step=1.0
    )

with col3:
    Year = st.number_input(
        "Year",
        value=2015.0,
        min_value=1950.0,
        max_value=2026.0,
        step=1.0
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Predict Car Price", use_container_width=True):

    # IMPORTANT:
    # Keep the column names and order exactly the same
    # as the features used during model training.

    input_data = pd.DataFrame({
        "Brand": [Brand],
        "Body": [Body],
        "Mileage": [Mileage],
        "EngineV": [EngineV],
        "Engine Type": [Engine_Type],
        "Registration": [Registration],
        "Year": [Year]
    })

    try:
        # Predict Log_price
        log_prediction = model.predict(input_data)[0]

        # Convert Log_price back to original Price
        predicted_price = np.exp(log_prediction)

        st.success("Prediction completed successfully!")

        st.metric(
            "💰 Estimated Car Price",
            f"${predicted_price:,.2f}"
        )

        st.info(
            f"Model output (Log Price): {log_prediction:.4f}"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
st.markdown("---")

st.subheader("📌 Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.write("**Algorithm**")
    st.write("Linear Regression")

with info2:
    st.write("**Target Variable**")
    st.write("Log_price")

with info3:
    st.write("**Final Output**")
    st.write("Estimated Car Price")

st.markdown("---")

st.caption(
    "Built with ❤️ using Python, Pandas, NumPy, Scikit-learn and Streamlit"
)

