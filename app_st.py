import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="BMW Sales Analysis",
    layout="wide"
)

st.title("🚗 BMW Sales Classification Analysis")
st.write("Exploratory Data Analysis + Prediction (No Model Training)")

# =============================
# Load Data
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("cleaning_data.csv")

df = load_data()

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    return joblib.load("module_SVC.pkl")

model = load_model()

# =============================
# Sidebar
# =============================
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "EDA", "Prediction"]
)

# =============================
# Dataset Overview
# =============================
if page == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Shape")
        st.write(df.shape)

    with col2:
        st.write("### Missing Values")
        st.write(df.isna().sum())

# =============================
# EDA Section
# =============================
elif page == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    numeric_cols = [
        "engine_size_l",
        "mileage_km",
        "price_usd",
        "sales_volume",
        "age_car"
    ]

    cat_cols = [
        "model",
        "region",
        "color",
        "fuel_type",
        "transmission",
        "sales_classification"
    ]

    st.write("### Numerical Features")
    selected_num = st.selectbox("Choose Numerical Feature", numeric_cols)
    fig = px.histogram(df, x=selected_num, color="sales_classification")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Categorical Features")
    selected_cat = st.selectbox("Choose Categorical Feature", cat_cols)
    fig2 = px.histogram(df, x=selected_cat, color="sales_classification")
    st.plotly_chart(fig2, use_container_width=True)

# =============================
# Prediction Section
# =============================
elif page == "Prediction":
    st.subheader("🤖 Sales Classification Prediction")

    col1, col2 = st.columns(2)

    with col1:
        model_name = st.selectbox("Model", df["model"].unique())
        region = st.selectbox("Region", df["region"].unique())
        color = st.selectbox("Color", df["color"].unique())
        fuel = st.selectbox("Fuel Type", df["fuel_type"].unique())
        transmission = st.selectbox("Transmission", df["transmission"].unique())

    with col2:
        engine = st.number_input("Engine Size (L)", 1.0, 6.0, 2.0)
        mileage = st.number_input("Mileage (KM)", 0.0, 500000.0, 50000.0)
        price = st.number_input("Price USD", 10000.0, 200000.0, 50000.0)
        volume = st.number_input("Sales Volume", 0, 10000, 3000)
        age = st.number_input("Car Age", 0, 20, 5)

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "model": model_name,
            "region": region,
            "color": color,
            "fuel_type": fuel,
            "transmission": transmission,
            "engine_size_l": engine,
            "mileage_km": mileage,
            "price_usd": price,
            "sales_volume": volume,
            "age_car": age
        }])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)

        label = "High" if pred == 0 else "Low"

        st.success(f"📌 Predicted Sales Classification: **{label}**")
        st.write("### Prediction Probability")
        st.bar_chart(pd.DataFrame(proba, columns=["High", "Low"]))

