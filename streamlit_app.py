import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# ---------------- BACKGROUND IMAGE ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/abstract-big-data-digital-technology-background-design_1017-22920.jpg?semt=ais_hybrid&w=740&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }

    .stButton>button {
        background-color: #1F77B4;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
    }

    .stNumberInput>div>input {
        background-color: #F0F2F6;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown("<h1>🍽️ Food Demand Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<h3>Predict Future Checkout Price using regression models</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop ID column if exists
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    st.success("✅ Dataset Loaded Successfully")
    st.write("### Sample Dataset")
    

    # ---------------- LINEAR REGRESSION ----------------
    features = ['week', 'center_id', 'meal_id', 'base_price',
                'emailer_for_promotion', 'homepage_featured']

    X = df[features]
    y = df['checkout_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # ---------------- ARIMA MODEL ----------------
    arima_model = ARIMA(y, order=(1, 1, 1))
    arima_fit = arima_model.fit()

    # ---------------- USER INPUT ----------------
    st.subheader("🔢 Enter Input Values for Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        week = st.number_input("Week", min_value=1, max_value=200, value=1, step=1)
        base_price = st.number_input("Base Price (₹)", min_value=50.0, max_value=1000.0, value=100.0, step=5.0)

    with c2:
        center_id = st.number_input("Center ID", min_value=1, max_value=100, value=1, step=1)
        meal_id = st.number_input("Meal ID", min_value=1, max_value=3000, value=1, step=10)

    with c3:
        emailer = st.selectbox("Emailer Promotion", [0, 1])
        homepage = st.selectbox("Homepage Featured", [0, 1])
    # ---------------- PREDICTION ----------------
    if st.button("🔮 Predict Checkout Price"):
        input_df = pd.DataFrame({
            'week': [week],
            'center_id': [center_id],
            'meal_id': [meal_id],
            'base_price': [base_price],
            'emailer_for_promotion': [emailer],
            'homepage_featured': [homepage]
        })

        # Linear Regression Prediction
        lr_pred = lr_model.predict(input_df)[0]

        # ARIMA Prediction
        arima_pred = arima_fit.forecast(steps=1).iloc[0]

        # Final Combined Prediction
        final_prediction = (lr_pred + arima_pred) / 2

        st.markdown("---")
        st.markdown(
            f"""
            <h2>✅ Predicted Checkout Price</h2>
            <h1 style="color:#2ECC71;">₹ {final_prediction:.2f}</h1>
            """,
            unsafe_allow_html=True
        )
        st.success("Prediction generated using Linear Regression + ARIMA Model")
    else:
        st.info("start forecasting.")
