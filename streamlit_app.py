
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from statsmodels.tsa.arima.model import ARIMA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: #F9E79F;
    text-align: center;
}
.stButton>button {
    background-color: #F9E79F;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🍽️ Food Demand Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3>Linear Regression + ARIMA Based Prediction</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Drop ID column
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

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
    #arima_model = ARIMA(y, order=(1, 1, 1))
    #arima_fit = arima_model.fit()

    # ---------------- USER INPUT ----------------
    st.subheader("🔢 Enter Input Values")

    c1, c2, c3 = st.columns(3)

    with c1:
        week = st.number_input("Week", 1, 200, 150, step=1)
        base_price = st.number_input("Base Price", 50.0, 1000.0, 300.0, step=5.0)

    with c2:
        center_id = st.number_input("Center ID", 1, 100, 10, step=1)
        meal_id = st.number_input("Meal ID", 1, 3000, 1000, step=10)

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
        final_prediction=lr_pred

        # ARIMA Prediction (next step)
        #arima_pred = arima_fit.forecast(steps=1)[0]

        # Final Combined Prediction
        #final_prediction = (lr_pred + arima_pred) / 2

        st.markdown("---")
        st.markdown(
            f"""
            <h2>✅ Predicted Checkout Price</h2>
            <h1 style="color:#2ECC71;">₹ {final_prediction:.2f}</h1>
            """,
            unsafe_allow_html=True
        )

        st.success("Prediction generated using Linear Regression.")

else:
    st.info("⬅️ Upload dataset to start forecasting")