import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load('fraud_model.pkl')
le_merchant = joblib.load('le_merchant.pkl')
le_location = joblib.load('le_location.pkl')

# Page config
st.set_page_config(
    page_title="UPI Fraud Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 UPI Fraud Transaction Detector")
st.markdown("**Powered by Random Forest + Explainable AI (SHAP)**")
st.markdown("---")

# Input form
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (₹)", min_value=1, max_value=100000, value=5000)
    hour = st.slider("Transaction Hour (24hr)", 0, 23, 14)
    failed_attempts = st.slider("Failed Attempts Before This", 0, 5, 0)

with col2:
    merchant_type = st.selectbox(
        "Merchant Type",
        ['grocery', 'electronics', 'food', 'travel', 'unknown', 'crypto']
    )
    location = st.selectbox(
        "Transaction Location",
        ['home_city', 'nearby_city', 'foreign', 'unknown']
    )
    new_device = st.radio("New Device?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

# Predict button
if st.button("🔎 Analyze Transaction", use_container_width=True):

    # Encode inputs
    merchant_encoded = le_merchant.transform([merchant_type])[0]
    location_encoded = le_location.transform([location])[0]

    input_data = pd.DataFrame([{
        'amount': amount,
        'hour': hour,
        'merchant_type': merchant_encoded,
        'location': location_encoded,
        'new_device': new_device,
        'failed_attempts': failed_attempts
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    fraud_prob = round(probability[1] * 100, 2)
    legit_prob = round(probability[0] * 100, 2)

    st.markdown("---")
    st.subheader("🧠 Analysis Result")

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — {fraud_prob}% confidence")
        st.markdown("This transaction shows multiple high-risk signals.")
    else:
        st.success(f"✅ LEGITIMATE — {legit_prob}% confidence")
        st.markdown("This transaction appears safe.")

    # Confidence bar
    st.markdown("#### Confidence Breakdown")
    st.progress(int(fraud_prob), text=f"Fraud Risk: {fraud_prob}%")

    # SHAP explanation
    st.markdown("---")
    st.subheader("📊 Why did the model decide this?")
    st.markdown("*(Explainable AI — SHAP Analysis)*")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[..., 1][0],
            base_values=explainer.expected_value[1],
            data=input_data.iloc[0],
            feature_names=input_data.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.caption("Built with Random Forest + SHAP | Razorpay Hackathon 2026")