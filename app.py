import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load model & metadata
# ==============================
model = joblib.load("model.pkl")      # model nhẹ (<100MB) hoặc retrain nhỏ
columns = joblib.load("columns.pkl")  # danh sách cột

st.set_page_config(page_title="Hotel Booking Cancellation", layout="centered")

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Dự đoán khả năng **hủy đặt phòng khách sạn**")

# ==============================
# Input form
# ==============================
with st.form("booking_form"):
    lead_time = st.number_input("Lead time (days)", 0, 500, 50)
    adr = st.number_input("ADR (Average Daily Rate)", 0.0, 500.0, 100.0)
    total_special_requests = st.slider("Total special requests", 0, 5, 1)
    previous_cancellations = st.slider("Previous cancellations", 0, 10, 0)

    submit = st.form_submit_button("Predict")

# ==============================
# Prediction
# ==============================
if submit:
    input_data = {
        "lead_time": lead_time,
        "adr": adr,
        "total_of_special_requests": total_special_requests,
        "previous_cancellations": previous_cancellations
    }

    df_input = pd.DataFrame([input_data])

    # align columns
    df_input = df_input.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"❌ Khách hàng CÓ KHẢ NĂNG HỦY (Prob = {prob:.2f})")
    else:
        st.success(f"✅ Khách hàng KHÔNG HỦY (Prob = {1-prob:.2f})")
