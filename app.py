import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# LOAD MODEL & METADATA
# =====================================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Dự đoán khả năng hủy đặt phòng")
st.markdown(
    "Ứng dụng dự đoán **khả năng khách hàng hủy đặt phòng** dựa trên mô hình Random Forest."
)

st.divider()

# =====================================
# INPUT FORM
# =====================================
st.subheader("📋 Nhập thông tin đặt phòng")

lead_time = st.number_input(
    "Lead time (số ngày từ lúc đặt đến ngày nhận phòng)",
    min_value=0,
    max_value=500,
    value=50
)

adr = st.number_input(
    "ADR (giá trung bình mỗi đêm)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0
)

total_of_special_requests = st.slider(
    "Số yêu cầu đặc biệt",
    min_value=0,
    max_value=5,
    value=1
)

previous_cancellations = st.slider(
    "Số lần hủy trước đây",
    min_value=0,
    max_value=10,
    value=0
)

required_car_parking_spaces = st.slider(
    "Số chỗ đỗ xe yêu cầu",
    min_value=0,
    max_value=5,
    value=0
)

market_segment = st.selectbox(
    "Market segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"]
)

customer_type = st.selectbox(
    "Customer type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

assigned_room_type = st.selectbox(
    "Assigned room type",
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]
)

st.divider()

# =====================================
# CREATE INPUT DATAFRAME
# =====================================
input_dict = {
    "lead_time": lead_time,
    "adr": adr,
    "total_of_special_requests": total_of_special_requests,
    "previous_cancellations": previous_cancellations,
    "required_car_parking_spaces": required_car_parking_spaces,
    "market_segment": market_segment,
    "customer_type": customer_type,
    "assigned_room_type": assigned_room_type
}

input_df = pd.DataFrame([input_dict])

# One-hot / align columns
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# =====================================
# PREDICTION
# =====================================
if st.button("🔮 Dự đoán"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("📊 Kết quả dự đoán")

    if pred == 1:
        st.error(f"❌ Khách hàng CÓ KHẢ NĂNG HỦY ĐẶT PHÒNG\n\nXác suất: **{prob:.2%}**")
    else:
        st.success(f"✅ Khách hàng KHÔNG CÓ KHẢ NĂNG HỦY\n\nXác suất hủy: **{prob:.2%}**")

st.divider()

st.caption("📌 Mô hình: Random Forest | Dữ liệu: Hotel Booking Demand")
