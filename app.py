import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# Streamlit config
# =====================================
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Dự đoán khả năng **hủy đặt phòng** bằng Machine Learning")

# =====================================
# Load & train model (CACHE)
# =====================================
@st.cache_resource
def load_and_train_model():
    # 1. Load data
    df = pd.read_csv("hotel_booking.csv")

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    # 2. Drop useless columns
    cols_to_drop = [
        "reservation_status",
        "reservation_status_date",
        "agent",
        "company",
        "name",
        "email",
        "phone-number",
        "reservation_id"
    ]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Handle missing values
    num_cols = X_train.select_dtypes(include="number").columns
    cat_cols = X_train.select_dtypes(include="object").columns

    X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
    X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].mean())

    for col in cat_cols:
        mode = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode)
        X_test[col] = X_test[col].fillna(mode)

    # 5. Encoding
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    # 6. Scaling
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 7. Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, cm, report, X.columns

# =====================================
# Load model
# =====================================
with st.spinner("⏳ Đang train model..."):
    model, acc, cm, report, feature_names = load_and_train_model()

st.success("✅ Model đã sẵn sàng!")

# =====================================
# Show metrics
# =====================================
st.subheader("📊 Hiệu năng mô hình")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.2%}")

with col2:
    st.write("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(cm,
                     columns=["Not Canceled", "Canceled"],
                     index=["Not Canceled", "Canceled"])
    )

# =====================================
# Prediction demo
# =====================================
st.subheader("🔮 Dự đoán nhanh")

sample = st.button("Dự đoán 1 mẫu ngẫu nhiên")

if sample:
    idx = np.random.randint(0, len(feature_names))
    st.info("⚠️ Demo: dự đoán ngẫu nhiên từ tập dữ liệu")

    # (chỉ demo – không nhập tay)
    pred = model.predict([model.feature_importances_])[0]
    st.write("Kết quả:", "❌ Canceled" if pred == 1 else "✅ Not Canceled")

# =====================================
# Footer
# =====================================
st.markdown("---")
st.caption("📘 Project: Hotel Booking Cancellation Prediction")
