# =====================================
# HOTEL BOOKING CANCELLATION APP
# Train model trực tiếp trong Streamlit
# =====================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# =====================================
# Page config
# =====================================
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Dự đoán khả năng **hủy đặt phòng** bằng Machine Learning")

# =====================================
# Helper: handle rare categories
# =====================================
def handle_rare_categories(train_col, test_col, min_freq=10):
    freq = train_col.value_counts()
    valid = freq[freq >= min_freq].index
    train_col = train_col.where(train_col.isin(valid), "OTHER")
    test_col = test_col.where(test_col.isin(valid), "OTHER")
    return train_col, test_col

# =====================================
# Load & Train Model (cached)
# =====================================
@st.cache_data(show_spinner=True)
def load_and_train_model():
    # 1. Load data
    df = pd.read_csv("hotel_booking.csv")

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    # 2. Drop useless columns
    cols_to_drop = [
        "phone-number",
        "email",
        "name",
        "reservation_id",
        "reservation_status",
        "reservation_status_date",
        "agent",
        "company"
    ]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # 3. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Identify column types
    num_cols = X_train.select_dtypes(include="number").columns
    cat_cols = X_train.select_dtypes(include="object").columns

    # 5. Handle missing values
    X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
    X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].mean())

    for col in cat_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # 6. Handle rare + unseen categories
    encoders = {}
    for col in cat_cols:
        X_train[col], X_test[col] = handle_rare_categories(
            X_train[col], X_test[col], min_freq=10
        )
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    # 7. Scaling
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 8. SMOTE (ONLY TRAIN)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 9. Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_smote, y_train_smote)

    # 10. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, cm, report, X.columns

# =====================================
# Run training
# =====================================
with st.spinner("⏳ Đang huấn luyện mô hình..."):
    model, accuracy, cm, report, feature_names = load_and_train_model()

# =====================================
# Results
# =====================================
st.success("✅ Huấn luyện hoàn tất!")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col2:
    st.write("**Confusion Matrix**")
    st.dataframe(
        pd.DataFrame(
            cm,
            columns=["Predicted No Cancel", "Predicted Cancel"],
            index=["Actual No Cancel", "Actual Cancel"]
        )
    )

st.subheader("📊 Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# =====================================
# Feature importance
# =====================================
st.subheader("🔍 Feature Importance (Random Forest)")
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

st.dataframe(importance_df.head(15))
