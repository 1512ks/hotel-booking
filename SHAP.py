# =====================================
# SHAP ANALYSIS FOR RANDOM FOREST
# =====================================

import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. Load model & data
# =====================================
model = joblib.load("model.pkl")      # Random Forest đã tối ưu
X_test = joblib.load("X_test.pkl")    # Test set sau tiền xử lý (KHÔNG SMOTE)

print("Model loaded:", type(model))
print("X_test shape:", X_test.shape)

# (Khuyến nghị) Lấy mẫu để SHAP chạy nhanh
X_test_sample = X_test.sample(
    n=min(1000, len(X_test)),
    random_state=42
)

# =====================================
# 2. Create SHAP explainer
# =====================================
explainer = shap.TreeExplainer(model)

# =====================================
# 3. Compute SHAP values
# =====================================
shap_values = explainer.shap_values(X_test_sample)

# Bài toán nhị phân → class 1 (is_canceled = 1)
shap_values_class1 = shap_values[1]

# =====================================
# 4. SHAP summary plot (BAR)
# =====================================
shap.summary_plot(
    shap_values_class1,
    X_test_sample,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance (Class: Canceled)")
plt.show()

# =====================================
# 5. SHAP summary plot (BEESWARM)
# =====================================
shap.summary_plot(
    shap_values_class1,
    X_test_sample,
    show=False
)
plt.title("SHAP Summary Plot")
plt.show()

# =====================================
# 6. Top features table
# =====================================
feature_importance = pd.DataFrame({
    "feature": X_test_sample.columns,
    "mean_abs_shap": np.abs(shap_values_class1).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

print("\nTop 10 important features based on SHAP:")
print(feature_importance.head(10))
