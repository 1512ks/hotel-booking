# =====================================
# 1. Import
# =====================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =====================================
# 2. Load data
# =====================================
df = pd.read_csv("hotel_booking.csv")

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']
print("Shape ban đầu:", df.shape)


# =====================================
# 3. Drop useless / ID columns
# =====================================
cols_to_drop = [
    'phone-number',
    'email',
    'name',
    'reservation_id',
    'reservation_status',
    'reservation_status_date',
    'agent',
    'company'
]
X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])


# =====================================
# 4. Train / Test split (RẤT QUAN TRỌNG)
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =====================================
# 5. Handle missing values (fit on train)
# =====================================
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns

# Numeric
X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
X_test[num_cols]  = X_test[num_cols].fillna(X_train[num_cols].mean())

# Categorical
for col in cat_cols:
    mode_val = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode_val)
    X_test[col]  = X_test[col].fillna(mode_val)


# =====================================
# 6. Handle RARE categories (QUAN TRỌNG)
# =====================================
def handle_rare_categories(train_col, test_col, min_freq=10):
    valid = train_col.value_counts()
    valid = valid[valid >= min_freq].index
    train_col = train_col.where(train_col.isin(valid), 'OTHER')
    test_col  = test_col.where(test_col.isin(valid), 'OTHER')
    return train_col, test_col

for col in cat_cols:
    X_train[col], X_test[col] = handle_rare_categories(
        X_train[col], X_test[col], min_freq=10
    )


# =====================================
# 7. Encoding (LabelEncoder – SAFE)
# =====================================
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col]  = le.transform(X_test[col])


# =====================================
# 8. Scaling (ONLY numeric columns)
# =====================================
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])
import joblib

joblib.dump(X_test, "X_test.pkl")
joblib.dump(X_train, "X_train.pkl")

print("Saved X_train.pkl & X_test.pkl")
