import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt

# Load the SMOTE-balanced data from train.py
with open('C:/Users/Admin/Desktop/HHTQD/train.py', 'rb') as f:
    # Assuming train.py exports X_train_smote, y_train_smote, X_test, y_test
    # You may need to adjust based on your actual train.py structure
    pass

# If data is saved as pickle files, load them:
with open('C:/Users/Admin/Desktop/HHTQD/X_train_smote.pkl', 'rb') as f:
    X_train_smote = pickle.load(f)

with open('C:/Users/Admin/Desktop/HHTQD/y_train_smote.pkl', 'rb') as f:
    y_train_smote = pickle.load(f)

with open('C:/Users/Admin/Desktop/HHTQD/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('C:/Users/Admin/Desktop/HHTQD/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
with open('C:/Users/Admin/Desktop/HHTQD/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)