import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========================
# Load model
# ========================
model = joblib.load("../models/final_model.pkl")

# ========================
# Load dataset
# ========================
# Change "heart.csv" to the name/path of your CSV file
df = pd.read_csv("../data/heart_disease_selected.csv")

# Expected feature order
feature_order = ["thalach", "oldpeak", "age", "chol", "trestbps", "ca", "thal", "cp", "exang"]

# Separate features and target
X = df[feature_order]
y = df["target"]   # change if your label column has a different name

# ========================
# Predictions
# ========================
y_pred = model.predict(X)

# Show first 10 test cases
print("\n=== Sample Predictions (first 10 rows) ===")
for i in range(10):
    row = X.iloc[i]
    probs = model.predict_proba([row])[0]
    print(f"\nCase {i+1}")
    print(f"Input: {row.to_dict()}")
    print(f"Expected Class: {y.iloc[i]} | Predicted Class: {y_pred[i]}")
    for j, p in enumerate(probs):
        print(f"  Class {j}: {p*100:.2f}%")

# ========================
# Metrics
# ========================
print("\n=== Overall Evaluation ===")
print(f"Accuracy: {accuracy_score(y, y_pred)*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))
