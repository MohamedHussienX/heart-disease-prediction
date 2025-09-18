# Cell: Test Cases for Model Validation
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_path = "../models/final_model.pkl"
model = joblib.load(model_path)

# ===========================
# Test Cases for each risk class
# ===========================
test_cases = [
    # Class 0 - Very Low Risk (Healthy)
    {
        "name": "Class 0 - Very Low Risk (Healthy)",
        "input": {
            "age": 35, "trestbps": 110, "chol": 160, "thalach": 180, 
            "oldpeak": 0.5, "ca": 0, "thal": 3, "cp": 1, "exang": 0
        },
        "expected": 0,
        "description": "Young, low blood pressure, excellent heart rate, no exercise-induced angina"
    },
    
    # Class 1 - Low Risk
    {
        "name": "Class 1 - Low Risk",
        "input": {
            "age": 45, "trestbps": 120, "chol": 180, "thalach": 160, 
            "oldpeak": 1.0, "ca": 0, "thal": 3, "cp": 1, "exang": 0
        },
        "expected": 1,
        "description": "Middle-aged, normal vitals, mild ST depression"
    },
    
    # Class 2 - Moderate Risk
    {
        "name": "Class 2 - Moderate Risk",
        "input": {
            "age": 55, "trestbps": 140, "chol": 220, "thalach": 140, 
            "oldpeak": 1.5, "ca": 1, "thal": 6, "cp": 2, "exang": 0
        },
        "expected": 2,
        "description": "Higher age, elevated BP and cholesterol, one vessel colored"
    },
    
    # Class 3 - High Risk
    {
        "name": "Class 3 - High Risk",
        "input": {
            "age": 60, "trestbps": 150, "chol": 250, "thalach": 130, 
            "oldpeak": 2.0, "ca": 2, "thal": 7, "cp": 3, "exang": 1
        },
        "expected": 3,
        "description": "Senior, hypertension, high cholesterol, exercise-induced angina"
    },
    
    # Class 4 - Very High Risk
    {
        "name": "Class 4 - Very High Risk",
        "input": {
            "age": 65, "trestbps": 160, "chol": 300, "thalach": 120, 
            "oldpeak": 3.0, "ca": 3, "thal": 7, "cp": 4, "exang": 1
        },
        "expected": 4,
        "description": "Elderly, severe hypertension, very high cholesterol, multiple risk factors"
    },
    
    # Edge Cases
    {
        "name": "Edge Case - Minimum Values",
        "input": {
            "age": 29, "trestbps": 94, "chol": 126, "thalach": 71, 
            "oldpeak": 0.0, "ca": 0, "thal": 3, "cp": 1, "exang": 0
        },
        "expected": "any",
        "description": "Testing minimum possible values from dataset"
    },
    
    {
        "name": "Edge Case - Maximum Values",
        "input": {
            "age": 77, "trestbps": 200, "chol": 564, "thalach": 202, 
            "oldpeak": 6.2, "ca": 3, "thal": 7, "cp": 4, "exang": 1
        },
        "expected": "any",
        "description": "Testing maximum possible values from dataset"
    }
]

# ===========================
# Run Test Predictions
# ===========================
print("🧪 MODEL TESTING SUITE")
print("=" * 50)

all_passed = True
features_order = ["thalach", "oldpeak", "age", "chol", "trestbps", "ca", "thal", "cp", "exang"]

for i, case in enumerate(test_cases, 1):
    # Create DataFrame with correct feature order
    df_test = pd.DataFrame([case["input"]])
    df_test = df_test[features_order]
    
    # Make prediction
    pred_class = model.predict(df_test)[0]
    probs = model.predict_proba(df_test)[0]
    
    # Check if test passed
    if case["expected"] == "any":
        passed = True  # Edge cases can have any valid prediction
        status = "✅ EDGE CASE"
    else:
        passed = pred_class == case["expected"]
        status = "✅ PASS" if passed else "❌ FAIL"
        all_passed = all_passed and passed
    
    print(f"\n{i}. {status} - {case['name']}")
    print(f"   Description: {case['description']}")
    print(f"   Predicted: Class {pred_class} | Expected: Class {case['expected']}")
    
    # Show confidence scores
    print("   Confidence Scores:")
    for class_idx, prob in enumerate(probs):
        print(f"     Class {class_idx}: {prob:.3f}")
    
    # Show top features (if using tree-based model)
    if hasattr(model, 'feature_importances_'):
        print("   Top influencing features:")
        feature_importance = list(zip(features_order, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for feat, importance in feature_importance[:3]:
            print(f"     {feat}: {importance:.3f}")

# Final summary
print("\n" + "=" * 50)
if all_passed:
    print("🎉 ALL TESTS PASSED! Model is working correctly.")
else:
    print("⚠️  SOME TESTS FAILED. Check model performance.")
print(f"Total tests run: {len(test_cases)}")

# ===========================
# Additional Validation Tests
# ===========================
print("\n📊 ADDITIONAL VALIDATION:")
print(f"Model type: {type(model).__name__}")
print(f"Number of classes: {model.n_classes_ if hasattr(model, 'n_classes_') else 'Unknown'}")
print(f"Features used: {len(features_order)}")
# Cell: Real-world Scenario Tests
print("🌍 REAL-WORLD SCENARIO TESTS")

# Realistic patient scenarios
real_world_cases = [
    {
        "scenario": "Athlete - Low Risk",
        "data": {"age": 28, "trestbps": 110, "chol": 150, "thalach": 190, 
                "oldpeak": 0.2, "ca": 0, "thal": 3, "cp": 1, "exang": 0}
    },
    {
        "scenario": "Middle-aged with Stress - Moderate Risk", 
        "data": {"age": 52, "trestbps": 145, "chol": 230, "thalach": 135, 
                "oldpeak": 1.8, "ca": 1, "thal": 6, "cp": 2, "exang": 0}
    },
    {
        "scenario": "Senior with History - High Risk",
        "data": {"age": 68, "trestbps": 155, "chol": 280, "thalach": 125, 
                "oldpeak": 2.5, "ca": 2, "thal": 7, "cp": 3, "exang": 1}
    }
]

for scenario in real_world_cases:
    df_test = pd.DataFrame([scenario["data"]])[features_order]
    prediction = model.predict(df_test)[0]
    confidence = max(model.predict_proba(df_test)[0])
    
    risk_levels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    print(f"\n{scenario['scenario']}:")
    print(f"  Risk Level: {risk_levels[prediction]} (Class {prediction})")
    print(f"  Confidence: {confidence:.1%}")