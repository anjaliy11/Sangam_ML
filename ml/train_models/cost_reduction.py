import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


data = pd.read_json("data/enhanced_data.json")

# Feature engineering
data["resource_efficiency"] = data["resources_allocated"] / (data["available_resources"] + 1e-6)
data["complexity_efficiency_ratio"] = data["task_complexity"] / (data["task_priority"] + 1)
data["interaction_effect"] = data["resources_allocated"] * data["communication_frequency"]
data["historical_impact"] = data["historical_delay"] * data["task_priority"]
data["adjusted_frequency"] = np.sqrt(data["communication_frequency"])
data["completion_deviation"] = data["actual_completion_time"] - data["expected_completion_time"]
data["cost_reduction_potential"] = (data["cost_estimate"] - data["actual_cost"]) / data["cost_estimate"]
data["department_location_interaction"] = (
    data["department"].astype(str) + "_" + data["site_location"].astype(str)
)
data["resource_overallocation"] = data["resources_allocated"] / (data["task_complexity"] + 1e-6)
data["time_overrun_ratio"] = data["actual_completion_time"] / (data["expected_completion_time"] + 1e-6)
data["communication_inefficiency"] = data["communication_frequency"] / (data["task_priority"] + 1e-6)
data["department_location_encoded"] = pd.factorize(data["department_location_interaction"])[0]

# Target variables
data["is_under_budget"] = (data["cost_estimate"] - data["actual_cost"] > 0).astype(int)

# Feature list
features = [
    "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
    "resource_efficiency", "complexity_efficiency_ratio", "interaction_effect", "historical_impact",
    "adjusted_frequency", "completion_deviation", "cost_reduction_potential",
    "resource_overallocation", "time_overrun_ratio", "communication_inefficiency",
    "department_location_encoded"
]

X = data[features]
y_classification = data["is_under_budget"]
y_regression = data["cost_reduction_potential"]

# Handle missing and infinite values
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y_classification = y_classification[X.index]
y_regression = y_regression[X.index]

# Split data
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classification model
clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
clf_model.fit(X_train_scaled, y_class_train)

# Train regression model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_reg_train)

# Evaluate models
y_class_pred = clf_model.predict(X_test_scaled)
y_reg_pred = xgb_model.predict(X_test_scaled)

class_accuracy = accuracy_score(y_class_test, y_class_pred)
reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
reg_r2 = r2_score(y_reg_test, y_reg_pred)

print("\nClassification Results (Under Budget Prediction):")
print(f"Accuracy: {class_accuracy:.2f}")

print("\nRegression Results (Cost Optimization Potential):")
print(f"MSE: {reg_mse:.2f}")
print(f"R² Score: {reg_r2:.2f}")

# Plot feature importance for regression model
plt.figure(figsize=(12, 8))
plt.title("Feature Importance (Regression)")
plt.barh(features, xgb_model.feature_importances_, color="teal")
plt.xlabel("Importance")
plt.show()

# Save models and scaler
os.makedirs("ml/models", exist_ok=True)

try:
    joblib.dump(clf_model, "ml/models/under_budget_classifier.pkl")
    joblib.dump(xgb_model, "ml/models/cost_reduction_regressor.pkl")
    joblib.dump(scaler, "ml/models/cost_scaler.pkl")
    print("Models and scaler saved successfully.")
except Exception as e:
    print(f"Error while saving models or scaler: {e}")

# Prediction function
def predict_project_outcomes(new_data):
    features = [
        "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
        "resource_efficiency", "complexity_efficiency_ratio", "interaction_effect", "historical_impact",
        "adjusted_frequency", "completion_deviation", "cost_reduction_potential",
        "resource_overallocation", "time_overrun_ratio", "communication_inefficiency",
        "department_location_encoded"
    ]

    # Load models and scaler
    clf_model = joblib.load("ml/models/under_budget_classifier.pkl")
    reg_model = joblib.load("ml/models/cost_reduction_regressor.pkl")
    scaler = joblib.load("ml/models/cost_scaler.pkl")

    # Create DataFrame for new data
    new_df = pd.DataFrame([new_data])

    # Derived features
    new_df["resource_efficiency"] = new_df["resources_allocated"] / (new_df["available_resources"] + 1e-6)
    new_df["complexity_efficiency_ratio"] = new_df["task_complexity"] / (new_df["task_priority"] + 1)
    new_df["interaction_effect"] = new_df["resources_allocated"] * new_df["communication_frequency"]
    new_df["historical_impact"] = new_df["historical_delay"] * new_df["task_priority"]
    new_df["adjusted_frequency"] = np.sqrt(new_df["communication_frequency"])
    new_df["completion_deviation"] = new_df["actual_completion_time"] - new_df["expected_completion_time"]
    new_df["cost_reduction_potential"] = (new_df["cost_estimate"] - new_df["actual_cost"]) / new_df["cost_estimate"]
    new_df["resource_overallocation"] = new_df["resources_allocated"] / (new_df["task_complexity"] + 1e-6)
    new_df["time_overrun_ratio"] = new_df["actual_completion_time"] / (new_df["expected_completion_time"] + 1e-6)
    new_df["communication_inefficiency"] = (
        new_df["communication_frequency"] / (new_df["task_priority"] + 1e-6)
    )
    new_df["department_location_interaction"] = (
        new_df["department"].astype(str) + "_" + new_df["site_location"].astype(str)
    )
    new_df["department_location_encoded"] = pd.factorize(new_df["department_location_interaction"])[0]

    # Ensure the DataFrame has required features
    X_new = new_df[features]

    # Scale features
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    under_budget_prediction = clf_model.predict(X_new_scaled)
    cost_optimization_prediction = reg_model.predict(X_new_scaled)

    # Return results
    return {
        "is_under_budget": bool(under_budget_prediction[0]),
        "cost_optimization_potential": float(cost_optimization_prediction[0])
    }
if __name__ == "_main_":
    new_project_data = {
        "task_priority": 2,
        "task_complexity": 5,
        "resources_allocated": 100,
        "communication_frequency": 10,
        "available_resources": 120,
        "historical_delay": 3,
        "actual_completion_time": 20,
        "expected_completion_time": 15,
        "cost_estimate": 5000,
        "actual_cost": 5500,
        "department": "Water Supply",
        "site_location": "Location-1"
    }

    predictions = predict_project_outcomes(new_project_data)
    print("\nPredictions for the new project:")
    print(f"Under Budget: {predictions['is_under_budget']}")
    print(f"Cost Optimization Potential: {predictions['cost_optimization_potential']:.2f}")
