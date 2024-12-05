import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, auc, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_json("data/enhanced_data.json")

# Validate required columns
required_columns = ["site_location", "department", "conflict_indicator"]


if missing_columns := [
    col for col in required_columns if col not in data.columns
 ]:
 if missing_columns:
    raise ValueError(f"Required columns are missing: {missing_columns}")

# Encode categorical features
data["site_location_encoded"] = LabelEncoder().fit_transform(data["site_location"])
data["department_encoded"] = LabelEncoder().fit_transform(data["department"])

# Validate and clean the target variable
if "conflict_indicator" not in data.columns:
    raise ValueError("The required target column 'conflict_indicator' is missing.")

# Handle missing values in the target column
data = data.dropna(subset=["conflict_indicator"])

# Ensure the target variable is binary
data = data[data["conflict_indicator"].isin([0, 1])]

# Revalidate the target variable after cleaning
if len(data["conflict_indicator"].unique()) != 2:
    raise ValueError("The target variable 'conflict_indicator' must be binary (0 or 1) after preprocessing.")

# Define features and target
features = [
    "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
    "resource_utilization", "complexity_to_priority_ratio", "adjusted_frequency", "delay_factor", 
    "site_location_encoded", "department_encoded"
]

# Ensure all required features exist in the dataset


X = data[features]
y = data["conflict_indicator"]

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1
)

# Train the model
grid_search.fit(X_train_scaled, y_train)

# Best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Conflict", "Conflict"]))

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Feature importance plot
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color="blue")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()
os.makedirs("ml/models", exist_ok=True)
# Save the model
joblib.dump(best_model, "ml/models/conflict_prediction_xgb.pkl")
print("Optimized XGBoost model saved as 'ml/models/conflict_prediction_xgb.pkl'")

# Function to predict new input
def predict_new_input(new_data):
    
    # Load the saved model and encoders
    model = joblib.load("ml/models/conflict_prediction_xgb.pkl")
    scaler = joblib.load("ml/models/scaler.pkl")
    site_location_encoder = joblib.load("ml/models/site_location_encoder.pkl")
    department_encoder = joblib.load("ml/models/department_encoder.pkl")

    # Preprocess input
    new_data["site_location_encoded"] = site_location_encoder.transform([new_data["site_location"]])[0]
    new_data["department_encoded"] = department_encoder.transform([new_data["department"]])[0]
    
    # Convert to DataFrame for consistency
    input_df = pd.DataFrame([new_data])
    input_df_scaled = scaler.transform(input_df[features])

    # Predict
    prediction = model.predict(input_df_scaled)[0]
    prediction_prob = model.predict_proba(input_df_scaled)[0]

    return {
        "predicted_class": "Conflict" if prediction == 1 else "No Conflict",
        "probabilities": {"No Conflict": prediction_prob[0], "Conflict": prediction_prob[1]},
    }

# Example usage
new_input = {
    "task_priority": 3,
    "task_complexity": 2,
    "resources_allocated": 5,
    "communication_frequency": 10,
    "resource_utilization": 0.75,
    "complexity_to_priority_ratio": 0.67,
    "adjusted_frequency": 8,
    "delay_factor": 0.3,
    "site_location": "Location-1",
    "department": "Roads"
}

result = predict_new_input(new_input)
print(result)
