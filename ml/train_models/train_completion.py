import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
import shap
import matplotlib.pyplot as plt

# Paths for input and output
enhanced_data_path = "data/enhanced_data.json"
training_data_path = "data/training_data.json"

# Load datasets
enhanced_data = pd.read_json(enhanced_data_path)
training_data = pd.read_json(training_data_path)

# Combine datasets
data = pd.concat([enhanced_data, training_data], ignore_index=True)

# Add new features
data["project_duration"] = (
    pd.to_datetime(data["project_end_date"]) - pd.to_datetime(data["project_start_date"])
).dt.days

# Prevent division by zero
data["communication_per_day"] = data["communication_frequency"] / (data["project_duration"] + 1e-6)
data["urgency"] = data["task_priority"] * data["project_duration"]

# Define features and targets
features = [
    "task_priority",
    "task_complexity",
    "resources_allocated",
    "communication_frequency",
    "resource_utilization",
    "complexity_to_priority_ratio",
    "project_duration",
    "communication_per_day",
    "urgency",
    "latitude",
    "longitude",
    "project_duration_variance",
    "cost_overrun_percentage",
    "completion_efficiency",
]
X = data[features]

# Define targets: task and project completion time
y_task = data["actual_completion_time"] - data["expected_completion_time"]
y_project = (
    data["actual_project_completion_time"] - data["expected_project_completion_time"]
)

# Replace inf values with NaN and drop missing values
y_task.replace([np.inf, -np.inf], np.nan, inplace=True)
y_project.replace([np.inf, -np.inf], np.nan, inplace=True)

X, y_task = X[y_task.notna()], y_task[y_task.notna()]
X, y_project = X[y_project.notna()], y_project[y_project.notna()]

X, y_project = X.align(y_project, join='inner', axis=0)

# Drop features with all missing values
missing_features = X.columns[X.isnull().all()]
if not missing_features.empty:
    print(f"Dropping features with all missing values: {list(missing_features)}")
    X = X.drop(columns=missing_features)
# Impute missing data
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Align lengths
if len(X_imputed) != len(y_task):
    min_length = min(len(X_imputed), len(y_task))
    X_imputed, y_task = X_imputed[:min_length], y_task[:min_length]

if len(X_imputed) != len(y_project):
    min_length = min(len(X_imputed), len(y_project))
    X_imputed, y_project = X_imputed[:min_length], y_project[:min_length]

# Split the dataset into training and testing sets
X_train, X_test, y_train_task, y_test_task = train_test_split(
    X_imputed, y_task, test_size=0.2, random_state=42
)
_, _, y_train_project, y_test_project = train_test_split(
    X_imputed, y_project, test_size=0.2, random_state=42
)

# Scaling data
scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, X_train.shape[0]))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train XGBoost model for task completion time
xgb_model_task = XGBRegressor(random_state=42, objective="reg:squarederror")
param_grid_task = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05],
    "max_depth": [3, 5],
    "min_child_weight": [1, 5],
    "gamma": [0, 0.1],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 5],
}

grid_search_task = GridSearchCV(
    estimator=xgb_model_task,
    param_grid=param_grid_task,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
)

grid_search_task.fit(X_train_poly, y_train_task)
best_xgb_model_task = grid_search_task.best_estimator_

# Train XGBoost model for project completion time
xgb_model_project = XGBRegressor(random_state=42, objective="reg:squarederror")
param_grid_project = param_grid_task  # Reusing the same grid

grid_search_project = GridSearchCV(
    estimator=xgb_model_project,
    param_grid=param_grid_project,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
)

grid_search_project.fit(X_train_poly, y_train_project)
best_xgb_model_project = grid_search_project.best_estimator_

# Evaluate the models
y_pred_task = best_xgb_model_task.predict(X_test_poly)
y_pred_project = best_xgb_model_project.predict(X_test_poly)

mse_task = mean_squared_error(y_test_task, y_pred_task)
mae_task = mean_absolute_error(y_test_task, y_pred_task)
r2_task = r2_score(y_test_task, y_pred_task)

mse_project = mean_squared_error(y_test_project, y_pred_project)
mae_project = mean_absolute_error(y_test_project, y_pred_project)
r2_project = r2_score(y_test_project, y_pred_project)

print("\nTask Completion Time Model Results:")
print(f"MSE: {mse_task:.2f}")
print(f"MAE: {mae_task:.2f}")
print(f"R² Score: {r2_task:.2f}")

print("\nProject Completion Time Model Results:")
print(f"MSE: {mse_project:.2f}")
print(f"MAE: {mae_project:.2f}")
print(f"R² Score: {r2_project:.2f}")

# Save models, scaler, and polynomial transformer
joblib.dump(best_xgb_model_task, "ml/models/task_completion_xgb.pkl")
joblib.dump(best_xgb_model_project, "ml/models/project_completion_xgb.pkl")
joblib.dump(scaler, "ml/models/completion_scaler.pkl")
joblib.dump(poly, "ml/models/completion_poly_features.pkl")

# SHAP Feature Importance
explainer_task = shap.Explainer(best_xgb_model_task, X_test_poly)
shap_values_task = explainer_task(X_test_poly)
shap.summary_plot(shap_values_task, X_test_poly)

explainer_project = shap.Explainer(best_xgb_model_project, X_test_poly)
shap_values_project = explainer_project(X_test_poly)
shap.summary_plot(shap_values_project, X_test_poly)

# Prediction function for both task and project completion times
def predict_task_and_project_completion(input_data):
    input_df = pd.DataFrame([input_data])
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    input_poly = poly.transform(input_scaled)

    predicted_task_time = best_xgb_model_task.predict(input_poly)[0]
    predicted_project_time = best_xgb_model_project.predict(input_poly)[0]

    task_status = "Completed" if predicted_task_time <= 0 else "Pending"
    project_status = "Completed" if predicted_project_time <= 0 else "Pending"

    return {
        "predicted_task_time": predicted_task_time,
        "predicted_project_time": predicted_project_time,
        "task_status": task_status,
        "project_status": project_status,
    }

# Example prediction
new_input = {
    "task_priority": 3,
    "task_complexity": 8,
    "resources_allocated": 50,
    "communication_frequency": 7,
    "resource_utilization": 0.6,
    "complexity_to_priority_ratio": 2.67,
    "project_duration": 90,
    "communication_per_day": 0.08,
    "urgency": 270,
    "latitude": 28.6139,
    "longitude": 77.2090,
    "project_duration_variance": 5,
    "cost_overrun_percentage": 10,
    "completion_efficiency": 0.9,
}

prediction = predict_task_and_project_completion(new_input)
print(f"\nPredicted Task Completion Time: {prediction['predicted_task_time']:.2f}")
print(f"Predicted Project Completion Time: {prediction['predicted_project_time']:.2f}")
print(f"Task Status: {prediction['task_status']}")
print(f"Project Status: {prediction['project_status']}")

