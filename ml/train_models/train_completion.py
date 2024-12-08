import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
import joblib
import shap

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
data["communication_per_day"] = data["communication_frequency"] / (data["project_duration"] + 1e-6)
data["urgency"] = data["task_priority"] * data["project_duration"]

# Define features and targets
features = [
    "task_priority", "task_complexity", "resources_allocated",
    "communication_frequency", "resource_utilization",
    "project_duration", "communication_per_day", "urgency",
    "latitude", "longitude", "project_duration_variance",
    "cost_overrun_percentage", "completion_efficiency"
]
X = data[features]

# Define targets
y_task = data["actual_completion_time"] - data["expected_completion_time"]
y_project = data["actual_project_completion_time"] - data["expected_project_completion_time"]

# Drop rows where targets are missing
X_task = X.loc[y_task.notna()]
y_task = y_task.dropna()
X_project = X.loc[y_project.notna()]
y_project = y_project.dropna()

# Impute missing values for features
imputer = SimpleImputer(strategy="median")
X_task_imputed = imputer.fit_transform(X_task)
X_project_imputed = imputer.transform(X_project)

# Scaling data with RobustScaler
scaler = RobustScaler()
X_task_scaled = scaler.fit_transform(X_task_imputed)
X_project_scaled = scaler.transform(X_project_imputed)

# Polynomial features for interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_task_poly = poly.fit_transform(X_task_scaled)
X_project_poly = poly.transform(X_project_scaled)

# Split the dataset into training and testing sets
X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(
    X_task_poly, y_task, test_size=0.2, random_state=42
)
X_train_project, X_test_project, y_train_project, y_test_project = train_test_split(
    X_project_poly, y_project, test_size=0.2, random_state=42
)

# Define RandomForest model and hyperparameters
param_distributions = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# Function to train and evaluate RandomForest
def train_random_forest(X_train, y_train, X_test, y_test, target_name):
    print(f"\nTraining RandomForest for {target_name}...")
    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        rf, param_distributions, n_iter=20, cv=3, scoring="r2", n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{target_name} Results:")
    print(f"R² Score: {r2:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}")
    return best_model

# Train models
best_rf_task = train_random_forest(X_train_task, y_train_task, X_test_task, y_test_task, "Task Completion")
best_rf_project = train_random_forest(X_train_project, y_train_project, X_test_project, y_test_project, "Project Completion")

# Save models and transformers
joblib.dump(best_rf_task, "ml/models/task_completion_model.pkl")
joblib.dump(best_rf_project, "ml/models/project_completion_model.pkl")
joblib.dump(scaler, "ml/models/completion_scaler.pkl")
joblib.dump(poly, "ml/models/completion_poly_features.pkl")
joblib.dump(imputer, "ml/models/completion_imputer.pkl")

# SHAP Feature Importance

# Use the original (pre-transformed) data that the model was trained on
explainer_task = shap.Explainer(best_rf_task, X_task_scaled)
shap_values_task = explainer_task(X_test_task)  # Pass the transformed data here
shap.summary_plot(shap_values_task, X_test_task)

explainer_project = shap.Explainer(best_rf_project, X_project_scaled)
shap_values_project = explainer_project(X_test_project)
shap.summary_plot(shap_values_project, X_test_project)
# Prediction function
def predict_task_and_project_completion(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[[col for col in features if col in input_df.columns]]

    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    input_poly = poly.transform(input_scaled)

    predicted_task_time = best_rf_task.predict(input_poly)[0]
    predicted_project_time = best_rf_project.predict(input_poly)[0]

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
print("\nPrediction Results:")
print(f"Predicted Task Completion Time: {prediction['predicted_task_time']:.2f}")
print(f"Predicted Project Completion Time: {prediction['predicted_project_time']:.2f}")
print(f"Task Status: {prediction['task_status']}")
print(f"Project Status: {prediction['project_status']}")
