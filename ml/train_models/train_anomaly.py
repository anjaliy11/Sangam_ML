# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     classification_report,
#     roc_curve,
#     precision_recall_curve,
# )
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import joblib
# import pandas as pd
# import numpy as np

# # Load your dataset
# data = pd.read_json("data/enhanced_data.json")

# # Feature Engineering (Add more features as necessary)
# data['time_difference'] = data['actual_completion_time'] - data['expected_completion_time']
# data['resource_allocation_ratio'] = data['resources_allocated'] / (data['available_resources'] + 1e-6)

# # Encode categorical features
# data["site_location_encoded"] = pd.factorize(data["site_location"])[0]
# data["department_encoded"] = pd.factorize(data["department"])[0]

# # Define features and target (assuming "conflict_indicator" as anomaly label for this purpose)
# features = [
#     "task_priority", "task_complexity", "communication_frequency", "historical_delay",
#     "time_difference", "resource_allocation_ratio", "site_location_encoded", "department_encoded"
# ]
# X = data[features]
# y = data["conflict_indicator"]  # Assuming conflict_indicator is the label for anomaly detection

# y.replace([np.inf, -np.inf], np.nan, inplace=True)
# y = y.dropna()
# X = X.loc[y.index]
# # Split the data into train-test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# # Ensure X and y are aligned after dropping NaNs

# # Handle class imbalance using SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_resampled)
# X_test_scaled = scaler.transform(X_test)

# # Define and train Isolation Forest model
# isolation_forest = IsolationForest(contamination=0.1, random_state=42)
# isolation_forest.fit(X_train_scaled)

# # Predict anomalies
# y_pred = isolation_forest.predict(X_test_scaled)
# y_pred = np.where(y_pred == 1, 0, 1)  # Convert to binary format (1 for anomaly, 0 for normal)

# # Performance Metrics
# accuracy = (y_pred == y_test).mean()
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)

# # Display results
# print("\nOptimized Isolation Forest Anomaly Detection Results:")
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print(f"ROC AUC: {roc_auc:.2f}")

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=["Normal", "Anomalous"]))

# # Save the Isolation Forest model and scaler
# joblib.dump(isolation_forest, "ml/models/anomaly_detection_isolation_forest.pkl")
# joblib.dump(scaler, "ml/models/anomaly_scaler.pkl")
# print("\nAnomaly Detection Model and Scaler saved.")

# # Plot Precision-Recall Curve
# precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred)
# plt.figure(figsize=(10, 6))
# plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve", color="blue")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend(loc="best")
# plt.grid()
# plt.show()

# # Plot ROC Curve
# fpr, tpr, _ = roc_curve(y_test, y_pred)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="skyblue")
# plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="best")
# plt.grid()
# plt.show()



from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_json("data/enhanced_data.json")

# Feature Engineering
data['time_difference'] = data['actual_completion_time'] - data['expected_completion_time']
data['resource_allocation_ratio'] = data['resources_allocated'] / (data['available_resources'] + 1e-6)

# Encode categorical features
data["site_location_encoded"] = pd.factorize(data["site_location"])[0]
data["department_encoded"] = pd.factorize(data["department"])[0]

# Define features and target
features = [
    "task_priority", "task_complexity", "communication_frequency", "historical_delay",
    "time_difference", "resource_allocation_ratio", "site_location_encoded", "department_encoded"
]
X = data[features]
y = data["conflict_indicator"]

# Handle missing or infinite values
y.replace([np.inf, -np.inf], np.nan, inplace=True)
y = y.dropna()
X = X.loc[y.index]  # Align X and y

# Feature selection
selector = SelectKBest(score_func=f_classif, k="all")
X_selected = selector.fit_transform(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define and tune Isolation Forest
isolation_forest = IsolationForest(
    contamination=0.05,  # Adjusted contamination
    n_estimators=200,    # Increased estimators for robustness
    max_samples=0.9,     # Use most of the data for training
    random_state=42
)
isolation_forest.fit(X_train_scaled)

# Predict anomalies
y_pred = isolation_forest.predict(X_test_scaled)
y_pred = np.where(y_pred == 1, 0, 1)  # Convert to binary format (1 for anomaly, 0 for normal)

# Performance Metrics
accuracy = (y_pred == y_test).mean()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Display results
print("\nEnhanced Isolation Forest Anomaly Detection Results:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomalous"]))

# Save the Isolation Forest model and scaler
joblib.dump(isolation_forest, "ml/models/anomaly_detection_isolation_forest.pkl")
joblib.dump(scaler, "ml/models/anomaly_scaler.pkl")
print("\nEnhanced Anomaly Detection Model and Scaler saved.")

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve", color="blue")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.grid()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="skyblue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.grid()
plt.show()

# Predict anomalies for new input
new_input = pd.DataFrame([{
    "task_priority": 4,
    "task_complexity": 7,
    "communication_frequency": 3,
    "historical_delay": 10,
    "time_difference": 5,
    "resource_allocation_ratio": 0.75,
    "site_location_encoded": 1,
    "department_encoded": 2,

}])

# Apply transformations to the new input data
new_input_scaled = scaler.transform(new_input)
anomaly_pred = isolation_forest.predict(new_input_scaled)
anomaly_pred = 1 if anomaly_pred == -1 else 0  # Convert to binary format

print(f"Prediction for new input: {'Anomalous' if anomaly_pred == 1 else 'Normal'}")