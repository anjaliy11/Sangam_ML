import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import random

departments = ["Roads", "Electricity", "Sanitation", "Water Supply", "Gas Pipelines"]
locations = {
    "Location-1": (28.7041, 77.1025),  # Example: Delhi
    "Location-2": (28.5355, 77.3910),  # Noida
    "Location-3": (28.4595, 77.0266),  # Gurgaon
    "Location-4": (28.4089, 77.3178),  # Faridabad
    "Location-5": (28.6139, 77.2090)   # Central Delhi
}

data = []
for _ in range(1000):
    department_1 = random.choice(departments)
    department_2 = random.choice([d for d in departments if d != department_1])
    site_location = random.choice(list(locations.keys()))
    lat, lon = locations[site_location]
    historical_conflicts = random.randint(0, 20)
    project_overlap = random.randint(1, 10)
    distance = random.uniform(0.5, 5.0)  
    communication_frequency = random.randint(1, 10)
    conflict_indicator = 1 if historical_conflicts > 10 and distance < 2 else 0

    data.append({
        "department_1": department_1,
        "department_2": department_2,
        "site_location": site_location,
        "latitude": lat,
        "longitude": lon,
        "historical_conflicts": historical_conflicts,
        "project_overlap": project_overlap,
        "distance": distance,
        "communication_frequency": communication_frequency,
        "conflict_indicator": conflict_indicator
    })

df = pd.DataFrame(data)


df.to_json("data/conflict_departments.json", orient="records", indent=4)

data = pd.read_json("data/conflict_departments.json")

label_encoder_dept1 = LabelEncoder()
label_encoder_dept2 = LabelEncoder()

data["department_1_encoded"] = label_encoder_dept1.fit_transform(data["department_1"])
data["department_2_encoded"] = label_encoder_dept2.fit_transform(data["department_2"])

features = [
    "department_1_encoded", "department_2_encoded", "latitude", "longitude",
    "historical_conflicts", "project_overlap", "distance", "communication_frequency"
]
X = data[features]
y = data["conflict_indicator"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


xgb_model.fit(X_train_scaled, y_train)


y_pred = xgb_model.predict(X_test_scaled)
y_pred_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Conflict", "Conflict"]))


precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

joblib.dump(xgb_model, "ml/models/conflict_department_model.pkl")
joblib.dump(scaler, "ml/models/conflict_scaler_with_coordinates.pkl")
joblib.dump(label_encoder_dept1, "ml/models/department_1_encoder.pkl")
joblib.dump(label_encoder_dept2, "ml/models/department_2_encoder.pkl")
print("Model and encoders saved.")


xgb_model = joblib.load("ml/models/conflict_department_model.pkl")
scaler = joblib.load("ml/models/conflict_scaler_with_coordinates.pkl")
label_encoder_dept1 = joblib.load("ml/models/department_1_encoder.pkl")
label_encoder_dept2 = joblib.load("ml/models/department_2_encoder.pkl")


new_input = pd.DataFrame([{
    "department_1_encoded": label_encoder_dept1.transform(["Roads"])[0],
    "department_2_encoded": label_encoder_dept2.transform(["Electricity"])[0],
    "latitude": 28.7041,  
    "longitude": 77.1025,  
    "historical_conflicts": 12,
    "project_overlap": 7,
    "distance": 1.5,
    "communication_frequency": 3
}])

new_input_scaled = scaler.transform(new_input)
conflict_prob = xgb_model.predict_proba(new_input_scaled)[:, 1][0]
conflict_pred = "Conflict" if conflict_prob > 0.5 else "No Conflict"

department_1 = label_encoder_dept1.inverse_transform([new_input["department_1_encoded"].iloc[0]])[0]
department_2 = label_encoder_dept2.inverse_transform([new_input["department_2_encoded"].iloc[0]])[0]

print(f"Prediction: {conflict_pred} between {department_1} and {department_2} (Probability: {conflict_prob:.2f})")
