import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

departments = ["Roads", "Electricity", "Sanitation", "Water Supply", "Gas Pipelines"]
resources = ["Manpower", "Funds", "Equipment"]
locations = ["Location-1", "Location-2", "Location-3", "Location-4", "Location-5"]

data = []
for _ in range(1000):
    department = np.random.choice(departments)
    site_location = np.random.choice(locations)
    manpower = np.random.randint(1, 100)  
    funds = np.random.randint(10000, 500000)  
    equipment = np.random.randint(1, 50)  

    historical_conflicts = np.random.randint(0, 20)
    project_overlap = np.random.randint(1, 10)
    communication_frequency = np.random.randint(1, 10)

    data.append({
        "department": department,
        "site_location": site_location,
        "historical_conflicts": historical_conflicts,
        "project_overlap": project_overlap,
        "communication_frequency": communication_frequency,
        "manpower": manpower,
        "funds": funds,
        "equipment": equipment
    })

df = pd.DataFrame(data)


df.to_json("data/resource_allocation.json", orient="records", indent=4)

data = pd.read_json("data/resource_allocation.json")

label_encoder_dept = LabelEncoder()
label_encoder_site = LabelEncoder()

data["department_encoded"] = label_encoder_dept.fit_transform(data["department"])
data["site_location_encoded"] = label_encoder_site.fit_transform(data["site_location"])

features = [
    "department_encoded", "site_location_encoded",
    "historical_conflicts", "project_overlap", "communication_frequency"
]
targets = ["manpower", "funds", "equipment"]

X = data[features]
y = data[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
))
xgb_model.fit(X_train_scaled, y_train)

y_pred = xgb_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print(f"Mean Squared Error: {mse:.2f}")


joblib.dump(xgb_model, "ml/models/resource_allocation_model.pkl")
joblib.dump(scaler, "ml/models/resource_allocation_scaler.pkl")
joblib.dump(label_encoder_dept, "ml/models/resource_department_encoder.pkl")
joblib.dump(label_encoder_site, "ml/models/resource_site_encoder.pkl")
print("Model and encoders saved.")


def predict_resource_allocation(input_data):

    try:

        xgb_model = joblib.load("ml/models/resource_allocation_model.pkl")
        scaler = joblib.load("ml/models/resource_allocation_scaler.pkl")
        label_encoder_dept = joblib.load("ml/models/resource_department_encoder.pkl")
        label_encoder_site = joblib.load("ml/models/resource_site_encoder.pkl")
        
        department_encoded = label_encoder_dept.transform([input_data["department"]])[0]
        site_location_encoded = label_encoder_site.transform([input_data["site_location"]])[0]
    
        input_df = pd.DataFrame([{
            "department_encoded": department_encoded,
            "site_location_encoded": site_location_encoded,
            "historical_conflicts": input_data["historical_conflicts"],
            "project_overlap": input_data["project_overlap"],
            "communication_frequency": input_data["communication_frequency"]
        }])
        

        input_scaled = scaler.transform(input_df)
        

        predicted_resources = xgb_model.predict(input_scaled)[0]
        
        return {
            "manpower": int(predicted_resources[0]),
            "funds": int(predicted_resources[1]),
            "equipment": int(predicted_resources[2])
        }
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    new_input ={
        "department": "Roads",
        "site_location": "Location-1",
        "historical_conflicts": 10,
        "project_overlap": 5,
        "communication_frequency": 7
    } 
    
    result = predict_resource_allocation(new_input)
    print("Predicted Resource Allocation:", result) 