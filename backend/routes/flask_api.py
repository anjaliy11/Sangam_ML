from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)


CORS(app)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    filename="flask_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


host = os.getenv('HOST', '0.0.0.0')  
port = int(os.getenv('PORT', 5000))



# <--------completion--------->
task_model = joblib.load("ml/models/task_completion_model.pkl")
project_model = joblib.load("ml/models/project_completion_model.pkl")
scaler = joblib.load("ml/models/completion_scaler.pkl")
poly = joblib.load("ml/models/completion_poly_features.pkl")
imputer = joblib.load("ml/models/completion_imputer.pkl")

# Load and combine training data for statistics
try:
    enhanced_data = pd.read_json("data/enhanced_data.json")
    training_data_extra = pd.read_json("data/training_data.json")
    training_data = pd.concat([enhanced_data, training_data_extra], ignore_index=True)
except Exception as e:
    raise Exception(f"Error loading training data: {str(e)}")

# Define the required features
FEATURES = [
    "task_priority", "task_complexity", "resources_allocated",
    "communication_frequency", "resource_utilization",
    "project_duration", "communication_per_day", "urgency",
    "latitude", "longitude", "project_duration_variance",
    "cost_overrun_percentage", "completion_efficiency"
    
]


# Step 1: Handle Missing Features
def handle_missing_data(input_data):
    """
    Validate and fill missing features in the input data.
    """
    feature_stats = {
        feature: training_data[feature].mean() if feature in training_data else 0
        for feature in FEATURES
    }

    # Fill missing features with fallback statistics
    for feature in FEATURES:
        if feature not in input_data or input_data[feature] is None:
            input_data[feature] = feature_stats[feature]

    return input_data


# Step 2: Prediction Function
def predict_task_and_project_completion(input_data):
    """
    Prepare input data, preprocess, and make predictions for task and project completion.
    """
    try:
        # Validate and fill missing features
        input_data = handle_missing_data(input_data)

        # Prepare input in the required order
        input_cleaned = {feature: input_data[feature] for feature in FEATURES}

        # Convert to DataFrame
        input_df = pd.DataFrame([input_cleaned])

        # Preprocessing: Impute, Scale, and Transform
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        input_poly = poly.transform(input_scaled)

        # Predict task and project completion times
        predicted_task_time = task_model.predict(input_poly)[0]
        predicted_project_time = project_model.predict(input_poly)[0]

        # Determine statuses
        task_status = "Completed" if predicted_task_time <= 0 else "Pending"
        project_status = "Completed" if predicted_project_time <= 0 else "Pending"

        return {
            "predicted_task_time": round(predicted_task_time, 2),
            "predicted_project_time": round(predicted_project_time, 2),
            "task_status": task_status,
            "project_status": project_status
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# Step 3: Flask Endpoint
@app.route("/predict_completion", methods=["POST"])
def predict_completion():
    """
    Flask endpoint for task and project completion prediction.
    """
    try:
        # Parse input JSON
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate input format
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid input format. Expected a JSON object."}), 400

        # Predict completion details
        result = predict_task_and_project_completion(input_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500



# <-----------------cost-reduction------------->

clf_model = joblib.load("ml/models/under_budget_classifier.pkl")
reg_model = joblib.load("ml/models/cost_reduction_regressor.pkl")
scaler = joblib.load("ml/models/cost_scaler.pkl")

def handle_missing_values(data, reference_data):
    for key in reference_data:
        if key not in data or pd.isnull(data[key]):
            data[key] = np.random.choice(reference_data[key].dropna())
    return data

# Feature engineering function
def process_input(data):
    data["resource_efficiency"] = data["resources_allocated"] / (data["available_resources"] + 1e-6)
    data["complexity_efficiency_ratio"] = data["task_complexity"] / (data["task_priority"] + 1)
    data["interaction_effect"] = data["resources_allocated"] * data["communication_frequency"]
    data["historical_impact"] = data["historical_delay"] * data["task_priority"]
    data["adjusted_frequency"] = np.sqrt(data["communication_frequency"])
    data["completion_deviation"] = data["actual_completion_time"] - data["expected_completion_time"]
    data["cost_reduction_potential"] = (data["cost_estimate"] - data["actual_cost"]) / data["cost_estimate"] * 100
    data["resource_overallocation"] = data["resources_allocated"] / (data["task_complexity"] + 1e-6)
    data["time_overrun_ratio"] = data["actual_completion_time"] / (data["expected_completion_time"] + 1e-6)
    data["communication_inefficiency"] = data["communication_frequency"] / (data["task_priority"] + 1e-6)
    data["department_location_interaction"] = f"{data['department']}_{data['site_location']}"
    data["department_location_encoded"] = pd.factorize([data["department_location_interaction"]])[0][0]
    
    features = [
        "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
        "resource_efficiency", "complexity_efficiency_ratio", "interaction_effect", "historical_impact",
        "adjusted_frequency", "completion_deviation", "cost_reduction_potential",
        "resource_overallocation", "time_overrun_ratio", "communication_inefficiency",
        "department_location_encoded"
    ]
    return pd.DataFrame([data])[features]

@app.route('/predict_cost_reduction', methods=['POST'])
def predict_cost_reduction():
    try:
        input_data = request.json

        # Load sample data as reference for missing value imputation
        reference_data = pd.read_json("data/enhanced_data.json")
        input_data = handle_missing_values(input_data, reference_data)

        input_df = process_input(input_data)

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make predictions
        under_budget_pred = clf_model.predict(input_scaled)
        cost_reduction_pred = reg_model.predict(input_scaled)

        response = {
             "is_under_budget": bool(under_budget_pred[0]),
             "cost_optimization_potential": float(cost_reduction_pred[0]) * 100
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    
    # <--------resource allocation--------->

# Load models and scalers for resource allocation
try:
    xgb_model_resource = joblib.load("ml/models/resource_allocation_model.pkl")
    scaler_resource = joblib.load("ml/models/resource_allocation_scaler.pkl")
    label_encoder_dept = joblib.load("ml/models/resource_department_encoder.pkl")
    label_encoder_site = joblib.load("ml/models/resource_site_encoder.pkl")
except Exception as e:
    logger.error(f"Error loading models or encoders: {str(e)}")
    raise

@app.route('/predict_resource_allocation', methods=['POST'])
def predict_resource_allocation():
    try:
        # Get input data from the request
        input_data = request.get_json()
        logger.debug(f"Received input: {input_data}")

        # Validate required fields
        required_fields = ["department", "site_location", "historical_conflicts", 
                           "project_overlap", "communication_frequency"]

        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Encode department and site location
        department_encoded = label_encoder_dept.transform([input_data["department"]])[0]
        site_location_encoded = label_encoder_site.transform([input_data["site_location"]])[0]
        logger.debug(f"Encoded department: {department_encoded}, site location: {site_location_encoded}")

        # Prepare input for prediction
        new_input = pd.DataFrame([{
            "department_encoded": department_encoded,
            "site_location_encoded": site_location_encoded,
            "historical_conflicts": input_data["historical_conflicts"],
            "project_overlap": input_data["project_overlap"],
            "communication_frequency": input_data["communication_frequency"]
        }])

        logger.debug(f"Prepared input for prediction: {new_input}")

        # Scale the input features
        new_input_scaled = scaler_resource.transform(new_input)
        logger.debug(f"Scaled input features: {new_input_scaled}")

        # Make prediction
        predicted_resources = xgb_model_resource.predict(new_input_scaled)[0]
        logger.debug(f"Predicted resources: {predicted_resources}")

        # Prepare response
        response = {
            "manpower": int(predicted_resources[0]),
            "funds": int(predicted_resources[1]),
            "equipment": int(predicted_resources[2])
        }

        logger.debug(f"Prediction result: {response}")
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error occurred during resource allocation prediction: {str(e)}")
        return jsonify({"error": f"Error occurred: {str(e)}"}), 500
    
# <----------------conflict-------------->
# Load models and encoders
model = joblib.load("ml/models/conflict_prediction_xgb.pkl")
scaler = joblib.load("ml/models/scaler.pkl")
site_location_encoder = joblib.load("ml/models/site_location_encoder.pkl")
department_encoder = joblib.load("ml/models/department_encoder.pkl")

# Load training data for handling missing values
training_data = pd.read_json("data/enhanced_data.json")

# Define feature columns as per the model's requirement
features = [
    "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
    "resource_utilization", "complexity_to_priority_ratio", "adjusted_frequency", "delay_factor",
    "site_location_encoded", "department_encoded"
]

# Define function for handling missing features
def handle_missing_data(input_data):
    # Load statistics for feature filling
    feature_stats = {
        "task_priority": training_data["task_priority"].mean(),
        "task_complexity": training_data["task_complexity"].mean(),
        "resources_allocated": training_data["resources_allocated"].mean(),
        "communication_frequency": training_data["communication_frequency"].mean(),
        "resource_utilization": training_data["resource_utilization"].mean(),
        "complexity_to_priority_ratio": training_data["complexity_to_priority_ratio"].mean(),
        "adjusted_frequency": training_data["adjusted_frequency"].mean(),
        "delay_factor": training_data["delay_factor"].mean(),
        "site_location": training_data["site_location"].mode()[0],
        "department": training_data["department"].mode()[0],
    }

    # Fill missing features with defaults or random values
    for feature, default_value in feature_stats.items():
        if feature not in input_data or input_data[feature] is None:
            if isinstance(default_value, float):  # For numeric features
                input_data[feature] = np.random.normal(
                    loc=training_data[feature].mean(),
                    scale=training_data[feature].std()
                )
            else:  # For categorical features
                input_data[feature] = default_value

    return input_data
@app.route('/predict_conflict', methods=['POST'])
def predict_conflict_():
    try:
        # Parse input data from the request
        input_data = request.get_json()

        # Automatically encode categorical features if provided
        site_location_encoded = None
        department_encoded = None

        if 'site_location' in input_data:
            try:
                site_location_encoded = site_location_encoder.transform([input_data["site_location"]])[0]
                input_data["site_location_encoded"] = site_location_encoded
            except Exception as e:
                return jsonify({"error": f"Error encoding 'site_location': {str(e)}"}), 400

        if 'department' in input_data:
            try:
                department_encoded = department_encoder.transform([input_data["department"]])[0]
                input_data["department_encoded"] = department_encoded
            except Exception as e:
                return jsonify({"error": f"Error encoding 'department': {str(e)}"}), 400

        # Validate and fill missing features
        input_data = handle_missing_data(input_data)

        # Prepare the data for prediction
        new_data = {
            "task_priority": input_data["task_priority"],
            "task_complexity": input_data["task_complexity"],
            "resources_allocated": input_data["resources_allocated"],
            "communication_frequency": input_data["communication_frequency"],
            "resource_utilization": input_data["resource_utilization"],
            "complexity_to_priority_ratio": input_data["complexity_to_priority_ratio"],
            "adjusted_frequency": input_data["adjusted_frequency"],
            "delay_factor": input_data.get("delay_factor", 0),  # Default to 0 if missing
            "site_location_encoded": input_data["site_location_encoded"],
            "department_encoded": input_data["department_encoded"],
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([new_data])

        # Log DataFrame columns for debugging
        print("Input DataFrame columns:", input_df.columns)

        # Ensure the feature columns are in the correct order for scaling
        missing_features = [feature for feature in features if feature not in input_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        # Scale the input data
        input_df_scaled = scaler.transform(input_df[features])

        # Predict
        prediction = model.predict(input_df_scaled)[0]
        prediction_prob = model.predict_proba(input_df_scaled)[0]

        # Prepare the response
        response = {
            "predicted_class": "Conflict" if prediction == 1 else "No Conflict",
            "probabilities": {
                "No Conflict": float(prediction_prob[0]),  # Convert to Python float
                "Conflict": float(prediction_prob[1]),    # Convert to Python float
            },
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "details": str(e)}), 500

    
    # <-------department conflict--------->
# Load models and encoders for department conflict prediction
try:
    xgb_model_conflict = joblib.load("ml/models/conflict_department_model.pkl")
    scaler_conflict = joblib.load("ml/models/conflict_scaler_with_coordinates.pkl")
    label_encoder_dept1 = joblib.load("ml/models/department_1_encoder.pkl")
    label_encoder_dept2 = joblib.load("ml/models/department_2_encoder.pkl")
except Exception as e:
    logger.error(f"Error loading models or encoders: {str(e)}")
    raise

@app.route('/predict_department_conflict', methods=['POST'])
def predict_conflict_department():
    try:
        input_data = request.get_json()
        logger.debug(f"Received input: {input_data}")

        # Validate required fields
        required_fields = ["department_1", "department_2", "latitude", "longitude", 
        "historical_conflicts", "project_overlap", "distance", "communication_frequency"]

        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Encode departments
        department_1_encoded = label_encoder_dept1.transform([input_data["department_1"]])[0]
        department_2_encoded = label_encoder_dept2.transform([input_data["department_2"]])[0]
        logger.debug(f"Encoded departments: {department_1_encoded}, {department_2_encoded}")

        # Prepare input for prediction
        new_input = pd.DataFrame([{
            "department_1_encoded": department_1_encoded,
            "department_2_encoded": department_2_encoded,
            "latitude": input_data["latitude"],
            "longitude": input_data["longitude"],
            "historical_conflicts": input_data["historical_conflicts"],
            "project_overlap": input_data["project_overlap"],
            "distance": input_data["distance"],
            "communication_frequency": input_data["communication_frequency"]
        }])

        logger.debug(f"Prepared input for prediction: {new_input}")

        # Scale the input features
        new_input_scaled = scaler_conflict.transform(new_input)
        logger.debug(f"Scaled input features: {new_input_scaled}")

        # Make prediction
        conflict_prob = xgb_model_conflict.predict_proba(new_input_scaled)[:, 1][0]

        # Convert numpy.float32 to float for JSON serialization
        conflict_prob = float(conflict_prob)

        conflict_pred = "Conflict" if conflict_prob > 0.5 else "No Conflict"
        logger.debug(f"Prediction result: {conflict_pred}, Probability: {conflict_prob}")

        # Decode departments back to their original names
        department_1 = label_encoder_dept1.inverse_transform([department_1_encoded])[0]
        department_2 = label_encoder_dept2.inverse_transform([department_2_encoded])[0]

        # Prepare response
        response = {
            "prediction": conflict_pred,
            "department_1": department_1,
            "department_2": department_2,
            "probability": round(conflict_prob, 2)  # You can round it as well
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error occurred during department conflict prediction: {str(e)}")
        return jsonify({"error": f"Error occurred: {str(e)}"}), 500

# <---------------------Anomaly Detection------------------>

# Load pre-trained models for anomaly detection
model_anomaly = joblib.load("ml/models/anomaly_detection_isolation_forest.pkl")
scaler_anomaly = joblib.load("ml/models/anomaly_scaler.pkl")

FEATURES = [
    "task_priority",
    "task_complexity",
    "communication_frequency",
    "historical_delay",
    "time_difference",
    "resource_allocation_ratio",
    "site_location_encoded",
    "department_encoded",
]

@app.route("/predict_anomaly", methods=["POST"])
def predict_anomaly():
    try:
        # Parse input JSON
        input_data = request.get_json()

        # Validate input
        if not input_data:
            raise ValueError("No input data provided.")

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        if missing_features := [
            feature for feature in FEATURES if feature not in input_df.columns
        ]:
            raise ValueError(f"Missing features in input data: {missing_features}")

        # Scale input data
        scaled_input = scaler_anomaly.transform(input_df[FEATURES])

        # Predict anomaly
        prediction = model_anomaly.predict(scaled_input)
        prediction = 1 if prediction[0] == -1 else 0  # Convert to binary format

        result = {
            "prediction": "Anomalous" if prediction == 1 else "Normal",
            "details": input_data
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error occurred in anomaly detection: {str(e)}")
        return jsonify({"error": str(e)}), 500
# Start the Flask app
if __name__ == "__main__":
    app.run(host=host, port=port, debug=True)