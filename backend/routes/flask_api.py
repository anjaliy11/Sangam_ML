from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os
import traceback
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
# <----------------------Department Conflict Prediction------------------>

# Load models and encoders for department conflict prediction
try:
    xgb_model_conflict = joblib.load("ml/models/conflict_department_model.pkl")
    scaler_conflict = joblib.load("ml/models/conflict_scaler_with_coordinates.pkl")
    label_encoder_dept1 = joblib.load("ml/models/department_1_encoder.pkl")
    label_encoder_dept2 = joblib.load("ml/models/department_2_encoder.pkl")
except Exception as e:
    logger.error(f"Error loading models or encoders: {str(e)}")
    raise

@app.route('/predict_department', methods=['POST'])
def predict_conflict():
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
    
   
    # <----------cost_reduction---------->
    
clf_model = joblib.load("ml/models/under_budget_classifier.pkl")
reg_model = joblib.load("ml/models/cost_reduction_regressor.pkl")
scaler = joblib.load("ml/models/cost_scaler.pkl")
site_location_encoder = joblib.load("ml/models/site_location_encoder.pkl")
department_encoder = joblib.load("ml/models/department_encoder.pkl")

# Feature list
features = [
    "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
    "resource_efficiency", "complexity_efficiency_ratio", "interaction_effect", "historical_impact",
    "adjusted_frequency", "completion_deviation", "cost_reduction_potential",
    "resource_overallocation", "time_overrun_ratio", "communication_inefficiency",
    "department_location_encoded"
]

# Function to handle missing data
def handle_missing_data(input_data, training_data):
    # Fill missing data with default values or based on the training data distribution
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

# Prediction function
def predict_project_outcomes(new_data):
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

    # Convert cost optimization potential to percentage and ensure it's a Python float
    cost_optimization_percentage = float(cost_optimization_prediction[0]) * 100

    # Return results
    return {
        "is_under_budget": bool(under_budget_prediction[0]),
        "cost_optimization_potential_percentage": cost_optimization_percentage
    }

@app.route('/predict_cost_reduction', methods=['POST'])
def predict_cost():
    try:
        # Parse input data from the request
        input_data = request.get_json()

        # Load training data for handling missing values (if necessary)
        training_data = pd.read_json("data/enhanced_data.json")

        # Handle missing data (optional step based on your model requirements)
        input_data = handle_missing_data(input_data, training_data)

        # Predict project outcomes
        predictions = predict_project_outcomes(input_data)

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e), "details": str(e)}), 500


# Start the Flask app
if __name__ == "__main__":
    app.run(host=host, port=port, debug=True)
