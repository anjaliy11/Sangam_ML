from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import logging
import traceback
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS if necessary
CORS(app)

# Configure logging
logging.basicConfig(
    filename="flask_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

print("Starting Flask app...")


# <--------------resource_allocator------------>
# Load the models and scalers
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Initialize Flask app and logger
app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load the models, scalers, and encoders
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

try:
    conflict_model = joblib.load("ml/models/conflict_prediction_xgb.pkl")
    scaler_conflict = joblib.load("ml/models/scaler.pkl")
    site_location_encoder = joblib.load("ml/models/site_location_encoder.pkl")
    department_encoder = joblib.load("ml/models/department_encoder.pkl")
except Exception as e:
    logger.error(f"Error loading models or encoders: {str(e)}")
    raise

# Define features
features = [
    "task_priority", "task_complexity", "resources_allocated", "communication_frequency",
    "resource_utilization", "complexity_to_priority_ratio", "adjusted_frequency", "delay_factor",
    "site_location_encoded", "department_encoded"
]


@app.route('/predict_conflict', methods=['POST'])
def predict_conflict_():
    try:
        input_data = request.get_json()
        logger.debug(f"Received input: {input_data}")

        # Validate input
        required_fields = ["site_location", "department"]
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Transform categorical data
        site_location_encoded = site_location_encoder.transform([input_data["site_location"]])[0]
        department_encoded = department_encoder.transform([input_data["department"]])[0]

        # Prepare input data
        new_data = {
            "task_priority": input_data.get("task_priority", 0),
            "task_complexity": input_data.get("task_complexity", 0),
            "resources_allocated": input_data.get("resources_allocated", 0),
            "communication_frequency": input_data.get("communication_frequency", 0),
            "resource_utilization": input_data.get("resource_utilization", 0),
            "complexity_to_priority_ratio": input_data.get("complexity_to_priority_ratio", 0),
            "adjusted_frequency": input_data.get("adjusted_frequency", 0),
            "delay_factor": input_data.get("delay_factor", 0),
            "site_location_encoded": site_location_encoded,
            "department_encoded": department_encoded
        }
        input_df = pd.DataFrame([new_data])

        # Scale data and predict
        input_df_scaled = scaler_conflict.transform(input_df[features])
        prediction = conflict_model.predict(input_df_scaled)[0]
        prediction_prob = conflict_model.predict_proba(input_df_scaled)[0]

        # Prepare response
        response = {
            "predicted_class": "Conflict" if prediction == 1 else "No Conflict",
            "probabilities": {
                "No Conflict": round(prediction_prob[0], 2),
                "Conflict": round(prediction_prob[1], 2)
            }
        }
        return jsonify(response)

    except Exception as e:
        logger.exception(f"Error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 500

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


if __name__ == "__main__":
    app.run(debug=True)
    
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
