import pandas as pd
import numpy as np
import random
import json

# Departments and Locations (same as original script)
departments = ["Roads", "Water Supply", "Electricity", "Sanitation", "Gas Pipelines"]
locations = {
    "Location-1": {"latitude": 40.7128, "longitude": -74.0060},
    "Location-2": {"latitude": 34.0522, "longitude": -118.2437},
    "Location-3": {"latitude": 41.8781, "longitude": -87.6298},
    "Location-4": {"latitude": 29.7604, "longitude": -95.3698},
    "Location-5": {"latitude": 33.4484, "longitude": -112.0740},
}

# Generate training dataset
data = []
for i in range(1000):
    project_id = f"P{i+1:04d}"
    department = random.choice(departments)
    site_location = random.choice(list(locations.keys()))
    latitude = locations[site_location]["latitude"]
    longitude = locations[site_location]["longitude"]
    task_priority = random.randint(1, 5)
    task_complexity = random.randint(1, 10)
    available_resources = random.randint(10, 100)
    resources_allocated = random.randint(5, available_resources)
    communication_frequency = random.randint(1, 10)
    historical_delay = random.randint(0, 30)
    expected_completion_time = random.randint(20, 150)
    actual_completion_time = expected_completion_time + np.random.normal(loc=0, scale=5)
    
    # Add expected and actual project completion time
    expected_project_completion_time = random.randint(100, 200)  # Random expected project completion time
    actual_project_completion_time = expected_project_completion_time + np.random.normal(loc=0, scale=10)  # Actual time with some deviation
    
    cost_estimate = random.randint(50000, 200000)
    actual_cost = cost_estimate + random.randint(-10000, 20000)

    # Derived fields for training
    project_duration_variance = actual_completion_time - expected_completion_time
    cost_overrun_percentage = ((actual_cost - cost_estimate) / cost_estimate) * 100
    historical_delay_weighted = historical_delay * task_complexity
    completion_efficiency = min(expected_completion_time / actual_completion_time, 1)
    conflict_severity_score = historical_delay * task_complexity * task_priority

    # Risk level categorization
    risk_level_bins = [-1, 10, 50, 100, np.inf]
    risk_level_labels = ["Low", "Moderate", "High", "Critical"]
    risk_level = pd.cut(
        [historical_delay * task_complexity], bins=risk_level_bins, labels=risk_level_labels
    )[0]

    project = {
        "project_id": project_id,
        "department": department,
        "site_location": site_location,
        "latitude": latitude,
        "longitude": longitude,
        "task_priority": task_priority,
        "task_complexity": task_complexity,
        "available_resources": available_resources,
        "resources_allocated": resources_allocated,
        "communication_frequency": communication_frequency,
        "historical_delay": historical_delay,
        "expected_completion_time": expected_completion_time,
        "actual_completion_time": actual_completion_time,
        "expected_project_completion_time": expected_project_completion_time,
        "actual_project_completion_time": actual_project_completion_time,
        "cost_estimate": cost_estimate,
        "actual_cost": actual_cost,
        "project_duration_variance": project_duration_variance,
        "cost_overrun_percentage": cost_overrun_percentage,
        "historical_delay_weighted": historical_delay_weighted,
        "completion_efficiency": completion_efficiency,
        "conflict_severity_score": conflict_severity_score,
        "risk_level": risk_level,
    }
    data.append(project)

# Save the training dataset to a JSON file
output_path = "data/training_data.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Training dataset with required fields generated and saved to {output_path}.")
