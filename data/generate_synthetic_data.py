import pandas as pd
import numpy as np
import random
import json

departments = ["Roads", "Water Supply", "Electricity", "Sanitation", "Gas Pipelines"]
locations = {
    "Location-1": {"latitude": 40.7128, "longitude": -74.0060},
    "Location-2": {"latitude": 34.0522, "longitude": -118.2437},
    "Location-3": {"latitude": 41.8781, "longitude": -87.6298},
    "Location-4": {"latitude": 29.7604, "longitude": -95.3698},
    "Location-5": {"latitude": 33.4484, "longitude": -112.0740},
}


data = []
for i in range(1000):
    project_id = f"P{i+1:04d}"
    department = random.choice(departments)
    site_location = random.choice(list(locations.keys()))
    latitude = locations[site_location]["latitude"]
    longitude = locations[site_location]["longitude"]
    task_priority = random.randint(1, 5)  # Task priority (1-5)
    task_complexity = random.randint(1, 10)  # Task complexity (1-10)
    available_resources = random.randint(10, 100)  # Total available resources
    resources_allocated = random.randint(5, available_resources)  # Resources allocated
    communication_frequency = random.randint(1, 10)  # Communication updates (1-10)
    historical_delay = random.randint(0, 30)  # Historical delays (in days)
    expected_completion_time = random.randint(20, 150)  # Expected duration (in days)
    actual_completion_time = (
        expected_completion_time + np.random.normal(loc=0, scale=5)
    )  # Adding variability to completion time
    cost_estimate = random.randint(50000, 200000)  # Project cost estimate
    actual_cost = cost_estimate + random.randint(-10000, 20000)  # Adding variability to actual cost

    # Generate synthetic project start and end dates
    project_start_date = pd.Timestamp(f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}")
    project_end_date = project_start_date + pd.Timedelta(days=expected_completion_time)

    # Derived features
    resource_utilization = resources_allocated / (available_resources + 1e-6)
    complexity_to_priority_ratio = task_complexity / (task_priority + 1)
    delay_factor = historical_delay * task_priority
    adjusted_frequency = np.sqrt(communication_frequency)
    cost_reduction_potential = (cost_estimate - actual_cost) / cost_estimate

    # Conflict logic based on location and department
    conflict_indicator = 0
    if (department in ["Roads", "Electricity"] and site_location == "Location-1") or \
       (task_complexity > 7 and communication_frequency < 4) or \
       (historical_delay > 10 and resources_allocated > 70):
        conflict_indicator = 1 

    cost_reduction_category = "High" if cost_reduction_potential > 0.1 else "Low"

    project = {
        "project_id": project_id,
        "department": department,
        "task_priority": task_priority,
        "task_complexity": task_complexity,
        "available_resources": available_resources,
        "resources_allocated": resources_allocated,
        "communication_frequency": communication_frequency,
        "historical_delay": historical_delay,
        "expected_completion_time": expected_completion_time,
        "actual_completion_time": actual_completion_time,
        "cost_estimate": cost_estimate,
        "actual_cost": actual_cost,
        "site_location": site_location,
        "latitude": latitude,
        "longitude": longitude,
        "project_start_date": str(project_start_date),
        "project_end_date": str(project_end_date),
        "conflict_indicator": conflict_indicator,
        "cost_reduction_potential": cost_reduction_potential,
        "cost_reduction_category": cost_reduction_category,
        "resource_utilization": resource_utilization,
        "complexity_to_priority_ratio": complexity_to_priority_ratio,
        "delay_factor": delay_factor,
        "adjusted_frequency": adjusted_frequency,
    }
    data.append(project)

# Save updated data to a JSON file
output_path = "data/enhanced_data.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Synthetic dataset with additional features generated and saved to {output_path}.")
