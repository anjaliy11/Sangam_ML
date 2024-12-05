
const express = require('express');
const axios = require('axios');
const router = express.Router();

// Base URL for Flask API
const FLASK_API_URL = 'http://localhost:5001';

// Route for predicting task completion time
router.post('/predict_task_completion', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_task_completion`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting task completion:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// Route to handle cost reduction prediction
// router.post('/predict_cost_reduction', async (req, res) => {
//     try {
//         const mlApiUrl = `${FLASK_API_URL}/predict_cost_reduction`;
//         const response = await axios.post(mlApiUrl, req.body);
//         res.json(response.data);
//     } catch (error) {
//         console.error('Error predicting cost optimization:', error.message);
//         res.status(500).json({ error: error.message });
//     }
// });

// Route to handle conflict prediction
router.post('/predict_conflict', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_conflict`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting conflict:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// Route for predicting resource allocation
router.post('/predict_resource_allocation', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_resource_allocation`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting resource allocation:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// {
//     "department": "Roads",
//     "site_location": "Location-1",
//     "historical_conflicts": 10,
//     "project_overlap": 5,
//     "communication_frequency": 7
// } 
// Route for anomaly detection
router.post('/predict_anomaly', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_anomaly`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error detecting anomaly:', error.message);
        res.status(500).json({ error: error.message });
    }
});
// {
//     "task_priority": 4,
//     "task_complexity": 7,
//     "communication_frequency": 3,
//     "historical_delay": 10,
//     "time_difference": 5,
//     "resource_allocation_ratio": 0.75,
//     "site_location_encoded": 1,
//     "department_encoded": 2

// }
router.post('/predict_department', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_department`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error :', error.message);
        res.status(500).json({ error: error.message });
    }
});

module.exports = router;

// {
//     "department_1": "Roads",
//     "department_2": "Electricity",
//     "latitude": 28.7041,
//     "longitude": 77.1025,
//     "historical_conflicts": 12,
//     "project_overlap": 7,
//     "distance": 1.5,
//     "communication_frequency": 3
// }
