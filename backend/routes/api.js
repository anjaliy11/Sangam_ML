const express = require('express');
const axios = require('axios');
const router = express.Router();

// Base URL for Flask API
const FLASK_API_URL = 'http://localhost:5001';

// Route for predicting task completion time
router.post('/predict-task-completion', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_task_completion`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting task completion:', error.message);
        res.status(500).send('Error predicting task completion time');
    }
});

// Route for predicting train delay
router.post('/predict-train-delay', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_train_delay`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting train delay:', error.message);
        res.status(500).send('Error predicting train delay');
    }
});

// Route for predicting resource allocation
router.post('/predict-resources', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_resources`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting resource allocation:', error.message);
        res.status(500).send('Error predicting resource allocation');
    }
});

// Route for anomaly detection
router.post('/predict-anomaly', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_anomaly`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error detecting anomaly:', error.message);
        res.status(500).send('Error detecting anomaly');
    }
});

// Route for predicting department conflict
router.post('/predict-department-conflict', async (req, res) => {
    try {
        const mlApiUrl = `${FLASK_API_URL}/predict_department_conflict`;
        const response = await axios.post(mlApiUrl, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting department conflict:', error.message);
        res.status(500).send('Error predicting department conflict');
    }
});

module.exports = router;
