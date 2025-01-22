# Predictive Analysis for Manufacturing Operations

## Project Overview

This project demonstrates how to build a simple predictive analysis model for manufacturing data. The goal is to predict machine downtime or production defects using machine learning techniques. A RESTful API is built using FastAPI that allows you to upload manufacturing data, train a model, and make predictions.

## Requirements

To run this project, you need the following Python libraries:

- FastAPI
- Uvicorn (for running the FastAPI application)
- scikit-learn (for machine learning)
- pandas (for data manipulation)
- imbalanced-learn (for SMOTE)
- joblib (for saving/loading models)

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Application
- **Step 1:** Clone or download the repository
- **Step 2:** Install dependencies
```bash
pip install -r requirements.txt
```
- **Step 3:** Run the API server. Start the FastAPI server using uvicorn:

```bash
uvicorn main:app --reload
```
The server will run locally, and you can access the API documentation via http://127.0.0.1:8000/docs.

## API Endpoints
1. ### Upload CSV file
- **Endpoint:** /upload/
- **Method:** POST
- **Description:** Upload a CSV file containing manufacturing data (e.g., machine temperature, run time, etc.).
- **Input Example:**
https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
(has no null values)

2. ### Train Model
- **Endpoint:** /train/
- **Method:** POST
- **Description:** Train the model on the uploaded dataset and evaluate its performance.
- **Response Example:**
```json
{
  "message": "Model trained successfully",
  "report": "classification report here"
}
```

3. ### Make Prediction
- **Endpoint:** /predict/
- **Method:** POST
- **Description:** Make a prediction using the trained model based on input data (e.g., machine temperature and run time).
- **Input Example:**
```json
{
    "ProductionVolume": 3240,
    "ProductionCost": 150.75,
    "SupplierQuality": 0.95,
    "DeliveryDelay": 5,
    "DefectRate": 0.03,
    "QualityScore": 8.5,
    "MaintenanceHours": 20,
    "DowntimePercentage": 2.5,
    "InventoryTurnover": 3.1,
    "StockoutRate": 0.02,
    "WorkerProductivity": 15.2,
    "SafetyIncidents": 1,
    "EnergyConsumption": 500,
    "EnergyEfficiency": 0.85,
    "AdditiveProcessTime": 25.4,
    "AdditiveMaterialCost": 85.0
}
```
- **Response Example:**
```json
{
  "DefectStatus": "Yes",
  "Confidence": 0.75
}
```

## How to Test Locally
You can test the API locally using Postman or any HTTP client. Here are some sample requests:

- Upload your CSV file to the /upload/ endpoint.
- Train the model with /train/.
- Send data for prediction to /predict/.

## Model Building Steps
A simple model training approach is taken up. Basic EDA steps are first performed to check for null values and other descripancies.

1. ### Data Preprocessing:

- The dataset is split into features (X) and the target variable (y).
- Numerical encoding is applied to categorical features.

2. ### Handling Class Imbalance:

- SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the target classes.

3. ### Model Selection:

- A Random Forest Classifier is used due to its robustness and ability to handle various feature types.

4. ### Training:

- The preprocessed data is split into training and testing sets (80% train, 20% test).
- The model is trained on the training set.

5. ### Evaluation:

- The model is evaluated on the test set using metrics like precision, recall, and F1-score.
- The classification report is returned as part of the training response.

The trained model is saved using Joblib for future inference.

## Conclusion
This project provides a simple approach to building a predictive model for manufacturing operations. The FastAPI-based API allows users to upload data, train the model, and make predictions for real-world decision-making scenarios.
