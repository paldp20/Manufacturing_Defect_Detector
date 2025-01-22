import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    X = data.drop("DefectStatus", axis=1)  # Features
    y = data["DefectStatus"]  # Target variable
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def predict(model, input_data):
    probabilities = model.predict_proba(input_data)  # Predict probabilities
    prediction = np.argmax(probabilities, axis=1)  # Get the predicted class (0 or 1)
    confidence = np.max(probabilities, axis=1)  # Get the confidence level
    return prediction[0], confidence[0]  # Return predicted class and confidence level

def save_model(model, model_filename):
    joblib.dump(model, model_filename)

def load_model(model_filename):
    model = joblib.load(model_filename)
    return model
