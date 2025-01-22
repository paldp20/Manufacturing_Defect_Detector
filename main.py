from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from model import preprocess_data, train_model, evaluate_model, predict, save_model, load_model

app = FastAPI()

data = None
model = None

# Upload the CSV file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global data
    contents = await file.read()
    stringio = StringIO(contents.decode("utf-8"))
    data = pd.read_csv(stringio)
    print(f"Null values:/n {data.isnull().sum()}")
    return {"message": "File uploaded successfully"}

# Train the model
@app.post("/train/")
async def train():
    global data, model
    if data is None:
        return {"error": "No data uploaded"}

    X, y = preprocess_data(data)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    report = evaluate_model(X_test, y_test, model)
    save_model(model, "model.pkl")

    return {"message": "Model trained successfully", "report": report}

# Predict using the trained model
@app.post("/predict/")
async def predict_data(data_input: dict):
    global model
    if model is None:
        model = load_model("model.pkl")

    input_df = pd.DataFrame([data_input])

    prediction, confidence = predict(model, input_df)

    return {
        "DefectStatus": "Yes" if prediction == 1 else "No",
        "Confidence": round(confidence, 2)
    }
