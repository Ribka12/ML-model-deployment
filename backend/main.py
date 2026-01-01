from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI(title="ML Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained PIPELINES
log_model = joblib.load("models/logistic_model.joblib")
dt_model = joblib.load("models/decision_tree_model.joblib")

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "API is running"}

def validate_input(features):
    if len(features) != 30:
        raise ValueError("Exactly 30 features are required")

@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    validate_input(data.features)
    X = np.array(data.features).reshape(1, -1)

    prediction = int(log_model.predict(X)[0])
    probability = log_model.predict_proba(X)[0].tolist()

    return {
        "model": "Logistic Regression",
        "prediction": prediction,
        "confidence": round(probability[prediction] * 100, 2)
    }

@app.post("/predict/tree")
def predict_tree(data: InputData):
    validate_input(data.features)
    X = np.array(data.features).reshape(1, -1)

    prediction = int(dt_model.predict(X)[0])
    probability = dt_model.predict_proba(X)[0].tolist()

    return {
        "model": "Decision Tree",
        "prediction": prediction,
        "confidence": round(probability[prediction] * 100, 2)
    }
