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

    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            raise ValueError(f"Feature {i+1} must be a number")

@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    validate_input(data.features)
    X = np.array(data.features).reshape(1, -1)

    pred = int(log_model.predict(X)[0])
    probs = log_model.predict_proba(X)[0]

    return {
        "model": "Logistic Regression",
        "prediction": "Benign" if pred == 1 else "Malignant",
        "confidence": round(max(probs) * 100, 2),
        "probabilities": {
            "malignant": round(probs[0] * 100, 2),
            "benign": round(probs[1] * 100, 2),
        }
    }


@app.post("/predict/tree")
def predict_tree(data: InputData):
    validate_input(data.features)
    X = np.array(data.features).reshape(1, -1)

    pred = int(dt_model.predict(X)[0])
    probs = dt_model.predict_proba(X)[0]

    return {
        "model": "Decision Tree",
        "prediction": "Benign" if pred == 1 else "Malignant",
        "confidence": round(max(probs) * 100, 2),
        "probabilities": {
            "malignant": round(probs[0] * 100, 2),
            "benign": round(probs[1] * 100, 2),
        }
    }
