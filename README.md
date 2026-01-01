 # ML-model-deployment

**DONE_BY : Ribka Mengiste UGR/9680/15 Software Stream**


This repository contains the **backend and frontend** for deploying two machine learning models trained on the **Breast Cancer Wisconsin dataset**:

1. **Logistic Regression**
2. **Decision Tree**

The models are exported as `.joblib` files and integrated into a **FastAPI backend**. The frontend is a simple HTML/JavaScript interface to send input features and receive predictions.

---


---

## ⚙️ Backend (FastAPI)

The backend exposes the following endpoints:

| Endpoint                | Method | Description                           |
|-------------------------|--------|---------------------------------------|
| `/`                     | GET    | Health check, returns "API is running" |
| `/predict/logistic`     | POST   | Predict using Logistic Regression      |
| `/predict/tree`         | POST   | Predict using Decision Tree            |

### Input Format

POST request body (JSON):

```json
{
  "features": [f1, f2, f3, ..., f30]
}

{
  "model": "Logistic Regression",
  "prediction": 1
}
