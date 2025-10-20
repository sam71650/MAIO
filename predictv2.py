from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


# --------------------------
# Config
# --------------------------
MODEL_PATH = "model/baseline_model.joblib"
SCALER_PATH = "model/scaler.joblib"
MODEL_VERSION = "v0.1"


# Load trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------
# Input schema
# --------------------------
class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: PatientFeatures):
    try:
        x = np.array([[
            features.age, features.sex, features.bmi, features.bp,
            features.s1, features.s2, features.s3, features.s4,
            features.s5, features.s6
        ]])
        x_scaled = scaler.transform(x)
        prediction = model.predict(x_scaled)[0]
        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
