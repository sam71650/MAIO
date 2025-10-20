from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

# --------------------------
# Config
# --------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ridge_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
SELECTOR_PATH = MODEL_DIR / "selector.joblib"
MODEL_VERSION = "v0.2"

# --------------------------
# Load trained model, scaler, selector
# --------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)

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

        # Preprocess
        x_scaled = scaler.transform(x)
        x_selected = selector.transform(x_scaled)

        # Predict
        prediction = model.predict(x_selected)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
