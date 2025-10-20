# predict_v0_2.py  — FastAPI for v0.2 (Ridge)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np

# paths relative to V2/ directory
MODEL_PATH  = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"


# load artifacts at startup
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Diabetes Progression API v0.2")

class Features(BaseModel):
    age: float; sex: float; bmi: float; bp: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.2.0"}

@app.post("/predict")
def predict(f: Features):
    try:
        x = np.array([[f.age, f.sex, f.bmi, f.bp, f.s1, f.s2, f.s3, f.s4, f.s5, f.s6]], dtype=float)
        xs = scaler.transform(x)
        y  = float(model.predict(xs)[0])
        return {"prediction": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
