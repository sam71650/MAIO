import joblib
import numpy as np
from pathlib import Path


# --------------------------
# Paths for v0.2 artifacts
# --------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ridge_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
SELECTOR_PATH = MODEL_DIR / "selector.joblib"


# --------------------------
# Tests
# --------------------------
def test_model_and_scaler_exist():
   
    assert MODEL_PATH.exists(), f"{MODEL_PATH} not exist"
    assert SCALER_PATH.exists(), f"{SCALER_PATH} not exist"
    assert SELECTOR_PATH.exists(), f"{SELECTOR_PATH} not exist"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)

    # dummy input with correct feature shape (10 features)
    X_dummy = np.zeros((1, 10))
    X_dummy_scaled = scaler.transform(X_dummy)
    X_dummy_selected = selector.transform(X_dummy_scaled)
    pred = model.predict(X_dummy_selected)

    # assert prediction shape
    assert pred.shape == (1,)
