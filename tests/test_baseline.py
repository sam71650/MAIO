# tests/test_baseline.py

import joblib
import numpy as np
from baseline import MODEL_PATH, SCALER_PATH


def test_model_and_scaler_exist():
    """Check if model and scaler files were saved."""
    assert MODEL_PATH.exists(), f"{MODEL_PATH} does not exist"
    assert SCALER_PATH.exists(), f"{SCALER_PATH} does not exist"


def test_model_prediction():
    """Check if model can make a simple prediction."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # dummy input with correct feature shape (10 features)
    X_dummy = np.zeros((1, 10))
    X_dummy_scaled = scaler.transform(X_dummy)
    pred = model.predict(X_dummy_scaled)
    assert pred.shape == (1,)
