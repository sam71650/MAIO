import joblib
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Configuration
RANDOM_STATE = 42
MODEL_DIR = Path("model_v0_2")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_VERSION = "v0.2.0"

# Load dataset
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge regression model
model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"v0.2 Ridge trained. Test RMSE: {rmse:.3f}")

# Save model, scaler, and metrics
with open(METRICS_PATH, "w") as f:
    json.dump(
        {"model_version": MODEL_VERSION, "rmse": rmse, "model_type": "Ridge"},
        f,
        indent=2
    )

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(
    f"Metrics -> {METRICS_PATH}\n"
    f"Model -> {MODEL_PATH}\n"
    f"Scaler -> {SCALER_PATH}"
)
