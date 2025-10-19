import joblib
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --------------------------
# Config
# --------------------------
RANDOM_STATE = 42
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --------------------------
# Load data
# --------------------------
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# --------------------------
# Preprocess
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Train model
# --------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --------------------------
# Evaluate
# --------------------------
y_pred = model.predict(X_test_scaled)

# Compute RMSE in a compatible way for all scikit-learn versions
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
print(f"Test RMSE: {rmse:.3f}")

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump({"rmse": rmse}, f)

# --------------------------
# Save model & scaler
# --------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
