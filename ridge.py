import joblib
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, precision_score, recall_score

# --------------------------
# Config
# --------------------------
RANDOM_STATE = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "ridge_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

HIGH_RISK_PERCENTILE = 75  # top 25% as high-risk
TOP_K_FEATURES = 8         # number of features to select

# --------------------------
# Load data
# --------------------------
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# --------------------------
# Preprocessing
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(score_func=f_regression, k=TOP_K_FEATURES)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# --------------------------
# Train Ridge model
# --------------------------
model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
model.fit(X_train_selected, y_train)

# --------------------------
# Predict
# --------------------------
y_pred = model.predict(X_test_selected)

# Fix for older scikit-learn versions:
try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # manual RMSE

print(f"v0.2 Ridge Regression Test RMSE: {rmse:.3f}")

# --------------------------
# High-risk flag
# --------------------------
threshold = np.percentile(y_train, HIGH_RISK_PERCENTILE)
high_risk_pred = (y_pred >= threshold).astype(int)
high_risk_true = (y_test >= threshold).astype(int)

precision = precision_score(high_risk_true, high_risk_pred)
recall = recall_score(high_risk_true, high_risk_pred)
print(f"High-risk precision: {precision:.3f}, recall: {recall:.3f}")

# --------------------------
# Save metrics
# --------------------------
metrics = {
    "rmse": rmse,
    "high_risk_threshold": float(threshold),
    "precision": float(precision),
    "recall": float(recall)
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)

# --------------------------
# Save model & scaler
# --------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(selector, MODEL_DIR / "selector.joblib")

print(f"Model saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
print(f"Feature selector saved to {MODEL_DIR / 'selector.joblib'}")
