# MAIO

Virtual Diabetes Clinic Triage - ML Service

This repository contains a baseline ML service that predicts short-term diabetes progression. It uses scikit-learn's Diabetes dataset as a placeholder for real patient data. The service exposes a REST API via FastAPI and is packaged as a Docker image for easy deployment.
Features

    Predict disease progression (/predict)
    Health check (/health)
    Self-contained Docker image
    Versioned releases via GitHub Actions

API Usage
Health Check

curl http://localhost:8000/health


Response:
{
  "status": "ok",
  "model_version": "v0.1.0"
}

Predict
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"age\":0.02,\"sex\":-0.044,\"bmi\":0.06,\"bp\":-0.03,\"s1\":-0.02,\"s2\":0.03,\"s3\":-0.02,\"s4\":0.02,\"s5\":0.02,\"s6\":-0.001}"


Response:
{
  "prediction": 235.9496372217627
}
