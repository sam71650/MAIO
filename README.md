# MAIO

This project demonstrates how to **build, version, and deploy machine learning models** using **FastAPI** and **Docker**.  
Two versions of a regression model (v0.1 and v0.2) are deployed as REST APIs and published to the **GitHub Container Registry (GHCR)**.

---

## Overview

| Version | Description |
|----------|--------------|
| **v0.1** | Baseline Ridge Regression model |
| **v0.2** | Improved Ridge Regression model (new scaler and tuned parameters) |

Each version runs as a separate FastAPI service with:
- `/health` → Health check (returns version)
- `/predict` → Returns model prediction

---

## Repository Structure

maio/
│
├── v0.1/
│ ├── .github/workflows/ # CI/CD workflows for v0.1
│ ├── models/ # Model artifacts for baseline (v0.1)
│ ├── baseline.py # Baseline model training (v0.1)
│ ├── predict.py # FastAPI app for v0.1
│ ├── tests/ # Unit tests for v0.1
│ ├── requirements.txt # Dependencies for v0.1
│ ├── Dockerfile # Docker build for v0.1
│ └── README.md # Docs for v0.1
│
├── v0.2/
│ ├── .github/workflows/ # CI/CD workflows for v0.2
│ ├── models/ # Model artifacts for ridge (v0.2)
│ ├── ridge.py # Ridge model training (v0.2)
│ ├── predictv2.py # FastAPI app for v0.2
│ ├── tests/ # Unit tests for v0.2
│ ├── requirements.txt # Dependencies for v0.2
│ ├── Dockerfile # Docker build for v0.2
│ └── README.md # Docs for v0.2
│
├── README.md # Main project documentation
└── CHANGELOG.md # Changelog for all versions


yaml
Copy code

---

## Requirements

## Install Dependencies

You’ll need:

- Python 3.10+  
- FastAPI  
- Uvicorn  
- scikit-learn  
- joblib  
- Docker  

## 🐳 Pull Images

```bash
docker pull ghcr.io/sam71650/maio:v0.1.0
```
```bash
docker pull ghcr.io/sam71650/maio:v0.2.0
```

🚀 Run Containers

```bash
docker run -d -p 8011:8000 --name maio_v1 ghcr.io/sam71650/maio:v0.1.0
```
If you want to run version 2 just run this :
```bash
docker run -d -p 8012:8000 --name maio_v2 ghcr.io/sam71650/maio:v0.2.0
```

```bash
docker ps
```

Health check
```bash
curl http://localhost:8000/health
```

Prediction
```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" ^
 -d "{\"age\":0.02,\"sex\":-0.044,\"bmi\":0.06,\"bp\":-0.03,\"s1\":-0.02,\"s2\":0.03,\"s3\":-0.02,\"s4\":0.02,\"s5\":0.02,\"s6\":-0.001}"
```

✅ Results

Version	Port	Health Response	Prediction Output
v0.1	8011	{"status":"ok","model_version":"v0.1"}	{"prediction":235.9496372217627}
v0.2	8012	{"status":"ok","model_version":"v0.2"}	{"prediction":195.78655231473178}

Published Docker Images
Version	Docker Image
v0.1	ghcr.io/sam71650/maio:v0.1.0
v0.2	ghcr.io/sam71650/maio:v0.2.0


Conclusion
This project successfully demonstrates:
Building and versioning ML models
Deploying with FastAPI and Docker
Publishing to GitHub Container Registry
Verifying two model versions (v0.1 and v0.2) with consistent APIs
Both versions return valid predictions and confirm a successful versioned deployment workflow.
