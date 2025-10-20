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

```bash
pip install -r requirements.txt


