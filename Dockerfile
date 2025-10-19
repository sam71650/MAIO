# Dockerfile for v0.2
# --------------------------
# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements (install dependencies)
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib pydantic numpy

# Copy model folder and API
COPY model_v0_2/ model_v0_2/
COPY predict_v0_2.py predict_v0_2.py

# Expose port
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "predict_v0_2:app", "--host", "0.0.0.0", "--port", "8000"]
