

# Dockerfile for v0.1
# --------------------------
# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib pydantic numpy

# Copy model folder and API code
COPY model/ model/
COPY predict.py predict.py

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]

