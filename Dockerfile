# --------------------------
# Stage 1: Build
# --------------------------
FROM python:3.11-slim AS build

# Set working directory
WORKDIR /app

# Copy dependencies file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# --------------------------
# Stage 2: Runtime
# --------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy everything from the build stage
COPY --from=build /app /app

# Expose FastAPI default port
EXPOSE 8000

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl --fail http://localhost:8000/health || exit 1

# Start FastAPI app
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
