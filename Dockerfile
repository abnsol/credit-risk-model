# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to save space
# Using --upgrade pip to ensure pip is up-to-date
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# This copies your entire project structure into the container
COPY . /app

# Set environment variables for MLflow (if using a remote tracking server later)
# For local MLflow tracking (default behavior), these might not be strictly needed,
# but good to include for future scalability.
# ENV MLFLOW_TRACKING_URI="http://mlflow_server:5000" # Example for remote MLflow server
# ENV MLFLOW_S3_ENDPOINT_URL="http://minio:9000" # Example for S3-compatible artifact store

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# src.api.main:app refers to the 'app' object in src/api/main.py
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]