# Multi-stage build for ML Pipeline
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY run_pipeline.py .
COPY src/ ./src/

# Create directories for data and output
RUN mkdir -p /app/data /app/output/models /app/output/metrics /app/output/artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command - run the pipeline
CMD ["python", "run_pipeline.py"]
