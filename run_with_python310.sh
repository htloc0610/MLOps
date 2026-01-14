#!/bin/bash
# Run Vertex AI Pipeline using Docker with Python 3.10

echo "=========================================="
echo "Running Vertex AI Pipeline with Docker"
echo "=========================================="

# Build Docker image with Python 3.10
echo "Building Docker image with Python 3.10..."
docker build -t vertex-pipeline:py310 -f- . <<'EOF'
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

CMD ["python", "run_pipeline.py"]
EOF

# Run pipeline in Docker with mounted credentials
echo ""
echo "Running pipeline in Docker container..."
docker run --rm \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
  vertex-pipeline:py310

echo ""
echo "=========================================="
echo "Pipeline execution completed!"
echo "=========================================="
