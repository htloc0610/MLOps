# Vertex AI Pipeline Setup Guide

This guide provides detailed step-by-step instructions for setting up and running the Vertex AI house price prediction pipeline.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [GCP Project Setup](#gcp-project-setup)
3. [IAM Permissions](#iam-permissions)
4. [Environment Configuration](#environment-configuration)
5. [Docker Image Setup](#docker-image-setup)
6. [Dataset Upload](#dataset-upload)
7. [Running the Pipeline](#running-the-pipeline)
8. [Monitoring and Results](#monitoring-and-results)
9. [Cleanup](#cleanup)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.9+**: [Download](https://www.python.org/downloads/)
- **Docker Desktop**: [Download](https://docs.docker.com/desktop/)
- **gcloud CLI**: [Download](https://cloud.google.com/sdk/docs/install)

### GCP Requirements
- Active GCP account with billing enabled
- GCP project created
- Owner or Editor role on the project

---

## GCP Project Setup

### 1. Find Your Project ID

```bash
# List all your GCP projects
gcloud projects list

# Set your project as default
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Artifact Registry API
gcloud services enable artifactregistry.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable Compute Engine API
gcloud services enable compute.googleapis.com
```

### 3. Authenticate with gcloud

```bash
# Login to your Google account
gcloud auth login

# Set application default credentials
gcloud auth application-default login
```

---

## IAM Permissions

### Required Permissions

You need to grant the Vertex AI Service Agent access to your Artifact Registry:

1. **Find your project number:**
   ```bash
   gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)"
   ```

2. **Grant Artifact Registry Reader role:**
   ```bash
   # Replace PROJECT_NUMBER with your actual project number
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
       --role="roles/artifactregistry.reader"
   ```

3. **Verify permissions in GCP Console:**
   - Go to IAM & Admin > IAM
   - Look for `PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com`
   - Confirm it has "Artifact Registry Reader" role

---

## Environment Configuration

### 1. Create GCS Bucket

```bash
# Set bucket name (must be globally unique)
export BUCKET_NAME="mlops-houseprice-$(date +%s)"

# Create bucket in your region
gcloud storage buckets create gs://$BUCKET_NAME \
    --location=europe-west1 \
    --uniform-bucket-level-access

# Verify bucket creation
gcloud storage buckets list | grep $BUCKET_NAME
```

### 2. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```bash
PROJECT_ID=your-actual-project-id
REGION=europe-west1
BUCKET_NAME=your-actual-bucket-name
PIPELINE_ROOT_FOLDER=pipeline_root_houseprice
DATA_PATH=data/Housing.csv
REPOSITORY=vertex-ai-pipeline-example
IMAGE_NAME=training
IMAGE_TAG=latest
```

### 3. Update run_pipeline.py

Edit `run_pipeline.py` and update lines 20-30 with your configuration:

```python
PROJECT_ID = "your-actual-project-id"
REGION = "europe-west1"
BUCKET_NAME = "your-actual-bucket-name"
```

---

## Docker Image Setup

### 1. Create Artifact Registry Repository

```bash
# Set environment variables
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export REPOSITORY="vertex-ai-pipeline-example"

# Create repository
gcloud artifacts repositories create $REPOSITORY \
    --repository-format=docker \
    --location=$REGION \
    --description="Repository for Vertex AI pipeline components"

# Verify repository creation
gcloud artifacts repositories list --location=$REGION
```

### 2. Configure Docker Authentication

```bash
# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker $REGION-docker.pkg.dev
```

### 3. Build Docker Image

```bash
# Navigate to project directory
cd house_prediction

# Set image variables
export IMAGE_NAME="training"
export IMAGE_TAG="latest"

# Build image (macOS users: include --platform flag)
# For macOS:
docker build --platform linux/amd64 -t $IMAGE_NAME:$IMAGE_TAG .

# For Linux/Windows:
docker build -t $IMAGE_NAME:$IMAGE_TAG .
```

### 4. Tag and Push Image

```bash
# Tag image for Artifact Registry
docker tag $IMAGE_NAME:$IMAGE_TAG \
    $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG

# Push to Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG

# Verify image in Artifact Registry
gcloud artifacts docker images list \
    $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY
```

### 5. Update Component Base Images

After successfully pushing the image, update the `BASE_IMAGE` variable in each component file:

**Files to update:**
- `src/data_ingestion.py`
- `src/preprocessing.py`
- `src/training.py`
- `src/evaluation.py`

Replace this line in each file:
```python
BASE_IMAGE = "europe-west1-docker.pkg.dev/your-project-id/vertex-ai-pipeline-example/training:latest"
```

With your actual image path:
```python
BASE_IMAGE = "europe-west1-docker.pkg.dev/YOUR_ACTUAL_PROJECT_ID/vertex-ai-pipeline-example/training:latest"
```

---

## Dataset Upload

### 1. Download Dataset

Download the Housing Prices Dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
- File: `Housing.csv`

### 2. Upload to GCS

```bash
# Upload dataset to GCS bucket
gcloud storage cp Housing.csv gs://$BUCKET_NAME/data/

# Verify upload
gcloud storage ls gs://$BUCKET_NAME/data/
```

---

## Running the Pipeline

### 1. Install Python Dependencies (Optional - for local testing)

```bash
pip install -r requirements.txt
```

### 2. Verify Configuration

Double-check that you've updated:
- ✅ `run_pipeline.py` configuration (lines 20-30)
- ✅ `BASE_IMAGE` in all component files
- ✅ Dataset uploaded to GCS
- ✅ Docker image pushed to Artifact Registry

### 3. Run the Pipeline

```bash
python run_pipeline.py
```

Expected output:
```
======================================================================
VERTEX AI PIPELINE - HOUSE PRICE PREDICTION
======================================================================

Configuration:
  Project ID: your-project-id
  Region: europe-west1
  Bucket: your-bucket-name
  Pipeline Root: gs://your-bucket-name/pipeline_root_houseprice
  Base Image: europe-west1-docker.pkg.dev/...
  Dataset: gs://your-bucket-name/data/Housing.csv

======================================================================

Compiling pipeline...
✓ Pipeline compiled successfully to 'houseprice_pipeline.json'

Starting pipeline execution...
Submitting pipeline job to Vertex AI...
```

---

## Monitoring and Results

### 1. Monitor in GCP Console

Navigate to:
```
https://console.cloud.google.com/vertex-ai/pipelines?project=YOUR_PROJECT_ID
```

You should see:
- Pipeline DAG visualization
- Component execution status
- Logs for each component
- Execution timeline

### 2. View Pipeline Artifacts

```bash
# List all pipeline artifacts
gcloud storage ls -r gs://$BUCKET_NAME/pipeline_root_houseprice/

# Download evaluation HTML report
gcloud storage cp \
    gs://$BUCKET_NAME/pipeline_root_houseprice/.../evaluation/html \
    ./evaluation_report.html
```

### 3. Expected Results

After successful execution:
- ✅ All 4 components completed (green checkmarks)
- ✅ Model R² score > 0.7
- ✅ Evaluation HTML report generated
- ✅ Model artifacts stored in GCS

---

## Cleanup

To avoid unnecessary GCP costs, clean up resources after testing:

### 1. Delete Pipeline Artifacts

```bash
gcloud storage rm -r gs://$BUCKET_NAME/pipeline_root_houseprice/
```

### 2. Delete Docker Image

```bash
gcloud artifacts docker images delete \
    $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG \
    --quiet
```

### 3. Delete Artifact Registry Repository

```bash
gcloud artifacts repositories delete $REPOSITORY \
    --location=$REGION \
    --quiet
```

### 4. Delete GCS Bucket (Optional)

```bash
# Delete entire bucket and all contents
gcloud storage rm -r gs://$BUCKET_NAME
```

---

## Troubleshooting

### Error: "Failed to create pipeline job"

**Cause:** Missing IAM permissions

**Solution:**
```bash
# Grant Artifact Registry Reader role to Vertex AI Service Agent
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"
```

### Error: "Dataset not found"

**Cause:** Dataset not uploaded or wrong path

**Solution:**
```bash
# Verify dataset exists
gcloud storage ls gs://$BUCKET_NAME/data/Housing.csv

# Re-upload if missing
gcloud storage cp Housing.csv gs://$BUCKET_NAME/data/
```

### Error: "Docker build fails"

**Cause:** Docker Desktop not running or network issues

**Solution:**
1. Start Docker Desktop
2. Check internet connection
3. Try building again

### Error: "Permission denied" when pushing Docker image

**Cause:** Docker not authenticated with Artifact Registry

**Solution:**
```bash
gcloud auth configure-docker $REGION-docker.pkg.dev
```

### Pipeline component fails with import errors

**Cause:** Missing dependencies in Docker image

**Solution:**
1. Verify `requirements.txt` is complete
2. Rebuild Docker image
3. Push updated image to Artifact Registry
4. Re-run pipeline

### Pipeline stuck or taking too long

**Cause:** Large dataset or resource constraints

**Solution:**
1. Check component logs in GCP Console
2. Verify dataset size is reasonable
3. Consider increasing machine resources in component definitions

---

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Kubeflow Pipelines SDK v2](https://www.kubeflow.org/docs/components/pipelines/v2/)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [GCS Documentation](https://cloud.google.com/storage/docs)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review GCP Console logs
3. Consult Vertex AI documentation
4. Check component-specific error messages

---

**Last Updated:** January 2026
