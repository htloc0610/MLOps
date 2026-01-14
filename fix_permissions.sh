#!/bin/bash
# Fix IAM permissions for Vertex AI Pipeline (Robust Version)

PROJECT_ID="project-1cd612d2-3ea2-4818-a72"
USER_EMAIL=$(gcloud config get-value account)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
DEFAULT_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

echo "========================================================"
echo "Fixing Permissions for Project: $PROJECT_ID"
echo "User: $USER_EMAIL"
echo "Default Service Account: $DEFAULT_SA"
echo "========================================================"

# 1. Enable APIs
echo "Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 2. Grant Admin Role to User (Broadest permission to rule out issues)
echo "Granting 'roles/aiplatform.admin' to User..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/aiplatform.admin"

echo "Granting 'roles/serviceusage.serviceUsageConsumer' to User..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/serviceusage.serviceUsageConsumer"

echo "Granting 'roles/storage.admin' to User..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/storage.admin"

echo "Granting 'roles/iam.serviceAccountUser' to User..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/iam.serviceAccountUser"

# 3. Grant Permissions to Default Service Account (Used by the pipeline job)
echo "Granting 'roles/aiplatform.serviceAgent' to Default Service Account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$DEFAULT_SA" \
    --role="roles/aiplatform.serviceAgent"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$DEFAULT_SA" \
    --role="roles/storage.objectAdmin"

echo "========================================================"
echo "Permissions updated!" 
echo ""
echo "IMPORTANT: IAM changes can take 2-5 minutes to propagate."
echo "========================================================"
