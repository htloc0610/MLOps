#!/bin/bash
# Fix IAM permissions for Vertex AI Pipeline

PROJECT_ID="project-1cd612d2-3ea2-4818-a72"
USER_EMAIL=$(gcloud config get-value account)

echo "========================================================"
echo "Fixing Permissions for User: $USER_EMAIL"
echo "Project: $PROJECT_ID"
echo "========================================================"

# 1. Vertex AI User (Access to Vertex AI resources)
echo "Granting 'roles/aiplatform.user'..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/aiplatform.user"

# 2. Service Account User (Required to attach service accounts to jobs)
echo "Granting 'roles/iam.serviceAccountUser'..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/iam.serviceAccountUser"

# 3. Storage Object Admin (Access to GCS artifacts)
echo "Granting 'roles/storage.objectAdmin'..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:$USER_EMAIL" \
    --role="roles/storage.objectAdmin"

echo "========================================================"
echo "Permissions updated! You may need to wait 1-2 minutes for propagation."
echo "Then try running ./run_with_python310.sh again."
echo "========================================================"
