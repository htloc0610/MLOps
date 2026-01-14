# Vertex AI Pipeline - House Price Prediction

Complete implementation of a Vertex AI pipeline for house price prediction using the Kubeflow Pipelines SDK.

## 📋 Project Structure

```
house_prediction/
├── Dockerfile              # Base image for pipeline components
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── run_pipeline.py        # Main pipeline orchestration script
├── SETUP.md              # Detailed setup instructions
└── src/                  # Pipeline components
    ├── __init__.py
    ├── data_ingestion.py    # Load data from GCS
    ├── preprocessing.py     # Clean and transform data
    ├── training.py         # Train RandomForest model
    └── evaluation.py       # Evaluate and visualize results
```

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Platform Setup**
   - Active GCP project with billing enabled
   - Vertex AI API enabled
   - gcloud CLI installed and authenticated
   - Docker Desktop installed and running

2. **Required GCP Resources**
   - GCS bucket created
   - Artifact Registry repository created
   - Proper IAM permissions configured

### Step 1: Configure Environment

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. The `.env` file is already configured with your GCP project details:
   - ✅ `PROJECT_ID=project-1cd612d2-3ea2-4818-a72`
   - ✅ `REGION=us-central1`
   - ✅ `BUCKET_NAME=housing-data-project-1cd612d2-3ea2-4818-a72`
   - ✅ `REPOSITORY=mlops-repo`
   - ✅ `IMAGE_NAME=training-image`

3. You can modify hyperparameters in `.env` if needed:
   ```bash
   N_ESTIMATORS=100
   MAX_DEPTH=10
   RANDOM_STATE=42
   ```

### Step 2: Upload Dataset

Download the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) and upload to GCS:

```bash
gsutil cp Housing.csv gs://YOUR_BUCKET_NAME/data/
```

### Step 3: Build and Push Docker Image

```bash
# Set environment variables
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export REPOSITORY="vertex-ai-pipeline-example"
export IMAGE_NAME="training"
export IMAGE_TAG="latest"

# Create Artifact Registry repository
gcloud artifacts repositories create $REPOSITORY \
    --repository-format=docker \
    --location=$REGION \
    --description="Repository for Vertex AI pipeline components"

# Configure Docker authentication
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build Docker image (for macOS, use --platform linux/amd64)
docker build --platform linux/amd64 -t $IMAGE_NAME:$IMAGE_TAG .

# Tag image for Artifact Registry
docker tag $IMAGE_NAME:$IMAGE_TAG \
    $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG

# Push to Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG
```

### Step 4: Run the Pipeline

```bash
python run_pipeline.py
```

Monitor the pipeline execution in the GCP Console:
```
https://console.cloud.google.com/vertex-ai/pipelines
```

## 📊 Pipeline Components

### 1. Data Ingestion
- Loads `Housing.csv` from GCS bucket
- Validates data structure
- Outputs dataset artifact

### 2. Preprocessing
- Handles missing values
- Scales numerical features (StandardScaler)
- Encodes categorical features (One-Hot Encoding)
- Outputs preprocessed dataset

### 3. Training
- Trains RandomForestRegressor
- Configurable hyperparameters
- Logs training and validation metrics
- Outputs trained model artifact

### 4. Evaluation
- Evaluates model performance
- Generates visualizations:
  - Actual vs Predicted prices
  - Residual plot
  - Feature importance
  - Residual distribution
- Creates HTML report with metrics

## 📈 Expected Results

After successful execution, you'll find:
- **Pipeline artifacts** in `gs://YOUR_BUCKET/pipeline_root_houseprice/`
- **Trained model** (joblib format)
- **Evaluation metrics** (MSE, R², MAE, RMSE)
- **HTML report** with visualizations
- **Pipeline DAG** in Vertex AI console

## 🧹 Cleanup

To avoid unnecessary costs:

```bash
# Delete pipeline artifacts
gsutil rm -r gs://YOUR_BUCKET/pipeline_root_houseprice/

# Delete Docker image
gcloud artifacts docker images delete \
    $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG

# Delete Artifact Registry repository
gcloud artifacts repositories delete $REPOSITORY --location=$REGION
```

## 🔒 Security Best Practices

This implementation follows OWASP security guidelines:
- **Environment Variables**: All sensitive configuration stored in `.env` file (not committed to git)
- **No Hardcoded Credentials**: Uses gcloud authentication and environment variables
- Uses Google's mirror registry for base images
- Proper IAM permission separation
- Input validation in all components
- Comprehensive error handling and logging
- `.gitignore` configured to protect `.env` file

## 📚 Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Kubeflow Pipelines SDK](https://www.kubeflow.org/docs/components/pipelines/)
- [Housing Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

## ⚠️ Important Notes

- Ensure Docker Desktop is running before building images
- macOS users must use `--platform linux/amd64` flag
- Pipeline execution incurs GCP costs
- Keep your GCP credentials secure
- Review IAM permissions before running

## 🐛 Troubleshooting

**Error: "Failed to create pipeline job"**
- Check IAM permissions for Vertex AI Service Agent
- Verify Artifact Registry permissions

**Error: "Dataset not found"**
- Confirm dataset is uploaded to correct GCS path
- Check bucket permissions

**Docker build fails**
- Ensure Docker Desktop is running
- Check internet connection for package downloads

For detailed setup instructions, see [SETUP.md](SETUP.md)
