"""
Vertex AI Pipeline for House Price Prediction

This script defines and runs a complete ML pipeline on Google Cloud Vertex AI.
The pipeline includes data ingestion, preprocessing, training, and evaluation.

Before running:
1. Copy .env.example to .env and configure your GCP project details
2. Build and push the Docker image to Artifact Registry
3. Upload the Housing.csv dataset to your GCS bucket
4. Ensure you have proper IAM permissions configured
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from kfp.v2 import compiler
from google.cloud import aiplatform
from kfp.v2.dsl import pipeline

# Import component functions
from src.data_ingestion import data_ingestion
from src.preprocessing import preprocessing
from src.training import training
from src.evaluation import evaluation

# ============================================================================
# LOAD CONFIGURATION FROM .env FILE
# ============================================================================

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# GCP Project Configuration
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')
PIPELINE_ROOT_FOLDER = os.getenv('PIPELINE_ROOT_FOLDER', 'pipeline_root_houseprice')

# Artifact Registry Configuration
REPOSITORY = os.getenv('REPOSITORY')
IMAGE_NAME = os.getenv('IMAGE_NAME')
IMAGE_TAG = os.getenv('IMAGE_TAG', 'latest')

# Dataset Configuration
DATA_PATH = os.getenv('DATA_PATH', 'data/Housing.csv')

# Hyperparameters
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', '100'))
MAX_DEPTH = int(os.getenv('MAX_DEPTH', '10'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))

# Construct paths
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/{PIPELINE_ROOT_FOLDER}"
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

# ============================================================================
# PIPELINE DEFINITION
# ============================================================================

@pipeline(
    name="houseprice-pipeline",
    description="End-to-end pipeline for house price prediction",
    pipeline_root=PIPELINE_ROOT
)
def houseprice_pipeline(
    bucket_name: str = BUCKET_NAME,
    data_path: str = DATA_PATH
):
    """
    Defines the complete ML pipeline workflow.
    
    Pipeline steps:
    1. Data Ingestion: Load dataset from GCS
    2. Preprocessing: Clean and transform data
    3. Training: Train RandomForest model
    4. Evaluation: Evaluate model and generate report
    
    Args:
        bucket_name: GCS bucket containing the dataset
        data_path: Path to the dataset file within the bucket
    """
    
    # Step 1: Data Ingestion
    ingestion_task = data_ingestion(
        bucket_name=bucket_name,
        data_path=data_path
    )
    
    # Step 2: Preprocessing
    preprocessing_task = preprocessing(
        input_dataset=ingestion_task.outputs["dataset"]
    )
    
    # Step 3: Model Training
    training_task = training(
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        hyperparameters={
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE
        }
    )
    
    # Step 4: Model Evaluation
    evaluation_task = evaluation(
        model=training_task.outputs["model"],
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"]
    )

# ============================================================================
# PIPELINE COMPILATION AND EXECUTION
# ============================================================================

def compile_pipeline():
    """Compile the pipeline to a JSON specification."""
    import json
    
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=houseprice_pipeline,
        package_path='houseprice_pipeline.json'
    )
    
    try:
        with open('houseprice_pipeline.json', 'r', encoding='utf-8') as f:
            pipeline_spec = json.load(f)
        
        with open('houseprice_pipeline.json', 'w', encoding='utf-8') as f:
            json.dump(pipeline_spec, f, ensure_ascii=True, indent=2)
        
        print("✓ Pipeline compiled and encoded successfully to 'houseprice_pipeline.json'")
    except Exception as e:
        print(f"✓ Pipeline compiled to 'houseprice_pipeline.json' (encoding warning: {e})")

def run_pipeline():
    """Initialize Vertex AI and submit the pipeline job."""
    print(f"Initializing Vertex AI (Project: {PROJECT_ID}, Region: {REGION})...")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    print("Creating pipeline job...")
    pipeline_job = aiplatform.PipelineJob(
        display_name="houseprice-pipeline-job",
        template_path="houseprice_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False  # Set to True to enable caching between runs
    )
    
    print("Submitting pipeline job to Vertex AI...")
    print(f"Pipeline artifacts will be stored at: {PIPELINE_ROOT}")
    print("\nYou can monitor the pipeline execution in the GCP Console:")
    print(f"https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")
    
    pipeline_job.run(
        service_account=None,  # Uses default compute service account
        sync=True  # Wait for pipeline to complete (set to False for async)
    )
    
    print("\n✓ Pipeline execution completed!")
    print(f"View results at: {PIPELINE_ROOT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Validate that environment variables are loaded
    required_vars = {
        'PROJECT_ID': PROJECT_ID,
        'REGION': REGION,
        'BUCKET_NAME': BUCKET_NAME,
        'REPOSITORY': REPOSITORY,
        'IMAGE_NAME': IMAGE_NAME
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print("❌ ERROR: Missing required environment variables in .env file:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease ensure .env file exists and contains all required variables.")
        print("You can copy .env.example to .env and update the values.")
        sys.exit(1)
    
    print("=" * 70)
    print("VERTEX AI PIPELINE - HOUSE PRICE PREDICTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Project ID: {PROJECT_ID}")
    print(f"  Region: {REGION}")
    print(f"  Bucket: {BUCKET_NAME}")
    print(f"  Pipeline Root: {PIPELINE_ROOT}")
    print(f"  Base Image: {BASE_IMAGE}")
    print(f"  Dataset: gs://{BUCKET_NAME}/{DATA_PATH}")
    print("\n" + "=" * 70 + "\n")
    
    # Compile the pipeline
    compile_pipeline()
    
    # Run the pipeline
    print("\nStarting pipeline execution...")
    run_pipeline()
