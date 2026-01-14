from kfp.v2.dsl import (
    Dataset,
    Output,
    component,
)

# Docker image from Artifact Registry
# Format: {REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}
BASE_IMAGE = "us-central1-docker.pkg.dev/project-1cd612d2-3ea2-4818-a72/mlops-repo/training-image:latest"

@component(
    base_image=BASE_IMAGE,
    output_component_file="data_ingestion.yaml"
)
def data_ingestion(
    dataset: Output[Dataset],
    bucket_name: str,
    data_path: str = "data/Housing.csv"
):
    """
    Loads and prepares the house price dataset.
    
    Args:
        dataset: Output artifact to store the prepared dataset
        bucket_name: GCS bucket name where the dataset is stored
        data_path: Path to the dataset file within the bucket
    """
    import pandas as pd
    import logging
    
    try:
        logging.info("Starting data ingestion...")
        
        # Load the dataset from the GCS bucket
        gcs_path = f"gs://{bucket_name}/{data_path}"
        logging.info(f"Loading dataset from {gcs_path}...")
        df = pd.read_csv(gcs_path)
        
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        
        # Save the dataset to the output artifact
        logging.info(f"Saving dataset to {dataset.path}...")
        df.to_csv(dataset.path, index=False)
        
        logging.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        raise
