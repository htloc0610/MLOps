from kfp.v2.dsl import (
    Dataset,
    Input,
    Output,
    component,
)

# Docker image from Artifact Registry
BASE_IMAGE = "us-central1-docker.pkg.dev/project-1cd612d2-3ea2-4818-a72/mlops-repo/training-image:latest"

@component(
    base_image=BASE_IMAGE,
    output_component_file="preprocessing.yaml"
)
def preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
):
    """
    Preprocesses the dataset for training.
    
    Args:
        input_dataset: Input dataset from the data ingestion step
        preprocessed_dataset: Output artifact for the preprocessed dataset
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import logging
    
    try:
        logging.info("Starting data preprocessing...")
        
        # Load the dataset
        df = pd.read_csv(input_dataset.path)
        logging.info(f"Loaded dataset with shape: {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logging.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
            # Fill missing values with median for numerical columns
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
                    logging.info(f"Filled missing values in {col} with median")
        
        # Separate features and target
        # Assuming 'price' is the target column (adjust if different)
        if 'price' in df.columns:
            target_col = 'price'
        elif 'Price' in df.columns:
            target_col = 'Price'
        else:
            # Use the last column as target if 'price' not found
            target_col = df.columns[-1]
            logging.warning(f"'price' column not found, using '{target_col}' as target")
        
        # Scale numerical features (excluding target)
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Only scale numerical columns
        numerical_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numerical_cols:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            logging.info(f"Scaled {len(numerical_cols)} numerical features")
        
        # Handle categorical features if any
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            logging.info(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
            # Simple one-hot encoding
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logging.info(f"Applied one-hot encoding. New shape: {df.shape}")
        
        # Save preprocessed dataset
        df.to_csv(preprocessed_dataset.path, index=False)
        logging.info(f"Preprocessed dataset saved to: {preprocessed_dataset.path}")
        logging.info(f"Final shape: {df.shape}")
        logging.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise
