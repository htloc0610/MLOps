from kfp.v2.dsl import (
    Dataset,
    Input,
    Model,
    Output,
    Metrics,
    component,
)

# Docker image from Artifact Registry
BASE_IMAGE = "us-central1-docker.pkg.dev/project-1cd612d2-3ea2-4818-a72/mlops-repo/training-image:latest"

@component(
    base_image=BASE_IMAGE,
    output_component_file="training.yaml"
)
def training(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    hyperparameters: dict
):
    """
    Trains the model on the preprocessed dataset.
    
    Args:
        preprocessed_dataset: Input preprocessed dataset
        model: Output artifact for the trained model
        metrics: Output artifact for training metrics
        hyperparameters: Dictionary of hyperparameters
    """
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import logging
    import json
    
    try:
        logging.info("Starting model training...")
        
        # Load preprocessed dataset
        df = pd.read_csv(preprocessed_dataset.path)
        logging.info(f"Loaded preprocessed dataset with shape: {df.shape}")
        
        # Split features and target
        # Assuming the target column is 'price' or 'Price' or the last column
        if 'price' in df.columns:
            target_col = 'price'
        elif 'Price' in df.columns:
            target_col = 'Price'
        else:
            target_col = df.columns[-1]
            logging.warning(f"Using '{target_col}' as target column")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=hyperparameters.get('random_state', 42)
        )
        
        logging.info(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
        
        # Initialize and train the model
        rf_model = RandomForestRegressor(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 10),
            random_state=hyperparameters.get('random_state', 42),
            n_jobs=-1
        )
        
        logging.info(f"Training Random Forest with hyperparameters: {hyperparameters}")
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = rf_model.predict(X_train)
        y_val_pred = rf_model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Log metrics
        metrics.log_metric("train_mse", train_mse)
        metrics.log_metric("train_r2", train_r2)
        metrics.log_metric("train_mae", train_mae)
        metrics.log_metric("validation_mse", val_mse)
        metrics.log_metric("validation_r2", val_r2)
        metrics.log_metric("validation_mae", val_mae)
        
        # Save the model
        joblib.dump(rf_model, model.path)
        
        logging.info(f"Model saved to: {model.path}")
        logging.info(f"Training Metrics - MSE: {train_mse:.2f}, R2: {train_r2:.4f}, MAE: {train_mae:.2f}")
        logging.info(f"Validation Metrics - MSE: {val_mse:.2f}, R2: {val_r2:.4f}, MAE: {val_mae:.2f}")
        logging.info("Model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
