"""
Preprocessing Module - Local Version

This module handles data preprocessing including:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Train/test splitting
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

logger = logging.getLogger(__name__)


def preprocessing(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    Preprocess the dataset for model training.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
        
    Raises:
        ValueError: If preprocessing fails
    """
    try:
        logger.info("Starting data preprocessing...")
        logger.info(f"Input dataset shape: {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
            
            # Fill missing values in numerical columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"  Filled missing values in '{col}' with median: {median_val:.2f}")
        
        # Identify target column
        # Try common names first, then use last column
        target_col = None
        for col_name in ['price', 'Price', 'PRICE', 'target', 'Target']:
            if col_name in df.columns:
                target_col = col_name
                break
        
        if target_col is None:
            target_col = df.columns[-1]
            logger.warning(f"Target column not found, using last column: '{target_col}'")
        else:
            logger.info(f"Target column identified: '{target_col}'")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Target: '{target_col}'")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            logger.info(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
            
            # Simple label encoding for categorical variables
            for col in categorical_cols:
                X[col] = pd.Categorical(X[col]).codes
                logger.info(f"  Encoded '{col}' to numeric")
        
        # Get numerical columns only
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Numerical features: {len(numerical_cols)} columns")
        
        # Split into train and test sets
        logger.info(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"  Train set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        
        # Feature scaling (StandardScaler)
        logger.info("Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"  Scaler fitted on {X_train.shape[1]} features")
        logger.info(f"  Mean: {scaler.mean_[:3]}... (first 3 features)")
        logger.info(f"  Std: {scaler.scale_[:3]}... (first 3 features)")
        
        logger.info("✓ Preprocessing completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


def get_feature_names(df: pd.DataFrame, target_col: str = None) -> list:
    """
    Get list of feature names from DataFrame.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column to exclude
        
    Returns:
        List of feature column names
    """
    if target_col is None:
        # Try to identify target column
        for col_name in ['price', 'Price', 'PRICE', 'target', 'Target']:
            if col_name in df.columns:
                target_col = col_name
                break
        if target_col is None:
            target_col = df.columns[-1]
    
    return [col for col in df.columns if col != target_col]
