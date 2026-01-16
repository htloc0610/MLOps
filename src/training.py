"""
Training Module - Local Version

This module handles model training using scikit-learn Random Forest.
No Google Cloud or KFP dependencies required.
"""

import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any

logger = logging.getLogger(__name__)


def training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameters: Dict[str, Any] = None
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train: Training features (scaled)
        y_train: Training target values
        hyperparameters: Dictionary of model hyperparameters
            - n_estimators: Number of trees (default: 100)
            - max_depth: Maximum tree depth (default: 10)
            - random_state: Random seed (default: 42)
            - min_samples_split: Minimum samples to split (default: 2)
            - min_samples_leaf: Minimum samples per leaf (default: 1)
            
    Returns:
        Trained RandomForestRegressor model
        
    Raises:
        Exception: If training fails
    """
    try:
        logger.info("Starting model training...")
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        params = {
            'n_estimators': hyperparameters.get('n_estimators', 100),
            'max_depth': hyperparameters.get('max_depth', 10),
            'random_state': hyperparameters.get('random_state', 42),
            'min_samples_split': hyperparameters.get('min_samples_split', 2),
            'min_samples_leaf': hyperparameters.get('min_samples_leaf', 1),
            'n_jobs': -1,  # Use all CPU cores
            'verbose': 0
        }
        
        logger.info(f"Training Random Forest with parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Initialize model
        model = RandomForestRegressor(**params)
        
        # Train model
        logger.info(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        logger.info("Training metrics:")
        logger.info(f"  MSE:  {train_mse:,.2f}")
        logger.info(f"  RMSE: {train_rmse:,.2f}")
        logger.info(f"  MAE:  {train_mae:,.2f}")
        logger.info(f"  R²:   {train_r2:.4f}")
        
        # Model info
        logger.info(f"Model trained with {len(model.estimators_)} trees")
        
        logger.info("✓ Model training completed successfully!")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list = None,
    top_n: int = 10
) -> Dict[str, float]:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained RandomForestRegressor
        feature_names: List of feature names (optional)
        top_n: Number of top features to return
        
    Returns:
        Dictionary of feature names and their importance scores
    """
    try:
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance and get top N
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return {}
