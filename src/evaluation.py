"""
Evaluation Module - Local Version

This module handles model evaluation and visualization.
No Google Cloud or KFP dependencies required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def evaluation(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path = None,
    feature_names: list = None
) -> Dict[str, float]:
    """
    Evaluate the trained model and generate visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        output_dir: Directory to save plots (optional)
        feature_names: List of feature names for importance plot (optional)
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        Exception: If evaluation fails
    """
    try:
        logger.info("Starting model evaluation...")
        
        # Make predictions
        logger.info(f"Predicting on {X_test.shape[0]} test samples...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        # Log metrics
        logger.info("=" * 70)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 70)
        logger.info(f"  Mean Squared Error (MSE):       {mse:,.2f}")
        logger.info(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
        logger.info(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
        logger.info(f"  R² Score:                       {r2:.4f}")
        logger.info(f"  MAPE:                           {mape:.2f}%")
        logger.info("=" * 70)
        
        # Generate visualizations if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating visualizations in {output_dir}...")
            
            # Create evaluation plots
            create_evaluation_plots(
                y_test, y_pred, model, 
                feature_names, output_dir
            )
        
        logger.info("✓ Model evaluation completed successfully!")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def create_evaluation_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model: RandomForestRegressor,
    feature_names: list = None,
    output_dir: Path = None
):
    """
    Create comprehensive evaluation plots.
    
    Args:
        y_test: Actual test values
        y_pred: Predicted values
        model: Trained model
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Actual vs Predicted scatter plot
        ax1 = plt.subplot(2, 2, 1)
        ax1.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax1.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residual plot
        ax2 = plt.subplot(2, 2, 2)
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance (if available)
        ax3 = plt.subplot(2, 2, 3)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]
            
            # Get top 10 features
            indices = np.argsort(importances)[-10:]
            
            ax3.barh(range(len(indices)), importances[indices], align='center')
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([feature_names[i] for i in indices])
            ax3.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax3.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
        else:
            ax3.text(0.5, 0.5, 'Feature importance not available', 
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. Residuals distribution
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax4.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        ax4.set_xlabel('Residuals', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            plot_path = output_dir / 'evaluation_plots.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Plots saved to {plot_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        plt.close()


def save_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path
):
    """
    Save predictions to CSV file.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        output_path: Path to save CSV file
    """
    try:
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred,
            'absolute_error': np.abs(y_test - y_pred),
            'percentage_error': np.abs((y_test - y_pred) / y_test) * 100
        })
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"✓ Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")


def save_metrics(
    metrics: Dict[str, float],
    output_path: Path
):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Metrics saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
