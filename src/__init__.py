"""
MLOps Pipeline Components - Local Version

This package contains all components for the local ML pipeline:
- data_ingestion: Load data from local filesystem
- preprocessing: Clean, transform, and split data
- training: Train Random Forest model
- evaluation: Evaluate model and generate visualizations
"""

from .data_ingestion import data_ingestion
from .preprocessing import preprocessing, get_feature_names
from .training import training, get_feature_importance
from .evaluation import (
    evaluation,
    create_evaluation_plots,
    save_predictions,
    save_metrics
)

__all__ = [
    'data_ingestion',
    'preprocessing',
    'get_feature_names',
    'training',
    'get_feature_importance',
    'evaluation',
    'create_evaluation_plots',
    'save_predictions',
    'save_metrics',
]

__version__ = '1.0.0'
__author__ = 'MLOps Team'
