"""
Data Ingestion Module - Local Version

This module handles loading the housing dataset from local filesystem.
No Google Cloud dependencies required.
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def data_ingestion(data_path: str) -> pd.DataFrame:
    """
    Load dataset from local file system.
    
    Args:
        data_path: Path to the dataset file (local path or file path)
        
    Returns:
        DataFrame containing the loaded dataset
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        Exception: For other data loading errors
    """
    try:
        logger.info("Starting data ingestion...")
        logger.info(f"Loading dataset from {data_path}...")
        
        # Check if file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        # Load the dataset
        df = pd.read_csv(data_path)
        
        logger.info(f"✓ Dataset loaded successfully!")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Basic data validation
        if df.empty:
            raise ValueError("Loaded dataset is empty!")
        
        logger.info("Data ingestion completed successfully!")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise
