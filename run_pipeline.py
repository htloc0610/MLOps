"""
Local Pipeline Runner for House Price Prediction

This script runs the ML pipeline locally using modular components.
All components run sequentially on your local machine.

Before running:
1. Place Housing.csv in the data/ directory
2. Install requirements: pip install -r requirements.txt
3. Run: python run_pipeline.py
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime
import json

# Import local pipeline components
from src.data_ingestion import data_ingestion
from src.preprocessing import preprocessing, get_feature_names
from src.training import training, get_feature_importance
from src.evaluation import evaluation, save_predictions, save_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Local paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Dataset configuration
DATA_FILE = "Housing.csv"
DATA_PATH = DATA_DIR / DATA_FILE

# Hyperparameters
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', '100'))
MAX_DEPTH = int(os.getenv('MAX_DEPTH', '10'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories for output."""
    directories = [DATA_DIR, OUTPUT_DIR, MODELS_DIR, METRICS_DIR, ARTIFACTS_DIR, PLOTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directories created at {OUTPUT_DIR.absolute()}")

# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_local_pipeline():
    """Run the complete ML pipeline locally using modular components."""
    
    logger.info("\n" + "=" * 70)
    logger.info("LOCAL ML PIPELINE - HOUSE PRICE PREDICTION")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info("=" * 70)
    
    try:
        # Setup
        setup_directories()
        
        # ====================================================================
        # STEP 1: DATA INGESTION
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 70)
        
        df = data_ingestion(str(DATA_PATH))
        
        # ====================================================================
        # STEP 2: PREPROCESSING
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: PREPROCESSING")
        logger.info("=" * 70)
        
        # Get feature names before preprocessing
        feature_names = get_feature_names(df)
        
        X_train, X_test, y_train, y_test, scaler = preprocessing(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Save scaler
        scaler_path = ARTIFACTS_DIR / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"✓ Scaler saved to {scaler_path}")
        
        # ====================================================================
        # STEP 3: MODEL TRAINING
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 70)
        
        hyperparameters = {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'random_state': RANDOM_STATE
        }
        
        model = training(X_train, y_train, hyperparameters)
        
        # Save model
        model_path = MODELS_DIR / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved to {model_path}")
        
        # Save feature importance
        feature_importance = get_feature_importance(model, feature_names, top_n=10)
        importance_path = ARTIFACTS_DIR / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=2)
        logger.info(f"✓ Feature importance saved to {importance_path}")
        
        # ====================================================================
        # STEP 4: MODEL EVALUATION
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 70)
        
        metrics = evaluation(
            model,
            X_test,
            y_test,
            output_dir=PLOTS_DIR,
            feature_names=feature_names
        )
        
        # Save metrics
        metrics_path = METRICS_DIR / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_metrics(metrics, metrics_path)
        
        # Save predictions
        predictions_path = ARTIFACTS_DIR / "predictions.csv"
        save_predictions(y_test, model.predict(X_test), predictions_path)
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("✓✓✓ PIPELINE COMPLETED SUCCESSFULLY! ✓✓✓")
        logger.info("=" * 70)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
        logger.info(f"\nGenerated files:")
        logger.info(f"  📦 Model:      {model_path.resolve().relative_to(Path.cwd())}")
        logger.info(f"  📊 Metrics:    {metrics_path.resolve().relative_to(Path.cwd())}")
        logger.info(f"  📈 Plots:      {(PLOTS_DIR / 'evaluation_plots.png').resolve().relative_to(Path.cwd())}")
        logger.info(f"  📋 Predictions: {predictions_path.resolve().relative_to(Path.cwd())}")
        logger.info(f"  🔧 Scaler:     {scaler_path.resolve().relative_to(Path.cwd())}")
        logger.info(f"  ⭐ Feature Importance: {importance_path.resolve().relative_to(Path.cwd())}")
        logger.info("\nKey Metrics:")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE:     {metrics['rmse']:,.2f}")
        logger.info(f"  MAE:      {metrics['mae']:,.2f}")
        logger.info("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("❌❌❌ PIPELINE FAILED ❌❌❌")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70 + "\n")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if dataset exists
    if not DATA_PATH.exists():
        logger.error(f"\n❌ ERROR: Dataset not found at {DATA_PATH.absolute()}")
        logger.error(f"\nPlease ensure {DATA_FILE} is placed in the {DATA_DIR} directory.")
        logger.error(f"Expected path: {DATA_PATH.absolute()}")
        logger.error(f"\nTip: Run 'python download_dataset.py' to get a sample dataset\n")
        sys.exit(1)
    
    # Run pipeline
    success = run_local_pipeline()
    sys.exit(0 if success else 1)
