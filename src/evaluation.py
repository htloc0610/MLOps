from kfp.v2.dsl import (
    Dataset,
    Input,
    Model,
    Output,
    Metrics,
    HTML,
    component,
)

# Docker image from Artifact Registry
BASE_IMAGE = "us-central1-docker.pkg.dev/project-1cd612d2-3ea2-4818-a72/mlops-repo/training-image:latest"

@component(
    base_image=BASE_IMAGE,
    output_component_file="evaluation.yaml"
)
def evaluation(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics],
    html: Output[HTML]
):
    """
    Evaluates the model's performance and generates visualizations.
    
    Args:
        model: Input trained model
        preprocessed_dataset: Input preprocessed dataset
        metrics: Output artifact for evaluation metrics
        html: Output artifact for visualization HTML
    """
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    import logging
    import base64
    from io import BytesIO
    
    try:
        logging.info("Starting model evaluation...")
        
        # Load the model and dataset
        rf_model = joblib.load(model.path)
        df = pd.read_csv(preprocessed_dataset.path)
        
        logging.info(f"Loaded model and dataset with shape: {df.shape}")
        
        # Split features and target (same as training)
        if 'price' in df.columns:
            target_col = 'price'
        elif 'Price' in df.columns:
            target_col = 'Price'
        else:
            target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Use the same split as training for consistency
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mse ** 0.5
        
        # Log metrics
        metrics.log_metric("test_mse", mse)
        metrics.log_metric("test_r2", r2)
        metrics.log_metric("test_mae", mae)
        metrics.log_metric("test_rmse", rmse)
        
        logging.info(f"Evaluation Metrics - MSE: {mse:.2f}, R2: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Price')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 10 Feature Importances')
        axes[1, 0].invert_yaxis()
        
        # 4. Residuals distribution
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4285f4;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .metric-label {{
                    font-size: 14px;
                    opacity: 0.9;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 28px;
                    font-weight: bold;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .timestamp {{
                    color: #888;
                    font-size: 12px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏠 House Price Prediction - Model Evaluation Report</h1>
                
                <h2>Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{r2:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Mean Squared Error</div>
                        <div class="metric-value">{mse:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Root Mean Squared Error</div>
                        <div class="metric-value">{rmse:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Mean Absolute Error</div>
                        <div class="metric-value">{mae:.2f}</div>
                    </div>
                </div>
                
                <h2>Visualizations</h2>
                <img src="data:image/png;base64,{image_base64}" alt="Model Evaluation Plots">
                
                <div class="timestamp">
                    Report generated by Vertex AI Pipeline
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(html.path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Evaluation report saved to: {html.path}")
        logging.info("Model evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise
