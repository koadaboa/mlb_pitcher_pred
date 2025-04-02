# src/scripts/model_comparison.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from src.data.db import get_pitcher_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.train import calculate_betting_metrics
from src.data.utils import setup_logger
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

def load_models(models_dir):
    """
    Load trained model from a directory
    
    Args:
        models_dir (Path): Directory containing model files
        
    Returns:
        dict: Dictionary with loaded model
    """
    models = {}
    
    # Look for model files
    model_files = []
    for pattern in ["strikeout_model.pkl", "optimized_lightgbm_model.pkl"]:
        model_files.extend(list(models_dir.glob(pattern)))
    
    if not model_files:
        logger.error(f"No LightGBM model files found in {models_dir}")
        return {}
    
    # Load the first model found
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                model_dict = pickle.load(f)
                models['lightgbm'] = model_dict
                logger.info(f"Loaded LightGBM model from {model_file}")
                # Only need one model
                break
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {e}")
    
    return models

def visualize_feature_importance(model_dict, save_dir, top_n=15):
    """
    Visualize feature importance from the LightGBM model
    
    Args:
        model_dict (dict): Dictionary with model data
        save_dir (Path): Directory to save visualizations
        top_n (int): Number of top features to show
    """
    if 'importance' not in model_dict:
        logger.warning("No feature importance data available")
        return
    
    # Get feature importance
    importance_df = model_dict['importance'].copy()
    
    # Keep only top N features
    top_importance = importance_df.head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=(14, 10))
    sns.barplot(x='importance', y='feature', data=top_importance)
    plt.title(f'Top {top_n} Feature Importance for LightGBM')
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance.png")
    plt.close()
    
    logger.info(f"Feature importance visualization saved to {save_dir}")

def run_comparison(models_dir=None, output_dir=None):
    """
    Evaluate the trained LightGBM model
    
    Args:
        models_dir (Path): Directory containing model files
        output_dir (Path): Directory to save comparison results
    """
    # Set default directories
    if models_dir is None:
        models_dir = Path("models")
    
    if output_dir is None:
        output_dir = models_dir / "evaluation"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    logger.info(f"Loading model from {models_dir}")
    models = load_models(models_dir)
    
    if not models:
        logger.error("No model loaded, cannot proceed with evaluation")
        return None, None
    
    # Get data for evaluation
    logger.info("Loading data for evaluation...")
    pitcher_data = get_pitcher_data()
    
    # Define test years
    test_years = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    test_df = pitcher_data[pitcher_data['season'].isin(test_years)]
    
    if test_df.empty:
        logger.error(f"No test data available for years {test_years}")
        return None, None
    
    # Extract metrics for evaluation
    comparison_data = []
    
    # Evaluate the LightGBM model
    model_dict = models['lightgbm']
    
    # Check if the model is already evaluated (has metrics)
    if 'metrics' in model_dict:
        metrics = model_dict['metrics']
        
        # Add to comparison data
        comparison_data.append({
            'model': 'lightgbm',
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'MAPE': metrics.get('mape', np.nan),
            'Over/Under Accuracy': metrics.get('over_under_accuracy', np.nan),
            'Within 1 Strikeout': metrics.get('within_1_strikeout', np.nan),
            'Within 2 Strikeouts': metrics.get('within_2_strikeouts', np.nan),
            'Within 3 Strikeouts': metrics.get('within_3_strikeouts', np.nan),
            'Bias': metrics.get('bias', np.nan),
            'Max Error': metrics.get('max_error', np.nan)
        })
    else:
        # Model doesn't have metrics yet, evaluate it
        logger.info("Evaluating model...")
        
        # Extract features and model
        features = model_dict['features']
        
        # Prepare test data
        X_test = test_df[features].copy()
        y_test = test_df['strikeouts'].copy()
        
        # Make predictions
        model = model_dict['model']
        y_pred = model.predict(X_test)
        
        # Calculate standard metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate betting metrics
        betting_metrics = calculate_betting_metrics(y_test, y_pred)
        
        # Store metrics in the model dictionary for future use
        model_dict['metrics'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            **betting_metrics
        }
        
        # Add to comparison data
        comparison_data.append({
            'model': 'lightgbm',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': betting_metrics['mape'],
            'Over/Under Accuracy': betting_metrics['over_under_accuracy'],
            'Within 1 Strikeout': betting_metrics['within_1_strikeout'],
            'Within 2 Strikeouts': betting_metrics['within_2_strikeouts'],
            'Within 3 Strikeouts': betting_metrics['within_3_strikeouts'],
            'Bias': betting_metrics['bias'],
            'Max Error': betting_metrics.get('max_error', np.nan)
        })
        
        logger.info(f"LightGBM: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, "
                   f"Within 1 K: {betting_metrics['within_1_strikeout']:.2f}%, "
                   f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%")
    
    # Create results DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add a simple print of the key metrics
    print("\n===== MODEL PERFORMANCE SUMMARY =====")
    print(f"{'Model':<15} {'RMSE':>8} {'MAE':>8} {'Within 1K':>10}")
    print("-" * 45)
    for _, row in comparison_df.iterrows():
        model_name = row['model']
        rmse = row['RMSE']
        mae = row['MAE']
        within_1k = row['Within 1 Strikeout']
        print(f"{model_name:<15} {rmse:>8.3f} {mae:>8.3f} {within_1k:>10.2f}%")
    
    # Save to CSV
    comparison_df.to_csv(output_dir / "model_evaluation.csv", index=False)
    
    # Create visualizations
    # Feature importance visualization
    visualize_feature_importance(models['lightgbm'], output_dir)
    
    # Create predictions vs actual plot
    plt.figure(figsize=(10, 8))
    
    # Get predicted and actual values
    features = models['lightgbm']['features']
    model = models['lightgbm']['model']
    X_test = test_df[features].copy()
    y_test = test_df['strikeouts'].copy()
    y_pred = model.predict(X_test)
    
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Strikeouts')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'strikeout_predictions.png')
    plt.close()
    
    # Save evaluation results as JSON
    with open(output_dir / "model_evaluation.json", 'w') as f:
        json.dump(comparison_data[0], f, indent=4)
    
    logger.info(f"Model evaluation complete. Results saved to {output_dir}")
    return comparison_df, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained LightGBM model')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_comparison(models_dir, output_dir)