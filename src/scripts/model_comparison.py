# src/scripts/model_comparison.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models(models_dir):
    """
    Load all trained models from a directory
    
    Args:
        models_dir (Path): Directory containing model files
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    
    # Look for model files
    model_files = list(models_dir.glob("strikeout_*_model.pkl"))
    
    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return {}
    
    for model_file in model_files:
        # Extract model type from filename
        model_type = model_file.stem.split('_')[1]
        
        # Skip the 'model' part if it's part of the filename
        if model_type == 'model':
            continue
        
        # Load model
        try:
            with open(model_file, 'rb') as f:
                model_dict = pickle.load(f)
                models[model_type] = model_dict
                logger.info(f"Loaded {model_type} model from {model_file}")
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {e}")
    
    return models

def visualize_feature_importance(models, save_dir, top_n=15):
    """
    Visualize feature importance across different models
    
    Args:
        models (dict): Dictionary of loaded models
        save_dir (Path): Directory to save visualizations
        top_n (int): Number of top features to show
    """
    # Create a combined feature importance DataFrame
    all_importances = []
    
    for model_type, model_dict in models.items():
        importance_df = model_dict['importance'].copy()
        importance_df['model'] = model_type
        all_importances.append(importance_df)
    
    if not all_importances:
        logger.warning("No feature importance data available")
        return
    
    combined_importance = pd.concat(all_importances)
    
    # Get top features across all models
    top_features = combined_importance.groupby('feature')['importance'].mean().nlargest(top_n).index.tolist()
    
    # Filter to top features only
    top_importance = combined_importance[combined_importance['feature'].isin(top_features)]
    
    # Plot feature importance by model
    plt.figure(figsize=(14, 10))
    sns.barplot(x='importance', y='feature', hue='model', data=top_importance)
    plt.title(f'Top {top_n} Feature Importance by Model')
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance_by_model.png")
    plt.close()
    
    # Plot average feature importance
    plt.figure(figsize=(14, 10))
    avg_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
    avg_importance = avg_importance[avg_importance['feature'].isin(top_features)].sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=avg_importance)
    plt.title(f'Average Feature Importance Across All Models')
    plt.tight_layout()
    plt.savefig(save_dir / "average_feature_importance.png")
    plt.close()
    
    logger.info(f"Feature importance visualizations saved to {save_dir}")

def run_comparison(models_dir=None, output_dir=None):
    """
    Run a comprehensive comparison of trained models
    
    Args:
        models_dir (Path): Directory containing model files
        output_dir (Path): Directory to save comparison results
    """
    # Set default directories
    if models_dir is None:
        models_dir = Path("models")
    
    if output_dir is None:
        output_dir = models_dir / "comparison"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load models
    logger.info(f"Loading models from {models_dir}")
    models = load_models(models_dir)
    
    if not models:
        logger.error("No models loaded, cannot proceed with comparison")
        return
    
    # Extract metrics for comparison
    comparison_data = []
    for model_name, model_dict in models.items():
        metrics = model_dict['metrics']
        model_type = model_dict.get('model_type', model_name)
        comparison_data.append({
            'model': model_name,
            'model_type': model_type,
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
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create visualizations
    # 1. Standard metrics bar chart
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
    melted_df = pd.melt(comparison_df, id_vars=['model'], value_vars=metrics_to_plot)
    sns.barplot(x='model', y='value', hue='variable', data=melted_df)
    plt.title('Error Metrics Comparison')
    plt.ylabel('Value (lower is better)')
    plt.tight_layout()
    plt.savefig(output_dir / "error_metrics_comparison.png")
    plt.close()
    
    # 2. Accuracy metrics bar chart
    plt.figure(figsize=(12, 6))
    accuracy_metrics = ['R²', 'Over/Under Accuracy', 'Within 1 Strikeout', 
                      'Within 2 Strikeouts', 'Within 3 Strikeouts']
    melted_df = pd.melt(comparison_df, id_vars=['model'], value_vars=accuracy_metrics)
    sns.barplot(x='model', y='value', hue='variable', data=melted_df)
    plt.title('Accuracy Metrics Comparison')
    plt.ylabel('Value (higher is better)')
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_metrics_comparison.png")
    plt.close()
    
    # 3. Bias comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='Bias', data=comparison_df)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Model Bias Comparison')
    plt.ylabel('Bias (close to 0 is better)')
    plt.tight_layout()
    plt.savefig(output_dir / "bias_comparison.png")
    plt.close()
    
    # 4. Feature importance comparison
    visualize_feature_importance(models, output_dir)
    
    # Save comparison JSON
    with open(output_dir / "model_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    # Determine best model based on different metrics
    best_models = {
        'RMSE': comparison_df.loc[comparison_df['RMSE'].idxmin(), 'model'],
        'MAE': comparison_df.loc[comparison_df['MAE'].idxmin(), 'model'],
        'R²': comparison_df.loc[comparison_df['R²'].idxmax(), 'model'],
        'MAPE': comparison_df.loc[comparison_df['MAPE'].idxmin(), 'model'],
        'Over/Under': comparison_df.loc[comparison_df['Over/Under Accuracy'].idxmax(), 'model'],
        'Within 2 Strikeouts': comparison_df.loc[comparison_df['Within 2 Strikeouts'].idxmax(), 'model']
    }
    
    # Save best models JSON
    with open(output_dir / "best_models.json", 'w') as f:
        json.dump(best_models, f, indent=4)
    
    # Print best models
    logger.info("Best models by metric:")
    for metric, model in best_models.items():
        logger.info(f"  {metric}: {model}")
    
    logger.info(f"Model comparison complete. Results saved to {output_dir}")
    return comparison_df, best_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare trained strikeout prediction models')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_comparison(models_dir, output_dir)