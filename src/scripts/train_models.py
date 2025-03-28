# src/scripts/train_models.py
import logging
import os
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.db import get_pitcher_data
from src.features.selection import select_features_for_strikeout_model
from src.models.train import train_strikeout_model, save_model
from src.visualization.plots import create_visualizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pitcher_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compare_models(models_dict, save_dir=None):
    """
    Compare trained models and visualize results
    
    Args:
        models_dict (dict): Dictionary of trained models
        save_dir (Path, optional): Directory to save comparison results
    """
    if save_dir is None:
        save_dir = Path("models/comparison")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract metrics for comparison
    comparison_data = []
    for model_name, model_dict in models_dict.items():
        metrics = model_dict['metrics']
        comparison_data.append({
            'model': model_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'MAPE': metrics['mape'],
            'Over/Under Accuracy': metrics['over_under_accuracy'],
            'Within 1 Strikeout': metrics['within_1_strikeout'],
            'Within 2 Strikeouts': metrics['within_2_strikeouts'],
            'Within 3 Strikeouts': metrics['within_3_strikeouts'],
            'Bias': metrics['bias']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv(save_dir / "model_comparison.csv", index=False)
    
    # Create visualizations
    
    # 1. Standard metrics bar chart
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
    melted_df = pd.melt(comparison_df, id_vars=['model'], value_vars=metrics_to_plot)
    sns.barplot(x='model', y='value', hue='variable', data=melted_df)
    plt.title('Error Metrics Comparison')
    plt.ylabel('Value (lower is better)')
    plt.tight_layout()
    plt.savefig(save_dir / "error_metrics_comparison.png")
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
    plt.savefig(save_dir / "accuracy_metrics_comparison.png")
    plt.close()
    
    # 3. Bias comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='Bias', data=comparison_df)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Model Bias Comparison')
    plt.ylabel('Bias (close to 0 is better)')
    plt.tight_layout()
    plt.savefig(save_dir / "bias_comparison.png")
    plt.close()
    
    # Save comparison JSON
    with open(save_dir / "model_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    logger.info(f"Model comparison saved to {save_dir}")
    return comparison_df

def train_and_save_models(train_years=(2019, 2021, 2022), test_years=(2023, 2024), 
                          tune_hyperparameters=True):
    """Train and save the strikeout prediction models"""
    # Create output directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Get the data
    logger.info("Loading pitcher data...")
    pitcher_data = get_pitcher_data()
    
    # Select features for strikeout model
    logger.info("Selecting features for strikeout model...")
    so_features = select_features_for_strikeout_model(pitcher_data)
    logger.info(f"Selected features for strikeout model: {so_features}")
    
    # Train models with different algorithms
    models = {}
    model_types = ['rf', 'xgboost', 'lightgbm']
    
    for model_type in model_types:
        logger.info(f"Training {model_type} strikeout prediction model...")
        model_dict = train_strikeout_model(
            pitcher_data,
            so_features,
            train_years=train_years,
            test_years=test_years,
            model_type=model_type,
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Add training and test years to model dictionary for reference
        model_dict['train_years'] = train_years
        model_dict['test_years'] = test_years
        
        # Save model
        logger.info(f"Saving {model_type} model...")
        save_model(model_dict, models_dir / f"strikeout_{model_type}_model.pkl")
        
        # Add to models dictionary
        models[model_type] = model_dict
    
    # Compare models
    logger.info("Comparing models...")
    comparison = compare_models(models, save_dir=models_dir / "comparison")
    
    # Find best model
    best_model = comparison.loc[comparison['RMSE'].idxmin(), 'model']
    logger.info(f"Best model based on RMSE: {best_model}")
    
    # Create a symbolic link or copy best model to a standard name
    best_model_path = models_dir / f"strikeout_{best_model}_model.pkl"
    standard_model_path = models_dir / "strikeout_model.pkl"
    
    # Copy best model to standard name
    import shutil
    shutil.copy2(best_model_path, standard_model_path)
    logger.info(f"Copied best model ({best_model}) to {standard_model_path}")
    
    logger.info("Model training and comparison complete!")
    
    return {
        'models': models,
        'comparison': comparison,
        'best_model': best_model
    }

if __name__ == "__main__":
    train_and_save_models(tune_hyperparameters=True)