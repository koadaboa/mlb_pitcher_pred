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
from src.scripts.create_ensemble import get_ensemble_weights
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

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
        try:
            if 'importance' in model_dict:
                importance_df = model_dict['importance'].copy()
                importance_df['model'] = model_type
                all_importances.append(importance_df)
            elif 'model' in model_dict and hasattr(model_dict['model'], 'feature_importances_'):
                # Extract feature importance directly from model
                features = model_dict.get('features', [])
                if features:
                    importance = pd.DataFrame({
                        'feature': features,
                        'importance': model_dict['model'].feature_importances_
                    })
                    importance['model'] = model_type
                    all_importances.append(importance)
        except Exception as e:
            logger.warning(f"Could not extract feature importance for {model_type}: {e}")
            continue
    
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
    
    # Get data for evaluation
    logger.info("Loading data for evaluation...")
    pitcher_data = get_pitcher_data()
    
    # Define test years
    test_years = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    test_df = pitcher_data[pitcher_data['season'].isin(test_years)]
    
    if test_df.empty:
        logger.error(f"No test data available for years {test_years}")
        return
    
    # Extract metrics for comparison
    comparison_data = []
    
    for model_name, model_dict in models.items():
        # Check if the model is already evaluated (has metrics)
        if 'metrics' in model_dict:
            metrics = model_dict['metrics']
            model_type = model_dict.get('model_type', model_name)
            
            # Add to comparison data
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
        else:
            # Model doesn't have metrics yet, evaluate it
            logger.info(f"Evaluating model: {model_name}")
            
            # Extract features and models
            if 'features' in model_dict:
                features = model_dict['features']
            elif 'base_models' in model_dict and len(model_dict['base_models']) > 0:
                # For ensemble models, use features from first base model
                features = model_dict['base_models'][0]['features']
            else:
                logger.error(f"Cannot determine features for model {model_name}")
                continue
            
            # Prepare test data
            X_test = test_df[features].copy()
            y_test = test_df['strikeouts'].copy()
            
            if 'base_models' in model_dict and 'weights' in model_dict:
                # For weighted ensemble
                weights = model_dict['weights']
                base_models = model_dict['base_models']
                predictions = np.zeros(len(X_test))
                
                # Debug logging
                logger.info(f"Ensemble evaluation - Base models: {len(base_models)}")
                
                # Handle None weights
                if weights is None:
                    logger.warning(f"Weights are None for model {model_name} - using equal weights")
                    # Create equal weights for all base models
                    weights = {model['model_type']: 1.0/len(base_models) for model in base_models}
                    # Update the model dict with these weights
                    model_dict['weights'] = weights
                
                # Debug logging
                logger.info(f"Using weights: {weights}")
                
                weight_sum = 0.0
                prediction_count = 0
                
                # Make predictions with base models
                for base_model in base_models:
                    try:
                        base_type = base_model.get('model_type', '')
                        logger.info(f"Processing base model: {base_type}")
                        
                        if not base_type:
                            logger.warning("Base model type is empty - skipping")
                            continue
                            
                        weight = weights.get(base_type, 1.0 / len(base_models))
                        weight_sum += weight
                        
                        if 'model' not in base_model:
                            logger.warning(f"No model found in base model {base_type}")
                            continue
                            
                        model = base_model['model']
                        pred = model.predict(X_test)
                        predictions += weight * pred
                        prediction_count += 1
                        
                        logger.info(f"Added prediction from {base_type} with weight {weight:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error making prediction with base model {base_type}: {e}")
                        continue
                
                # Normalize weights if they don't sum to 1 and we have predictions
                if prediction_count > 0 and abs(weight_sum - 1.0) > 0.01:
                    logger.warning(f"Weights sum to {weight_sum:.4f}, normalizing predictions")
                    predictions = predictions / weight_sum
                
                y_pred = predictions
                
            elif 'model' in model_dict:
                # Standard model with 'model' key
                model = model_dict['model']
                y_pred = model.predict(X_test)
            elif 'base_models' in model_dict and hasattr(model_dict, 'predict'):
                # For other ensemble types with predict method
                y_pred = model_dict.predict(X_test)
            else:
                logger.error(f"Cannot make predictions with model {model_name}")
                continue
            
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
                'model': model_name,
                'model_type': model_dict.get('model_type', model_name),
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
            
            logger.info(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, "
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