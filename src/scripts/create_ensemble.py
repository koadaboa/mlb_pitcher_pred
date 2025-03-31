# src/scripts/create_ensemble.py
import pickle
import pandas as pd
import numpy as np
import os
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from src.data.db import get_pitcher_data
from src.models.train import calculate_betting_metrics
from src.data.utils import setup_logger
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

def load_optimized_models(model_paths):
    """
    Load optimized models from pickle files
    
    Args:
        model_paths (list): List of paths to model pickle files
        
    Returns:
        list: List of loaded model dictionaries
    """
    models = []
    
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                model_dict = pickle.load(f)
                # Add source path for reference
                model_dict['source_path'] = str(path)
                models.append(model_dict)
                logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    
    return models

def get_ensemble_weights(models, X_val, y_val, method='inverse_error'):
    """
    Calculate ensemble weights based on validation performance
    
    Args:
        models (list): List of model dictionaries
        X_val (DataFrame): Validation features
        y_val (Series): Validation target
        method (str): Weighting method ('equal', 'inverse_error', 'betting_accuracy')
        
    Returns:
        dict: Dictionary of model weights
    """
    weights = {}
    
    if method == 'equal':
        # Equal weights for all models
        weight = 1.0 / len(models)
        weights = {model['model_type']: weight for model in models}
    
    elif method == 'inverse_error':
        # Weights inversely proportional to RMSE
        errors = {}
        for model_dict in models:
            model_type = model_dict['model_type']
            model = model_dict['model']
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            errors[model_type] = rmse
        
        # Inverse of errors
        inverse_errors = {k: 1.0/v for k, v in errors.items()}
        
        # Normalize to sum to 1
        total = sum(inverse_errors.values())
        weights = {k: v/total for k, v in inverse_errors.items()}
    
    elif method == 'betting_accuracy':
        # Weights proportional to 'within_2_strikeouts' and 'over_under_accuracy'
        accuracies = {}
        for model_dict in models:
            model_type = model_dict['model_type']
            model = model_dict['model']
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate betting metrics
            betting_metrics = calculate_betting_metrics(y_val, y_pred)
            
            # Combined score (within 2 strikeouts + over/under accuracy)
            combined_score = (betting_metrics['within_2_strikeouts'] + 
                             betting_metrics['over_under_accuracy']) / 2
            
            accuracies[model_type] = combined_score
        
        # Normalize to sum to 1
        total = sum(accuracies.values())
        weights = {k: v/total for k, v in accuracies.items()}
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights

def simple_weighted_ensemble(models, weights, X):
    """
    Make predictions using a simple weighted average ensemble
    
    Args:
        models (list): List of model dictionaries
        weights (dict): Dictionary of model weights
        X (DataFrame): Features
        
    Returns:
        array: Weighted ensemble predictions
    """
    predictions = np.zeros(len(X))
    
    for model_dict in models:
        model_type = model_dict['model_type']
        model = model_dict['model']
        weight = weights.get(model_type, 0)
        
        if weight > 0:
            # Make predictions and add weighted contribution
            y_pred = model.predict(X)
            predictions += weight * y_pred
    
    return predictions

class StackingEnsemble:
    """
    Stacking ensemble model that uses a meta-model to combine base model predictions
    """
    
    def __init__(self, base_models, meta_model=None):
        """
        Initialize stacking ensemble
        
        Args:
            base_models (list): List of model dictionaries
            meta_model: Meta-model (if None, Ridge regression will be used)
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.base_model_types = [model['model_type'] for model in base_models]
    
    def fit(self, X, y):
        """
        Fit the stacking ensemble
        
        Args:
            X (DataFrame): Training features
            y (Series): Training target
        """
        # Generate meta-features using cross-validation predictions
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        # Use a simple holdout approach for simplicity
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=StrikeoutModelConfig.RANDOM_STATE)
        
        for i, model_dict in enumerate(self.base_models):
            model = model_dict['model']
            
            # Train on X_train and predict on X_val for meta-features
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            
            # Store predictions as meta-features
            meta_features[:, i] = model.predict(X)
        
        # Train meta-model on meta-features
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the stacking ensemble
        
        Args:
            X (DataFrame): Features
            
        Returns:
            array: Ensemble predictions
        """
        # Generate meta-features for prediction
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, model_dict in enumerate(self.base_models):
            model = model_dict['model']
            meta_features[:, i] = model.predict(X)
        
        # Make predictions with meta-model
        return self.meta_model.predict(meta_features)

def create_ensemble(models, ensemble_type='weighted', weighting_method='betting_accuracy'):
    """
    Create an ensemble model from optimized base models
    
    Args:
        models (list): List of model dictionaries
        ensemble_type (str): Type of ensemble ('weighted' or 'stacking')
        weighting_method (str): Method for calculating weights (for weighted ensemble)
        
    Returns:
        dict: Ensemble model dictionary
    """
    if ensemble_type == 'weighted':
        # Explicitly calculate weights rather than leaving them as None
        X_sample = np.random.rand(10, len(models[0]['features']))
        dummy_target = np.random.rand(10)
        
        # Calculate weights using the specified method
        weights = get_ensemble_weights(models, X_sample, dummy_target, weighting_method)
        
        # Create a function-based ensemble with pre-calculated weights
        ensemble_dict = {
            'model_type': 'weighted_ensemble',
            'base_models': models,
            'weighting_method': weighting_method,
            'weights': weights
        }
        
        logger.info(f"Created weighted ensemble with weights: {weights}")
        return ensemble_dict
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

def evaluate_models(models, data, test_years, ensemble_types=['weighted', 'stacking']):
    """
    Evaluate individual models and ensembles on test data
    
    Args:
        models (list): List of model dictionaries
        data (DataFrame): Full dataset
        test_years (list): Years to use for testing
        ensemble_types (list): Types of ensembles to evaluate
        
    Returns:
        tuple: (Results DataFrame, best ensemble)
    """
    # Filter to test years
    test_df = data[data['season'].isin(test_years)]
    logger.info(f"Evaluating models on {len(test_df)} rows from years {test_years}")
    
    if test_df.empty:
        logger.error(f"No test data available for years {test_years}")
        return None, None
    
    # Split test data into validation and test sets
    # Validation set is used for ensemble weights, test set for final evaluation
    X_all = test_df[models[0]['features']].copy()
    y_all = test_df['strikeouts'].copy()
    
    X_val, X_test, y_val, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=StrikeoutModelConfig.RANDOM_STATE)
    
    logger.info(f"Split test data into {len(X_val)} validation and {len(X_test)} test rows")
    
    # Evaluate individual models
    results = []
    
    for model_dict in models:
        model_type = model_dict['model_type']
        model = model_dict['model']
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate betting metrics
        betting_metrics = calculate_betting_metrics(y_test, y_pred)
        
        # Add to results
        results.append({
            'model': model_type,
            'type': 'base',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': betting_metrics['mape'],
            'Over/Under Accuracy': betting_metrics['over_under_accuracy'],
            'Within 1 Strikeout': betting_metrics['within_1_strikeout'],
            'Within 2 Strikeouts': betting_metrics['within_2_strikeouts'],
            'Within 3 Strikeouts': betting_metrics['within_3_strikeouts'],
            'Bias': betting_metrics['bias'],
            'model_dict': model_dict
        })
        
        logger.info(f"{model_type}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, "
                   f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%, "
                   f"Over/Under: {betting_metrics['over_under_accuracy']:.2f}%")
    
    # Create ensembles
    ensembles = []
    
    # Weighted ensembles with different weighting methods
    weighting_methods = ['equal', 'inverse_error', 'betting_accuracy']
    for method in weighting_methods:
        # Calculate weights using validation set
        weights = get_ensemble_weights(models, X_val, y_val, method=method)
        
        # Create ensemble
        weighted_ensemble = {
            'model_type': f'weighted_ensemble_{method}',
            'base_models': models,
            'weighting_method': method,
            'weights': weights
        }
        
        ensembles.append(weighted_ensemble)
        
        logger.info(f"Created weighted ensemble with {method} weighting: {weights}")
    
    # Stacking ensemble
    try:
        # Create and fit stacking ensemble on validation data
        stacking = StackingEnsemble(models)
        stacking.fit(X_val, y_val)
        
        stacking_ensemble = {
            'model_type': 'stacking_ensemble',
            'model': stacking,
            'base_models': models
        }
        
        ensembles.append(stacking_ensemble)
        logger.info("Created stacking ensemble")
    except Exception as e:
        logger.error(f"Error creating stacking ensemble: {e}")
    
    # Evaluate ensembles
    for ensemble in ensembles:
        ensemble_type = ensemble['model_type']
        
        # Make predictions
        if 'weighted_ensemble' in ensemble_type:
            y_pred = simple_weighted_ensemble(
                models=ensemble['base_models'],
                weights=ensemble['weights'],
                X=X_test
            )
        else:  # stacking
            y_pred = ensemble['model'].predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate betting metrics
        betting_metrics = calculate_betting_metrics(y_test, y_pred)
        
        # Add to results
        results.append({
            'model': ensemble_type,
            'type': 'ensemble',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': betting_metrics['mape'],
            'Over/Under Accuracy': betting_metrics['over_under_accuracy'],
            'Within 1 Strikeout': betting_metrics['within_1_strikeout'],
            'Within 2 Strikeouts': betting_metrics['within_2_strikeouts'],
            'Within 3 Strikeouts': betting_metrics['within_3_strikeouts'],
            'Bias': betting_metrics['bias'],
            'model_dict': ensemble
        })
        
        logger.info(f"{ensemble_type}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, "
                   f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%, "
                   f"Over/Under: {betting_metrics['over_under_accuracy']:.2f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model_dict'} 
                             for r in results])
    
    # Find the best ensemble based on RMSE
    ensemble_results = [r for r in results if r['type'] == 'ensemble']
    if ensemble_results:
        best_ensemble = min(ensemble_results, key=lambda x: x['RMSE'])
        logger.info(f"Best ensemble: {best_ensemble['model']} with RMSE={best_ensemble['RMSE']:.4f}")
        best_ensemble_dict = best_ensemble['model_dict']
    else:
        best_ensemble_dict = None
    
    return results_df, best_ensemble_dict

def save_ensemble(ensemble, output_path):
    """
    Save ensemble model to pickle file
    
    Args:
        ensemble (dict): Ensemble model dictionary
        output_path (Path): Path to save ensemble
    """
    # Create output directory
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    logger.info(f"Ensemble saved to {output_path}")

def visualize_results(results_df, output_dir):
    """
    Create visualizations for model comparison
    
    Args:
        results_df (DataFrame): Results DataFrame
        output_dir (Path): Directory to save visualizations
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Add model type column for nicer labels
    results_df['model_type'] = results_df['model'].apply(lambda x: 
        'Base: ' + x if x in ['rf', 'xgboost', 'lightgbm'] else 
        ('Ensemble: ' + x.replace('weighted_ensemble_', '').replace('_', ' ') 
         if 'weighted_ensemble' in x else 
         'Ensemble: stacking'))
    
    # 1. Error metrics comparison
    plt.figure(figsize=(14, 7))
    metrics = ['RMSE', 'MAE', 'MAPE']
    error_df = pd.melt(results_df, id_vars=['model_type'], value_vars=metrics, 
                      var_name='metric', value_name='value')
    
    g = sns.catplot(x='model_type', y='value', hue='metric', data=error_df, kind='bar',
                  height=6, aspect=2, legend=False)
    plt.legend(title='Metric', loc='best')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value (lower is better)')
    plt.title('Error Metrics Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / "error_metrics_comparison.png")
    plt.close()
    
    # 2. Accuracy metrics comparison
    plt.figure(figsize=(14, 7))
    accuracy_metrics = ['R²', 'Over/Under Accuracy', 'Within 1 Strikeout', 
                      'Within 2 Strikeouts', 'Within 3 Strikeouts']
    acc_df = pd.melt(results_df, id_vars=['model_type'], value_vars=accuracy_metrics,
                    var_name='metric', value_name='value')
    
    g = sns.catplot(x='model_type', y='value', hue='metric', data=acc_df, kind='bar',
                  height=6, aspect=2, legend=False)
    plt.legend(title='Metric', loc='best')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value (higher is better)')
    plt.title('Accuracy Metrics Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_metrics_comparison.png")
    plt.close()
    
    # 3. Bias comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_type', y='Bias', data=results_df)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Bias (close to 0 is better)')
    plt.title('Model Bias Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / "bias_comparison.png")
    plt.close()
    
    # 4. Overall ranking
    # Normalize metrics to 0-1 scale where 1 is best
    norm_df = results_df.copy()
    
    # Error metrics (lower is better)
    for metric in ['RMSE', 'MAE', 'MAPE', 'Bias']:
        if metric == 'Bias':
            # For bias, 0 is best
            norm_df[f'norm_{metric}'] = 1 - np.abs(norm_df[metric]) / np.abs(norm_df[metric]).max()
        else:
            # For other error metrics, lower is better
            min_val = norm_df[metric].min()
            max_val = norm_df[metric].max()
            if max_val > min_val:
                norm_df[f'norm_{metric}'] = 1 - (norm_df[metric] - min_val) / (max_val - min_val)
            else:
                norm_df[f'norm_{metric}'] = 1
    
    # Accuracy metrics (higher is better)
    for metric in ['R²', 'Over/Under Accuracy', 'Within 1 Strikeout', 
                 'Within 2 Strikeouts', 'Within 3 Strikeouts']:
        min_val = norm_df[metric].min()
        max_val = norm_df[metric].max()
        if max_val > min_val:
            norm_df[f'norm_{metric}'] = (norm_df[metric] - min_val) / (max_val - min_val)
        else:
            norm_df[f'norm_{metric}'] = 1
    
    # Calculate overall score as average of normalized metrics
    norm_cols = [col for col in norm_df.columns if col.startswith('norm_')]
    norm_df['overall_score'] = norm_df[norm_cols].mean(axis=1)
    
    # Sort by overall score
    norm_df = norm_df.sort_values('overall_score', ascending=False)
    
    # Plot overall score
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_type', y='overall_score', data=norm_df)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Overall Score (higher is better)')
    plt.title('Model Overall Performance Score')
    plt.tight_layout()
    plt.savefig(output_dir / "overall_score.png")
    plt.close()
    
    # 5. Save rankings to CSV
    rankings = norm_df[['model', 'model_type', 'type', 'RMSE', 'MAE', 'R²', 
                      'Within 2 Strikeouts', 'Over/Under Accuracy', 'Bias', 'overall_score']]
    rankings.to_csv(output_dir / "model_rankings.csv", index=False)
    
    # 6. Create a summary JSON file
    summary = {
        'best_model': norm_df.iloc[0]['model'],
        'best_model_type': norm_df.iloc[0]['model_type'],
        'best_model_score': float(norm_df.iloc[0]['overall_score']),
        'best_base_model': norm_df[norm_df['type'] == 'base'].iloc[0]['model'],
        'best_base_model_score': float(norm_df[norm_df['type'] == 'base'].iloc[0]['overall_score']),
        'metrics': {
            metric: float(norm_df.iloc[0][metric]) for metric in 
            ['RMSE', 'MAE', 'R²', 'Within 2 Strikeouts', 'Over/Under Accuracy', 'Bias']
        }
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create and evaluate ensemble models')
    parser.add_argument('--model-paths', type=str, nargs='+', required=True,
                        help='Paths to optimized model pickle files')
    parser.add_argument('--test-years', type=int, nargs='+', default=[2023, 2024],
                        help='Years to use for testing')
    parser.add_argument('--output-dir', type=str, default='models/ensemble',
                        help='Directory to save ensemble models and results')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    model_paths = [Path(p) for p in args.model_paths]
    output_dir = Path(args.output_dir)
    
    # Load optimized models
    models = load_optimized_models(model_paths)
    
    if not models:
        logger.error("No models loaded, exiting")
        return
    
    # Get data
    logger.info("Loading data...")
    pitcher_data = get_pitcher_data()
    
    # Evaluate models and create ensembles
    results_df, best_ensemble = evaluate_models(
        models=models,
        data=pitcher_data,
        test_years=args.test_years
    )
    
    if results_df is not None:
        # Create visualizations
        visualize_results(results_df, output_dir)
        
        # Save best ensemble
        if best_ensemble is not None:
            save_ensemble(best_ensemble, output_dir / "best_ensemble_model.pkl")
            
            # Save a copy of the best ensemble with standardized name
            save_ensemble(best_ensemble, Path("models") / "strikeout_ensemble_model.pkl")
    
    logger.info("Ensemble evaluation complete!")

if __name__ == "__main__":
    main()