#!/usr/bin/env python
# src/scripts/model_pipeline.py
import argparse
from pathlib import Path
import sys
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime

from src.data.utils import setup_logger, ensure_dir
from src.data.db import get_pitcher_data
from src.features.selection import select_features_for_strikeout_model
from src.models.train import train_strikeout_model, save_model, calculate_betting_metrics
from src.scripts.optimize_models import optimize_model
from src.scripts.create_ensemble import load_optimized_models, create_ensemble
from src.scripts.create_ensemble import get_ensemble_weights, simple_weighted_ensemble
from src.scripts.model_comparison import run_comparison, visualize_feature_importance
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

def run_pipeline(command, **kwargs):
    """
    Run the model pipeline with the specified command
    
    Args:
        command (str): Command to execute (train, optimize, ensemble, evaluate, all)
        **kwargs: Additional command-specific arguments
        
    Returns:
        dict: Results of command execution
    """
    # Create necessary directories
    ensure_dir("models")
    
    # Execute the appropriate command
    if command == 'train':
        return _train_models(**kwargs)
    elif command == 'optimize':
        return _optimize_models(**kwargs)
    elif command == 'ensemble':
        return _create_ensemble(**kwargs)
    elif command == 'evaluate':
        return _evaluate_models(**kwargs)
    elif command == 'all':
        # Train all models first
        train_kwargs = {k: v for k, v in kwargs.items() if k in 
                      ['model_types', 'train_years', 'test_years', 'tune_hyperparameters']}
        models_result = _train_models(**train_kwargs)
        
        # Then optimize them
        optimize_kwargs = {k: v for k, v in kwargs.items() if k in 
                         ['model_type', 'train_years', 'n_trials', 'metric']}
        optimize_result = _optimize_models(**optimize_kwargs)
        
        # Create ensemble
        ensemble_kwargs = {k: v for k, v in kwargs.items() if k in 
                         ['model_paths', 'ensemble_type', 'weighting_method']}
        ensemble_result = _create_ensemble(**ensemble_kwargs)
        
        # Finally evaluate everything
        evaluate_kwargs = {k: v for k, v in kwargs.items() if k in 
                         ['models_dir', 'output_dir']}
        eval_result = _evaluate_models(**evaluate_kwargs)
        
        return {
            'training': models_result,
            'optimization': optimize_result,
            'ensemble': ensemble_result,
            'evaluation': eval_result
        }
    else:
        logger.error(f"Unknown command: {command}")
        return None

def _train_models(model_types=None, train_years=None, test_years=None, tune_hyperparameters=True):
    """
    Train multiple model types
    
    Args:
        model_types (list): Model types to train
        train_years (tuple): Years to use for training
        test_years (tuple): Years to use for testing
        tune_hyperparameters (bool): Whether to tune hyperparameters
        
    Returns:
        dict: Dictionary of trained models
    """
    # Set defaults
    if model_types is None or len(model_types) == 0:
        model_types = ['rf', 'xgboost', 'lightgbm']
        
    if train_years is None:
        train_years = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
        
    if test_years is None:
        test_years = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    
    # Create output directory
    models_dir = Path("models")
    ensure_dir(models_dir)
    
    # Get the data
    logger.info("Loading pitcher data...")
    pitcher_data = get_pitcher_data()
    
    # Select features for strikeout model
    logger.info("Selecting features for strikeout model...")
    so_features = select_features_for_strikeout_model(pitcher_data)
    logger.info(f"Selected {len(so_features)} features for strikeout model")
    
    # Train models with different algorithms
    models = {}
    
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
        
        if model_dict is not None:
            # Save model
            model_path = models_dir / f"strikeout_{model_type}_model.pkl"
            save_model(model_dict, model_path)
            
            # Add to models dictionary
            models[model_type] = model_dict
            logger.info(f"Successfully trained and saved {model_type} model")
        else:
            logger.error(f"Failed to train {model_type} model")
    
    if models:
        # Compare models and find the best one
        logger.info("Comparing models...")
        comparison, best_models = run_comparison(models_dir)
        
        # Create a symbolic link or copy best model to a standard name
        if 'Within 1 Strikeout' in best_models:
            best_model = best_models['Within 1 Strikeout']
            best_model_path = models_dir / f"strikeout_{best_model}_model.pkl"
            standard_model_path = models_dir / "strikeout_model.pkl"
            
            # Copy best model to standard name
            import shutil
            shutil.copy2(best_model_path, standard_model_path)
            logger.info(f"Copied best model for 'Within 1 Strikeout' metric ({best_model}) to {standard_model_path}")
    
    logger.info("Model training complete")
    return models

def _optimize_models(model_type='xgboost', train_years=None, n_trials=50, metric='within_1_strikeout'):
    """
    Optimize models using Bayesian optimization
    
    Args:
        model_type (str): Model type to optimize
        train_years (tuple): Years to use for training
        n_trials (int): Number of optimization trials
        metric (str): Metric to optimize
        
    Returns:
        dict: Optimization results
    """
    # Set defaults
    if train_years is None:
        train_years = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    
    logger.info(f"Starting optimization for {model_type} model with {n_trials} trials...")
    logger.info(f"Optimizing for metric: {metric}")
    
    # Define model types to optimize
    if model_type == 'all':
        models_to_optimize = ['rf', 'xgboost', 'lightgbm']
    else:
        models_to_optimize = [model_type]
    
    results = {}
    
    for model_type in models_to_optimize:
        try:
            # Run optimization
            logger.info(f"Optimizing {model_type} model...")
            study, best_model, output_dir = optimize_model(
                model_type=model_type,
                train_years=train_years,
                n_trials=n_trials,
                primary_metric=metric
            )
            
            results[model_type] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'optimization_metric': metric,
                'output_dir': str(output_dir)
            }
            
            logger.info(f"Optimization for {model_type} completed. Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error optimizing {model_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results

def _create_ensemble(model_paths=None, ensemble_type='weighted', weighting_method='betting_accuracy'):
    """
    Create ensemble models from optimized base models
    
    Args:
        model_paths (list): Paths to model pickle files
        ensemble_type (str): Type of ensemble ('weighted' or 'stacking')
        weighting_method (str): Method for weighting models
        
    Returns:
        dict: Ensemble results
    """
    # Find model paths if not provided
    if model_paths is None or len(model_paths) == 0:
        # Look for optimized models
        opt_paths = list(Path("models/optimization").glob("*/optimized_*_model.pkl"))
        if opt_paths:
            model_paths = [str(p) for p in opt_paths]
            logger.info(f"Found {len(model_paths)} optimized models")
        else:
            # Fall back to regular models
            reg_paths = list(Path("models").glob("strikeout_*_model.pkl"))
            model_paths = [str(p) for p in reg_paths if 'ensemble' not in p.stem]
            logger.info(f"Found {len(model_paths)} regular models")
    
    if not model_paths:
        logger.error("No models found to create ensemble")
        return None
    
    # Convert paths to Path objects if they're strings
    model_paths = [Path(p) if isinstance(p, str) else p for p in model_paths]
    
    # Load models
    logger.info(f"Loading {len(model_paths)} models for ensemble...")
    models = load_optimized_models(model_paths)
    
    if not models:
        logger.error("Failed to load any models")
        return None
    
    # Get data for evaluation
    logger.info("Loading data for ensemble evaluation...")
    pitcher_data = get_pitcher_data()
    
    # Evaluate models and create ensembles
    output_dir = Path("models/ensemble")
    ensure_dir(output_dir)
    
    # Define test years (default from latest model)
    test_years = models[0].get('test_years', StrikeoutModelConfig.DEFAULT_TEST_YEARS)
    
    # Create ensemble
    logger.info(f"Creating {ensemble_type} ensemble with {weighting_method} weighting...")
    ensemble_dict = create_ensemble(models, ensemble_type, weighting_method)
    
    # Save ensemble
    ensemble_path = output_dir / f"{ensemble_type}_{weighting_method}_ensemble.pkl"
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_dict, f)
    
    # Also save as standard ensemble model
    standard_path = Path("models") / "strikeout_ensemble_model.pkl"
    with open(standard_path, 'wb') as f:
        pickle.dump(ensemble_dict, f)
    
    logger.info(f"Ensemble saved to {ensemble_path} and {standard_path}")
    
    # Evaluate on test data
    logger.info("Evaluating ensemble performance...")
    test_df = pitcher_data[pitcher_data['season'].isin(test_years)]
    
    if test_df.empty:
        logger.warning(f"No test data available for years {test_years}")
        return {'ensemble_path': str(ensemble_path)}
    
    # Prepare test data
    features = models[0]['features']
    X_test = test_df[features].copy()
    y_test = test_df['strikeouts'].copy()
    
    # Make predictions with ensemble
    if ensemble_type == 'weighted':
        y_pred = simple_weighted_ensemble(
            models=models,
            weights=ensemble_dict['weights'],
            X=X_test
        )
    else:  # stacking
        y_pred = ensemble_dict['model'].predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(((y_test - y_pred) ** 2).mean()),
        'mae': np.abs(y_test - y_pred).mean(),
    }
    
    # Add betting metrics
    betting_metrics = calculate_betting_metrics(y_test, y_pred)
    metrics.update(betting_metrics)
    
    logger.info(f"Ensemble metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    logger.info(f"Within 1 strikeout: {metrics['within_1_strikeout']:.2f}%")
    logger.info(f"Within 2 strikeouts: {metrics['within_2_strikeouts']:.2f}%")
    
    return {
        'ensemble_path': str(ensemble_path),
        'metrics': metrics,
        'ensemble_type': ensemble_type,
        'weighting_method': weighting_method
    }

def _evaluate_models(models_dir=None, output_dir=None):
    """
    Evaluate and compare trained models
    
    Args:
        models_dir (str): Directory containing model files
        output_dir (str): Directory to save comparison results
        
    Returns:
        dict: Evaluation results
    """
    # Set default directories
    if models_dir is None:
        models_dir = Path("models")
    else:
        models_dir = Path(models_dir)
    
    if output_dir is None:
        output_dir = models_dir / "comparison"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Run comparison
    logger.info(f"Comparing models from {models_dir}")
    comparison_df, best_models = run_comparison(models_dir, output_dir)
    
    if comparison_df is None:
        logger.error("Model comparison failed")
        return None
    
    # Create a summary file
    summary_path = output_dir / "evaluation_summary.json"
    summary = {
        'best_models': best_models,
        'model_count': len(comparison_df),
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {
            'avg_rmse': float(comparison_df['RMSE'].mean()),
            'best_within_1_strikeout': float(comparison_df['Within 1 Strikeout'].max()),
            'best_within_2_strikeouts': float(comparison_df['Within 2 Strikeouts'].max())
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Evaluation summary saved to {summary_path}")
    logger.info(f"Best model for 'Within 1 Strikeout': {best_models.get('Within 1 Strikeout', 'N/A')}")
    
    return {
        'comparison': comparison_df.to_dict() if comparison_df is not None else None,
        'best_models': best_models,
        'summary_path': str(summary_path)
    }

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='MLB model pipeline for strikeout prediction')
    
    # Main command argument
    parser.add_argument('command', choices=['train', 'optimize', 'ensemble', 'evaluate', 'all'],
                       help='Command to execute')
    
    # Training arguments
    parser.add_argument('--model-types', type=str, nargs='+', 
                       choices=['rf', 'xgboost', 'lightgbm'], 
                       help='Model types to train')
    parser.add_argument('--train-years', type=int, nargs='+',
                       help='Years to use for training')
    parser.add_argument('--test-years', type=int, nargs='+',
                       help='Years to use for testing')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Whether to tune hyperparameters')
    
    # Optimization arguments
    parser.add_argument('--model-type', type=str, 
                       choices=['rf', 'xgboost', 'lightgbm', 'all'],
                       help='Model type to optimize')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--metric', type=str, 
                       choices=['within_1_strikeout', 'within_2_strikeouts', 
                              'over_under_accuracy', 'neg_mean_squared_error'],
                       help='Metric to optimize')
    
    # Ensemble arguments
    parser.add_argument('--model-paths', type=str, nargs='+',
                       help='Paths to model pickle files')
    parser.add_argument('--ensemble-type', type=str, 
                       choices=['weighted', 'stacking'],
                       help='Type of ensemble')
    parser.add_argument('--weighting-method', type=str,
                       choices=['equal', 'inverse_error', 'betting_accuracy'],
                       help='Method for weighting models')
    
    # Evaluation arguments
    parser.add_argument('--models-dir', type=str,
                       help='Directory containing model files')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save results')
    
    print("Debug - sys.argv:", sys.argv)
    
    args = parser.parse_args()
    print("Debug - parsed args:", args)
    
    # Save command separately before converting to dict
    command = args.command 
    
    # Convert args to kwargs
    kwargs = vars(args)
    del kwargs['command']
    
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Run pipeline (use the saved command)
    result = run_pipeline(command, **kwargs)
    
    if result is not None:
        logger.info(f"Successfully completed command: {command}")
        return 0
    else:
        logger.error(f"Failed to complete command: {command}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())