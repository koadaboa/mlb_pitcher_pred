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
from src.scripts.model_comparison import run_comparison
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

def run_pipeline(command, **kwargs):
    """
    Run the model pipeline with the specified command
    
    Args:
        command (str): Command to execute (train, optimize, evaluate, all)
        **kwargs: Additional command-specific arguments
        
    Returns:
        dict: Results of command execution
    """
    # Create necessary directories
    ensure_dir("models")
    
    # Execute the appropriate command
    if command == 'train':
        return _train_model(**kwargs)
    elif command == 'optimize':
        return _optimize_model(**kwargs)
    elif command == 'evaluate':
        return _evaluate_model(**kwargs)
    elif command == 'all':
        # Train model first
        train_kwargs = {k: v for k, v in kwargs.items() if k in 
                      ['train_years', 'test_years', 'tune_hyperparameters']}
        model_result = _train_model(**train_kwargs)
        
        # Then optimize it
        optimize_kwargs = {k: v for k, v in kwargs.items() if k in 
                         ['train_years', 'n_trials', 'metric']}
        optimize_result = _optimize_model(**optimize_kwargs)
        
        # Finally evaluate everything
        evaluate_kwargs = {k: v for k, v in kwargs.items() if k in 
                         ['models_dir', 'output_dir']}
        eval_result = _evaluate_model(**evaluate_kwargs)
        
        return {
            'training': model_result,
            'optimization': optimize_result,
            'evaluation': eval_result
        }
    else:
        logger.error(f"Unknown command: {command}")
        return None

def _train_model(train_years=None, test_years=None, tune_hyperparameters=True):
    """
    Train a LightGBM model for strikeout prediction
    
    Args:
        train_years (tuple): Years to use for training
        test_years (tuple): Years to use for testing
        tune_hyperparameters (bool): Whether to tune hyperparameters
        
    Returns:
        dict: Dictionary of trained model
    """
    # Set defaults
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
    
    # Train LightGBM model
    logger.info("Training LightGBM strikeout prediction model...")
    model_dict = train_strikeout_model(
        pitcher_data,
        so_features,
        train_years=train_years,
        test_years=test_years,
        tune_hyperparameters=tune_hyperparameters
    )
    
    if model_dict is not None:
        # Save model
        model_path = models_dir / "strikeout_lightgbm_model.pkl"
        save_model(model_dict, model_path)
        
        # Also save a copy as the standard model
        standard_path = models_dir / "strikeout_model.pkl"
        save_model(model_dict, standard_path)
        
        logger.info(f"Successfully trained and saved LightGBM model")
        
        return {'lightgbm': model_dict}
    else:
        logger.error("Failed to train LightGBM model")
        return None

def _optimize_model(train_years=None, n_trials=50, metric='within_1_strikeout'):
    """
    Optimize LightGBM model using Bayesian optimization
    
    Args:
        train_years (tuple): Years to use for training
        n_trials (int): Number of optimization trials
        metric (str): Metric to optimize
        
    Returns:
        dict: Optimization results
    """
    # Set defaults
    if train_years is None:
        train_years = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    
    logger.info(f"Starting optimization for LightGBM model with {n_trials} trials...")
    logger.info(f"Optimizing for metric: {metric}")
    
    try:
        # Run optimization
        study, best_model, output_dir = optimize_model(
            train_years=train_years,
            n_trials=n_trials,
            primary_metric=metric
        )
        
        results = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_metric': metric,
            'output_dir': str(output_dir)
        }
        
        logger.info(f"Optimization for LightGBM completed. Results saved to {output_dir}")
        return {'lightgbm': results}
        
    except Exception as e:
        logger.error(f"Error optimizing LightGBM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _evaluate_model(models_dir=None, output_dir=None):
    """
    Evaluate trained LightGBM model
    
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
        output_dir = models_dir / "evaluation"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Run comparison
    logger.info(f"Evaluating model from {models_dir}")
    comparison_df, _ = run_comparison(models_dir, output_dir)
    
    if comparison_df is None:
        logger.error("Model evaluation failed")
        return None
    
    # Create a summary file
    summary_path = output_dir / "evaluation_summary.json"
    summary = {
        'model_count': len(comparison_df),
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {
            'rmse': float(comparison_df['RMSE'].mean()),
            'within_1_strikeout': float(comparison_df['Within 1 Strikeout'].mean()),
            'within_2_strikeouts': float(comparison_df['Within 2 Strikeouts'].mean())
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Evaluation summary saved to {summary_path}")
    
    return {
        'comparison': comparison_df.to_dict() if comparison_df is not None else None,
        'summary_path': str(summary_path)
    }

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='MLB model pipeline for strikeout prediction')
    
    # Main command argument
    parser.add_argument('command', choices=['train', 'optimize', 'evaluate', 'all'],
                       help='Command to execute')
    
    # Training arguments
    parser.add_argument('--train-years', type=int, nargs='+',
                       help='Years to use for training')
    parser.add_argument('--test-years', type=int, nargs='+',
                       help='Years to use for testing')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Whether to tune hyperparameters')
    
    # Optimization arguments
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--metric', type=str, 
                       choices=['within_1_strikeout', 'within_2_strikeouts', 
                              'over_under_accuracy', 'neg_mean_squared_error'],
                       help='Metric to optimize')
    
    # Evaluation arguments
    parser.add_argument('--models-dir', type=str,
                       help='Directory containing model files')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
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