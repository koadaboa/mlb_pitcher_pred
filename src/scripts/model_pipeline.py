import argparse
from pathlib import Path
import sys
import json
from datetime import datetime
import traceback

from src.data.utils import setup_logger, ensure_dir
from src.data.db import get_pitcher_data
from src.features.selection import select_features
from src.models.train import train_strikeout_model, save_model
from src.scripts.optimize_models import optimize_model
from src.scripts.model_comparison import run_comparison
from config import StrikeoutModelConfig

logger = setup_logger(__name__, log_file = "logs/model_pipeline.log")

def run_pipeline(command, **kwargs):
    """
    Run the model pipeline with the specified command - optimized for LightGBM focus
    
    Args:
        command (str): Command to execute (train, optimize, evaluate, all)
        **kwargs: Additional command-specific arguments
        
    Returns:
        dict: Results of command execution
    """
    # Create necessary directories
    ensure_dir("models")
    
    # Extract feature selection method if provided
    feature_selection_method = kwargs.pop('feature_selection', 'rfecv')
    
    # Execute the appropriate command
    if command == 'train':
        kwargs['feature_selection'] = feature_selection_method
        return _train_models(**kwargs)
    
    elif command == 'optimize':
        # Extract only the parameters that _optimize_models accepts
        optimize_kwargs = {k: v for k, v in kwargs.items() if k in 
                          ['train_years', 'n_trials', 'metric']}
        return _optimize_models(**optimize_kwargs)
    
    elif command == 'evaluate':
        evaluate_kwargs = {}
        if 'models_dir' in kwargs:
            evaluate_kwargs['models_dir'] = kwargs['models_dir']
        if 'output_dir' in kwargs:
            evaluate_kwargs['output_dir'] = kwargs['output_dir']
            
        return _evaluate_model(**evaluate_kwargs)
    
    elif command == 'all':
        # Train LightGBM model
        train_kwargs = {
            'feature_selection': feature_selection_method,
            **{k: v for k, v in kwargs.items() if k in 
               ['train_years', 'test_years', 'tune_hyperparameters']}
        }
        models_result = _train_models(**train_kwargs)
        
        # Optimize LightGBM model
        optimize_kwargs = {
            **{k: v for k, v in kwargs.items() if k in 
               ['train_years', 'n_trials', 'metric']}
        }
        optimize_result = _optimize_models(**optimize_kwargs)
        
        # Evaluate model
        evaluate_kwargs = {}
        if 'models_dir' in kwargs:
            evaluate_kwargs['models_dir'] = kwargs['models_dir']
        if 'output_dir' in kwargs:
            evaluate_kwargs['output_dir'] = kwargs['output_dir']
            
        eval_result = _evaluate_model(**evaluate_kwargs)
        
        return {
            'training': models_result,
            'optimization': optimize_result,
            'evaluation': eval_result
        }
    else:
        logger.error(f"Unknown command: {command}")
        return None

def _train_models(train_years=None, test_years=None, tune_hyperparameters=True, feature_selection='rfecv'):
    """
    Train LightGBM model for strikeout prediction
    
    Args:
        train_years (tuple): Years to use for training
        test_years (tuple): Years to use for testing
        tune_hyperparameters (bool): Whether to tune hyperparameters
        feature_selection (str): Feature selection method
        
    Returns:
        dict: Dictionary with trained model
    """
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
    
    # Select features using the specified method
    logger.info(f"Selecting features using {feature_selection} method...")

    so_features = select_features(pitcher_data, method=feature_selection)
    logger.info(f"Selected {len(so_features)} features")
    
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
        model_path = models_dir / "strikeout_model.pkl"
        save_model(model_dict, model_path)
        
        # Log performance metrics
        metrics = model_dict.get('metrics', {})
        logger.info(f"LightGBM model performance:")
        logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
        logger.info(f"  Within 1 Strikeout: {metrics.get('within_1_strikeout', 'N/A'):.2f}%")
        logger.info(f"  Within 2 Strikeouts: {metrics.get('within_2_strikeouts', 'N/A'):.2f}%")

        # Get and log feature importance
        if 'importance' in model_dict:
            importance_df = model_dict['importance']
            logger.info("\nTOP 10 FEATURES BY IMPORTANCE:")
            for i, (feature, importance) in enumerate(zip(
                importance_df['feature'].iloc[:10], 
                importance_df['importance'].iloc[:10]
            )):
                logger.info(f"  {i+1}. {feature}: {importance:.6f}")
        elif 'model' in model_dict and hasattr(model_dict['model'], 'feature_importances_'):
            # If we don't have importance DataFrame but model has feature_importances_
            model = model_dict['model']
            features = model_dict.get('features', [])
            
            if features and len(features) == len(model.feature_importances_):
                # Create list of (feature, importance) tuples
                importances = list(zip(features, model.feature_importances_))
                # Sort by importance (descending)
                importances.sort(key=lambda x: x[1], reverse=True)
                
                logger.info("\nTOP 10 FEATURES BY IMPORTANCE:")
                for i, (feature, importance) in enumerate(importances[:10]):
                    logger.info(f"  {i+1}. {feature}: {importance:.6f}")
    else:
        logger.error(f"Failed to train LightGBM model")
        return None

    # Save feature information to files
    feature_info_dir = models_dir / "feature_info"
    ensure_dir(feature_info_dir)

    # Save selected features
    with open(feature_info_dir / "selected_features.json", "w") as f:
        json.dump({
            "selected_features": so_features,
            "feature_count": len(so_features),
            "selection_method": feature_selection,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)

    # Save feature importance if available
    if 'importance' in model_dict:
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(
                model_dict['importance']['feature'], 
                model_dict['importance']['importance']
            )
        }
        
        with open(feature_info_dir / "feature_importance.json", "w") as f:
            json.dump({
                "feature_importance": importance_dict,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4)

    logger.info(f"Feature information saved to {feature_info_dir}")
    
    logger.info("Model training complete")
    return {"lightgbm": model_dict}

def _optimize_models(train_years=None, n_trials=100, metric='within_1_strikeout', **kwargs):
    """
    Optimize LightGBM model using Bayesian optimization
    
    Args:
        train_years (tuple): Years to use for training
        n_trials (int): Number of optimization trials
        metric (str): Metric to optimize
        **kwargs: Additional keyword arguments (ignored)
        
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
        
        logger.info(f"Optimization completed. Results saved to {output_dir}")
        return results
        
    except Exception as e:
        logger.error(f"Error optimizing LightGBM: {e}")
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
    parser = argparse.ArgumentParser(description='MLB model pipeline for strikeout prediction - LightGBM focus')
    
    # Main command argument
    parser.add_argument('command', choices=['train', 'optimize', 'evaluate', 'all'],
                       help='Command to execute')
    
    # Training arguments
    parser.add_argument('--train-years', type=int, nargs='+',
                       help='Years to use for training')
    parser.add_argument('--test-years', type=int, nargs='+',
                       help='Years to use for testing')
    parser.add_argument('--tune-hyperparameters', action='store_true', default=True,
                       help='Whether to tune hyperparameters (default: True)')
    parser.add_argument('--no-tune', dest='tune_hyperparameters', action='store_false',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--feature-selection', type=str, 
                       choices=['manual', 'rfecv', 'both'], default='rfecv',
                       help='Feature selection method')
    
    # Optimization arguments
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--metric', type=str, 
                       choices=['within_1_strikeout', 'within_2_strikeouts', 
                              'over_under_accuracy', 'neg_mean_squared_error'],
                       default='within_1_strikeout',
                       help='Metric to optimize')
    
    # Evaluation arguments
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing model files')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Print mode information
    logger.info("Running LightGBM model pipeline")
    logger.info(f"Feature selection method: {args.feature_selection}")
    logger.info(f"Hyperparameter tuning: {'Enabled' if args.tune_hyperparameters else 'Disabled'}")
    
    # Save command separately before converting to dict
    command = args.command 
    
    # Convert args to kwargs
    kwargs = vars(args)
    del kwargs['command']
    
    # Run pipeline
    result = run_pipeline(command, **kwargs)
    
    if result is not None:
        logger.info(f"Successfully completed command: {command}")
        return 0
    else:
        logger.error(f"Failed to complete command: {command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())