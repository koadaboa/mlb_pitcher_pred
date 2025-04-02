# src/scripts/optimize_models.py
import optuna
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import time
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.data.db import get_pitcher_data
from src.features.selection import select_features
from src.models.train import calculate_betting_metrics
from src.data.utils import setup_logger
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

def custom_cv_score(model, X, y, n_splits=5, random_state=StrikeoutModelConfig.RANDOM_STATE, scorer='neg_mean_squared_error'):
    """
    Calculate cross-validation score using KFold
    
    Args:
        model: Model object
        X: Features
        y: Target
        n_splits: Number of folds
        random_state: Random seed
        scorer (str): Scoring metric
        
    Returns:
        float: Mean CV score
    """
    scores = []
    predictions = []
    actuals = []
    
    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        if isinstance(X_val, pd.DataFrame):
            y_pred = model.predict(X_val)
        else:
            # Convert to DataFrame with proper column names if it's not already
            X_val_df = pd.DataFrame(X_val, columns=X.columns)
            y_pred = model.predict(X_val_df)
        predictions.extend(y_pred)
        actuals.extend(y_val)
        
        # Calculate score based on scorer
        if scorer == 'neg_mean_squared_error':
            score = -mean_squared_error(y_val, y_pred)
        elif scorer == 'neg_mean_absolute_error':
            score = -mean_absolute_error(y_val, y_pred)
        elif scorer == 'r2':
            score = r2_score(y_val, y_pred)
        elif scorer == 'within_1_strikeout':
            score = np.mean(np.abs(y_val - y_pred) <= 1) * 100
        elif scorer == 'within_2_strikeouts':
            score = np.mean(np.abs(y_val - y_pred) <= 2) * 100
        elif scorer == 'over_under_accuracy':
            threshold = np.mean(y_train)
            true_over = y_val > threshold
            pred_over = y_pred > threshold
            score = np.mean(true_over == pred_over) * 100
        else:
            raise ValueError(f"Unknown scorer: {scorer}")
        
        scores.append(score)
    
    # Calculate betting metrics for logging
    betting_metrics = calculate_betting_metrics(actuals, predictions)
    
    return np.mean(scores), betting_metrics

def save_best_model(study, X, y, output_dir):
    """
    Train and save the best model
    
    Args:
        study (optuna.Study): Completed Optuna study
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        output_dir (Path): Directory to save model
    """
    # Get best parameters
    best_params = study.best_params
    
    # Create model with best parameters
    model = lgb.LGBMRegressor(**best_params, random_state=StrikeoutModelConfig.RANDOM_STATE)
    
    # Train model on full dataset
    model.fit(X, y)
    
    # Create model dictionary
    model_dict = {
        'model': model,
        'model_type': 'lightgbm',
        'params': best_params,
        'best_value': study.best_value,
        'features': X.columns.tolist(),
        'study_name': study.study_name,
        'optimization_metric': study.user_attrs.get('optimization_metric', 'unknown')
    }
    
    # Save model
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / "optimized_lightgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    # Also save as the standard model name
    standard_path = output_dir / "strikeout_model.pkl"
    with open(standard_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    logger.info(f"Best LightGBM model saved to {model_path}")
    
    # Return the best model
    return model_dict

def visualize_optimization_results(study, output_dir):
    """
    Create visualizations of optimization results
    
    Args:
        study (optuna.Study): Completed Optuna study
        output_dir (Path): Directory to save visualizations
    """
    # Create visualization directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # 1. Plot optimization history
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(viz_dir / "optimization_history.png")
        plt.close()
        
        # 2. Plot parameter importances
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(viz_dir / "param_importances.png")
        plt.close()
        
        # 3. Plot score vs. key parameters (if enough trials)
        if len(study.trials) >= 20:
            plt.figure(figsize=(15, 10))
            important_params = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth']
            for i, param in enumerate(important_params):
                plt.subplot(2, 2, i+1)
                optuna.visualization.matplotlib.plot_slice(study, params=[param])
                plt.title(f"Score vs. {param}")
            plt.tight_layout()
            plt.savefig(viz_dir / "score_vs_parameters.png")
            plt.close()
        
        logger.info(f"Optimization visualizations saved to {viz_dir}")
    
    except Exception as e:
        logger.error(f"Error creating optimization visualizations: {e}")

def optimize_model(train_years, features=None, n_trials=100, primary_metric='within_1_strikeout', **kwargs):
    """
    Optimize a LightGBM model using Optuna
    
    Args:
        train_years (list): Years to use for training
        features (list, optional): Features to use. If None, will be selected automatically.
        n_trials (int): Number of Optuna trials
        primary_metric (str): Primary metric to optimize
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        optuna.Study: Completed Optuna study
    """
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/optimization/lightgbm_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get data
    logger.info("Loading data...")
    pitcher_data = get_pitcher_data()
    
    # Select features if not provided
    if features is None:
        logger.info("Selecting features...")
        features = select_features(pitcher_data)
    
    # Filter to training years and prepare data
    train_df = pitcher_data[pitcher_data['season'].isin(train_years)]
    logger.info(f"Using {len(train_df)} rows from years {train_years} for optimization")
    
    if train_df.empty:
        logger.error(f"No data available for years {train_years}")
        return None
    
    # Prepare features and target
    X = train_df[features].copy()
    y = train_df['strikeouts'].copy()
    
    # Create study
    study_name = f"lightgbm_optimization_{timestamp}"
    storage_url = f"sqlite:///{output_dir}/optimization.db"
    
    # Set direction based on metric (higher is better or lower is better)
    if primary_metric in ['within_1_strikeout', 'within_2_strikeouts', 'over_under_accuracy', 'r2']:
        direction = 'maximize'
    else:  # 'neg_mean_squared_error', 'neg_mean_absolute_error'
        direction = 'maximize'  # These are already negated
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=True
    )
    
    # Store optimization metric in study user attributes
    study.set_user_attr('optimization_metric', primary_metric)
    
    # Set objective function
    objective = lambda trial: lgb_objective(trial, X, y, primary_metric)
    
    # Run optimization
    logger.info(f"Starting LightGBM optimization with {n_trials} trials...")
    logger.info(f"Optimizing for {primary_metric} ({direction})")
    
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)
    end_time = time.time()
    
    # Log results
    logger.info(f"Optimization completed in {(end_time - start_time) / 60:.2f} minutes")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    
    # Create visualizations
    visualize_optimization_results(study, output_dir)
    
    # Save best model
    best_model = save_best_model(study, X, y, output_dir)
    
    return study, best_model, output_dir

def lgb_objective(trial, X, y, primary_metric):
    """Optuna objective for LightGBM with focus on 1-strikeout precision"""
    
    # Define hyperparameters to optimize - refined ranges for strikeout precision
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # Lower max value
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # Lower for better generalization
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 100),  # Higher min values
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),  # More focused range
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 3),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 3),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 3),
        'random_state': StrikeoutModelConfig.RANDOM_STATE
    }
    
    # Create model
    model = lgb.LGBMRegressor(**params)
    
    # Calculate CV score with emphasis on 1-strikeout precision
    score, betting_metrics = custom_cv_score(model, X, y, scorer=primary_metric)
    
    # Log progress with focus on 1-strikeout precision
    logger.info(f"Trial {trial.number}: {primary_metric}={score:.4f}, "
                f"Within 1 K: {betting_metrics['within_1_strikeout']:.2f}%, "
                f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%, "
                f"Over/Under: {betting_metrics['over_under_accuracy']:.2f}%")
    
    # Store additional metrics in trial user attributes for later analysis
    trial.set_user_attr('betting_metrics', betting_metrics)
    
    return score

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimize LightGBM model for pitcher strikeout prediction')
    parser.add_argument('--years', type=int, nargs='+', default=[2019, 2021, 2022],
                        help='Training years to use')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--metric', type=str, 
                        choices=['neg_mean_squared_error', 'neg_mean_absolute_error', 
                                'r2', 'within_1_strikeout', 'within_2_strikeouts', 'over_under_accuracy'],
                        default='within_1_strikeout',
                        help='Primary metric to optimize')
    # Add test_years parameter for compatibility with model_pipeline.py
    parser.add_argument('--test-years', type=int, nargs='+', 
                        help='Testing years (not used in optimization but added for compatibility)')
    
    args = parser.parse_args()
    
    try:
        # Run optimization
        logger.info(f"Optimizing LightGBM model...")
        study, best_model, output_dir = optimize_model(
            train_years=args.years,
            n_trials=args.trials,
            primary_metric=args.metric
        )
        
        logger.info(f"Optimization completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error optimizing LightGBM: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()