# src/scripts/optimize_models.py
import optuna
import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.data.db import get_pitcher_data
from src.features.selection import select_features_for_strikeout_model
from src.models.train import calculate_betting_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def custom_cv_score(model, X, y, n_splits=5, random_state=42, scorer='neg_mean_squared_error'):
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
        y_pred = model.predict(X_val)
        predictions.extend(y_pred)
        actuals.extend(y_val)
        
        # Calculate score based on scorer
        if scorer == 'neg_mean_squared_error':
            score = -mean_squared_error(y_val, y_pred)
        elif scorer == 'neg_mean_absolute_error':
            score = -mean_absolute_error(y_val, y_pred)
        elif scorer == 'r2':
            score = r2_score(y_val, y_pred)
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

def rf_objective(trial, X, y, primary_metric):
    """Optuna objective for RandomForest"""
    
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    # Create model
    model = RandomForestRegressor(**params)
    
    # Calculate CV score
    score, betting_metrics = custom_cv_score(model, X, y, scorer=primary_metric)
    
    # Log progress
    logger.info(f"Trial {trial.number}: {primary_metric}={score:.4f}, "
                f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%, "
                f"Over/Under: {betting_metrics['over_under_accuracy']:.2f}%, "
                f"Params: {params}")
    
    # Store additional metrics in trial user attributes for later analysis
    trial.set_user_attr('betting_metrics', betting_metrics)
    trial.set_user_attr('model_type', 'rf')
    
    return score

def xgb_objective(trial, X, y, primary_metric):
    """Optuna objective for XGBoost"""
    
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }
    
    # Create model
    model = xgb.XGBRegressor(**params)
    
    # Calculate CV score
    score, betting_metrics = custom_cv_score(model, X, y, scorer=primary_metric)
    
    # Log progress
    logger.info(f"Trial {trial.number}: {primary_metric}={score:.4f}, "
                f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%, "
                f"Over/Under: {betting_metrics['over_under_accuracy']:.2f}%, "
                f"Params: {params}")
    
    # Store additional metrics in trial user attributes for later analysis
    trial.set_user_attr('betting_metrics', betting_metrics)
    trial.set_user_attr('model_type', 'xgboost')
    
    return score

def lgb_objective(trial, X, y, primary_metric):
    """Optuna objective for LightGBM"""
    
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 5),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
        'random_state': 42
    }
    
    # Create model
    model = lgb.LGBMRegressor(**params)
    
    # Calculate CV score
    score, betting_metrics = custom_cv_score(model, X, y, scorer=primary_metric)
    
    # Log progress
    logger.info(f"Trial {trial.number}: {primary_metric}={score:.4f}, "
                f"Within 2 K: {betting_metrics['within_2_strikeouts']:.2f}%, "
                f"Over/Under: {betting_metrics['over_under_accuracy']:.2f}%, "
                f"Params: {params}")
    
    # Store additional metrics in trial user attributes for later analysis
    trial.set_user_attr('betting_metrics', betting_metrics)
    trial.set_user_attr('model_type', 'lightgbm')
    
    return score

def visualize_optimization_results(study, output_dir):
    """
    Create visualizations for optimization results
    
    Args:
        study (optuna.Study): Completed Optuna study
        output_dir (Path): Directory to save visualizations
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract model type from first trial
    model_type = study.trials[0].user_attrs['model_type']
    
    # Extract additional metrics
    trials_df = study.trials_dataframe()
    trials_df['within_2_strikeouts'] = [t.user_attrs['betting_metrics']['within_2_strikeouts'] 
                                        for t in study.trials]
    trials_df['over_under_accuracy'] = [t.user_attrs['betting_metrics']['over_under_accuracy'] 
                                       for t in study.trials]
    trials_df['mape'] = [t.user_attrs['betting_metrics']['mape'] 
                        for t in study.trials]
    
    # 1. Optimization history plot
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(f'Optimization History for {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_optimization_history.png")
    plt.close()
    
    # 2. Parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title(f'Parameter Importances for {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_param_importances.png")
    plt.close()
    
    # 3. Parallel coordinate plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title(f'Parallel Coordinate Plot for {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_parallel_coordinate.png")
    plt.close()
    
    # 4. Additional metrics progression
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(trials_df.index, trials_df['within_2_strikeouts'], 'o-')
    plt.xlabel('Trial')
    plt.ylabel('Within 2 Strikeouts (%)')
    plt.title('Within 2 Strikeouts Progression')
    
    plt.subplot(1, 2, 2)
    plt.plot(trials_df.index, trials_df['over_under_accuracy'], 'o-')
    plt.xlabel('Trial')
    plt.ylabel('Over/Under Accuracy (%)')
    plt.title('Over/Under Accuracy Progression')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_betting_metrics.png")
    plt.close()
    
    # 5. Save trials dataframe
    trials_df.to_csv(output_dir / f"{model_type}_trials.csv", index=False)

def save_best_model(study, X, y, model_type, output_dir):
    """
    Train and save the best model
    
    Args:
        study (optuna.Study): Completed Optuna study
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        model_type (str): Model type ('rf', 'xgboost', or 'lightgbm')
        output_dir (Path): Directory to save model
    """
    # Get best parameters
    best_params = study.best_params
    
    # Create model with best parameters
    if model_type == 'rf':
        model = RandomForestRegressor(**best_params, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(**best_params, random_state=42)
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(**best_params, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model on full dataset
    model.fit(X, y)
    
    # Create model dictionary
    model_dict = {
        'model': model,
        'model_type': model_type,
        'params': best_params,
        'best_value': study.best_value,
        'features': X.columns.tolist(),
        'study_name': study.study_name,
        'optimization_metric': study.user_attrs.get('optimization_metric', 'unknown')
    }
    
    # Save model
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / f"optimized_{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    logger.info(f"Best {model_type} model saved to {model_path}")
    
    # Return the best model
    return model_dict

def optimize_model(model_type, train_years, features=None, n_trials=100, primary_metric='within_2_strikeouts'):
    """
    Optimize a model using Optuna
    
    Args:
        model_type (str): Model type ('rf', 'xgboost', or 'lightgbm')
        train_years (list): Years to use for training
        features (list, optional): Features to use. If None, will be selected automatically.
        n_trials (int): Number of Optuna trials
        primary_metric (str): Primary metric to optimize
        
    Returns:
        optuna.Study: Completed Optuna study
    """
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/optimization/{model_type}_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get data
    logger.info("Loading data...")
    pitcher_data = get_pitcher_data()
    
    # Select features if not provided
    if features is None:
        logger.info("Selecting features...")
        features = select_features_for_strikeout_model(pitcher_data)
    
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
    study_name = f"{model_type}_optimization_{timestamp}"
    storage_url = f"sqlite:///{output_dir}/optimization.db"
    
    # Set direction based on metric (higher is better or lower is better)
    if primary_metric in ['within_2_strikeouts', 'over_under_accuracy', 'r2']:
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
    
    # Select objective function based on model type
    if model_type == 'rf':
        objective = lambda trial: rf_objective(trial, X, y, primary_metric)
    elif model_type == 'xgboost':
        objective = lambda trial: xgb_objective(trial, X, y, primary_metric)
    elif model_type == 'lightgbm':
        objective = lambda trial: lgb_objective(trial, X, y, primary_metric)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run optimization
    logger.info(f"Starting {model_type} optimization with {n_trials} trials...")
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
    best_model = save_best_model(study, X, y, model_type, output_dir)
    
    return study, best_model, output_dir

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimize ML models for pitcher strikeout prediction')
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'xgboost', 'lightgbm', 'all'],
                        help='Model type to optimize')
    parser.add_argument('--years', type=int, nargs='+', default=[2019, 2021, 2022],
                        help='Training years to use')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--metric', type=str, 
                        choices=['neg_mean_squared_error', 'neg_mean_absolute_error', 
                                'r2', 'within_2_strikeouts', 'over_under_accuracy'],
                        default='within_2_strikeouts',
                        help='Primary metric to optimize')
    
    args = parser.parse_args()
    
    # Run optimization for selected model(s)
    if args.model == 'all':
        models = ['rf', 'xgboost', 'lightgbm']
    else:
        models = [args.model]
    
    for model_type in models:
        try:
            study, best_model, output_dir = optimize_model(
                model_type=model_type,
                train_years=args.years,
                n_trials=args.trials,
                primary_metric=args.metric
            )
            logger.info(f"Optimization for {model_type} completed. Results saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error optimizing {model_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()