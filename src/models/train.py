# src/models/train.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

def calculate_betting_metrics(y_true, y_pred):
    """
    Calculate additional metrics relevant for betting on strikeouts
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        dict: Dictionary of betting-relevant metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate mean absolute percentage error (MAPE)
    # Only include non-zero true values to avoid division by zero
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    
    # Calculate maximum error (worst case scenario)
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Calculate accuracy within 1, 2, and 3 strikeouts
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    within_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100
    within_3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
    
    # Calculate over/under accuracy (good for betting)
    # Assuming a threshold (e.g., the betting line) as the mean of true values
    threshold = np.mean(y_true)
    true_over = y_true > threshold
    pred_over = y_pred > threshold
    over_under_accuracy = np.mean(true_over == pred_over) * 100
    
    # Calculate bias (tendency to over or under predict)
    bias = np.mean(y_pred - y_true)
    
    return {
        'mape': mape,
        'max_error': max_error,
        'within_1_strikeout': within_1,
        'within_2_strikeouts': within_2,
        'within_3_strikeouts': within_3,
        'over_under_accuracy': over_under_accuracy,
        'bias': bias
    }

def tune_model_hyperparameters(X_train, y_train, model_type='rf', cv=3, random_state=42):
    """
    Perform basic hyperparameter tuning for the selected model type
    
    Args:
        X_train (array): Training features
        y_train (array): Training target
        model_type (str): Model type ('rf', 'xgboost', or 'lightgbm')
        cv (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Best hyperparameters
    """
    logger.info(f"Tuning hyperparameters for {model_type} model...")
    
    # Define hyperparameter grids for each model type
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.9]
        }
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 70]
        }
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return {}
    
    # Create cross-validation object
    cv_folds = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {-grid_search.best_score_:.4f} MSE")
    
    return grid_search.best_params_

def train_strikeout_model(df, features, target='strikeouts', train_years=(2019, 2021, 2022), 
                          test_years=(2023, 2024), random_state=42, model_type='rf',
                          tune_hyperparameters=True):
    """
    Train a model to predict strikeouts using time-based splitting
    
    Args:
        df (pandas.DataFrame): DataFrame with features and target
        features (list): List of feature names to use
        target (str): Target variable name
        train_years (tuple): Years to use for training
        test_years (tuple): Years to use for testing
        random_state (int): Random state for reproducibility
        model_type (str): Model type ('rf', 'xgboost', or 'lightgbm')
        tune_hyperparameters (bool): Whether to tune hyperparameters
        
    Returns:
        dict: Dictionary with model, scaler, metrics, and feature importance
    """
    logger.info(f"Training {model_type} strikeout prediction model with {len(features)} features")
    logger.info(f"Training years: {train_years}, Testing years: {test_years}")
    
    # Ensure all necessary columns exist
    if target not in df.columns:
        logger.error(f"Target '{target}' not found in DataFrame")
        return None
    
    if 'season' not in df.columns:
        logger.error("'season' column not found in DataFrame, required for time-based splitting")
        return None
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        features = [f for f in features if f in df.columns]
    
    # Filter out rows with missing values
    model_df = df[features + [target, 'season']].dropna()
    logger.info(f"Using {len(model_df)} rows after dropping NA values")
    
    # Time-based splitting
    train_df = model_df[model_df['season'].isin(train_years)]
    test_df = model_df[model_df['season'].isin(test_years)]
    
    logger.info(f"Training set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    
    # Prepare features and target
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning if requested
    if tune_hyperparameters:
        best_params = tune_model_hyperparameters(
            X_train_scaled, y_train, 
            model_type=model_type, 
            random_state=random_state
        )
    else:
        best_params = {}
    
    # Train model based on type
    if model_type == 'rf':
        # Set default parameters if not tuned
        if not best_params:
            best_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2
            }
        
        model = RandomForestRegressor(
            random_state=random_state,
            **best_params
        )
    elif model_type == 'xgboost':
        # Set default parameters if not tuned
        if not best_params:
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
        
        model = xgb.XGBRegressor(
            random_state=random_state,
            **best_params
        )
    elif model_type == 'lightgbm':
        # Set default parameters if not tuned
        if not best_params:
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31
            }
        
        model = lgb.LGBMRegressor(
            random_state=random_state,
            **best_params
        )
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate standard metrics
    std_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Calculate betting-relevant metrics
    betting_metrics = calculate_betting_metrics(y_test, y_pred)
    
    # Combine all metrics
    metrics = {**std_metrics, **betting_metrics}
    
    # Get feature importance
    if model_type == 'rf' or model_type == 'xgboost' or model_type == 'lightgbm':
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # Create placeholder importance if not available
            importance = pd.DataFrame({
                'feature': features,
                'importance': np.ones(len(features)) / len(features)
            })
    else:
        # Create placeholder importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': np.ones(len(features)) / len(features)
        })
    
    # Log metrics
    logger.info(f"Model metrics: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
    logger.info(f"Betting metrics: MAPE={metrics['mape']:.2f}%, Over/Under Accuracy={metrics['over_under_accuracy']:.2f}%")
    logger.info(f"Within 2 strikeouts: {metrics['within_2_strikeouts']:.2f}%")
    logger.info(f"Top features by importance: {', '.join(importance['feature'].head(3).tolist())}")
    
    return {
        'model': model,
        'model_type': model_type,
        'params': best_params,
        'scaler': scaler,
        'metrics': metrics,
        'importance': importance,
        'features': features
    }

def save_model(model_dict, model_path):
    """
    Save model and associated artifacts
    
    Args:
        model_dict (dict): Dictionary with model and associated objects
        model_path (str): Path to save the model
    """
    path = Path(model_path)
    path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    logger.info(f"Model saved to {path}")