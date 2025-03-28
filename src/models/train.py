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

logger = logging.getLogger(__name__)

def train_strikeout_model(df, features, target='strikeouts', train_years=(2019, 2021, 2022), 
                          test_years=(2023, 2024), random_state=42):
    """
    Train a model to predict strikeouts using time-based splitting
    
    Args:
        df (pandas.DataFrame): DataFrame with features and target
        features (list): List of feature names to use
        target (str): Target variable name
        train_years (tuple): Years to use for training
        test_years (tuple): Years to use for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary with model, scaler, metrics, and feature importance
    """
    logger.info(f"Training strikeout prediction model with {len(features)} features")
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
    
    # Train Random Forest model
    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Model metrics: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
    logger.info(f"Top features by importance: {', '.join(importance['feature'].head(3).tolist())}")
    
    return {
        'model': rf,
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