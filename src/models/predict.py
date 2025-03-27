# src/models/predict.py
import pickle
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load a trained model
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        dict: Dictionary with model and associated objects
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    logger.info(f"Model loaded from {model_path}")
    return model_dict

def predict_strikeouts(model_dict, pitcher_data):
    """
    Predict strikeouts for a pitcher
    
    Args:
        model_dict (dict): Dictionary with model and associated objects
        pitcher_data (pandas.DataFrame): DataFrame with pitcher features
        
    Returns:
        array: Predicted strikeouts
    """
    features = model_dict['features']
    scaler = model_dict['scaler']
    model = model_dict['model']
    train_years = model_dict.get('train_years')
    
    # Add time validation (optional but good practice)
    if 'season' in pitcher_data.columns and train_years:
        pred_seasons = pitcher_data['season'].unique()
        if any(season < min(train_years) for season in pred_seasons):
            logger.warning(f"Warning: Predicting for seasons {pred_seasons} " 
                          f"which include seasons earlier than training data ({train_years})")
    
    # Check if all needed features are available
    missing_features = [f for f in features if f not in pitcher_data.columns]
    if missing_features:
        logger.warning(f"Missing features for prediction: {missing_features}")
        return None
    
    # Prepare features
    X = pitcher_data[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

def predict_outs(model_dict, pitcher_data):
    """
    Predict outs for a pitcher
    
    Args:
        model_dict (dict): Dictionary with model and associated objects
        pitcher_data (pandas.DataFrame): DataFrame with pitcher features
        
    Returns:
        array: Predicted outs
    """
    features = model_dict['features']
    scaler = model_dict['scaler']
    model = model_dict['model']
    train_years = model_dict.get('train_years')
    
    # Add time validation (optional but good practice)
    if 'season' in pitcher_data.columns and train_years:
        pred_seasons = pitcher_data['season'].unique()
        if any(season < min(train_years) for season in pred_seasons):
            logger.warning(f"Warning: Predicting for seasons {pred_seasons} " 
                          f"which include seasons earlier than training data ({train_years})")
    
    # Check if all needed features are available
    missing_features = [f for f in features if f not in pitcher_data.columns]
    if missing_features:
        logger.warning(f"Missing features for prediction: {missing_features}")
        return None
    
    # Prepare features
    X = pitcher_data[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Round to nearest integer since outs is a count
    predictions_rounded = np.round(predictions)
    
    # Ensure no negative values
    predictions_rounded = np.maximum(predictions_rounded, 0)
    
    return predictions_rounded