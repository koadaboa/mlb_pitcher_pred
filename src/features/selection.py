# src/features/selection.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def select_features_for_strikeout_model(df):
    """
    Select relevant features for strikeout prediction model
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        
    Returns:
        list: Selected features for strikeout model
    """
    # Check what columns are available
    available_columns = df.columns.tolist()
    logger.info(f"Available columns: {available_columns}")
    
    # Try to find prediction features first
    prediction_features = [
        'last_3_games_strikeouts_avg', 
        'last_5_games_strikeouts_avg',
        'last_3_games_velo_avg', 
        'last_5_games_velo_avg',
        'last_3_games_swinging_strike_pct_avg', 
        'last_5_games_swinging_strike_pct_avg',
        'days_rest'
    ]
    
    # Check which prediction features are available
    available_pred_features = [f for f in prediction_features if f in available_columns]
    
    # If we don't have prediction features, use raw game features
    if not available_pred_features:
        logger.warning("No prediction features found. Using raw game features instead.")
        # Use these raw features as fallback
        raw_features = [
            'release_speed_mean',
            'release_speed_max',
            'release_spin_rate_mean',
            'swinging_strike_pct',
            'called_strike_pct',
            'zone_rate'
        ]
        available_features = [f for f in raw_features if f in available_columns]
    else:
        available_features = available_pred_features
    
    # Add pitch mix features if available
    pitch_mix_cols = [col for col in available_columns if col.startswith('pitch_pct_')]
    available_features.extend(pitch_mix_cols)
    
    logger.info(f"Selected {len(available_features)} features for strikeout model: {available_features}")
    return available_features