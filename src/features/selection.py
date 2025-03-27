# src/features/selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

def select_features_for_strikeout_model(df):
    """
    Select relevant features for strikeout prediction model
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        
    Returns:
        pandas.DataFrame: Selected features for strikeout model
    """
    # Basic features always included
    basic_features = [
        'pitcher_id', 'player_name', 'game_date', 'season',
        'last_3_games_strikeouts_avg', 'last_5_games_strikeouts_avg',
        'last_3_games_velo_avg', 'last_5_games_velo_avg',
        'last_3_games_swinging_strike_pct_avg', 'last_5_games_swinging_strike_pct_avg',
        'days_rest', 'team_changed'
    ]
    
    # Pitch mix features - select all available
    pitch_mix_cols = [col for col in df.columns if col.startswith('pitch_pct_')]
    
    # Combine all features
    all_features = basic_features + pitch_mix_cols
    
    # Select only available columns
    available_features = [col for col in all_features if col in df.columns]
    
    # Target variable
    target = 'strikeouts'
    
    if target in df.columns:
        available_features.append(target)
    
    # Select the features
    selected_df = df[available_features].copy()
    
    # Handle missing values
    selected_df.fillna(0, inplace=True)
    
    logger.info(f"Selected {len(available_features)} features for strikeout model")
    return selected_df

def select_features_for_outs_model(df):
    """
    Select relevant features for outs prediction model
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        
    Returns:
        pandas.DataFrame: Selected features for outs model
    """
    # Basic features always included
    basic_features = [
        'pitcher_id', 'player_name', 'game_date', 'season',
        'last_3_games_outs_avg', 'last_5_games_outs_avg',
        'last_3_games_strikeouts_avg', 'last_5_games_strikeouts_avg',
        'last_3_games_velo_avg', 'last_5_games_velo_avg',
        'last_3_games_swinging_strike_pct_avg', 'last_5_games_swinging_strike_pct_avg',
        'days_rest', 'team_changed'
    ]
    
    # Pitch mix features - select all available
    pitch_mix_cols = [col for col in df.columns if col.startswith('pitch_pct_')]
    
    # Combine all features
    all_features = basic_features + pitch_mix_cols
    
    # Select only available columns
    available_features = [col for col in all_features if col in df.columns]
    
    # Target variable
    target = 'outs'
    
    if target in df.columns:
        available_features.append(target)
    
    # Select the features
    selected_df = df[available_features].copy()
    
    # Handle missing values
    selected_df.fillna(0, inplace=True)
    
    logger.info(f"Selected {len(available_features)} features for outs model")
    return selected_df