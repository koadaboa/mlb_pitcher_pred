# src/features/selection.py
import pandas as pd
import numpy as np
from src.data.utils import setup_logger

logger = setup_logger(__name__)

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
    
    # Define feature groups
    base_features = [
        'last_3_games_strikeouts_avg', 
        'last_5_games_strikeouts_avg',
        'last_3_games_velo_avg', 
        'last_5_games_velo_avg',
        'last_3_games_swinging_strike_pct_avg', 
        'last_5_games_swinging_strike_pct_avg',
        'days_rest'
    ]
    
    standard_dev_features = [
        'last_3_games_strikeouts_std', 
        'last_5_games_strikeouts_std',
        'last_3_games_velo_std', 
        'last_5_games_velo_std',
        'last_3_games_swinging_strike_pct_std', 
        'last_5_games_swinging_strike_pct_std'
    ]
    
    trend_features = [
        'trend_3_strikeouts', 
        'trend_5_strikeouts',
        'trend_3_release_speed_mean', 
        'trend_5_release_speed_mean',
        'trend_3_swinging_strike_pct', 
        'trend_5_swinging_strike_pct'
    ]
    
    momentum_features = [
        'momentum_3_strikeouts', 
        'momentum_5_strikeouts',
        'momentum_3_release_speed_mean', 
        'momentum_5_release_speed_mean',
        'momentum_3_swinging_strike_pct', 
        'momentum_5_swinging_strike_pct'
    ]
    
    entropy_features = [
        'pitch_entropy', 
        'prev_game_pitch_entropy'
    ]
    
    additional_features = [
        'last_3_games_called_strike_pct_avg', 
        'last_5_games_called_strike_pct_avg',
        'last_3_games_zone_rate_avg', 
        'last_5_games_zone_rate_avg'
    ]
    
    # NEW: Pitcher-specific baseline features
    pitcher_baseline_features = [
        'career_so_avg',
        'career_so_std',
        'career_so_per_batter',
        'career_so_consistency',
        'prev_so_deviation',
        'so_deviation_3g_avg',
        'so_deviation_5g_avg',
        'is_home_game',
        'home_away_so_exp'
    ]
    
    # NEW: Enhanced matchup features
    matchup_features = [
        'opponent_strikeout_rate',
        'opponent_whiff_rate',
        'opponent_chase_rate',
        'opponent_zone_contact_rate',
        'opponent_k_vs_avg',
        'opponent_whiff_vs_avg',
        'matchup_advantage',
        'recency_weighted_matchup'
    ]
    
    # If we don't have prediction features, use raw game features
    raw_features = [
        'release_speed_mean',
        'release_speed_max',
        'release_spin_rate_mean',
        'swinging_strike_pct',
        'called_strike_pct',
        'zone_rate'
    ]
    
    # Combine all feature groups
    all_prediction_features = (
        base_features + 
        standard_dev_features + 
        trend_features + 
        momentum_features + 
        entropy_features + 
        additional_features +
        pitcher_baseline_features +  # New features
        matchup_features            # New features
    )
    
    # Check which prediction features are available
    available_pred_features = [f for f in all_prediction_features if f in available_columns]
    
    if not available_pred_features:
        logger.warning("No prediction features found. Using raw game features instead.")
        available_features = [f for f in raw_features if f in available_columns]
    else:
        available_features = available_pred_features
    
    # Add pitch mix features if available
    pitch_mix_cols = [col for col in available_columns if col.startswith('prev_game_pitch_pct_')]
    available_features.extend(pitch_mix_cols)
    
    # Add opponent-specific history features
    opponent_history_cols = [col for col in available_columns if col.startswith('so_vs_')]
    available_features.extend(opponent_history_cols)
    
    logger.info(f"Selected {len(available_features)} features for strikeout model: {available_features}")
    return available_features