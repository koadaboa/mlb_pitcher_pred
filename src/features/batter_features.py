# src/features/batter_features.py (updated version)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from config import StrikeoutModelConfig

# Setup logger
logger = setup_logger('batter_features')

def create_batter_features(df=None, dataset_type="all"):
    """Create batter features from pre-aggregated game-level data
    
    Args:
        df (pandas.DataFrame, optional): Game-level data to process. If None, load from DB.
        dataset_type (str): Type of dataset ("train", "test", or "all")
    
    Returns:
        pandas.DataFrame: DataFrame with batter features
    """
    try:
        # Load game-level batter data if not provided
        if df is None:
            logger.info("Loading game-level batter data...")
            with DBConnection() as conn:
                query = "SELECT * FROM game_level_batters"
                df = pd.read_sql_query(query, conn)
            
            if df.empty:
                logger.error("No game-level batter data found. Run create_game_level_batters.py first.")
                return pd.DataFrame()
                
            logger.info(f"Loaded {len(df)} rows of game-level batter data")
        
        # Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Sort by batter_id and game_date
        df = df.sort_values(['batter_id', 'game_date'])
        
        # Container for all features
        all_features = []
        
        # Rolling window sizes
        window_sizes = StrikeoutModelConfig.WINDOW_SIZES
        
        # Process each batter
        unique_batters = df['batter_id'].unique()
        logger.info(f"Processing features for {len(unique_batters)} unique batters")
        
        for batter_id in unique_batters:
            # Get all games for this batter
            batter_games = df[df['batter_id'] == batter_id].copy()
            
            if len(batter_games) <= 1:
                continue
                
            # *** KEY FIX: Use shifted values before calculating rolling windows ***
            for window in window_sizes:
                # Shift values first to prevent leakage
                shifted_strikeouts = batter_games['strikeouts'].shift(1)
                shifted_pitches = batter_games['total_pitches'].shift(1)
                shifted_swinging_strikes = batter_games['swinging_strikes'].shift(1)
                
                # Rolling strikeout rate with shifted data
                batter_games[f'rolling_{window}g_k_pct'] = (
                    shifted_strikeouts.rolling(window, min_periods=1).sum() /
                    shifted_pitches.rolling(window, min_periods=1).sum() * 100
                )
                
                # Rolling swinging strike rate with shifted data
                batter_games[f'rolling_{window}g_swstr_pct'] = (
                    shifted_swinging_strikes.rolling(window, min_periods=1).sum() /
                    shifted_pitches.rolling(window, min_periods=1).sum() * 100
                )
                
                # Other metrics with shifting
                batter_games[f'rolling_{window}g_chase_pct'] = (
                    batter_games['chase_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_zone_contact_pct'] = (
                    batter_games['zone_contact_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                # By pitch type
                batter_games[f'rolling_{window}g_fb_whiff_pct'] = (
                    batter_games['fastball_whiff_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_breaking_whiff_pct'] = (
                    batter_games['breaking_whiff_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_offspeed_whiff_pct'] = (
                    batter_games['offspeed_whiff_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
            
            all_features.append(batter_games)
        
        # Combine all batter features
        batter_features = pd.concat(all_features, ignore_index=True)
        
        # Fill missing values
        for col in batter_features.select_dtypes(include=['float64']).columns:
            if batter_features[col].isnull().sum() > 0:
                logger.info(f"Filling {batter_features[col].isnull().sum()} missing values in {col}")
                batter_features[col] = batter_features[col].fillna(batter_features[col].median())
        
        # Store to database
        table_name = "batter_predictive_features"
        if dataset_type == "train":
            table_name = "train_batter_predictive_features"
        elif dataset_type == "test":
            table_name = "test_batter_predictive_features"
            
        with DBConnection() as conn:
            batter_features.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(batter_features)} batter features to {table_name}")
        
        return batter_features
        
    except Exception as e:
        logger.error(f"Error creating batter features: {str(e)}")
        return pd.DataFrame()