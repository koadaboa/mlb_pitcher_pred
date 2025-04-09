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

def create_batter_features():
    """Create batter features from pre-aggregated game-level data"""
    try:
        # Load game-level batter data
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
        
        # Calculate rolling window features for each batter
        logger.info("Creating rolling window features...")
        
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
                
            # Calculate rolling features for each window size
            for window in window_sizes:
                # Rolling strikeout rate
                batter_games[f'rolling_{window}g_k_pct'] = (
                    batter_games['strikeouts'].rolling(window, min_periods=1).sum() /
                    batter_games['total_pitches'].rolling(window, min_periods=1).sum() * 100
                )
                
                # Rolling swinging strike rate
                batter_games[f'rolling_{window}g_swstr_pct'] = (
                    batter_games['swinging_strikes'].rolling(window, min_periods=1).sum() /
                    batter_games['total_pitches'].rolling(window, min_periods=1).sum() * 100
                )
                
                # Rolling chase rate
                batter_games[f'rolling_{window}g_chase_pct'] = (
                    batter_games['chase_pct'].rolling(window, min_periods=1).mean()
                )
                
                # Rolling zone contact rate
                batter_games[f'rolling_{window}g_zone_contact_pct'] = (
                    batter_games['zone_contact_pct'].rolling(window, min_periods=1).mean()
                )
                
                # By pitch type
                batter_games[f'rolling_{window}g_fb_whiff_pct'] = (
                    batter_games['fastball_whiff_pct'].rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_breaking_whiff_pct'] = (
                    batter_games['breaking_whiff_pct'].rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_offspeed_whiff_pct'] = (
                    batter_games['offspeed_whiff_pct'].rolling(window, min_periods=1).mean()
                )
                
                # By handedness
                rhp_k = batter_games['strikeouts_vs_rhp'].rolling(window, min_periods=1).sum()
                lhp_k = batter_games['strikeouts_vs_lhp'].rolling(window, min_periods=1).sum()
                
                # Add the handedness features when we have enough data
                if rhp_k.sum() > 0:
                    batter_games[f'rolling_{window}g_k_vs_rhp'] = rhp_k
                
                if lhp_k.sum() > 0:
                    batter_games[f'rolling_{window}g_k_vs_lhp'] = lhp_k
            
            # Add to collection
            all_features.append(batter_games)
        
        # Combine all batter features
        batter_features = pd.concat(all_features, ignore_index=True)
        
        # Fill missing values
        for col in batter_features.select_dtypes(include=['float64']).columns:
            if batter_features[col].isnull().sum() > 0:
                logger.info(f"Filling {batter_features[col].isnull().sum()} missing values in {col}")
                batter_features[col] = batter_features[col].fillna(batter_features[col].median())
        
        # Store to database
        with DBConnection() as conn:
            batter_features.to_sql('batter_predictive_features', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(batter_features)} batter features to database")
        
        return batter_features
        
    except Exception as e:
        logger.error(f"Error creating batter features: {str(e)}")
        return pd.DataFrame()