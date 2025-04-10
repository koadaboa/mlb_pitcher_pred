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

#   src/features/batter_features.py
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

#   Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from config import StrikeoutModelConfig

#   Setup logger
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
        #   Load game-level batter data if not provided
        if df is None:
            logger.info("Loading game-level batter data...")
            with DBConnection() as conn:
                query = "SELECT * FROM game_level_batters"
                df = pd.read_sql_query(query, conn)
            
            if df.empty:
                logger.error("No game-level batter data found. Run create_game_level_batters.py first.")
                return pd.DataFrame()
                
            logger.info(f"Loaded {len(df)} rows of game-level batter data")
        
        #   Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        #   Sort by batter_id and game_date
        df = df.sort_values(['batter_id', 'game_date'])
        
        #   Container for all features
        all_features = []
        
        #   Rolling window sizes
        window_sizes = StrikeoutModelConfig.WINDOW_SIZES
        
        #   Process each batter
        unique_batters = df['batter_id'].unique()
        logger.info(f"Processing features for {len(unique_batters)} unique batters")
        
        for batter_id in unique_batters:
            #   Get all games for this batter
            batter_games = df[df['batter_id'] == batter_id].copy()
            
            if len(batter_games) <= 1:
                continue
                
            #   *** KEY FIX: Use shifted values before calculating rolling windows ***
            for window in window_sizes:
                #   Shift values first to prevent leakage
                shifted_strikeouts = batter_games['strikeouts'].shift(1)
                shifted_pitches = batter_games['total_pitches'].shift(1)
                shifted_swinging_strikes = batter_games['swinging_strikes'].shift(1)
                
                #   Rolling strikeout rate with shifted data
                batter_games[f'rolling_{window}g_k_pct'] = (
                    shifted_strikeouts.rolling(window, min_periods=1).sum() /
                    shifted_pitches.rolling(window, min_periods=1).sum() * 100
                )
                
                #   Rolling swinging strike rate with shifted data
                batter_games[f'rolling_{window}g_swstr_pct'] = (
                    shifted_swinging_strikes.rolling(window, min_periods=1).sum() /
                    shifted_pitches.rolling(window, min_periods=1).sum() * 100
                )
                
                #   Other metrics with shifting
                batter_games[f'rolling_{window}g_chase_pct'] = (
                    batter_games['chase_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_zone_contact_pct'] = (
                    batter_games['zone_contact_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                #   By pitch type
                batter_games[f'rolling_{window}g_fb_whiff_pct'] = (
                    batter_games['fastball_whiff_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_breaking_whiff_pct'] = (
                    batter_games['breaking_whiff_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
                
                batter_games[f'rolling_{window}g_offspeed_whiff_pct'] = (
                    batter_games['offspeed_whiff_pct'].shift(1).rolling(window, min_periods=1).mean()
                )
            
            #   *** NEW BATTER FEATURES ***
            #   Need to ensure these columns exist in your game_level_batters table
            if all(col in batter_games.columns for col in ['hits', 'doubles', 'triples', 'home_runs', 'walks', 'at_bats', 'sacrifice_flies']):
                #   Calculate OPS components (ensure no division by zero)
                batter_games['singles'] = batter_games['hits'] - batter_games['doubles'] - batter_games['triples'] - batter_games['home_runs']
                
                safe_ab = batter_games['at_bats'].replace(0, 1)  #   Avoid division by zero
                safe_pa = (batter_games['at_bats'] + batter_games['walks'] + batter_games['sacrifice_flies']).replace(0, 1)
                
                batter_games['ba'] = batter_games['hits'] / safe_ab
                batter_games['obp'] = (batter_games['hits'] + batter_games['walks']) / safe_pa
                batter_games['slg'] = (batter_games['singles'] + 2 * batter_games['doubles'] + 3 * batter_games['triples'] + 4 * batter_games['home_runs']) / safe_ab
                batter_games['ops'] = batter_games['obp'] + batter_games['slg']
                
                #   wOBA calculation (simplified - you might need more complex weights)
                batter_games['woba'] = (0.7*batter_games['walks'] + 0.9*batter_games['singles'] + 1.25*batter_games['doubles'] + 1.6*batter_games['triples'] + 2.0*batter_games['home_runs']) / safe_pa
                
                for window in window_sizes:
                    batter_games[f'rolling_{window}g_ops'] = batter_games['ops'].shift(1).rolling(window, min_periods=1).mean()
                    batter_games[f'rolling_{window}g_woba'] = batter_games['woba'].shift(1).rolling(window, min_periods=1).mean()
                    
                    #   Basic hitting streak (games with a hit)
                    batter_games['has_hit'] = (batter_games['hits'] > 0).astype(int)
                    batter_games['hitting_streak'] = batter_games['has_hit'].rolling(window, min_periods=1).sum()
                    batter_games[f'rolling_{window}g_hitting_streak'] = batter_games['hitting_streak'].shift(1).rolling(window, min_periods=1).max()
                    
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