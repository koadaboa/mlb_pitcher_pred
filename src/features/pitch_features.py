#   src/features/pitch_features.py (Updated)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

#   Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from config import DataConfig, StrikeoutModelConfig

#   Setup logger
logger = setup_logger('pitch_features')

def load_game_level_data():
    """Load game-level pitcher data from database"""
    logger.info("Loading game-level pitcher data...")
    with DBConnection() as conn:
        query = "SELECT * FROM game_level_pitchers"
        df = pd.read_sql_query(query, conn)
    
    logger.info(f"Loaded {len(df)} rows of game-level pitcher data")
    return df

def create_additional_features(df):
    """Create additional features to improve model performance"""
    logger.info("Creating additional features...")
    
    #   Convert game_date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    #   Sort by pitcher_id and game_date
    df = df.sort_values(['pitcher_id', 'game_date'])
    
    #   Add more advanced rolling metrics with proper shifting to prevent leakage
    for window in StrikeoutModelConfig.WINDOW_SIZES:
        #   Group by pitcher_id
        grouped = df.groupby('pitcher_id')
        
        #   *** KEY FIX: SHIFT before calculating rolling metrics ***
        #   Calculate rolling K/9 with proper shifting
        shifted_strikeouts = grouped['strikeouts'].shift(1)
        shifted_innings = grouped['innings_pitched'].shift(1)
        
        #   Calculate rolling statistics on SHIFTED values
        rolling_k_sum = shifted_strikeouts.rolling(window, min_periods=1).sum()
        rolling_ip_sum = shifted_innings.rolling(window, min_periods=1).sum()
        
        #   Apply to dataframe
        rolling_k9 = (rolling_k_sum * 9 / rolling_ip_sum).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_k9'] = df.index.map(rolling_k9)
        
        #   Calculate rolling K% with shifting
        shifted_bf = grouped['batters_faced'].shift(1)
        rolling_bf_sum = shifted_bf.rolling(window, min_periods=1).sum()
        rolling_k_pct = (rolling_k_sum / rolling_bf_sum).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_k_pct'] = df.index.map(rolling_k_pct)
        
        #   Calculate rolling SwStr% with shifting
        shifted_swstr = grouped['swinging_strike_percent'].shift(1)
        rolling_swstr = shifted_swstr.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'rolling_{window}g_swstr_pct'] = df.index.map(rolling_swstr)
        
        #   Calculate velocity trend with shifting
        shifted_velo = grouped['avg_velocity'].shift(1)
        rolling_velo = shifted_velo.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'rolling_{window}g_velocity'] = df.index.map(rolling_velo)
        
        #   Calculate K standard deviation (variability) with shifting
        rolling_k_std = shifted_strikeouts.rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        df[f'rolling_{window}g_K_std'] = df.index.map(rolling_k_std)
    
    #   Create pitch mix trend features with shifting
    for pitch_type in ['fastball', 'breaking', 'offspeed']:
        for window in [3, 5]:
            #   Calculate rolling pitch usage with proper shifting
            shifted_usage = df.groupby('pitcher_id')[f'{pitch_type}_percent'].shift(1)
            rolling_usage = shifted_usage.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'rolling_{window}g_{pitch_type}_pct'] = df.index.map(rolling_usage)
    
    #   Fix career metrics - use expanding window on SHIFTED data
    df['career_k9'] = df.groupby('pitcher_id').apply(
        lambda x: x.sort_values('game_date').loc[:, 'strikeouts'].shift(1).expanding().sum() * 9 / 
                  x.sort_values('game_date').loc[:, 'innings_pitched'].shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)
    
    df['career_k_pct'] = df.groupby('pitcher_id').apply(
        lambda x: x.sort_values('game_date').loc[:, 'strikeouts'].shift(1).expanding().sum() / 
                  x.sort_values('game_date').loc[:, 'batters_faced'].shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)
    
    #   Remaining features remain mostly the same
    df['is_home'] = (df['home_team'] == df['away_team']).astype(int)
    df['K_last_game'] = df.groupby('pitcher_id')['strikeouts'].shift(1)
    df['days_since_last_game'] = df.groupby('pitcher_id')['game_date'].diff().dt.days
    
    #   Rest day features
    df['rest_days_4_less'] = ((df['days_since_last_game'] > 0) & 
                             (df['days_since_last_game'] <= 4)).astype(float)
    df['rest_days_5'] = (df['days_since_last_game'] == 5).astype(float)
    df['rest_days_6_more'] = (df['days_since_last_game'] >= 6).astype(float)
    
    #   Month indicators
    df['game_month'] = pd.to_datetime(df['game_date']).dt.month
    for month in range(3, 11):
        df[f'is_month_{month}'] = (df['game_month'] == month).astype(int)
    
    #   Calculate recent form vs. career average
    df['recent_vs_career_k9'] = df['rolling_5g_k9'] / df['career_k9']
    df['throws_right'] = (df['p_throws'] == 'R').astype(float)
    
    #   *** NEW PITCHER FEATURES ***
    #   Pitch Type Sequencing (simplified - requires pitch-level data for more detail)
    #   This is a very basic approximation.  A more sophisticated approach
    #   would require access to the raw pitch-by-pitch data to analyze sequences.
    for pitch_type in ['fastball_percent', 'breaking_percent', 'offspeed_percent']:
        df[f'lag_1_{pitch_type}'] = df.groupby('pitcher_id')[pitch_type].shift(1)
        df[f'lag_2_{pitch_type}'] = df.groupby('pitcher_id')[pitch_type].shift(2)
    
    #   Game Context Features (example - requires joining with game-level data)
    #   These are placeholders.  You'll need to join with other tables
    #   to get actual game context (or calculate it earlier).
    df['inning'] = 5  #   Placeholder
    df['score_differential'] = 0  #   Placeholder
    df['is_close_game'] = (df['score_differential'].abs() <= 2).astype(int)  #   Placeholder
    df['is_playoff'] = 0  #   Placeholder
    
    logger.info(f"Created additional features. Total features: {len(df.columns)}")
    return df

def create_pitcher_features(df=None, dataset_type="all"):
    """Main function to create pitcher features
    
    Args:
        df (pandas.DataFrame, optional): Game-level data to process. If None, load from DB.
        dataset_type (str): Type of dataset ("train", "test", or "all")
    
    Returns:
        pandas.DataFrame: DataFrame with pitcher features
    """
    try:
        #   Load game-level data if not provided
        if df is None:
            logger.info("Loading game-level pitcher data...")
            with DBConnection() as conn:
                query = "SELECT * FROM game_level_pitchers"
                df = pd.read_sql_query(query, conn)
            
            logger.info(f"Loaded {len(df)} rows of game-level pitcher data")
        
        #   Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        #   Create additional features
        df = create_additional_features(df)
        
        #   Handle missing values
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].isnull().sum() > 0:
                logger.info(f"Filling {df[col].isnull().sum()} missing values in {col}")
                df[col] = df[col].fillna(df[col].median())
        
        #   Store to database with appropriate table name
        table_name = "predictive_pitch_features"
        if dataset_type == "train":
            table_name = "train_predictive_pitch_features"
        elif dataset_type == "test":
            table_name = "test_predictive_pitch_features"
        
        with DBConnection() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(df)} rows of predictive pitcher features to {table_name}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error creating pitcher features: {str(e)}")
        return pd.DataFrame()

def main():
    """Run the pitcher feature creation process"""
    logger.info("Starting pitcher feature creation process...")
    df = create_pitcher_features()
    if not df.empty:
        logger.info("Pitcher feature creation completed successfully")
        return True
    else:
        logger.error("Pitcher feature creation failed")
        return False

if __name__ == "__main__":
    main()