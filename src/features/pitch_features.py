# src/features/pitch_features.py
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from config import DataConfig, StrikeoutModelConfig

# Setup logger
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
    
    # Convert game_date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Add more advanced rolling metrics
    for window in StrikeoutModelConfig.WINDOW_SIZES:
        # Create group key for proper rolling calculations
        df = df.sort_values(['pitcher_id', 'game_date'])
        
        # Calculate rolling K/9
        rolling_k9 = (df.groupby('pitcher_id')['strikeouts']
                      .rolling(window, min_periods=1)
                      .sum() * 9 / 
                      df.groupby('pitcher_id')['innings_pitched']
                      .rolling(window, min_periods=1)
                      .sum()).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_k9'] = df.index.map(rolling_k9)
        
        # Calculate rolling K%
        rolling_k_pct = (df.groupby('pitcher_id')['strikeouts']
                        .rolling(window, min_periods=1)
                        .sum() / 
                        df.groupby('pitcher_id')['batters_faced']
                        .rolling(window, min_periods=1)
                        .sum()).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_k_pct'] = df.index.map(rolling_k_pct)
        
        # Calculate rolling SwStr%
        rolling_swstr = (df.groupby('pitcher_id')['swinging_strike_percent']
                         .rolling(window, min_periods=1)
                         .mean()).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_swstr_pct'] = df.index.map(rolling_swstr)
        
        # Calculate velocity trend
        rolling_velo = (df.groupby('pitcher_id')['avg_velocity']
                        .rolling(window, min_periods=1)
                        .mean()).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_velocity'] = df.index.map(rolling_velo)
        
        # Calculate K standard deviation (variability)
        rolling_k_std = (df.groupby('pitcher_id')['strikeouts']
                         .rolling(window, min_periods=2)
                         .std()).reset_index(level=0, drop=True)
        df[f'rolling_{window}g_K_std'] = df.index.map(rolling_k_std)
    
    # Create pitch mix trend features
    for pitch_type in ['fastball', 'breaking', 'offspeed']:
        for window in [3, 5]:
            # Calculate rolling pitch usage
            rolling_usage = (df.groupby('pitcher_id')[f'{pitch_type}_percent']
                            .rolling(window, min_periods=1)
                            .mean()).reset_index(level=0, drop=True)
            df[f'rolling_{window}g_{pitch_type}_pct'] = df.index.map(rolling_usage)
    
    # Create career metrics
    career_k9 = df.groupby('pitcher_id').apply(
        lambda x: x['strikeouts'].sum() * 9 / x['innings_pitched'].sum()
    ).to_dict()
    df['career_k9'] = df['pitcher_id'].map(career_k9)
    
    career_k_pct = df.groupby('pitcher_id').apply(
        lambda x: x['strikeouts'].sum() / x['batters_faced'].sum()
    ).to_dict()
    df['career_k_pct'] = df['pitcher_id'].map(career_k_pct)
    
    # Create home/away feature
    # In reality, you'd compare with the pitcher's team, but as a placeholder:
    df['is_home'] = (df['home_team'] == df['away_team']).astype(int)
    
    # Create streak features
    df['K_last_game'] = df.groupby('pitcher_id')['strikeouts'].shift(1)
    
    # Calculate days since last game
    df['days_since_last_game'] = df.groupby('pitcher_id')['game_date'].diff().dt.days
    
    # Create rest day categorical features
    df['rest_days_4_less'] = ((df['days_since_last_game'] > 0) & 
                              (df['days_since_last_game'] <= 4)).astype(float)
    df['rest_days_5'] = (df['days_since_last_game'] == 5).astype(float)
    df['rest_days_6_more'] = (df['days_since_last_game'] >= 6).astype(float)
    
    # Create season month indicators
    df['game_month'] = pd.to_datetime(df['game_date']).dt.month
    for month in range(3, 11):  # MLB season months
        df[f'is_month_{month}'] = (df['game_month'] == month).astype(int)
    
    # Calculate recent form vs. career average
    df['recent_vs_career_k9'] = df['rolling_5g_k9'] / df['career_k9']
    
    # Create handedness feature
    df['throws_right'] = (df['p_throws'] == 'R').astype(float)
    
    logger.info(f"Created additional features. Total features: {len(df.columns)}")
    return df

def create_pitcher_features():
    """Main function to create pitcher features"""
    try:
        # Load game-level data
        df = load_game_level_data()
        
        # Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Create additional features
        df = create_additional_features(df)
        
        # Handle missing values
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].isnull().sum() > 0:
                logger.info(f"Filling {df[col].isnull().sum()} missing values in {col}")
                df[col] = df[col].fillna(df[col].median())
        
        # Store to database
        with DBConnection() as conn:
            df.to_sql('predictive_pitch_features', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(df)} rows of predictive pitcher features to database")
        
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