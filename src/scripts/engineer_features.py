# src/scripts/engineer_features.py
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import time

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.features.pitch_features import create_pitcher_features
from src.features.batter_features import create_batter_features
from src.features.team_features import create_team_features, create_combined_features
from config import StrikeoutModelConfig

# Setup logger
logger = setup_logger('engineer_features')

def run_feature_engineering_pipeline():
    """Run the complete feature engineering pipeline with train/test separation"""
    start_time = time.time()
    
    # Define train/test seasons explicitly
    train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    
    logger.info(f"Running pipeline with train seasons {train_seasons}, test seasons {test_seasons}")
    
    try:
        # Split data by season BEFORE feature engineering
        with DBConnection() as conn:
            # Get pitcher data split by seasons
            train_pitcher_query = f"SELECT * FROM game_level_pitchers WHERE season IN {train_seasons}"
            train_pitcher_df = pd.read_sql_query(train_pitcher_query, conn)
            
            test_pitcher_query = f"SELECT * FROM game_level_pitchers WHERE season IN {test_seasons}"
            test_pitcher_df = pd.read_sql_query(test_pitcher_query, conn)
            
            # Get batter data split by seasons
            train_batter_query = f"SELECT * FROM game_level_batters WHERE season IN {train_seasons}"
            train_batter_df = pd.read_sql_query(train_batter_query, conn)
            
            test_batter_query = f"SELECT * FROM game_level_batters WHERE season IN {test_seasons}"
            test_batter_df = pd.read_sql_query(test_batter_query, conn)
        
        # Step 1: Create pitcher features separately for train/test
        logger.info("Creating pitcher features for train set...")
        train_pitcher_features = create_pitcher_features(train_pitcher_df, "train")
        
        logger.info("Creating pitcher features for test set...")
        test_pitcher_features = create_pitcher_features(test_pitcher_df, "test")
        
        # Step 2: Create batter features separately for train/test
        logger.info("Creating batter features for train set...")
        train_batter_features = create_batter_features(train_batter_df, "train")
        
        logger.info("Creating batter features for test set...")
        test_batter_features = create_batter_features(test_batter_df, "test")
        
        # Step 3: Create team features from train data only
        logger.info("Creating team features from training data only...")
        team_features = create_team_features(train_seasons)
        
        # Step 4: Create combined features separately
        logger.info("Creating combined features for train set...")
        train_combined = create_combined_features(train_pitcher_features, team_features, "train")
        
        logger.info("Creating combined features for test set...")
        test_combined = create_combined_features(test_pitcher_features, team_features, "test")
        
        pipeline_time = time.time() - start_time
        logger.info(f"Feature engineering pipeline completed in {pipeline_time:.2f} seconds.")
        return True
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting feature engineering pipeline...")
    success = run_feature_engineering_pipeline()
    if success:
        logger.info("Feature engineering pipeline completed successfully.")
    else:
        logger.error("Feature engineering pipeline failed.")