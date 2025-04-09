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

# Setup logger
logger = setup_logger('engineer_features')

def run_feature_engineering_pipeline():
    """Run the complete feature engineering pipeline"""
    start_time = time.time()
    
    try:
        # Step 1: Create pitcher features
        logger.info("Step 1: Creating pitcher features...")
        pitcher_df = create_pitcher_features()
        if pitcher_df.empty:
            logger.error("Failed to create pitcher features. Aborting pipeline.")
            return False
        logger.info("Pitcher features created successfully.")
        
        # Step 2: Create batter features
        logger.info("Step 2: Creating batter features...")
        batter_df = create_batter_features()
        if batter_df.empty:
            logger.warning("No batter features created. Continuing pipeline.")
        else:
            logger.info("Batter features created successfully.")
        
        # Step 3: Create team features
        logger.info("Step 3: Creating team features...")
        team_df = create_team_features()
        if team_df.empty:
            logger.warning("No team features created. Continuing pipeline.")
        else:
            logger.info("Team features created successfully.")
        
        # Step 4: Create combined features for modeling
        logger.info("Step 4: Creating combined features...")
        combined_df = create_combined_features()
        if combined_df.empty:
            logger.error("Failed to create combined features. Pipeline may be incomplete.")
            return False
        logger.info("Combined features created successfully.")
        
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