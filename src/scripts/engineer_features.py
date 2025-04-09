"""
Script for engineering predictive features for pitcher strikeout prediction.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.utils import setup_logger
from src.features.pitch_features import (
    create_predictive_features, 
    store_predictive_features,
    calculate_rolling_features,
    create_days_rest_features,
    create_handedness_features,
    combine_features,
    prepare_final_features
)
from src.data.aggregate_pitchers import aggregate_pitchers_to_game_level, get_game_level_pitcher_stats
from config import StrikeoutModelConfig, DBConfig

logger = setup_logger(__name__)

def create_and_store_features(limit=None, seasons=None, batch_size=None, rebuild_tables=False):
    """
    Create and store predictive features for pitcher strikeout prediction.
    
    Args:
        limit (int): Optional limit on rows to process
        seasons (list): Optional list of seasons to include
        batch_size (int): Optional batch size for processing large datasets
        rebuild_tables (bool): If True, rebuild existing tables
        
    Returns:
        bool: Success status
    """
    try:
        logger.info("Starting feature engineering process")
        
        # Aggregate pitch data to game level if needed
        # This ensures we have the game_level_pitchers table
        logger.info("Aggregating pitch data to game level")
        aggregate_success = aggregate_pitchers_to_game_level(force_rebuild=rebuild_tables)
        
        if not aggregate_success:
            logger.error("Failed to aggregate pitcher data to game level")
            return False
        
        # Use specified seasons or default
        if seasons is None:
            seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
        
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = DBConfig.BATCH_SIZE
            
        logger.info(f"Using batch size: {batch_size}")
        logger.info(f"Using seasons: {seasons}")
        
        # Get game level data
        logger.info("Retrieving game-level pitchers data")
        game_data = get_game_level_pitcher_stats(limit=limit, seasons=seasons)
        
        if game_data.empty:
            logger.error("Failed to retrieve game-level pitcher data")
            return False
        
        logger.info(f"Retrieved {len(game_data)} game-level records")
        
        # Create advanced predictive features
        logger.info("Creating advanced predictive features")
        
        # Convert game_date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(game_data['game_date']):
            game_data['game_date'] = pd.to_datetime(game_data['game_date'])
        
        # Calculate rolling features
        logger.info("Calculating rolling features")
        game_data = calculate_rolling_features(game_data)
        
        # Create rest day features
        logger.info("Creating rest day features")
        game_data = create_days_rest_features(game_data)
        
        # Create handedness features
        logger.info("Creating handedness features")
        game_data = create_handedness_features(game_data)
        
        # Combine features to create interaction terms
        logger.info("Creating feature interactions")
        game_data = combine_features(game_data)
        
        # Prepare final feature set (apply shifting to prevent data leakage)
        logger.info("Preparing final feature set")
        final_features = prepare_final_features(game_data)
        
        # Store features
        logger.info("Storing predictive features")
        success = store_predictive_features(final_features)
        
        if success:
            logger.info(f"Feature engineering completed successfully, created {len(final_features)} feature records")
        else:
            logger.error("Feature engineering failed during storage")
            
        return success
        
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        return False

def main():
    """Main function to parse arguments and execute feature engineering."""
    parser = argparse.ArgumentParser(description='Create predictive features for strikeout prediction')
    parser.add_argument('--limit', type=int, help='Limit on number of rows to process')
    parser.add_argument('--seasons', nargs='+', type=int, help='Seasons to include')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild tables even if they exist')
    
    args = parser.parse_args()
    
    create_and_store_features(
        limit=args.limit, 
        seasons=args.seasons, 
        batch_size=args.batch_size,
        rebuild_tables=args.rebuild
    )

if __name__ == "__main__":
    main()