# 2. data_processing.py - Aggregate and transform data

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.db import update_pitcher_mapping, store_processed_data
from src.data.process import aggregate_to_game_level, merge_traditional_stats
from src.data.fetch import get_statcast_data, get_traditional_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(force_refresh=False):
    """
    Process raw data into analysis-ready format
    
    Args:
        force_refresh (bool): Whether to force refresh existing data
    """
    # 1. Get raw data from database
    statcast_data = get_statcast_data()
    traditional_data = get_traditional_stats()
    
    if statcast_data.empty:
        logger.error("No raw statcast data found in database.")
        return False
    
    # 2. Aggregate statcast data to pitcher-game level
    logger.info("Aggregating statcast data to pitcher-game level...")
    game_level = aggregate_to_game_level(statcast_data)
    
    # 3. Merge with traditional stats if available
    if not traditional_data.empty:
        logger.info("Merging with traditional stats...")
        game_level = merge_traditional_stats(game_level, traditional_data)
    
    # 4. Update pitcher ID mappings
    logger.info("Updating pitcher ID mappings...")
    update_pitcher_mapping()
    
    # 5. Store processed data
    logger.info("Storing processed data...")
    store_processed_data(game_level, force_refresh=force_refresh)
    
    # 6. Export to CSV for convenience
    output_path = Path("data/processed_pitcher_data.csv")
    game_level.to_csv(output_path, index=False)
    logger.info(f"Exported processed data to {output_path}")
    
    logger.info("Data processing completed successfully!")
    return True

if __name__ == "__main__":
    process_data(force_refresh=False)