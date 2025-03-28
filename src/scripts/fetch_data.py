# src/scripts/data_acquisition.py

import logging
import os
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import necessary functions
from src.data.fetch import get_statcast_data
from src.data.db import init_database, store_statcast_data

def fetch_and_store_data(force_refresh=False):
    """Fetch and store data from external sources"""
    # Create directories if needed
    Path("data").mkdir(exist_ok=True)
    
    # Initialize the database
    init_database()
    
    # 1. Fetch statcast data
    logger.info("Fetching Statcast data...")
    statcast_data = get_statcast_data(force_refresh=force_refresh)
    if statcast_data.empty:
        logger.error("No statcast data available. Exiting.")
        return False
    
    # 2. Store raw data in the database
    logger.info("Storing raw data in database...")
    try:
        store_statcast_data(statcast_data, force_refresh=force_refresh)
        
        logger.info("Data acquisition completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error storing data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    fetch_and_store_data(force_refresh=False)