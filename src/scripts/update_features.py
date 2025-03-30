# src/scripts/update_features.py
import os
from pathlib import Path

from src.data.db import get_db_connection, update_database_schema
from src.features.engineering import create_enhanced_features, create_prediction_features
from src.data.process import export_data_to_csv
from src.data.utils import setup_logger

logger = setup_logger(__name__)

def update_features(force_refresh=False):
    """
    Update database schema and calculate enhanced features
    
    Args:
        force_refresh (bool): Whether to force refresh existing features
    """
    # 1. Update database schema to support new features
    logger.info("Updating database schema...")
    update_database_schema()
    
    # 2. Create enhanced prediction features
    logger.info("Creating enhanced prediction features...")
    create_prediction_features(force_refresh=force_refresh)
    
    # 3. Export updated data to CSV
    logger.info("Exporting processed data to CSV...")
    exported_files = export_data_to_csv()
    
    logger.info("Feature update completed successfully!")
    return True

if __name__ == "__main__":
    update_features(force_refresh=True)