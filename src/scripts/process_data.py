# src/scripts/data_processing.py
import logging
import pandas as pd
from pathlib import Path

from src.data.fetch import get_statcast_data
from src.features.engineering import create_prediction_features
from src.data.process import export_data_to_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(force_refresh=False):
    """
    Process raw data into analysis-ready format for strikeout prediction
    
    Args:
        force_refresh (bool): Whether to force refresh existing data
    """
    # 1. Create prediction features
    logger.info("Creating prediction features...")
    create_prediction_features(force_refresh=force_refresh)
    
    # 2. Export data to CSV files for analysis
    logger.info("Exporting processed data to CSV...")
    exported_files = export_data_to_csv()
    
    logger.info("Data processing completed successfully!")
    return True

if __name__ == "__main__":
    process_data(force_refresh=False)