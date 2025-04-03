#!/usr/bin/env python
# src/scripts/export_engineered_dataset.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pybaseball
from tqdm import tqdm
import logging

from src.data.utils import setup_logger, ensure_dir
from src.data.db import get_pitcher_data
from src.features.selection import select_features
from config import StrikeoutModelConfig

logger = setup_logger(__name__, log_file="logs/export_dataset.log")

def export_dataset(output_dir=None, starters_only=True):
    """
    Export dataset with all engineered features
    
    Args:
        output_dir: Directory to save the CSV file
        starters_only: Whether to filter to only starting pitchers
        
    Returns:
        Path to exported CSV file
    """
    if output_dir is None:
        output_dir = Path("data/output")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Load complete pitcher data with all features
    logger.info("Loading pitcher data with all engineered features...")
    pitcher_data = get_pitcher_data(force_refresh=False)
    
    if pitcher_data.empty:
        logger.error("No pitcher data available. Make sure to run the data_pipeline first.")
        return None
    
    logger.info(f"Loaded {len(pitcher_data)} rows of pitcher data with {len(pitcher_data.columns)} columns")
    
    # Export to CSV
    file_path = output_dir / "pitcher_prediction_dataset.csv"
    pitcher_data.tail(20000).to_csv(file_path, index=False)
    
    logger.info(f"Exported {len(pitcher_data)} rows with {len(pitcher_data.columns)} columns to {file_path}")
    
    return file_path

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Export pitcher prediction dataset with all engineered features')
    parser.add_argument('--output-dir', type=str, default='data/output',
                       help='Directory to save exported files')
    
    args = parser.parse_args()
    
    file_path = export_dataset(args.output_dir)
    
    if file_path:
        print(f"Successfully exported dataset to {file_path}")
        return 0
    else:
        print("Failed to export dataset")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())