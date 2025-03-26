# Main coordination module for MLB pitcher prediction pipeline
import logging
import os
import pickle
from pathlib import Path
import pandas as pd
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pitcher_stats.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from src.data.db import (
    init_database, 
    store_statcast_data, 
    store_traditional_stats, 
    update_pitcher_mapping,
    export_dataset_to_csv,
    troubleshoot_database,
    examine_data_structure,
    clear_database
)
from src.data.fetch import get_statcast_data, get_traditional_stats
from src.features.engineering import create_prediction_features
from src.visualization.plots import create_visualizations

def main(force_refresh=False, clear_db=False):
    """
    Main function to run the entire pipeline
    
    Args:
        force_refresh (bool): Whether to force refresh data even if it exists
        clear_db (bool): Whether to clear the database before starting
    """
    start_time = time.time()
    logger.info("Starting pitcher performance analysis pipeline...")
    
    # Create directories if they don't exist
    Path("data").mkdir(exist_ok=True)
    Path("data/visualizations").mkdir(exist_ok=True, parents=True)
    
    # Initialize the database
    init_database()
    
    # Clear database if requested
    if clear_db:
        logger.info("Clearing database as requested...")
        clear_database()
    
    # 1. Fetch statcast data
    statcast_data = get_statcast_data(force_refresh=force_refresh)
    if statcast_data.empty:
        logger.error("No statcast data available. Exiting.")
        return
    
    # Examine raw data structure
    examine_data_structure(statcast_data)
    
    # 2. Fetch traditional pitching stats
    traditional_data = get_traditional_stats(force_refresh=force_refresh)
    if traditional_data.empty:
        logger.warning("No traditional pitching data available. Continuing with statcast data only.")
    
    # 3. Store data in the database
    store_statcast_data(statcast_data, force_refresh=force_refresh)
    if not traditional_data.empty:
        store_traditional_stats(traditional_data, force_refresh=force_refresh)
    
    # 4. Update pitcher ID mappings
    update_pitcher_mapping()
    
    # 5. Create prediction features
    create_prediction_features(force_refresh=force_refresh)
    
    # 6. Export final dataset to CSV
    final_data = export_dataset_to_csv()
    
    # 7. Create visualizations
    create_visualizations(final_data)
    
    # 8. Run diagnostics
    troubleshoot_database()
    
    # 9. Verify data types in the final dataset
    verify_data_quality(final_data)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {elapsed_time/60:.1f} minutes!")

def verify_data_quality(df):
    """
    Verify the quality of the data in the final dataset
    
    Args:
        df (pandas.DataFrame): The dataset to verify
    """
    logger.info("Verifying data quality...")
    
    # Check data shape
    logger.info(f"Dataset shape: {df.shape}")
    
    # Check for null values in key columns
    key_cols = ['pitcher_id', 'player_name', 'game_id', 'game_date', 'season', 'strikeouts']
    for col in key_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            logger.info(f"Column {col}: {null_count} null values ({null_count/len(df)*100:.2f}%)")
        else:
            logger.warning(f"Key column {col} not found in dataset")
    
    # Check for missing ERA values
    era_col = None
    for possible_col in ['era', 'ERA', 'era_x', 'era_y']:
        if possible_col in df.columns:
            era_col = possible_col
            break
    
    if era_col:
        era_null = df[era_col].isnull().sum()
        era_zero = (df[era_col] == 0).sum()
        logger.info(f"ERA column ({era_col}): {era_null} null values, {era_zero} zero values")
        
        if era_null == len(df):
            logger.error("All ERA values are null - traditional stats join likely failed")
        elif era_zero / len(df) > 0.9:
            logger.warning(f"Over 90% of ERA values are zero - check traditional stats processing")
    else:
        logger.warning("No ERA column found in dataset")
    
    # Check for pitch mix columns
    pitch_cols = [col for col in df.columns if col.startswith('pitch_pct_')]
    logger.info(f"Found {len(pitch_cols)} pitch mix columns")
    
    if pitch_cols:
        # Calculate percentage of rows with any pitch data
        has_pitch_data = df[pitch_cols].notna().any(axis=1).mean() * 100
        pitch_data_sum = df[pitch_cols].sum().sum()
        logger.info(f"{has_pitch_data:.2f}% of rows have pitch mix data")
        logger.info(f"Total sum of pitch mix percentages: {pitch_data_sum:.1f}")
        
        if has_pitch_data < 10:
            logger.warning("Less than 10% of rows have pitch mix data - check pitch type extraction")
        
        if pitch_data_sum < len(df) * 0.1:
            logger.warning("Pitch mix data sum is very low - possible extraction issue")
    else:
        logger.warning("No pitch mix columns found - pitch type extraction likely failed")
    
    # Check for prediction features
    feature_cols = [
        'last_3_games_strikeouts_avg', 'last_5_games_strikeouts_avg',
        'last_3_games_k9_avg', 'last_5_games_k9_avg',
        'last_3_games_era_avg', 'last_5_games_era_avg'
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Found {len(available_features)}/{len(feature_cols)} feature columns")
    
    for col in available_features:
        null_count = df[col].isnull().sum()
        zero_count = (df[col] == 0).sum()
        logger.info(f"Feature {col}: {null_count} nulls, {zero_count} zeros")
    
    # Check seasons distribution
    if 'season' in df.columns:
        season_counts = df['season'].value_counts().sort_index()
        logger.info(f"Seasons distribution: {season_counts.to_dict()}")
    
    # Verify the data is properly exported to CSV
    csv_path = Path("data/pitcher_game_level_data.csv")
    if csv_path.exists():
        logger.info(f"CSV file exists at {csv_path} with size {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Check if the file can be read
        try:
            test_df = pd.read_csv(csv_path, nrows=5)
            logger.info(f"CSV can be read successfully with {len(test_df.columns)} columns")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
    else:
        logger.error(f"CSV file not found at {csv_path}")
    
    logger.info("Data quality verification completed!")

def debug_mode():
    """Run in debug mode with verbose logging"""
    # Set more verbose logging
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)
    
    logger.debug("Running in debug mode with verbose logging")
    
    # Check cached data
    cache_files = [
        "data/statcast_pitcher_data.pkl",
        "data/traditional_pitcher_data.pkl"
    ]
    
    for file in cache_files:
        path = Path(file)
        if path.exists():
            logger.debug(f"Cache file {file} exists with size {path.stat().st_size / 1024 / 1024:.1f} MB")
            
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Successfully loaded {file} with shape {data.shape}")
                
                if hasattr(data, 'columns'):
                    logger.debug(f"Columns: {', '.join(data.columns[:20])}...")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        else:
            logger.debug(f"Cache file {file} does not exist")
    
    # Check database file
    db_path = Path("data/pitcher_stats.db")
    if db_path.exists():
        logger.debug(f"Database file exists with size {db_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Run diagnostics
        troubleshoot_database()
    else:
        logger.debug("Database file does not exist")
    
    logger.debug("Debug mode completed")

def quick_process():
    """Run a quick process to just export the dataset from existing database"""
    logger.info("Running quick process to export dataset...")
    
    # Check if database exists
    db_path = Path("data/pitcher_stats.db")
    if not db_path.exists():
        logger.error("Database file does not exist. Cannot run quick process.")
        return
    
    # Export the dataset
    final_data = export_dataset_to_csv()
    
    # Create visualizations
    create_visualizations(final_data)
    
    logger.info("Quick process completed!")

if __name__ == "__main__":
    main(force_refresh=False, clear_db=False)