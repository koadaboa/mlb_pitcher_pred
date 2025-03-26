# Main coordination module for MLB pitcher prediction pipeline
import logging
import os
from pathlib import Path
import pandas as pd

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
from ..data.db import (
    init_database, 
    store_statcast_data, 
    store_traditional_stats, 
    update_pitcher_mapping,
    export_dataset_to_csv
)
from ..data.fetch import get_statcast_data, get_traditional_stats
from ..features.engineering import create_prediction_features
from ..visualization.plots import create_visualizations

def main():
    """
    Main function to run the entire pipeline
    """
    logger.info("Starting pitcher performance analysis pipeline...")
    
    # Create directories if they don't exist
    Path("data").mkdir(exist_ok=True)
    Path("data/visualizations").mkdir(exist_ok=True, parents=True)
    
    # Initialize the database
    init_database()
    
    # 1. Fetch statcast data
    statcast_data = get_statcast_data(force_refresh=False)
    if statcast_data.empty:
        logger.error("No statcast data available. Exiting.")
        return
    
    # 2. Fetch traditional pitching stats
    traditional_data = get_traditional_stats(force_refresh=False)
    if traditional_data.empty:
        logger.warning("No traditional pitching data available. Continuing with statcast data only.")
    
    # 3. Store data in the database
    store_statcast_data(statcast_data, force_refresh=False)
    if not traditional_data.empty:
        store_traditional_stats(traditional_data, force_refresh=False)
    
    # 4. Update pitcher ID mappings
    update_pitcher_mapping()
    
    # 5. Create prediction features
    create_prediction_features(force_refresh=False)
    
    # 6. Export final dataset to CSV
    final_data = export_dataset_to_csv()
    
    # 7. Create visualizations
    create_visualizations(final_data)
    
    logger.info("Pipeline completed successfully!")

def troubleshoot_database():
    """
    Troubleshoot database issues
    """
    from src.data.db import get_db_connection, execute_query
    
    logger.info("Troubleshooting database issues...")
    
    try:
        # Check database existence
        db_path = Path("data/pitcher_stats.db")
        if not db_path.exists():
            logger.error(f"Database file not found at {db_path}")
            return
        
        logger.info(f"Database file exists at {db_path} with size {db_path.stat().st_size} bytes")
        
        # Check tables
        conn = get_db_connection()
        tables = execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        logger.info(f"Tables in database: {tables['name'].tolist()}")
        
        # Check row counts in each table
        for table in tables['name']:
            count = execute_query(f"SELECT COUNT(*) as count FROM {table}")
            logger.info(f"Table {table}: {count.iloc[0]['count']} rows")
        
        # Check pitcher mapping
        mapping_stats = execute_query("""
            SELECT 
                COUNT(*) as total_pitchers,
                SUM(CASE WHEN statcast_id IS NOT NULL THEN 1 ELSE 0 END) as with_statcast,
                SUM(CASE WHEN traditional_id IS NOT NULL THEN 1 ELSE 0 END) as with_traditional,
                SUM(CASE WHEN statcast_id IS NOT NULL AND traditional_id IS NOT NULL THEN 1 ELSE 0 END) as fully_mapped
            FROM pitchers
        """)
        
        logger.info(f"Pitcher mapping stats: {mapping_stats.to_dict('records')[0]}")
        
        # Check for duplicate pitcher entries
        duplicate_query = """
            SELECT player_name, COUNT(*) as count
            FROM pitchers
            GROUP BY player_name
            HAVING COUNT(*) > 1
        """
        duplicates = execute_query(duplicate_query)
        if not duplicates.empty:
            logger.warning(f"Found {len(duplicates)} pitchers with duplicate entries:")
            for _, row in duplicates.iterrows():
                logger.warning(f"  {row['player_name']}: {row['count']} entries")
        
        # Check pitch mix data
        pitch_mix_query = """
            SELECT COUNT(DISTINCT game_stats_id) as games_with_pitch_mix
            FROM pitch_mix
        """
        pitch_mix_stats = execute_query(pitch_mix_query)
        logger.info(f"Games with pitch mix data: {pitch_mix_stats.iloc[0]['games_with_pitch_mix']}")
        
        # Check feature completeness
        feature_stats = execute_query("""
            SELECT 
                COUNT(*) as total_features,
                AVG(last_3_games_strikeouts_avg) as avg_k3,
                AVG(last_5_games_strikeouts_avg) as avg_k5,
                AVG(last_3_games_era_avg) as avg_era3,
                AVG(last_5_games_era_avg) as avg_era5
            FROM prediction_features
        """)
        
        logger.info(f"Feature stats: {feature_stats.to_dict('records')[0]}")
        
        logger.info("Database troubleshooting completed. See log for details.")
    
    except Exception as e:
        logger.error(f"Error during database troubleshooting: {e}")

def verify_data_quality():
    """
    Verify the quality of the data exported to CSV
    """
    logger.info("Verifying data quality...")
    
    try:
        # Load the CSV file
        csv_path = Path("data/pitcher_game_level_data.csv")
        if not csv_path.exists():
            logger.error(f"CSV file not found at {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
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
            logger.info(f"ERA column ({era_col}): {era_null} null values ({era_null/len(df)*100:.2f}%)")
        else:
            logger.warning("No ERA column found in dataset")
        
        # Check for pitch mix data
        pitch_cols = [col for col in df.columns if col.startswith('pitch_pct_')]
        logger.info(f"Found {len(pitch_cols)} pitch mix columns")
        
        if pitch_cols:
            # Calculate percentage of rows with any pitch data
            has_pitch_data = df[pitch_cols].notna().any(axis=1).mean() * 100
            logger.info(f"{has_pitch_data:.2f}% of rows have pitch mix data")
        
        # Check feature completeness
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
        
        logger.info("Data quality verification completed. See log for details.")
    
    except Exception as e:
        logger.error(f"Error during data quality verification: {e}")

if __name__ == "__main__":
    main()