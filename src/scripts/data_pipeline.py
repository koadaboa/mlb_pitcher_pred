#!/usr/bin/env python
# src/scripts/data_pipeline.py
import argparse
from pathlib import Path
import logging
import pandas as pd

from src.data.utils import setup_logger, ensure_dir
from src.data.db import init_database, clear_database, update_database_schema
from src.data.db import get_db_connection, store_statcast_data, store_game_context
from src.data.db import store_team_data, store_opponent_data
from src.data.fetch import get_statcast_data, fetch_team_data
from src.data.process import extract_game_context, map_opponents_to_games
from src.data.process import export_data_to_csv, aggregate_to_game_level
from src.features.engineering import create_prediction_features
from config import DataConfig, DBConfig

logger = setup_logger(__name__)

def run_pipeline(command, force_refresh=False, skip_statcast=False):
    """
    Run the data pipeline with the specified command
    
    Args:
        command (str): Command to execute (setup, fetch, process, features, all)
        force_refresh (bool): Whether to force refresh existing data
        skip_statcast (bool): Whether to skip fetching new Statcast data
        
    Returns:
        bool: Success status
    """
    # Create necessary directories
    ensure_dir("data")
    
    # Execute the appropriate command
    if command == 'setup':
        return _setup_database(force_refresh)
    elif command == 'fetch':
        return _fetch_data(force_refresh, skip_statcast)
    elif command == 'process':
        return _process_data(force_refresh)
    elif command == 'export':
        return _export_data()
    elif command == 'all':
        # Run all pipeline steps
        setup_success = _setup_database(force_refresh)
        if not setup_success:
            return False
            
        fetch_success = _fetch_data(force_refresh, skip_statcast)
        if not fetch_success:
            return False
            
        process_success = _process_data(force_refresh)
        if not process_success:
            return False
            
        export_success = _export_data()
        return export_success
    else:
        logger.error(f"Unknown command: {command}")
        return False

def _setup_database(clear_existing=False):
    """
    Set up the database structure
    
    Args:
        clear_existing (bool): Whether to clear existing database
        
    Returns:
        bool: Success status
    """
    try:
        # Create data directory if it doesn't exist
        ensure_dir(Path(DBConfig.PATH).parent)
        
        # Optionally clear the database
        if clear_existing:
            logger.info("Clearing existing database...")
            clear_database()
        
        # Initialize database structure
        logger.info("Initializing database structure...")
        init_database()
        
        # Update schema to latest version
        update_database_schema()
        
        logger.info("Database structure initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def _fetch_data(force_refresh=False, skip_statcast=False):
    """
    Fetch data from external sources
    
    Args:
        force_refresh (bool): Whether to force refresh cached data
        skip_statcast (bool): Whether to skip fetching new Statcast data
        
    Returns:
        bool: Success status
    """
    try:
        # Initialize the database if not already done
        init_database()
        
        # 1. Load already fetched statcast data if skip_statcast=True
        if skip_statcast:
            logger.info("Loading cached Statcast data...")
            try:
                import pickle
                with open("data/statcast_pitcher_data.pkl", 'rb') as f:
                    statcast_data = pickle.load(f)
                logger.info(f"Loaded cached data with {len(statcast_data)} records")
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
                return False
        else:
            # Fetch new statcast data
            logger.info("Fetching Statcast data...")
            statcast_data = get_statcast_data(force_refresh=force_refresh)
            
        if statcast_data.empty:
            logger.error("No statcast data available. Exiting.")
            return False
        
        # 2. Extract game context information
        logger.info("Extracting game context information...")
        game_context = extract_game_context(statcast_data)
        
        # 3. Fetch team data
        logger.info("Fetching team data...")
        team_data = fetch_team_data()
        
        # 4. Map opponents to games
        logger.info("Mapping opponents to games...")
        # Create a simple pitcher-team map
        pitcher_teams = statcast_data.groupby(['pitcher', 'season', 'home_team']).size().reset_index()
        pitcher_teams = pitcher_teams.sort_values('pitcher').drop_duplicates(['pitcher', 'season'])
        pitcher_team_map = dict(zip(pitcher_teams['pitcher'], pitcher_teams['home_team']))
        
        opponent_mapping = map_opponents_to_games(statcast_data, pitcher_team_map)
        
        # 5. Store raw data in the database
        logger.info("Storing raw data in database...")
        try:
            store_statcast_data(statcast_data, force_refresh=force_refresh)
            
            # Store additional data
            if not game_context.empty:
                store_game_context(game_context)
            
            if not team_data.empty:
                store_team_data(team_data)
                
            if not opponent_mapping.empty:
                store_opponent_data(opponent_mapping)
            
            logger.info("Data acquisition completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Error during data fetching: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def _process_data(force_refresh=False):
    """
    Process raw data into analysis-ready format
    
    Args:
        force_refresh (bool): Whether to force refresh existing data
        
    Returns:
        bool: Success status
    """
    try:
        # 1. Update database schema to ensure it's ready for new features
        logger.info("Updating database schema...")
        update_database_schema()
        
        # 2. Create prediction features (rolling averages, trends, etc.)
        logger.info("Creating standard prediction features...")
        create_prediction_features(force_refresh=force_refresh)
        
        logger.info("Data processing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def _export_data():
    """
    Export processed data to CSV files
    
    Returns:
        bool: Success status
    """
    try:
        # Export data to CSV files for analysis
        logger.info("Exporting processed data to CSV...")
        exported_files = export_data_to_csv()
        
        for file_type, path in exported_files.items():
            logger.info(f"Exported {file_type} to {path}")
        
        logger.info("Data export completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during data export: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='MLB data pipeline for strikeout prediction')
    parser.add_argument('command', choices=['setup', 'fetch', 'process', 'export', 'all'],
                       help='Command to execute')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh of cached data')
    parser.add_argument('--skip-statcast', action='store_true',
                       help='Skip fetching new Statcast data and use cached data')
    
    args = parser.parse_args()
    success = run_pipeline(args.command, args.force_refresh, args.skip_statcast)
    
    if success:
        logger.info(f"Successfully completed command: {args.command}")
        return 0
    else:
        logger.error(f"Failed to complete command: {args.command}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())