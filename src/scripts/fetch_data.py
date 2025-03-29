# src/scripts/fetch_data.py
import logging
import os
from pathlib import Path
import pandas as pd
import argparse
import pickle
import pybaseball

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import necessary functions
from src.data.fetch import get_statcast_data, fetch_team_data
from src.data.db import init_database, store_statcast_data
from src.data.process import extract_game_context, map_opponents_to_games

def fetch_and_store_data(force_refresh=False, skip_statcast=False):
    """Fetch and store data from external sources"""
    # Create directories if needed
    Path("data").mkdir(exist_ok=True)
    
    # Initialize the database
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
    # Create a simple pitcher-team map (this is a placeholder - you'll need to enhance this)
    # This assumes the first team a pitcher appears with in a season is their team
    pitcher_teams = statcast_data.groupby(['pitcher', 'season', 'home_team']).size().reset_index()
    pitcher_teams = pitcher_teams.sort_values('pitcher').drop_duplicates(['pitcher', 'season'])
    pitcher_team_map = dict(zip(pitcher_teams['pitcher'], pitcher_teams['home_team']))
    
    opponent_mapping = map_opponents_to_games(statcast_data, pitcher_team_map)
    
    # 5. Store raw data in the database
    logger.info("Storing raw data in database...")
    try:
        store_statcast_data(statcast_data, force_refresh=force_refresh)
        
        # Store additional data
        # These functions need to be implemented in src/data/db.py
        from src.data.db import store_game_context, store_team_data, store_opponent_data
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and store baseball data')
    parser.add_argument('--force-refresh', action='store_true', 
                        help='Force refresh of cached data')
    parser.add_argument('--skip-statcast', action='store_true',
                        help='Skip fetching new Statcast data and use cached data')
    args = parser.parse_args()
    
    fetch_and_store_data(force_refresh=args.force_refresh, skip_statcast=args.skip_statcast)