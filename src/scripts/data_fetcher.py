# src/data/db_test.py
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import pybaseball as pb
import logging
import time
from datetime import datetime, timedelta
from config import DBConfig, DataConfig
from src.data.utils import setup_logger, ensure_dir
from src.data.utils import DBConnection

logger = setup_logger(__name__)

# Database path
DB_PATH = DBConfig.PATH

def fetch_pitcher_id_mapping(seasons=None):
    """
    Fetch pitcher ID mappings between MLBAM and FanGraphs to identify starting pitchers
    
    Args:
        seasons (list): List of seasons to process
        
    Returns:
        pandas.DataFrame: DataFrame with pitcher ID mappings
    """
    if seasons is None:
        seasons = DataConfig.SEASONS
    
    logger.info(f"Fetching pitcher ID mappings for seasons: {seasons}")
    
    # Get Chadwick Register for ID mapping
    player_lookup = pb.chadwick_register()
    all_pitchers = []
    
    for season in seasons:
        try:
            # Get pitching stats for the season
            pitching_stats = pb.pitching_stats(season, season, qual=0)
            
            # Identify columns for games and games started
            games_col = 'G' if 'G' in pitching_stats.columns else 'Games'
            gs_col = 'GS' if 'GS' in pitching_stats.columns else 'Games Started'
            
            # Check if columns exist
            if games_col not in pitching_stats.columns or gs_col not in pitching_stats.columns:
                logger.warning(f"Missing required columns in season {season}")
                continue
            
            # Identify pitchers who are starters
            # Current season: more lenient (1+ starts)
            # Past seasons: stricter (5+ starts or 50%+ starts)
            if season == 2025:  # Current season
                starters = pitching_stats[pitching_stats[gs_col] >= 1].copy()
            else:
                starters = pitching_stats[
                    (pitching_stats[gs_col] >= 5) | 
                    ((pitching_stats[gs_col]/pitching_stats[games_col] >= 0.5) & 
                     (pitching_stats[games_col] >= 8))
                ].copy()
            
            starters['is_starter'] = 1
            starters['season'] = season
            
            # Get FanGraphs IDs
            id_col = 'playerid' if 'playerid' in starters.columns else 'IDfg'
            if id_col not in starters.columns:
                logger.warning(f"No player ID column found in season {season}")
                continue
                
            starters = starters.rename(columns={id_col: 'key_fangraphs'})
            
            # Add to collection
            all_pitchers.append(starters[['key_fangraphs', 'is_starter', 'season']])
            
            logger.info(f"Season {season}: Found {len(starters)} starting pitchers")
            
        except Exception as e:
            logger.error(f"Error processing season {season}: {e}")
    
    if not all_pitchers:
        return pd.DataFrame()
        
    # Combine all data
    all_pitchers_df = pd.concat(all_pitchers)
    
    # Join with Chadwick to get MLBAM IDs
    player_lookup_filtered = player_lookup[['key_fangraphs', 'key_mlbam', 'name_first', 'name_last']].dropna()
    
    # Convert to int for joining
    all_pitchers_df['key_fangraphs'] = pd.to_numeric(all_pitchers_df['key_fangraphs'], errors='coerce')
    player_lookup_filtered['key_fangraphs'] = pd.to_numeric(player_lookup_filtered['key_fangraphs'], errors='coerce')
    
    merged_df = pd.merge(
        all_pitchers_df,
        player_lookup_filtered,
        on='key_fangraphs',
        how='inner'
    )
    
    # Add full name
    merged_df['name'] = merged_df['name_first'] + ' ' + merged_df['name_last']
    
    return merged_df

def fetch_statcast_for_pitchers(pitcher_ids, seasons):
    """
    Fetch Statcast data for a list of pitchers
    
    Args:
        pitcher_ids (list): List of (pitcher_id, name) tuples
        seasons (list): List of seasons to fetch
        
    Returns:
        pandas.DataFrame: Combined DataFrame with all Statcast data
    """
    logger.info(f"Fetching Statcast data for {len(pitcher_ids)} pitchers across {len(seasons)} seasons")
    
    all_data = []
    successful_fetches = 0
    
    for pitcher_id, name in pitcher_ids:
        try:
            for season in seasons:
                # Define date range for the season
                if season == 2025:  # Current season
                    start_date = f"{season}-03-30"
                    end_date = "2025-04-07"  # Current date in our scenario
                else:
                    start_date = f"{season}-03-30"
                    end_date = f"{season}-11-01"
                
                logger.info(f"Fetching data for {name} (ID: {pitcher_id}) from {start_date} to {end_date}")
                
                # Fetch the data
                pitcher_data = pb.statcast_pitcher(start_date, end_date, pitcher_id)
                
                if pitcher_data.empty:
                    logger.warning(f"No data found for {name} in {season}")
                    continue
                
                # Add pitcher_id and season columns
                pitcher_data['pitcher_id'] = pitcher_id
                pitcher_data['season'] = season
                
                all_data.append(pitcher_data)
                successful_fetches += 1
                
                logger.info(f"Successfully fetched {len(pitcher_data)} records for {name} in {season}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {name} (ID: {pitcher_id}): {e}")
    
    if not all_data:
        logger.warning("No data was successfully fetched")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total combined data: {len(combined_data)} records from {successful_fetches} successful fetches")
    
    return combined_data

def fetch_team_batting_data(seasons):
    """
    Fetch team batting data from FanGraphs
    
    Args:
        seasons (list): List of seasons to fetch
        
    Returns:
        pandas.DataFrame: Combined DataFrame with all team batting data
    """
    logger.info(f"Fetching team batting data for seasons: {seasons}")
    
    all_data = []
    
    for season in seasons:
        try:
            # Fetch team batting data
            team_data = pb.team_batting(season, season)
            
            if team_data.empty:
                logger.warning(f"No team batting data found for season {season}")
                continue
            
            # Add season column if not present
            if 'Season' not in team_data.columns:
                team_data['Season'] = season
            
            all_data.append(team_data)
            logger.info(f"Successfully fetched team batting data for {season} ({len(team_data)} teams)")
            
        except Exception as e:
            logger.error(f"Error fetching team batting data for season {season}: {e}")
    
    if not all_data:
        logger.warning("No team batting data was successfully fetched")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total combined team batting data: {len(combined_data)} records")
    
    return combined_data

def fetch_batter_data(seasons):
    """
    Fetch Statcast data for all batters by using date ranges
    
    Args:
        seasons (list): List of seasons to fetch
        
    Returns:
        pandas.DataFrame: Combined DataFrame with batter data
    """
    logger.info(f"Fetching comprehensive batter data for seasons: {seasons}")
    
    all_data = []
    total_rows = 0
    
    for season in seasons:
        try:
            # Define date ranges to break up the season into manageable chunks
            if season == 2025:  # Current season (assuming it's just started)
                date_ranges = [
                    (f"{season}-03-28", "2025-04-06")  # Opening day to current date
                ]
            else:
                # Break the season into monthly chunks to handle the data volume
                date_ranges = [
                    (f"{season}-03-28", f"{season}-04-30"),  # March/April
                    (f"{season}-05-01", f"{season}-05-31"),  # May
                    (f"{season}-06-01", f"{season}-06-30"),  # June
                    (f"{season}-07-01", f"{season}-07-31"),  # July
                    (f"{season}-08-01", f"{season}-08-31"),  # August
                    (f"{season}-09-01", f"{season}-09-30"),  # September
                    (f"{season}-10-01", f"{season}-10-15")   # Early October (end of regular season)
                ]
            
            for start_date, end_date in date_ranges:
                logger.info(f"Fetching batter data from {start_date} to {end_date}")
                
                # Use statcast() to get all pitches for this date range
                # This gets all pitches thrown, which includes all batter data
                try:
                    period_data = pb.statcast(start_dt=start_date, end_dt=end_date)
                    
                    if period_data.empty:
                        logger.warning(f"No data returned for {start_date} to {end_date}")
                        continue
                    
                    # Add season column
                    period_data['season'] = season
                    
                    # Log success
                    logger.info(f"Successfully fetched {len(period_data)} pitch records from {start_date} to {end_date}")
                    
                    # Append to our collection
                    all_data.append(period_data)
                    total_rows += len(period_data)
                    
                    # Be respectful of rate limits
                    logger.info("Waiting for rate limit...")
                    time.sleep(5)  # 5 second pause between requests
                
                except Exception as e:
                    logger.error(f"Error fetching data for period {start_date} to {end_date}: {e}")
                    
                    # Try again with a smaller date range if possible
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    duration = (end_dt - start_dt).days
                    
                    if duration > 3:
                        mid_dt = start_dt + timedelta(days=duration // 2)
                        mid_date = mid_dt.strftime("%Y-%m-%d")
                        
                        logger.info(f"Retrying with smaller date ranges: {start_date} to {mid_date} and {mid_date} to {end_date}")
                        
                        try:
                            # Try first half
                            first_half = pb.statcast(start_dt=start_date, end_dt=mid_date)
                            if not first_half.empty:
                                first_half['season'] = season
                                all_data.append(first_half)
                                total_rows += len(first_half)
                                logger.info(f"Retrieved {len(first_half)} records for first half")
                            
                            time.sleep(5)  # Respect rate limits
                            
                            # Try second half
                            second_half = pb.statcast(start_dt=mid_date, end_dt=end_date)
                            if not second_half.empty:
                                second_half['season'] = season
                                all_data.append(second_half)
                                total_rows += len(second_half)
                                logger.info(f"Retrieved {len(second_half)} records for second half")
                        
                        except Exception as nested_error:
                            logger.error(f"Error during retry attempt: {nested_error}")
        
        except Exception as e:
            logger.error(f"Error processing season {season}: {e}")
    
    if not all_data:
        logger.warning("No batter data was successfully fetched")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total combined pitch data: {len(combined_data)} records")
    
    return combined_data

def store_data_to_sql(df, table_name, if_exists='replace'):
    """
    Store DataFrame to SQLite using pandas to_sql
    
    Args:
        df (pandas.DataFrame): DataFrame to store
        table_name (str): Name of the table
        if_exists (str): How to behave if table exists ('fail', 'replace', 'append')
        
    Returns:
        bool: Success status
    """
    if df.empty:
        logger.warning(f"Empty DataFrame provided for {table_name}, nothing to store")
        return False
    
    try:
        logger.info(f"Storing {len(df)} records to table {table_name}")
        
        with DBConnection() as conn:
            # Store the DataFrame to SQL
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            
        logger.info(f"Successfully stored data to {table_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing data to {table_name}: {e}")
        return False

def get_data_from_sql(table_name, limit=None):
    """
    Retrieve data from SQLite as a DataFrame
    
    Args:
        table_name (str): Name of the table
        limit (int): Optional limit on number of rows
        
    Returns:
        pandas.DataFrame: Retrieved data
    """
    try:
        with DBConnection() as conn:
            # Build query
            query = f"SELECT * FROM {table_name}"
            if limit is not None:
                query += f" LIMIT {limit}"
                
            # Use pandas read_sql_query to preserve column names
            df = pd.read_sql_query(query, conn)
            
            logger.info(f"Retrieved {len(df)} records from {table_name} with {len(df.columns)} columns")
            return df
            
    except Exception as e:
        logger.error(f"Error retrieving data from {table_name}: {e}")
        return pd.DataFrame()

def setup_database():
    """
    Main function to set up the database with Statcast data
    """
    seasons = DataConfig.SEASONS
    
    # Ensure database directory exists
    ensure_dir(Path(DB_PATH).parent)
    
    # 1. Get starting pitcher IDs
    pitcher_mapping = fetch_pitcher_id_mapping(seasons)
    
    if pitcher_mapping.empty:
        logger.error("Failed to get pitcher mapping, cannot proceed")
        return False
    
    # Store pitcher mapping
    store_data_to_sql(pitcher_mapping, 'pitcher_mapping')
    
    # Get list of pitcher IDs and names
    pitcher_ids = list(zip(pitcher_mapping['key_mlbam'], pitcher_mapping['name']))
    
    # 2. Fetch Statcast data for starting pitchers (sample for testing)
    sample_pitchers = pitcher_ids[:5]  # Just a few for testing
    pitcher_data = fetch_statcast_for_pitchers(sample_pitchers, seasons)
    
    # Store pitcher data
    if not pitcher_data.empty:
        store_data_to_sql(pitcher_data, 'statcast_pitchers')
    
    # 3. Fetch team batting data
    team_batting = fetch_team_batting_data(seasons)
    
    # Store team batting data
    if not team_batting.empty:
        store_data_to_sql(team_batting, 'team_batting')
    
    # 4. Fetch batter data (sample)
    batter_data = fetch_batter_data(seasons)
    
    # Store batter data
    if not batter_data.empty:
        store_data_to_sql(batter_data, 'statcast_batters')
    
    # 5. Test data retrieval
    test_pitcher_data = get_data_from_sql('statcast_pitchers', limit=5)
    if not test_pitcher_data.empty:
        logger.info(f"Successfully retrieved pitcher data with columns: {test_pitcher_data.columns[:5]}...")
    
    return True

if __name__ == "__main__":
    logger.info("Setting up database with simplified approach...")
    setup_database()
    logger.info("Database setup complete")