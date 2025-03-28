# Functions for fetching data from external sources like pybaseball
import pandas as pd
import time
from datetime import date, timedelta
import pickle
import os
import logging
from tqdm import tqdm
from pathlib import Path

# Import pybaseball library
from pybaseball import statcast, statcast_pitcher
from pybaseball import cache

logger = logging.getLogger(__name__)

# Enable pybaseball cache
cache.enable()

# Global variables
RATE_LIMIT_PAUSE = 5  # seconds to wait between API calls
SEASONS = [2019, 2021, 2022, 2023, 2024]

def fetch_statcast_safely(start_date, end_date, max_retries=3):
    """
    Fetch statcast data with error handling and rate limiting
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        max_retries (int): Maximum number of retries in case of failure
        
    Returns:
        pandas.DataFrame: Statcast data
    """
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Fetching data from {start_date} to {end_date}...")
            data = statcast(start_dt=start_date, end_dt=end_date)
            time.sleep(RATE_LIMIT_PAUSE)  # Respect rate limits
            return data
        except pd.errors.ParserError as e:
            logger.warning(f"Parser error encountered: {e}")
            logger.info("Attempting to handle parser error by manually adjusting request parameters...")
            # Try smaller date range as a workaround
            start_dt_obj = pd.to_datetime(start_date)
            end_dt_obj = pd.to_datetime(end_date)
            mid_dt_obj = start_dt_obj + (end_dt_obj - start_dt_obj) / 2
            mid_dt = mid_dt_obj.strftime('%Y-%m-%d')
            
            logger.info(f"Splitting request into two: {start_date} to {mid_dt} and {mid_dt} to {end_date}")
            try:
                df1 = statcast(start_dt=start_date, end_dt=mid_dt)
                time.sleep(RATE_LIMIT_PAUSE)
                df2 = statcast(start_dt=mid_dt, end_dt=end_date)
                time.sleep(RATE_LIMIT_PAUSE)
                return pd.concat([df1, df2], ignore_index=True)
            except Exception as nested_error:
                logger.warning(f"Nested error: {nested_error}")
                retries += 1
                logger.info(f"Retrying ({retries}/{max_retries})...")
                time.sleep(RATE_LIMIT_PAUSE * 2)  # Longer pause before retry
        except Exception as e:
            logger.warning(f"Error: {e}")
            retries += 1
            logger.info(f"Retrying ({retries}/{max_retries})...")
            time.sleep(RATE_LIMIT_PAUSE * 2)  # Longer pause before retry
    
    logger.error(f"Failed to fetch data after {max_retries} retries")
    return pd.DataFrame()  # Return empty DataFrame if all retries fail

def fetch_season_in_chunks(season, chunk_size=14):
    """
    Fetch statcast data for a whole season in smaller chunks to avoid timeout/memory issues
    
    Args:
        season (int): MLB season year
        chunk_size (int): Number of days per chunk
        
    Returns:
        pandas.DataFrame: Season's statcast data
    """
    # Define season start and end dates (approximate MLB season)
    if season == 2024:
        # 2024 season started on March 28
        season_start = f"{season}-03-28"
        # Use current date as end if we're in 2024
        if date.today().year == 2024:
            season_end = date.today().strftime('%Y-%m-%d')
        else:
            # Otherwise go through end of regular season (approximate)
            season_end = f"{season}-10-01"
    else:
        # Regular seasons (approximate dates)
        season_start = f"{season}-04-01"
        season_end = f"{season}-10-01"
    
    # Convert to datetime for easier manipulation
    start_dt = pd.to_datetime(season_start)
    end_dt = pd.to_datetime(season_end)
    
    all_data = []
    
    # Create chunks of dates
    current_start = start_dt
    total_chunks = (end_dt - start_dt).days // chunk_size + 1
    
    with tqdm(total=total_chunks, desc=f"Season {season}") as pbar:
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size), end_dt)
            
            chunk_data = fetch_statcast_safely(
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            )
            
            if not chunk_data.empty:
                chunk_data['season'] = season
                all_data.append(chunk_data)
            
            current_start = current_end + timedelta(days=1)
            pbar.update(1)
    
    if not all_data:
        logger.warning(f"No data collected for season {season}")
        return pd.DataFrame()
    
    # Combine all chunks
    season_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Collected {len(season_data)} rows for season {season}")
    
    return season_data

def get_statcast_data(force_refresh=False):
    """
    Fetch statcast data for multiple seasons
    
    Args:
        force_refresh (bool): Whether to force refresh cached data
        
    Returns:
        pandas.DataFrame: Combined statcast data
    """
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True, parents=True)
    
    cache_file = "data/statcast_pitcher_data.pkl"
    
    # Check if we have a recent cached version
    if not force_refresh and os.path.exists(cache_file):
        logger.info(f"Loading cached statcast data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    all_statcast_data = []
    
    for season in SEASONS:
        season_data = fetch_season_in_chunks(season)
        
        if not season_data.empty:
            # Filter to only include pitcher-relevant data
            pitcher_data = season_data[season_data['pitcher'].notna()].copy()
            pitcher_data['season'] = season
            all_statcast_data.append(pitcher_data)
            
            # Save season data separately as backup
            season_cache = f"data/statcast_pitcher_{season}.pkl"
            with open(season_cache, 'wb') as f:
                pickle.dump(pitcher_data, f)
            logger.info(f"Saved {season} data to {season_cache}")
    
    if not all_statcast_data:
        logger.warning("No statcast data retrieved")
        return pd.DataFrame()
    
    # Combine all season data
    combined_data = pd.concat(all_statcast_data, ignore_index=True)
    
    # Save combined data
    with open(cache_file, 'wb') as f:
        pickle.dump(combined_data, f)
    logger.info(f"Saved combined statcast data to {cache_file}")
    
    return combined_data