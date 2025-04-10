# src/scripts/data_fetcher.py
import sqlite3
import pandas as pd
import numpy as np
import pybaseball as pb
import logging
import time
import json
import os
import argparse
import traceback
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import DBConfig, DataConfig
from src.data.utils import setup_logger, ensure_dir, DBConnection

# Enhanced logger setup
logger = setup_logger(
    'data_fetcher', 
    log_file='logs/data_fetcher.log',
    level=logging.INFO
)

class CheckpointManager:
    """Manages checkpoints for resumable data fetching"""
    
    def __init__(self, checkpoint_dir='data/checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.current_checkpoint = {}
        self.load_overall_checkpoint()
    
    def load_overall_checkpoint(self):
        """Load the overall progress checkpoint"""
        overall_checkpoint_file = self.checkpoint_dir / 'overall_progress.json'
        
        if overall_checkpoint_file.exists():
            with open(overall_checkpoint_file, 'r') as f:
                self.current_checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {self.current_checkpoint}")
        else:
            # Initialize with default values
            self.current_checkpoint = {
                'pitcher_mapping_completed': False,
                'processed_pitcher_ids': [],
                'team_batting_completed': False,
                'processed_seasons_batter_data': {},
                'last_update': datetime.now().isoformat()
            }
            self.save_overall_checkpoint()
    
    def save_overall_checkpoint(self):
        """Save the overall progress checkpoint"""
        overall_checkpoint_file = self.checkpoint_dir / 'overall_progress.json'
        
        # Update timestamp
        self.current_checkpoint['last_update'] = datetime.now().isoformat()
        
        with open(overall_checkpoint_file, 'w') as f:
            json.dump(self.current_checkpoint, f, indent=4)
        
        logger.info("Updated overall checkpoint")
    
    def is_completed(self, task):
        """Check if a task is already completed"""
        return self.current_checkpoint.get(f"{task}_completed", False)
    
    def mark_completed(self, task):
        """Mark a task as completed"""
        self.current_checkpoint[f"{task}_completed"] = True
        self.save_overall_checkpoint()
    
    def add_processed_pitcher(self, pitcher_id):
        """Add a processed pitcher ID to the checkpoint"""
        if pitcher_id not in self.current_checkpoint['processed_pitcher_ids']:
            self.current_checkpoint['processed_pitcher_ids'].append(pitcher_id)
            self.save_overall_checkpoint()
    
    def is_pitcher_processed(self, pitcher_id):
        """Check if a pitcher ID has already been processed"""
        return pitcher_id in self.current_checkpoint['processed_pitcher_ids']
    
    def add_processed_season_date_range(self, season, date_range):
        """Add a processed season date range to the checkpoint"""
        if str(season) not in self.current_checkpoint['processed_seasons_batter_data']:
            self.current_checkpoint['processed_seasons_batter_data'][str(season)] = []
        
        if date_range not in self.current_checkpoint['processed_seasons_batter_data'][str(season)]:
            self.current_checkpoint['processed_seasons_batter_data'][str(season)].append(date_range)
            self.save_overall_checkpoint()
    
    def is_season_date_range_processed(self, season, date_range):
        """Check if a season date range has already been processed"""
        if str(season) not in self.current_checkpoint['processed_seasons_batter_data']:
            return False
        
        return date_range in self.current_checkpoint['processed_seasons_batter_data'][str(season)]

class DataFetcher:
    """Data fetcher with checkpointing and memory-efficient processing"""
    
    def __init__(self, args):
        self.args = args
        self.db_path = DBConfig.PATH
        self.seasons = args.seasons if args.seasons else DataConfig.SEASONS
        self.checkpoint_manager = CheckpointManager()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
        
        # Set pybaseball cache
        pb.cache.enable()
        
        # Ensure database directory exists
        ensure_dir(Path(self.db_path).parent)
    
    def handle_interrupt(self, signum, frame):
        """Handle interruption signals"""
        logger.warning(f"Received interrupt signal {signum}. Saving checkpoint and exiting gracefully...")
        
        # Force save checkpoint
        self.checkpoint_manager.save_overall_checkpoint()
        
        logger.info("Checkpoint saved. Exiting...")
        sys.exit(0)
    
    def fetch_with_retries(self, fetch_function, *args, max_retries=3, retry_delay=5, **kwargs):
        """Execute a fetch function with retries"""
        for attempt in range(max_retries):
            try:
                return fetch_function(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    retry_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {retry_time} seconds...")
                    time.sleep(retry_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    raise
    
    def fetch_pitcher_id_mapping(self):
        """Fetch pitcher ID mappings with checkpointing"""
        if self.checkpoint_manager.is_completed('pitcher_mapping'):
            logger.info("Pitcher mapping already completed, skipping...")
            
            # Load from database
            with DBConnection() as conn:
                try:
                    query = "SELECT * FROM pitcher_mapping"
                    pitcher_mapping = pd.read_sql_query(query, conn)
                    logger.info(f"Loaded {len(pitcher_mapping)} pitcher mappings from database")
                    return pitcher_mapping
                except Exception:
                    logger.warning("Failed to load pitcher mapping from database, fetching again")
        
        logger.info(f"Fetching pitcher ID mappings for seasons: {self.seasons}")
        
        # Get Chadwick Register for ID mapping with retries
        player_lookup = self.fetch_with_retries(pb.chadwick_register)
        all_pitchers = []
        
        for season in tqdm(self.seasons, desc="Processing seasons for pitcher mapping"):
            try:
                # Get pitching stats for the season
                pitching_stats = self.fetch_with_retries(pb.pitching_stats, season, season, qual=0)
                
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
                logger.error(traceback.format_exc())
        
        if not all_pitchers:
            logger.error("No pitcher data was collected")
            return pd.DataFrame()
            
        # Combine all data
        all_pitchers_df = pd.concat(all_pitchers, ignore_index=True)
        
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
        
        # Store to database
        store_data_to_sql(merged_df, 'pitcher_mapping')
        
        # Mark as completed in checkpoint
        self.checkpoint_manager.mark_completed('pitcher_mapping')
        
        return merged_df
    
    def fetch_statcast_for_pitcher(self, pitcher_id, name, seasons):
        """Fetch Statcast data for a single pitcher"""
        if self.checkpoint_manager.is_pitcher_processed(pitcher_id):
            logger.info(f"Pitcher {name} (ID: {pitcher_id}) already processed, skipping...")
            return pd.DataFrame()
        
        all_data = []
        
        for season in seasons:
            # Define date range for the season
            if season == 2025:  # Current season
                start_date = f"{season}-03-30"
                end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")  # Yesterday
            else:
                start_date = f"{season}-03-30"
                end_date = f"{season}-11-01"
            
            logger.info(f"Fetching data for {name} (ID: {pitcher_id}) from {start_date} to {end_date}")
            
            try:
                # Fetch the data with retries
                pitcher_data = self.fetch_with_retries(pb.statcast_pitcher, start_date, end_date, pitcher_id)
                
                if pitcher_data.empty:
                    logger.warning(f"No data found for {name} in {season}")
                    continue
                
                # Add pitcher_id and season columns
                pitcher_data['pitcher_id'] = pitcher_id
                pitcher_data['season'] = season
                
                all_data.append(pitcher_data)
                
                logger.info(f"Successfully fetched {len(pitcher_data)} records for {name} in {season}")
                
                # Respect rate limits
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error fetching data for {name} (ID: {pitcher_id}) in season {season}: {e}")
                logger.error(traceback.format_exc())
        
        if not all_data:
            logger.warning(f"No data was successfully fetched for {name} (ID: {pitcher_id})")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Mark this pitcher as processed
        self.checkpoint_manager.add_processed_pitcher(pitcher_id)
        
        return combined_data
    
    def fetch_all_pitchers(self, pitcher_mapping):
        """Fetch data for all pitchers with parallel processing"""
        # Get list of pitcher IDs and names
        pitcher_ids = list(zip(pitcher_mapping['key_mlbam'], pitcher_mapping['name']))
        
        # Filter out already processed pitchers
        unprocessed_pitchers = [
            (pid, name) for pid, name in pitcher_ids 
            if not self.checkpoint_manager.is_pitcher_processed(pid)
        ]
        
        logger.info(f"Processing {len(unprocessed_pitchers)} unprocessed pitchers out of {len(pitcher_ids)} total")
        
        # Use ThreadPoolExecutor for parallel processing if specified
        all_pitcher_data = []
        
        if self.args.parallel and unprocessed_pitchers:
            batch_size = min(40, len(unprocessed_pitchers))  # Process up to 10 pitchers at a time
            
            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = {}
                processed_count = 0
                
                # Process pitchers in batches to avoid memory issues
                for i in range(0, len(unprocessed_pitchers), batch_size):
                    batch = unprocessed_pitchers[i:i+batch_size]
                    
                    # Submit tasks for each pitcher in batch
                    for pitcher_id, name in batch:
                        future = executor.submit(
                            self.fetch_statcast_for_pitcher, 
                            pitcher_id, 
                            name, 
                            self.seasons
                        )
                        futures[future] = (pitcher_id, name)
                    
                    # Process completed tasks
                    for future in as_completed(futures):
                        pitcher_id, name = futures[future]
                        try:
                            data = future.result()
                            if not data.empty:
                                # Store directly to database to free memory
                                store_data_to_sql(data, 'statcast_pitchers', if_exists='append')
                                logger.info(f"Stored {len(data)} records for {name} (ID: {pitcher_id})")
                            
                            processed_count += 1
                            logger.info(f"Progress: {processed_count}/{len(unprocessed_pitchers)} pitchers processed")
                            
                        except Exception as e:
                            logger.error(f"Error processing pitcher {name} (ID: {pitcher_id}): {e}")
                            logger.error(traceback.format_exc())
                    
                    # Clear dictionary to free memory
                    futures.clear()
        else:
            # Sequential processing
            for i, (pitcher_id, name) in enumerate(tqdm(unprocessed_pitchers, desc="Processing pitchers")):
                try:
                    data = self.fetch_statcast_for_pitcher(pitcher_id, name, self.seasons)
                    if not data.empty:
                        # Store directly to database to free memory
                        store_data_to_sql(data, 'statcast_pitchers', if_exists='append')
                        logger.info(f"Stored {len(data)} records for {name} (ID: {pitcher_id})")
                    
                    logger.info(f"Progress: {i+1}/{len(unprocessed_pitchers)} pitchers processed")
                    
                except Exception as e:
                    logger.error(f"Error processing pitcher {name} (ID: {pitcher_id}): {e}")
                    logger.error(traceback.format_exc())
        
        logger.info("All pitchers processed successfully")
        return True
    
    def fetch_team_batting_data(self):
        """Fetch team batting data with checkpointing"""
        if self.checkpoint_manager.is_completed('team_batting'):
            logger.info("Team batting data already completed, skipping...")
            return pd.DataFrame()
        
        logger.info(f"Fetching team batting data for seasons: {self.seasons}")
        
        all_data = []
        
        for season in tqdm(self.seasons, desc="Processing team batting data"):
            try:
                # Fetch team batting data with retries
                team_data = self.fetch_with_retries(pb.team_batting, season, season)
                
                if team_data.empty:
                    logger.warning(f"No team batting data found for season {season}")
                    continue
                
                # Add season column if not present
                if 'Season' not in team_data.columns:
                    team_data['Season'] = season
                
                all_data.append(team_data)
                logger.info(f"Successfully fetched team batting data for {season} ({len(team_data)} teams)")
                
                # Respect rate limits
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching team batting data for season {season}: {e}")
                logger.error(traceback.format_exc())
        
        if not all_data:
            logger.warning("No team batting data was successfully fetched")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Store to database
        store_data_to_sql(combined_data, 'team_batting')
        
        # Mark as completed in checkpoint
        self.checkpoint_manager.mark_completed('team_batting')
        
        logger.info(f"Total team batting data: {len(combined_data)} records")
        return combined_data
    
    def fetch_batter_data_efficient(self):
        """Fetch batter data by date ranges with immediate storage"""
        logger.info(f"Fetching batter data efficiently for seasons: {self.seasons}")
        
        total_records = 0
        
        for season in self.seasons:
            logger.info(f"Processing season {season} for batter data")
            
            # Define date ranges to break up the season into manageable chunks
            if season == 2025:  # Current season
                date_ranges = [
                    (f"{season}-03-28", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))  # Up to yesterday
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
                    (f"{season}-10-01", f"{season}-10-15")   # Early October
                ]
            
            for start_date, end_date in tqdm(date_ranges, desc=f"Processing date ranges for season {season}"):
                range_key = f"{start_date}_{end_date}"
                
                # Check if this date range is already processed
                if self.checkpoint_manager.is_season_date_range_processed(season, range_key):
                    logger.info(f"Season {season}, date range {start_date} to {end_date} already processed, skipping...")
                    continue
                
                logger.info(f"Fetching batter data from {start_date} to {end_date}")
                
                # Try to get data in smaller chunks if needed
                success = False
                try:
                    # Use statcast() to get all pitches for this date range
                    period_data = self.fetch_with_retries(pb.statcast, start_dt=start_date, end_dt=end_date)
                    
                    if period_data.empty:
                        logger.warning(f"No data returned for {start_date} to {end_date}")
                        # Mark as processed even if empty to avoid retrying
                        self.checkpoint_manager.add_processed_season_date_range(season, range_key)
                        continue
                    
                    # Add season column
                    period_data['season'] = season
                    
                    # Store immediately to free memory
                    period_records = len(period_data)
                    store_data_to_sql(period_data, 'statcast_batters', if_exists='append')
                    
                    # Mark as processed
                    self.checkpoint_manager.add_processed_season_date_range(season, range_key)
                    
                    total_records += period_records
                    logger.info(f"Stored {period_records} records for {start_date} to {end_date}. Total: {total_records}")
                    
                    success = True
                    
                    # Be respectful of rate limits
                    time.sleep(DataConfig.RATE_LIMIT_PAUSE)
                
                except Exception as e:
                    logger.error(f"Error fetching data for period {start_date} to {end_date}: {e}")
                    logger.error(traceback.format_exc())
                
                # If failed, try with smaller date ranges
                if not success:
                    try:
                        # Convert to datetime objects
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        duration = (end_dt - start_dt).days
                        
                        if duration > 3:
                            # Split into smaller chunks
                            chunk_size = max(3, duration // 4)  # Use at most 4 chunks
                            
                            logger.info(f"Trying with smaller chunks of {chunk_size} days")
                            
                            current_start = start_dt
                            while current_start < end_dt:
                                current_end = min(current_start + timedelta(days=chunk_size), end_dt)
                                
                                current_start_str = current_start.strftime("%Y-%m-%d")
                                current_end_str = current_end.strftime("%Y-%m-%d")
                                
                                logger.info(f"Fetching chunk from {current_start_str} to {current_end_str}")
                                
                                try:
                                    chunk_data = self.fetch_with_retries(
                                        pb.statcast, 
                                        start_dt=current_start_str, 
                                        end_dt=current_end_str
                                    )
                                    
                                    if not chunk_data.empty:
                                        chunk_data['season'] = season
                                        chunk_records = len(chunk_data)
                                        
                                        # Store immediately
                                        store_data_to_sql(chunk_data, 'statcast_batters', if_exists='append')
                                        
                                        total_records += chunk_records
                                        logger.info(f"Stored {chunk_records} records for chunk. Total: {total_records}")
                                    
                                    # Mark sub-range as processed
                                    sub_range_key = f"{current_start_str}_{current_end_str}"
                                    self.checkpoint_manager.add_processed_season_date_range(season, sub_range_key)
                                
                                except Exception as e:
                                    logger.error(f"Error processing chunk {current_start_str} to {current_end_str}: {e}")
                                
                                # Move to next chunk
                                current_start = current_end + timedelta(days=1)
                                
                                # Respect rate limits
                                time.sleep(DataConfig.RATE_LIMIT_PAUSE)
                            
                            # Mark the original date range as processed
                            self.checkpoint_manager.add_processed_season_date_range(season, range_key)
                            
                    except Exception as e:
                        logger.error(f"Error during chunked processing: {e}")
                        logger.error(traceback.format_exc())
        
        logger.info(f"Batter data fetching completed. Total records: {total_records}")
        return True
    
    def run(self):
        """Run the data fetching pipeline with checkpointing"""
        logger.info(f"Starting data fetching pipeline for seasons: {self.seasons}")
        start_time = time.time()
        
        try:
            # 1. Get pitcher mapping
            pitcher_mapping = self.fetch_pitcher_id_mapping()
            
            if pitcher_mapping.empty:
                logger.error("Failed to get pitcher mapping, cannot proceed")
                return False
            
            # 2. Fetch Statcast data for pitchers
            self.fetch_all_pitchers(pitcher_mapping)
            
            # 3. Fetch team batting data
            self.fetch_team_batting_data()
            
            # 4. Fetch batter data
            self.fetch_batter_data_efficient()
            
            total_time = time.time() - start_time
            logger.info(f"Data fetching pipeline completed in {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data fetching pipeline: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Ensure checkpoint is saved
            self.checkpoint_manager.save_overall_checkpoint()

def store_data_to_sql(df, table_name, if_exists='replace'):
    """
    Store DataFrame to SQLite using pandas to_sql with chunking
    
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
        
        # Use chunking for large dataframes to avoid memory issues
        if len(df) > DBConfig.BATCH_SIZE:
            with DBConnection() as conn:
                # First chunk will use if_exists parameter
                chunk_size = DBConfig.BATCH_SIZE
                chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        chunk.to_sql(table_name, conn, if_exists=if_exists, index=False)
                    else:
                        chunk.to_sql(table_name, conn, if_exists='append', index=False)
                    
                    logger.info(f"Stored chunk {i+1}/{len(chunks)} for {table_name}")
        else:
            # Smaller dataframes can be stored in one go
            with DBConnection() as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        
        logger.info(f"Successfully stored data to {table_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing data to {table_name}: {e}")
        logger.error(traceback.format_exc())
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch MLB data with checkpointing")
    
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from last checkpoint"
    )
    
    parser.add_argument(
        "--seasons", 
        type=int, 
        nargs="+", 
        help="Seasons to fetch (default: all seasons in DataConfig)"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Use parallel processing for fetching pitcher data"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run the data fetcher
    logger.info("Starting data fetcher with checkpointing")
    fetcher = DataFetcher(args)
    success = fetcher.run()
    
    if success:
        logger.info("Data fetching completed successfully")
    else:
        logger.error("Data fetching failed")
        sys.exit(1)