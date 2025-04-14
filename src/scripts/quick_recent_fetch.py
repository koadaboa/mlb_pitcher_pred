# src/scripts/quick_fetch_recent.py

import pandas as pd
import pybaseball as pb
import logging
import time
from datetime import datetime, timedelta, date
from pathlib import Path
import sys
from tqdm import tqdm
import warnings

# Ensure src directory is in the path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Attempt to import necessary modules from the project
try:
    from src.config import DBConfig, DataConfig
    from src.data.utils import setup_logger, ensure_dir, DBConnection
    # Import the storage function and scraper/mapping functions
    from src.scripts.data_fetcher import store_data_to_sql # Re-use storage function
    from src.data.mlb_api import scrape_probable_pitchers, load_team_mapping
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Ensure src/config.py, src/data/utils.py, src/data/mlb_api.py, and src/scripts/data_fetcher.py exist.")
    MODULE_IMPORTS_OK = False
    # Define dummy logger if needed
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConfig: PATH = "data/pitcher_stats.db" # Need default path
    def store_data_to_sql(df, tn, dp, if_exists): print(f"Dummy store {tn}"); return True
    def load_team_mapping(p): print("Dummy load team map"); return None
    def scrape_probable_pitchers(tm): print("Dummy scrape"); return None, []


# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logger
log_dir = project_root / 'logs'
ensure_dir(log_dir)
logger = setup_logger('quick_fetch', log_file= log_dir / 'quick_fetch.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('quick_fetch_fallback')

# --- Configuration ---
TARGET_DATES = ['2025-04-09', '2025-04-10']
SCRAPE_DATE = datetime.now().strftime("%Y-%m-%d") # Date for scraping probable pitchers (usually today)
DB_PATH = project_root / DBConfig.PATH
RATE_LIMIT_PAUSE = 0.75 # Slightly reduced pause for fewer calls

# --- Helper Functions ---
def fetch_with_retries(fetch_function, *args, max_retries=3, retry_delay=4, **kwargs):
    """Execute a fetch function with retries (simplified)."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            time.sleep(RATE_LIMIT_PAUSE * (1.5**attempt)) # Exponential backoff on delay too
            return fetch_function(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {fetch_function.__name__}: {str(e)}")
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.error(f"All retries failed for {fetch_function.__name__}")
    raise last_exception


# --- Main Fetching Logic ---
def main():
    if not MODULE_IMPORTS_OK:
        logger.error("Exiting due to missing module imports.")
        return False

    logger.info(f"--- Starting Quick Fetch for Dates: {', '.join(TARGET_DATES)} ---")
    logger.info(f"Database Path: {DB_PATH}")
    ensure_dir(DB_PATH.parent)

    all_success = True

    # 1. Fetch Batter Statcast Data for Target Dates
    logger.info("Fetching Batter Statcast data...")
    batter_data_list = []
    for target_date in tqdm(TARGET_DATES, desc="Fetching Batter Dates"):
        logger.info(f"Fetching batter data for {target_date}...")
        try:
            daily_batter_data = fetch_with_retries(pb.statcast, start_dt=target_date, end_dt=target_date)
            if not daily_batter_data.empty:
                 daily_batter_data['season'] = int(target_date[:4]) # Add season
                 # Basic cleaning
                 num_cols = ['release_speed','launch_speed','launch_angle','woba_value']
                 for col in num_cols:
                     if col in daily_batter_data.columns: daily_batter_data[col]=pd.to_numeric(daily_batter_data[col],errors='coerce')
                 daily_batter_data.dropna(subset=['batter','pitcher','game_pk'], inplace=True)
                 batter_data_list.append(daily_batter_data)
                 logger.info(f"Fetched {len(daily_batter_data)} batter rows for {target_date}.")
            else:
                 logger.info(f"No batter data found for {target_date}.")
        except Exception as e:
            logger.error(f"Failed to fetch batter data for {target_date}: {e}")
            all_success = False # Mark failure but continue if possible

    if batter_data_list:
        full_batter_df = pd.concat(batter_data_list, ignore_index=True)
        logger.info(f"Storing {len(full_batter_df)} total batter rows...")
        if not store_data_to_sql(full_batter_df, 'statcast_batters', DB_PATH, if_exists='append'):
            all_success = False
    else:
        logger.warning("No batter data fetched for target dates.")

    # 2. Fetch Pitcher Statcast Data for Target Dates
    logger.info("Fetching Pitcher Statcast data...")
    # Load pitcher mapping first
    logger.info("Loading pitcher mapping...")
    try:
        with DBConnection(DB_PATH) as conn:
            if conn: pitcher_mapping = pd.read_sql_query("SELECT * FROM pitcher_mapping", conn)
            else: raise ConnectionError("DB connection failed.")
        if 'pitcher_id' not in pitcher_mapping.columns and 'key_mlbam' in pitcher_mapping.columns:
            pitcher_mapping.rename(columns={'key_mlbam':'pitcher_id'}, inplace=True)
        pitcher_mapping['pitcher_id'] = pitcher_mapping['pitcher_id'].astype(int) # Ensure int
        logger.info(f"Loaded {len(pitcher_mapping)} pitchers.")
    except Exception as e:
        logger.error(f"Failed to load pitcher mapping: {e}. Cannot fetch pitcher Statcast.")
        return False # Critical failure

    pitcher_data_list = []
    pitcher_ids = pitcher_mapping['pitcher_id'].unique()
    for target_date in TARGET_DATES:
         logger.info(f"Processing pitchers for {target_date}...")
         for pitcher_id in tqdm(pitcher_ids, desc=f"Pitchers {target_date}", leave=False):
              try:
                   daily_pitcher_data = fetch_with_retries(pb.statcast_pitcher, start_dt=target_date, end_dt=target_date, player_id=pitcher_id)
                   if not daily_pitcher_data.empty:
                        daily_pitcher_data['pitcher_id'] = pitcher_id
                        daily_pitcher_data['season'] = int(target_date[:4])
                        # Basic cleaning
                        num_cols = ['release_speed','release_spin_rate']
                        for col in num_cols:
                           if col in daily_pitcher_data.columns: daily_pitcher_data[col]=pd.to_numeric(daily_pitcher_data[col],errors='coerce')
                        daily_pitcher_data.dropna(subset=['game_pk','pitcher','batter','pitch_number'], inplace=True)
                        # Store pitcher by pitcher to manage memory
                        if not store_data_to_sql(daily_pitcher_data, 'statcast_pitchers', DB_PATH, if_exists='append'):
                             logger.warning(f"Failed to store data for pitcher {pitcher_id} on {target_date}.")
                             # Don't mark overall success as False here, maybe just log warning
                        # pitcher_data_list.append(daily_pitcher_data) # Alt: append and store at end
              except Exception as e:
                   logger.warning(f"Failed to fetch pitcher data for ID {pitcher_id} on {target_date}: {e}")
                   # Don't mark overall success as False here, continue

    # If appending to list instead of storing in loop:
    # if pitcher_data_list:
    #     full_pitcher_df = pd.concat(pitcher_data_list, ignore_index=True)
    #     logger.info(f"Storing {len(full_pitcher_df)} total pitcher rows...")
    #     if not store_data_to_sql(full_pitcher_df, 'statcast_pitchers', DB_PATH, if_exists='append'):
    #         all_success = False
    # else:
    #     logger.warning("No pitcher data fetched for target dates.")


    # 3. Fetch Scraped Probable Pitchers for *Today*
    logger.info(f"Fetching Scraped Probable Pitchers for {SCRAPE_DATE}...")
    team_map_df = load_team_mapping(DB_PATH)
    if team_map_df is not None:
        scraped_date, daily_data = scrape_probable_pitchers(team_map_df)
        if daily_data:
            logger.info(f"Scraped {len(daily_data)} games for {scraped_date or SCRAPE_DATE}")
            scraped_df = pd.DataFrame(daily_data)
            if 'game_date' not in scraped_df.columns or scraped_df['game_date'].isnull().all():
                scraped_df['game_date'] = scraped_date or SCRAPE_DATE # Use scraped or current date
            # Ensure columns / types
            expected_cols = ['gamePk','game_date','home_team_id','home_team_name','home_team_abbr','away_team_id','away_team_name','away_team_abbr','home_probable_pitcher_id','home_probable_pitcher_name','away_probable_pitcher_id','away_probable_pitcher_name']
            for col in expected_cols:
                if col not in scraped_df.columns: scraped_df[col] = pd.NA
            for col_id in ['gamePk','home_team_id','away_team_id','home_probable_pitcher_id','away_probable_pitcher_id']:
                scraped_df[col_id] = pd.to_numeric(scraped_df[col_id], errors='coerce').astype('Int64')
            scraped_df = scraped_df[expected_cols]
            # Store using REPLACE for the daily scrape table
            logger.info(f"Storing scraped data to 'mlb_api' table (replacing existing)...")
            if not store_data_to_sql(scraped_df, 'mlb_api', DB_PATH, if_exists='replace'):
                all_success = False
        else:
            logger.warning(f"No probable pitchers scraped for {SCRAPE_DATE}.")
    else:
        logger.warning("Could not load team mapping. Skipping probable pitcher scrape.")
        # Don't mark as failure if mapping is missing, just skip

    logger.info(f"--- Quick Fetch Finished ---")
    return all_success


if __name__ == "__main__":
    # No argument parsing needed for this specific script
    ensure_dir(project_root / 'data'); ensure_dir(project_root / 'logs')
    if not MODULE_IMPORTS_OK:
        logger.error("Exiting due to missing module imports.")
        sys.exit(1)

    success = main() # Execute the main logic

    if success:
        logger.info("Quick fetch script finished successfully.")
        sys.exit(0)
    else:
        logger.error("Quick fetch script finished with errors.")
        sys.exit(1)