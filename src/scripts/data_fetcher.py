# src/scripts/data_fetcher.py
# --- MODIFIED ---

import sqlite3
import pandas as pd
import numpy as np
import pybaseball as pb # Keep for statcast functions
import logging
import time
import json
import os
import argparse
import traceback
import signal
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from tqdm import tqdm # Keep for progress bars
from concurrent.futures import ThreadPoolExecutor, as_completed # Keep for parallel fetching
import warnings
import threading # Needed for parallel checkpoint updates

# Columns that uniquely identify a pitch in Statcast data
UNIQUE_PITCH_COLS = [
    "game_pk",
    "pitcher",
    "inning",
    "batter",
    "pitch_number",
]


def dedup_pitch_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate pitch rows based on key columns."""
    before = len(df)
    df = df.drop_duplicates(subset=UNIQUE_PITCH_COLS)
    dropped = before - len(df)
    if dropped > 0:
        logger.debug(f"Removed {dropped} duplicate pitch rows")
    return df


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Return only regular season rows if ``game_type`` column is present."""
    if "game_type" in df.columns:
        before = len(df)
        df = df[df["game_type"] == "R"]
        removed = before - len(df)
        if removed > 0:
            logger.debug(f"Filtered {removed} non-regular season rows")
    return df

# --- REMOVED requests and bs4 imports as they are no longer needed here ---
# try: import requests; import bs4
# except ImportError: pass

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path: sys.path.append(str(project_root))

try:
    from src.config import DBConfig, DataConfig, LogConfig  # Added LogConfig
    from src.utils import setup_logger, ensure_dir, DBConnection
    # Updated import: scrape_probable_pitchers no longer needs team_mapping_df
    from src.data.mlb_api import scrape_probable_pitchers # Removed load_team_mapping import here
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Imports failed: {e}"); MODULE_IMPORTS_OK = False
    # Fallback definitions
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('data_fetcher_fallback')
    class DataConfig:
        SEASONS = list(range(2019, datetime.now().year + 1)); RATE_LIMIT_PAUSE = 1; CHUNK_SIZE = 14; MAX_RETRIES=3; RETRY_DELAY=5
    class DBConfig: PATH = "data/pitcher_stats.db"
    class LogConfig: LOG_DIR = project_root / 'logs'
    def setup_logger(name, level=logging.INFO, log_file=None): return logger
    def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)
    class DBConnection: # Basic fallback
        def __init__(self, db_path): self.db_path = db_path
        def __enter__(self): import sqlite3; self.conn = sqlite3.connect(self.db_path); return self.conn
        def __exit__(self,et,ev,tb): self.conn.close()
    # Fallback for scrape_probable_pitchers if import fails
    def scrape_probable_pitchers(target_date_str): logger.error("Fallback scrape_probable_pitchers called."); return []


warnings.filterwarnings("ignore", category=FutureWarning)

log_dir = LogConfig.LOG_DIR if MODULE_IMPORTS_OK else project_root / 'logs'
ensure_dir(log_dir)
logger = setup_logger('data_fetcher', log_file=log_dir/'data_fetcher.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('data_fetcher_fallback')

# --- CheckpointManager Class (MODIFIED for new batter checkpoint) ---
class CheckpointManager:
    def __init__(self, checkpoint_dir=project_root / 'data' / '.checkpoints'): # Changed default location
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_file = self.checkpoint_dir / 'data_fetcher_progress.json' # Changed filename
        self.current_checkpoint = {}
        self.lock = threading.Lock() # Lock for thread-safe updates
        try:
            self.load_overall_checkpoint()
        except Exception as e:
            logger.error(f"CheckpointManager init error: {e}. Init fresh.")
            self._initialize_checkpoint()
            self.save_overall_checkpoint()

    def load_overall_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as fp:
                    self.current_checkpoint = json.load(fp)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}.")
            except json.JSONDecodeError as e:
                 logger.error(f"Failed to decode checkpoint JSON ({e}) from {self.checkpoint_file}. Initializing new checkpoint.")
                 self._initialize_checkpoint()
            except Exception as e:
                logger.error(f"Failed load checkpoint file {self.checkpoint_file} ({e}). Initializing new checkpoint.")
                self._initialize_checkpoint()
        else:
            logger.info("No checkpoint file found. Initializing new checkpoint.")
            self._initialize_checkpoint()
        self._ensure_keys()

    def _initialize_checkpoint(self):
        # Removed team_batting_completed, processed_seasons_batter_data
        # Added last_processed_batter_date
        self.current_checkpoint = {
            'processed_pitcher_ids': [],
            'last_processed_batter_date': {}, # Stores { "season_str": "YYYY-MM-DD", ... }
            'processed_mlb_api_dates': [],
            'last_update': datetime.now().isoformat()
        }

    def _ensure_keys(self):
        # Updated defaults for new structure
        defaults = {
            'processed_pitcher_ids': [],
            'last_processed_batter_date': {},
            'processed_mlb_api_dates': [],
            'last_update': datetime.now().isoformat()
        }
        updated = False
        for key, default_val in defaults.items():
            current_val = self.current_checkpoint.get(key)
            correct_type = isinstance(current_val, type(default_val))
            # Check if key exists, value is not None, and type is correct
            if key not in self.current_checkpoint or current_val is None or not correct_type:
                 # Skip logging warning for last_update initialization
                if key != 'last_update':
                     logger.warning(f"Initializing/Resetting checkpoint key '{key}' (was type {type(current_val)}). Setting to default: {default_val}")
                self.current_checkpoint[key] = default_val
                updated = True
        # Clean up old keys if they exist
        old_keys = ['pitcher_mapping_completed', 'processed_seasons_batter_data', 'team_batting_completed']
        for old_key in old_keys:
             if old_key in self.current_checkpoint:
                  logger.warning(f"Removing obsolete checkpoint key: '{old_key}'")
                  del self.current_checkpoint[old_key]
                  updated = True
        # If any key was missing/reset or old keys removed, save the updated structure
        if updated: self.save_overall_checkpoint()


    def save_overall_checkpoint(self):
        with self.lock: # Ensure thread safety during save
            self.current_checkpoint['last_update'] = datetime.now().isoformat()
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            try:
                # Ensure lists are sorted sets of primitives
                for key in ['processed_pitcher_ids', 'processed_mlb_api_dates']:
                    current_list = self.current_checkpoint.get(key, [])
                    if isinstance(current_list, list):
                        try:
                            # Convert potential numpy types to primitives, get unique, sort
                            s_list = [item.item() if hasattr(item, 'item') else item for item in current_list]
                            self.current_checkpoint[key] = sorted(list(set(s_list)))
                        except TypeError:
                            logger.warning(f"Sorting or cleaning checkpoint list '{key}' failed. Saving as is.")
                            # Save the potentially problematic list as is if sorting/cleaning fails
                            self.current_checkpoint[key] = current_list

                with open(temp_file, 'w') as fp:
                    json.dump(self.current_checkpoint, fp, indent=4)
                # Atomic rename to replace the old file
                os.replace(temp_file, self.checkpoint_file)
                logger.debug(f"Saved checkpoint to {self.checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint to {self.checkpoint_file}: {e}")
                # Attempt to remove the temporary file if save failed
                if temp_file.exists():
                    try: os.remove(temp_file)
                    except OSError: logger.error(f"Failed to remove temporary checkpoint file {temp_file}")

    # --- Pitcher Checkpoint Methods (Unchanged) ---
    def add_processed_pitcher(self, pitcher_id):
        with self.lock:
            p_list = self.current_checkpoint.setdefault('processed_pitcher_ids', [])
            # Convert numpy types if necessary
            item_id = pitcher_id.item() if hasattr(pitcher_id, 'item') else pitcher_id
            if item_id not in p_list:
                p_list.append(item_id)

    def is_pitcher_processed(self, pitcher_id):
        item_id = pitcher_id.item() if hasattr(pitcher_id, 'item') else pitcher_id
        return item_id in self.current_checkpoint.get('processed_pitcher_ids', [])

    # --- Batter Checkpoint Methods (NEW) ---
    def get_last_processed_batter_date(self, season):
        """Returns the last processed date string (YYYY-MM-DD) for the season, or None."""
        season_str = str(season)
        last_date_str = self.current_checkpoint.get('last_processed_batter_date', {}).get(season_str)
        if last_date_str:
            try:
                datetime.strptime(last_date_str, "%Y-%m-%d")
                return last_date_str
            except ValueError:
                logger.warning(f"Invalid date format '{last_date_str}' found in checkpoint for season {season}. Ignoring.")
                # Optionally remove the invalid date
                with self.lock:
                     if season_str in self.current_checkpoint.get('last_processed_batter_date', {}):
                          del self.current_checkpoint['last_processed_batter_date'][season_str]
                return None
        return None

    def update_last_processed_batter_date(self, season, date_str):
        """Updates the last processed date for the season. Assumes date_str is valid YYYY-MM-DD."""
        with self.lock:
            season_str = str(season)
            date_map = self.current_checkpoint.setdefault('last_processed_batter_date', {})
            # Only update if the new date is later than the existing one
            current_last_date_str = date_map.get(season_str)
            should_update = True
            if current_last_date_str:
                 try:
                      current_dt = datetime.strptime(current_last_date_str, "%Y-%m-%d").date()
                      new_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                      if new_dt <= current_dt:
                           should_update = False
                 except ValueError:
                      logger.warning(f"Cannot compare with invalid existing date '{current_last_date_str}' for season {season}. Overwriting.")
                      # Force update if existing date is invalid
                      should_update = True

            if should_update:
                date_map[season_str] = date_str
                logger.debug(f"Updated last processed batter date for season {season} to {date_str}")


    # --- MLB API Checkpoint Methods (Unchanged logic, added lock) ---
    def get_last_processed_mlb_api_date(self):
        processed_list = self.current_checkpoint.get('processed_mlb_api_dates', [])
        if not processed_list:
            return None
        # Assume list is sorted, get the last one
        last_date_str = processed_list[-1]
        try:
            datetime.strptime(last_date_str, "%Y-%m-%d")
            return last_date_str
        except (ValueError, TypeError):
            logger.warning(f"Invalid date format found in processed_mlb_api_dates: {last_date_str}. Trying previous dates.")
            # Iterate backwards to find the last valid date
            for dt_str in reversed(processed_list[:-1]):
                 try:
                      datetime.strptime(dt_str, "%Y-%m-%d"); return dt_str
                 except (ValueError, TypeError): continue
            logger.warning("No valid dates found in processed_mlb_api_dates.")
            return None # Return None if no valid date found

    def add_processed_mlb_api_date(self, date_str):
        with self.lock:
            processed_list = self.current_checkpoint.setdefault('processed_mlb_api_dates', [])
            if date_str not in processed_list:
                processed_list.append(date_str)
                # Keep the list sorted after adding
                try: self.current_checkpoint['processed_mlb_api_dates'] = sorted(processed_list)
                except TypeError: logger.warning("Could not sort processed_mlb_api_dates list.")


    def is_mlb_api_date_processed(self, date_str):
         # Added check for non-string input just in case
        if not isinstance(date_str, str): return False
        return date_str in self.current_checkpoint.get('processed_mlb_api_dates', [])


# --- DataFetcher Class (MODIFIED)---
class DataFetcher:
    def __init__(self, args):
        if not MODULE_IMPORTS_OK: raise ImportError("Core module imports failed. Cannot initialize DataFetcher.")
        self.args = args
        self.db_path = Path(DBConfig.PATH)

        # --- Mode Determination (Unchanged) ---
        self.single_date_historical_mode = (not args.mlb_api and args.date is not None)
        if self.single_date_historical_mode:
             logger.info("Running in Single-Date Historical Fetch mode.")
             try: self.target_fetch_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
             except ValueError: logger.error(f"Invalid date format for --date: {args.date}. Use YYYY-MM-DD."); sys.exit(1)
             self.seasons_to_fetch = [self.target_fetch_date_obj.year]
             self.end_date_limit = self.target_fetch_date_obj
        elif args.mlb_api:
             logger.info("Running in MLB API Scraper mode.")
             if not args.date: logger.error("--mlb-api requires --date."); sys.exit(1)
             try: self.target_fetch_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
             except ValueError: logger.error(f"Invalid date format for --date: {args.date}. Use YYYY-MM-DD."); sys.exit(1)
             self.seasons_to_fetch = [] # No historical seasons needed for this mode
             self.end_date_limit = self.target_fetch_date_obj # Limit doesn't really apply here
        else: # Default historical backfill mode
             logger.info("Running in Full Historical Backfill mode.")
             # Use DataConfig seasons if available, else default range
             default_seasons = list(range(2019, date.today().year + 1))
             config_seasons = getattr(DataConfig, 'SEASONS', default_seasons)
             self.seasons_to_fetch = sorted(args.seasons if args.seasons else config_seasons)
             self.end_date_limit = date.today() - timedelta(days=1)
             # Use the calculated end_date_limit as the target date for determining the latest season end
             self.target_fetch_date_obj = self.end_date_limit

        self.end_date_limit_str = self.end_date_limit.strftime('%Y-%m-%d')
        logger.info(f"Effective End Date Limit for historical fetches: {self.end_date_limit_str}")
        logger.info(f"Seasons to consider for fetching (if historical): {self.seasons_to_fetch}")

        self.checkpoint_manager = CheckpointManager()
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

        # --- Pybaseball Cache (Unchanged) ---
        try:
            pb.cache.enable()
            logger.info("Pybaseball cache enabled.")
        except Exception as e:
            logger.warning(f"Failed to enable pybaseball cache: {e}")

        ensure_dir(Path(self.db_path).parent)

        # --- REMOVED team_mapping_df loading here ---
        # self.team_mapping_df = None
        # if args.mlb_api:
        #     # Team mapping is no longer needed by scrape_probable_pitchers
        #     # self.team_mapping_df = load_team_mapping(self.db_path)
        #     # if self.team_mapping_df is None: logger.error("Mapping needed for --mlb-api failed load.")
        #     logger.info("Team mapping is no longer loaded/required for --mlb-api mode.")
        #     pass

        # --- NEW: Initialize error tracking sets ---
        self.problematic_pitcher_ids = set()
        self.failed_batter_fetches = set() # Store tuples (season, start_date, end_date)
        self.failed_mlb_api_fetches = set() # Store date strings

    def handle_interrupt(self, signum, frame):
        logger.warning(f"Interrupt signal ({signum}) received. Saving checkpoint and reporting issues...")
        self._report_issues() # Report issues on interrupt
        self.checkpoint_manager.save_overall_checkpoint()
        logger.info("Exiting due to interrupt signal.")
        sys.exit(0)

    def _report_issues(self):
        """Logs any tracked errors at the end of the run."""
        if self.problematic_pitcher_ids:
            sorted_ids = sorted(list(self.problematic_pitcher_ids))
            logger.warning(f"Completed with persistent fetch errors for pitcher IDs: {sorted_ids}")
            print(f"\nWARNING: Persistent fetch errors encountered for pitcher IDs: {sorted_ids}")
        if self.failed_batter_fetches:
            sorted_failures = sorted(list(self.failed_batter_fetches))
            logger.warning(f"Completed with persistent fetch errors for batter date ranges: {sorted_failures}")
            print(f"\nWARNING: Persistent fetch errors encountered for batter ranges (season, start, end): {sorted_failures}")
        if self.failed_mlb_api_fetches:
             sorted_dates = sorted(list(self.failed_mlb_api_fetches))
             logger.warning(f"Completed with persistent fetch errors for MLB API dates: {sorted_dates}")
             print(f"\nWARNING: Persistent fetch errors encountered for MLB API dates: {sorted_dates}")

        if not any([self.problematic_pitcher_ids, self.failed_batter_fetches, self.failed_mlb_api_fetches]):
            logger.info("Fetching process completed with no persistent errors tracked.")


    def fetch_with_retries(self, fetch_function, *args, max_retries=None, retry_delay=None, **kwargs):
        """Wrapper for pybaseball calls with retries and delays."""
        # Use config values if available, otherwise defaults
        retries = max_retries if max_retries is not None else getattr(DataConfig, 'MAX_RETRIES', 3)
        delay = retry_delay if retry_delay is not None else getattr(DataConfig, 'RETRY_DELAY', 5)
        pause_base = getattr(DataConfig, 'RATE_LIMIT_PAUSE', 1)

        last_exception = None
        for attempt in range(retries):
            try:
                # Apply a small base delay before each attempt
                base_sleep = (pause_base / 2) * (1.5**attempt)
                time.sleep(base_sleep)
                # Make the actual call
                data = fetch_function(*args, **kwargs)
                # Check if pybaseball returned an empty DataFrame, which is valid (no data)
                # If it's not a DataFrame or other error occurs, it will raise exception below
                if isinstance(data, pd.DataFrame):
                     # Return even if empty, signifies no data for the request
                     return data
                else:
                     # If pybaseball returns something unexpected (not DF, not Exception)
                     logger.warning(f"Unexpected return type from {fetch_function.__name__}: {type(data)}. Treating as failure.")
                     last_exception = TypeError(f"Unexpected return type: {type(data)}")
                     # Continue to retry logic

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{retries} failed for {fetch_function.__name__} with args {args[:2]}...: {e}") # Log only first few args
                # Exponential backoff for retries
                sleep_time = delay * (2**attempt)
                time.sleep(sleep_time)

        logger.error(f"All {retries} retries failed for {fetch_function.__name__} with args {args[:2]}...")
        # Instead of raising, return None to indicate definitive failure after retries
        # The calling function can then decide how to handle this (e.g., log to problematic sets)
        # raise last_exception # Original behavior
        return None # Return None to indicate fetch failure after retries

    # --- Pitcher ID Mapping (Unchanged from previous version) ---
    def fetch_pitcher_id_mapping(self):
        """Loads pitcher mapping (mlbid, playername) from the 'pitcher_mapping' table."""
        logger.info("Loading pitcher mapping from database...")
        mapping_table = 'pitcher_mapping'
        required_cols_db = ['mlbid', 'playername']
        final_cols = ['pitcher_id', 'name']
        try:
            with DBConnection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{mapping_table}'")
                if not cursor.fetchone():
                    logger.error(f"Pitcher mapping table '{mapping_table}' not found.")
                    return pd.DataFrame()
                cursor.execute(f"PRAGMA table_info({mapping_table})")
                available_cols = [info[1] for info in cursor.fetchall()]
                missing_cols = [col for col in required_cols_db if col not in available_cols]
                if missing_cols:
                    logger.error(f"Pitcher mapping table '{mapping_table}' missing required columns: {missing_cols}. Available: {available_cols}")
                    return pd.DataFrame()
                query = f"SELECT {', '.join(required_cols_db)} FROM {mapping_table}"
                pm = pd.read_sql_query(query, conn)
            if pm.empty:
                logger.warning(f"Pitcher mapping table '{mapping_table}' is empty.")
                return pd.DataFrame()
            rename_dict = dict(zip(required_cols_db, final_cols))
            pm = pm.rename(columns=rename_dict)
            pm['pitcher_id'] = pd.to_numeric(pm['pitcher_id'], errors='coerce')
            # Use nullable integer type Int64 AFTER dropping NaNs
            pm = pm.dropna(subset=['pitcher_id'])
            pm['pitcher_id'] = pm['pitcher_id'].astype('Int64')
            pm['name'] = pm['name'].astype(str)
            pm = pm.dropna(subset=final_cols) # Drop if name became NaN or pitcher_id conversion failed earlier
            if pm.empty:
                 logger.warning("Pitcher mapping loaded, but resulted in empty DataFrame after cleaning/conversion.")
                 return pd.DataFrame()
            logger.info(f"Successfully loaded {len(pm)} pitcher mappings from '{mapping_table}'.")
            return pm
        except sqlite3.Error as e:
            logger.error(f"SQLite error loading pitcher mapping: {e}", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error loading pitcher mapping: {e}", exc_info=True)
            return pd.DataFrame()

    # --- fetch_statcast_for_pitcher (REFACTORED into helpers) ---
    # This method is now removed, logic moved to helpers below

    # --- NEW Helper: Fetch Pitcher Statcast (Single Date) ---
    def _fetch_pitcher_statcast_single_date(self, pitcher_id, name, target_date_obj):
        """Fetches pitcher statcast data for a single specific date."""
        target_date_str = target_date_obj.strftime("%Y-%m-%d")
        target_season = target_date_obj.year
        data_exists = False
        try:
            with DBConnection(self.db_path) as conn:
                # Check if data already exists for this specific pitcher and date
                query = "SELECT COUNT(*) FROM statcast_pitchers WHERE DATE(game_date) = ? AND pitcher = ?"
                cursor = conn.cursor()
                # Ensure pitcher_id is a primitive type for the query
                pid_primitive = pitcher_id.item() if hasattr(pitcher_id, 'item') else pitcher_id
                cursor.execute(query, (target_date_str, pid_primitive))
                count = cursor.fetchone()[0]
                if count > 0: data_exists = True
        except Exception as e:
            logger.warning(f"DB check failed for pitcher {name} ({pitcher_id}) on {target_date_str}: {e}. Will attempt fetch.")

        if data_exists:
            logger.debug(f" -> Skipping P fetch {name} ({pitcher_id}): Data already exists for {target_date_str}")
            return pd.DataFrame() # Return empty DF, not None

        logger.debug(f" -> Fetch P {name} ({pitcher_id}) for single date: {target_date_str}")
        # Use fetch_with_retries, it now returns None on definitive failure
        pd_data = self.fetch_with_retries(pb.statcast_pitcher, target_date_str, target_date_str, pitcher_id)

        # Drop spring training or other non-regular season rows
        pd_data = filter_regular_season(pd_data)

        if pd_data is None: # Fetch failed after retries
            logger.error(f" -> Error fetching P {name} ({pitcher_id}) single date {target_date_str} after retries.")
            self.problematic_pitcher_ids.add(pitcher_id) # Add to problematic set
            return pd.DataFrame() # Return empty DF on failure

        if not pd_data.empty:
            try:
                # Add identifiers and clean numeric columns
                pid_primitive = pitcher_id.item() if hasattr(pitcher_id, 'item') else pitcher_id
                pd_data['pitcher_id'] = pid_primitive # Store primitive type
                pd_data['season'] = target_season
                numeric_cols = ['release_speed', 'release_spin_rate', 'launch_speed', 'launch_angle']
                for col in numeric_cols:
                    if col in pd_data.columns:
                        pd_data[col] = pd.to_numeric(pd_data[col], errors='coerce')
                # Ensure essential columns are present before dropping NaNs
                essential_cols = ['game_pk', 'pitcher', 'batter', 'pitch_number']
                if all(col in pd_data.columns for col in essential_cols):
                     # Use dropna without inplace=True
                     pd_data = pd_data.dropna(subset=essential_cols)
                     pd_data = dedup_pitch_df(pd_data)
                else:
                     logger.warning(f"Missing essential columns in fetched data for P {name} ({pitcher_id}) on {target_date_str}. Cannot clean NaNs reliably.")

                logger.debug(f" -> Fetched {len(pd_data)} rows for P {name} single date")
                return pd_data

            except Exception as e:
                logger.error(f"Error processing fetched data for P {name} single date {target_date_str}: {e}")
                self.problematic_pitcher_ids.add(pitcher_id) # Mark as problematic if processing fails
                return pd.DataFrame() # Return empty DF
        else:
            logger.debug(f" -> No data found for P {name} ({pitcher_id}) on {target_date_str}")
            return pd.DataFrame() # Return empty DF if pybaseball returned empty


    # --- NEW Helper: Fetch Pitcher Statcast (Historical) ---
    def _fetch_pitcher_statcast_historical(self, pitcher_id, name, seasons_list):
        """Fetches pitcher statcast data across multiple seasons for historical backfill."""
        # Skip if already processed based on checkpoint
        if self.checkpoint_manager.is_pitcher_processed(pitcher_id):
            logger.debug(f" -> Skipping P fetch {name} ({pitcher_id}): Already processed per checkpoint.")
            return pd.DataFrame() # Return empty DF, not None

        all_data = []
        logger.debug(f" -> Hist fetch P: {name} ({pitcher_id}) for seasons {seasons_list}")
        # Filter seasons based on the overall end date limit
        relevant_seasons = [s for s in seasons_list if s <= self.end_date_limit.year]

        fetch_failed = False # Flag if any season fetch fails definitively
        for season in relevant_seasons:
            # Determine date range for the season
            start_dt = date(season, 3, 1) # Approximate season start
            # Use end_date_limit if it's the current year, otherwise use a default end date
            end_dt = self.end_date_limit if season == self.end_date_limit.year else date(season, 11, 30)

            # Skip if the date range is invalid (start after end)
            if start_dt > end_dt:
                logger.debug(f" -> Skipping season {season} for P {name}: Start date {start_dt} is after end date {end_dt}.")
                continue

            start_str = start_dt.strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
            logger.debug(f" -> Fetching {name} ({season}): {start_str} to {end_str}")

            # Use fetch_with_retries
            pd_data = self.fetch_with_retries(pb.statcast_pitcher, start_str, end_str, pitcher_id)

            # Filter to regular season games only
            pd_data = filter_regular_season(pd_data)

            if pd_data is None: # Fetch failed after retries for this season
                logger.error(f" -> Error fetching Hist Statcast P {name} ({pitcher_id}) season {season} after retries.")
                fetch_failed = True # Mark that at least one fetch failed
                # Continue to try other seasons, but this pitcher ID will be marked problematic later
                continue # Skip appending data for this failed season

            if not pd_data.empty:
                try:
                    pid_primitive = pitcher_id.item() if hasattr(pitcher_id, 'item') else pitcher_id
                    pd_data['pitcher_id'] = pid_primitive
                    pd_data['season'] = season
                    all_data.append(pd_data)
                    logger.debug(f" -> Fetched {len(pd_data)} rows for {name} ({season})")
                except Exception as e:
                    logger.error(f" -> Error adding identifiers for {name} ({season}): {e}")
                    fetch_failed = True # Mark as problematic if basic processing fails


        # After trying all seasons:
        if fetch_failed:
            self.problematic_pitcher_ids.add(pitcher_id) # Add to problematic set if any season fetch failed

        if not all_data:
            logger.debug(f" -> No historical Statcast data found or fetched successfully for {name} ({pitcher_id}).")
            # If no data was *found* (all fetches returned empty), still mark as processed if no errors occurred.
            if not fetch_failed:
                 self.checkpoint_manager.add_processed_pitcher(pitcher_id)
                 logger.debug(f" -> Marked P {name} ({pitcher_id}) as processed (no data found/errors).")
            return pd.DataFrame() # Return empty DF

        # Concatenate and clean data if we have results
        try:
            combined_data = pd.concat(all_data, ignore_index=True)
            numeric_cols = ['release_speed', 'release_spin_rate', 'launch_speed', 'launch_angle']
            for col in numeric_cols:
                if col in combined_data.columns:
                    combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

            essential_cols = ['game_pk', 'pitcher', 'batter', 'pitch_number']
            if all(col in combined_data.columns for col in essential_cols):
                # Use dropna without inplace=True
                combined_data = combined_data.dropna(subset=essential_cols)
                combined_data = dedup_pitch_df(combined_data)
            else:
                logger.warning(f"Missing essential columns in combined historical data for P {name} ({pitcher_id}). Cannot clean NaNs reliably.")


            logger.debug(f"Combined {len(combined_data)} historical rows for P {name}.")
            # Mark as processed only if concatenation and cleaning were successful
            # Do not mark as processed if fetch_failed was true earlier
            if not fetch_failed:
                 self.checkpoint_manager.add_processed_pitcher(pitcher_id)
                 logger.debug(f" -> Marked P {name} ({pitcher_id}) as processed.")

            return combined_data

        except Exception as e:
            logger.error(f"Error combining or cleaning Hist Statcast for {name} ({pitcher_id}): {e}")
            self.problematic_pitcher_ids.add(pitcher_id) # Mark as problematic if combine/clean fails
            return pd.DataFrame() # Return empty DF


    # --- fetch_all_pitchers (MODIFIED to use helpers) ---
    def fetch_all_pitchers(self, pitcher_mapping):
        """Fetch data for pitchers based on the operating mode."""
        if pitcher_mapping is None or pitcher_mapping.empty:
            logger.error("Pitcher mapping is empty or failed to load. Cannot fetch pitcher data.")
            return False # Indicate failure

        # Validate pitcher_mapping DataFrame structure (already done in load function, but double check)
        if 'pitcher_id' not in pitcher_mapping.columns or 'name' not in pitcher_mapping.columns:
             logger.error("Loaded pitcher mapping is missing required columns ('pitcher_id', 'name').")
             return False
        # Ensure pitcher_id is suitable (Int64 allows NaN, which should have been dropped)
        # if not pd.api.types.is_integer_dtype(pitcher_mapping['pitcher_id']) and not pd.api.types.is_float_dtype(pitcher_mapping['pitcher_id']):
             # Check if it's Int64 or potentially float if NaNs were present before drop
        #     logger.warning(f"Pitcher IDs are not integers/floats after loading: {pitcher_mapping['pitcher_id'].dtype}. Trying to proceed.")


        pitcher_list = list(zip(pitcher_mapping['pitcher_id'], pitcher_mapping['name']))
        total_pitchers = len(pitcher_list)
        fetch_tasks = [] # List of tuples (pitcher_id, name, *args_for_fetch_func)

        if self.single_date_historical_mode:
            logger.info(f"Fetching pitcher Statcast for single date: {self.target_fetch_date_obj.strftime('%Y-%m-%d')}")
            # Prepare tasks for the single-date helper
            for pid, name in pitcher_list:
                fetch_tasks.append((pid, name, self.target_fetch_date_obj)) # pid, name, date_obj
            fetch_function = self._fetch_pitcher_statcast_single_date
            mode_desc = "Pitcher Statcast (Single Date)"
            process_desc = f"Processing {total_pitchers} pitchers for {self.target_fetch_date_obj}"
            use_checkpoint = False
        else: # Historical backfill mode
            logger.info(f"Fetching pitcher Statcast historically up to {self.end_date_limit_str} for seasons {self.seasons_to_fetch}")
            # Filter out already processed pitchers using the checkpoint
            unprocessed_pitchers = [(pid, name) for pid, name in pitcher_list if not self.checkpoint_manager.is_pitcher_processed(pid)]
            processed_count_prev = total_pitchers - len(unprocessed_pitchers)
            if processed_count_prev > 0: logger.info(f"Skipping {processed_count_prev} pitchers already processed per checkpoint.")

            if not unprocessed_pitchers:
                logger.info("No new pitchers require historical fetching based on checkpoint.")
                return True # Success, nothing to do

            total_pitchers = len(unprocessed_pitchers) # Update total for progress bar
            # Prepare tasks for the historical helper
            for pid, name in unprocessed_pitchers:
                fetch_tasks.append((pid, name, self.seasons_to_fetch)) # pid, name, seasons_list
            fetch_function = self._fetch_pitcher_statcast_historical
            mode_desc = "Pitcher Statcast (Historical)"
            process_desc = f"Fetching {total_pitchers} unprocessed pitchers historically"
            use_checkpoint = True

        logger.info(process_desc)
        processed_count = 0
        success_flag = True
        data_stored_count = 0

        # --- Parallel vs Sequential Execution ---
        if self.args.parallel:
            workers = min(DataConfig.MAX_WORKERS, os.cpu_count() or 1)
            logger.info(f"Using PARALLEL pitcher fetch ({workers} workers).")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Map futures back to pitcher info
                future_to_pitcher = {
                    executor.submit(fetch_function, pid, name, *fetch_args): (pid, name)
                    for pid, name, *fetch_args in fetch_tasks
                }
                progress_bar = tqdm(as_completed(future_to_pitcher), total=total_pitchers, desc=f"{mode_desc} (Parallel)")
                for future in progress_bar:
                    pid, name = future_to_pitcher[future]
                    try:
                        data_df = future.result() # This is the DataFrame returned by the helper
                        # data_df can be empty if no data found, fetch failed, or processing failed
                        if data_df is not None and not data_df.empty:
                             # Store the data (append mode is safe for both historical and single date)
                             save_ok = store_data_to_sql(data_df, 'statcast_pitchers', self.db_path, if_exists='append')
                             if save_ok:
                                 data_stored_count += len(data_df)
                                 logger.debug(f"Stored {len(data_df)} rows for P {name}")
                             else:
                                 logger.warning(f"Failed to store {len(data_df)} rows for P {name}.")
                                 success_flag = False
                                 self.problematic_pitcher_ids.add(pid) # Also flag if storage fails
                        # Note: Checkpoint update for historical is now handled *within* _fetch_pitcher_statcast_historical
                        processed_count += 1
                        # Save checkpoint periodically during historical runs
                        if use_checkpoint and processed_count % 100 == 0:
                             logger.info(f"Saving checkpoint mid-fetch ({processed_count}/{total_pitchers})...")
                             self.checkpoint_manager.save_overall_checkpoint()

                    except Exception as exc:
                        logger.error(f"Error processing result for pitcher {name} ({pid}): {exc}", exc_info=True)
                        success_flag = False
                        self.problematic_pitcher_ids.add(pid) # Ensure added to problematic set on unexpected future error
                    finally:
                        # Update progress bar description dynamically if needed
                        progress_bar.set_postfix_str(f"Stored: {data_stored_count}, Errors: {len(self.problematic_pitcher_ids)}")


        else: # Sequential Execution
            logger.info("Using SEQUENTIAL pitcher fetch.")
            progress_bar = tqdm(fetch_tasks, total=total_pitchers, desc=f"{mode_desc} (Sequential)")
            for task in progress_bar:
                pid, name, *fetch_args = task
                try:
                    data_df = fetch_function(pid, name, *fetch_args)
                    if data_df is not None and not data_df.empty:
                        save_ok = store_data_to_sql(data_df, 'statcast_pitchers', self.db_path, if_exists='append')
                        if save_ok:
                            data_stored_count += len(data_df)
                            logger.debug(f"Stored {len(data_df)} rows for P {name}")
                        else:
                            logger.warning(f"Failed to store {len(data_df)} rows for P {name}.")
                            success_flag = False
                            self.problematic_pitcher_ids.add(pid) # Flag if storage fails
                    # Checkpoint update handled within historical helper
                    processed_count += 1
                    # Save checkpoint periodically
                    if use_checkpoint and (processed_count % 100 == 0 or processed_count == total_pitchers):
                         logger.info(f"Saving checkpoint mid-fetch ({processed_count}/{total_pitchers})...")
                         self.checkpoint_manager.save_overall_checkpoint()

                except Exception as e:
                    logger.error(f"Critical error processing pitcher {name} ({pid}) sequentially: {e}", exc_info=True)
                    success_flag = False
                    self.problematic_pitcher_ids.add(pid)
                finally:
                    # Update progress bar description dynamically if needed
                    progress_bar.set_postfix_str(f"Stored: {data_stored_count}, Errors: {len(self.problematic_pitcher_ids)}")


        # Final checkpoint save after the loop finishes for historical mode
        if use_checkpoint:
             logger.info("Saving final checkpoint for pitcher historical fetch.")
             self.checkpoint_manager.save_overall_checkpoint()

        logger.info(f"Pitcher Statcast fetching phase complete. Processed {processed_count}/{total_pitchers}. Stored {data_stored_count} new rows.")
        if self.problematic_pitcher_ids:
             logger.warning(f"Encountered persistent errors fetching data for {len(self.problematic_pitcher_ids)} pitchers (see logs and final report).")

        return success_flag


    # --- fetch_team_batting_data (Unchanged - simple fetch/replace) ---
    def fetch_team_batting_data(self):
        """ Fetch team batting data. Skips if in single date mode. """
        if self.single_date_historical_mode:
            logger.info("Skipping team batting fetch in single-date mode.")
            return True

        logger.info(f"Fetching team batting stats up to year {self.end_date_limit.year}")
        all_team_data = []
        # Determine seasons to fetch based on overall limit
        seasons_hist = sorted(self.args.seasons if self.args.seasons else getattr(DataConfig, 'SEASONS', []))
        seasons_to_check = [s for s in seasons_hist if s <= self.end_date_limit.year]

        if not seasons_to_check:
             logger.info("No relevant seasons for team batting fetch based on end date limit.")
             return True

        for season in tqdm(seasons_to_check, desc="Fetching Team Batting Data"):
            # Use fetch_with_retries
            team_data = self.fetch_with_retries(pb.team_batting, season, season) # Fetches for the full season

            if team_data is None: # Fetch failed definitively
                logger.error(f"Failed to fetch team batting data for season {season} after retries.")
                # Decide if this should halt the process or just be logged
                # For now, just log and continue, but don't add partial data
                continue # Skip this season

            if not team_data.empty:
                team_data['Season'] = season # Add season identifier
                all_team_data.append(team_data)
            else:
                logger.debug(f"No team batting data found for season {season}.")


        if not all_team_data:
            logger.warning("No team batting data was fetched successfully.")
            return True # Still considered success if no data found/fetched without errors

        try:
             combined_data = pd.concat(all_team_data, ignore_index=True)
             logger.info(f"Storing {len(combined_data)} team batting records (replacing existing table)...")
             # Replace the entire table with the fetched data up to the limit
             success = store_data_to_sql(combined_data, 'team_batting', self.db_path, if_exists='replace')
             if success:
                 logger.info("Successfully stored team batting data.")
             else:
                 logger.error("Failed to store team batting data.")
                 return False # Return failure if storage fails
             return True
        except Exception as e:
             logger.error(f"Error combining or storing team batting data: {e}", exc_info=True)
             return False


    # --- fetch_batter_data_efficient (REFACTORED into helpers) ---
    # Removed method body, delegate to helpers below
    def fetch_batter_data_efficient(self):
        """Delegates fetching batter Statcast data based on mode."""
        if self.single_date_historical_mode:
             return self._fetch_batter_statcast_single_date(self.target_fetch_date_obj)
        else:
             return self._fetch_batter_statcast_historical()

    # --- NEW Helper: Fetch Batter Statcast (Single Date) ---
    def _fetch_batter_statcast_single_date(self, target_date_obj):
        """Fetches batter statcast data for a single specific date."""
        target_date_str = target_date_obj.strftime('%Y-%m-%d')
        target_season = target_date_obj.year
        data_exists = False
        fetch_key = f"batter_single_{target_date_str}" # For error tracking

        try:
            # Check DB if data for this date already exists
            with DBConnection(self.db_path) as conn:
                query = "SELECT COUNT(*) FROM statcast_batters WHERE DATE(game_date) = ?"
                cursor = conn.cursor()
                cursor.execute(query, (target_date_str,))
                count = cursor.fetchone()[0]
                if count > 0: data_exists = True
        except Exception as e:
            logger.warning(f"DB check failed for batters on {target_date_str}: {e}. Will attempt fetch.")

        if data_exists:
            logger.info(f"Skipping batter fetch: Data already exists for {target_date_str}")
            return True # Success

        logger.info(f"Fetching batter Statcast for single date: {target_date_str}")
        # Use fetch_with_retries
        pdata = self.fetch_with_retries(pb.statcast, start_dt=target_date_str, end_dt=target_date_str)

        # Filter out any non-regular season rows
        pdata = filter_regular_season(pdata)

        if pdata is None: # Fetch failed definitively
             logger.error(f"Error fetching batter data for single date {target_date_str} after retries.")
             self.failed_batter_fetches.add(('single_date', target_date_str, target_date_str))
             return False # Failure

        if pdata.empty:
             logger.info(f"No batter data found for {target_date_str}.")
             return True # Success (no data is not an error)

        # Process and store data
        try:
            pdata['season'] = target_season
            numeric_cols = ['release_speed', 'launch_speed', 'launch_angle', 'woba_value'] # Add others if needed
            for col in numeric_cols:
                if col in pdata.columns:
                    pdata[col] = pd.to_numeric(pdata[col], errors='coerce')

            essential_cols = ['batter', 'pitcher', 'game_pk']
            if all(col in pdata.columns for col in essential_cols):
                # Use dropna without inplace=True
                pdata = pdata.dropna(subset=essential_cols)
            else:
                 logger.warning(f"Missing essential columns in fetched batter data for {target_date_str}. Cannot clean NaNs reliably.")

            if pdata.empty:
                 logger.info(f"Batter data for {target_date_str} was empty after cleaning.")
                 return True # Still success if cleaning resulted in empty

            rows_to_store = len(pdata)
            success = store_data_to_sql(pdata, 'statcast_batters', self.db_path, if_exists='append')

            if success:
                logger.info(f"Stored {rows_to_store} batter rows for {target_date_str}.")
                return True
            else:
                logger.error(f"Failed to store batter data for {target_date_str}.")
                self.failed_batter_fetches.add(('single_date', target_date_str, target_date_str))
                return False # Failure

        except Exception as e:
            logger.error(f"Error processing/storing single date batter data for {target_date_str}: {e}", exc_info=True)
            self.failed_batter_fetches.add(('single_date', target_date_str, target_date_str))
            return False # Failure

    # --- NEW Helper: Fetch Batter Statcast (Historical - MODIFIED for new checkpoint & parallelism) ---
    def _fetch_batter_statcast_historical(self):
        """Fetches batter statcast data historically using checkpoints and optional parallelism."""
        logger.info(f"Starting historical batter Statcast fetch up to {self.end_date_limit_str}")
        total_stored_rows = 0
        overall_success = True
        # Determine seasons to process
        seasons_hist = sorted(self.args.seasons if self.args.seasons else getattr(DataConfig, 'SEASONS', []))
        seasons_to_process = [s for s in seasons_hist if s <= self.end_date_limit.year]

        if not seasons_to_process:
            logger.info("No relevant seasons for historical batter fetch based on end date limit.")
            return True

        for season in seasons_to_process:
            logger.info(f"--- Processing Batter Season: {season} ---")
            # Determine the effective end date for this season
            season_end_limit = self.end_date_limit if season == self.end_date_limit.year else date(season, 11, 30) # Default end Nov 30

            # Get the last processed date for this season from the checkpoint
            last_processed_date_str = self.checkpoint_manager.get_last_processed_batter_date(season)
            start_fetch_dt = date(season, 3, 1) # Default season start
            if last_processed_date_str:
                 try:
                      last_processed_dt = datetime.strptime(last_processed_date_str, "%Y-%m-%d").date()
                      # Start fetching from the day AFTER the last processed date
                      start_fetch_dt = last_processed_dt + timedelta(days=1)
                      logger.info(f"Resuming season {season} batter fetch from {start_fetch_dt.strftime('%Y-%m-%d')}")
                 except ValueError:
                      logger.warning(f"Invalid last processed date '{last_processed_date_str}' for season {season}. Starting from beginning.")
                 except TypeError: # Handle if None somehow got stored
                      logger.warning(f"Invalid type for last processed date for season {season}. Starting from beginning.")


            # Check if the calculated start date is already past the season's end limit
            if start_fetch_dt > season_end_limit:
                logger.info(f"Season {season} already fully processed up to {season_end_limit.strftime('%Y-%m-%d')}. Skipping.")
                continue

            # Generate date range chunks from the calculated start date
            ranges_to_fetch = []
            current_chunk_start = start_fetch_dt
            chunk_days = getattr(DataConfig, 'CHUNK_SIZE', 14) # Get chunk size from config or default
            while current_chunk_start <= season_end_limit:
                current_chunk_end = min(current_chunk_start + timedelta(days=chunk_days - 1), season_end_limit)
                ranges_to_fetch.append((current_chunk_start.strftime("%Y-%m-%d"), current_chunk_end.strftime("%Y-%m-%d")))
                current_chunk_start = current_chunk_end + timedelta(days=1)

            if not ranges_to_fetch:
                logger.info(f"No new date ranges to fetch for season {season}.")
                continue

            logger.info(f"Generated {len(ranges_to_fetch)} date ranges to fetch for season {season} (from {ranges_to_fetch[0][0]} to {ranges_to_fetch[-1][1]}).")

            processed_chunks_count = 0
            season_stored_rows = 0
            max_successful_date_in_batch = None # Track last successful date within this run

            # --- Inner function to process a single chunk ---
            def process_chunk(start_str, end_str):
                nonlocal season_stored_rows # Allow modifying outer scope variable
                fetch_key = (season, start_str, end_str) # Tuple for error tracking
                logger.debug(f"Fetching hist batter: {start_str} to {end_str}")
                pdata = self.fetch_with_retries(pb.statcast, start_dt=start_str, end_dt=end_str)

                # Keep only regular season rows
                pdata = filter_regular_season(pdata)

                if pdata is None: # Fetch failed definitively
                    logger.error(f" -> Error fetching hist batter range {start_str}-{end_str} for season {season} after retries.")
                    self.failed_batter_fetches.add(fetch_key)
                    return False, None # Indicate failure, no date processed

                if pdata.empty:
                    logger.debug(f" -> No data found for batter range {start_str}-{end_str}.")
                    # Return success, and the end_date of the range as potentially processed
                    return True, end_str

                # Process data if not empty
                try:
                    pdata['season'] = season
                    numeric_cols = ['release_speed', 'launch_speed', 'launch_angle', 'woba_value']
                    for col in numeric_cols:
                        if col in pdata.columns: pdata[col] = pd.to_numeric(pdata[col], errors='coerce')

                    essential_cols = ['batter', 'pitcher', 'game_pk']
                    if all(col in pdata.columns for col in essential_cols):
                        pdata = pdata.dropna(subset=essential_cols)
                    else:
                         logger.warning(f"Missing essential columns in fetched batter data for {start_str}-{end_str}. Cannot clean NaNs reliably.")

                    if pdata.empty:
                        logger.debug(f"Batter data for range {start_str}-{end_str} was empty after cleaning.")
                        return True, end_str # Success, range effectively processed

                    rows_to_store = len(pdata)
                    # Store data (append is safe)
                    success = store_data_to_sql(pdata, 'statcast_batters', self.db_path, if_exists='append')

                    if success:
                        logger.debug(f" -> Stored {rows_to_store} batter rows for range {start_str}-{end_str}.")
                        season_stored_rows += rows_to_store
                        # Return success and the end date of this successfully processed chunk
                        return True, end_str
                    else:
                        logger.error(f" -> Failed to store batter data for range {start_str}-{end_str}.")
                        self.failed_batter_fetches.add(fetch_key)
                        return False, None # Indicate failure

                except Exception as e:
                    logger.error(f" -> Error processing/storing hist batter range {start_str}-{end_str}: {e}", exc_info=True)
                    self.failed_batter_fetches.add(fetch_key)
                    return False, None # Indicate failure
            # --- End of inner function ---

            # --- Execute chunk processing (Parallel or Sequential) ---
            season_fetch_success = True # Track if any chunk in the season fails
            successful_end_dates = [] # Collect end dates of successfully processed chunks

            if self.args.parallel:
                workers = min(DataConfig.MAX_WORKERS, os.cpu_count() or 1)
                logger.info(f"Using PARALLEL batter fetch for season {season} ({workers} workers).")
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_range = {
                        executor.submit(process_chunk, start, end): (start, end)
                        for start, end in ranges_to_fetch
                    }
                    progress_bar = tqdm(as_completed(future_to_range), total=len(ranges_to_fetch), desc=f"Batter Chunks S{season} (Parallel)")
                    for future in progress_bar:
                        start, end = future_to_range[future]
                        try:
                            chunk_success, processed_end_date = future.result()
                            if chunk_success and processed_end_date:
                                successful_end_dates.append(processed_end_date)
                                processed_chunks_count += 1
                            elif not chunk_success:
                                season_fetch_success = False # Mark season as having errors
                            # Update progress bar description
                            progress_bar.set_postfix_str(f"Stored: {season_stored_rows}, Errors: {len(self.failed_batter_fetches)}")

                        except Exception as exc:
                             logger.error(f"Error processing result for batter range {start}-{end}: {exc}", exc_info=True)
                             self.failed_batter_fetches.add((season, start, end))
                             season_fetch_success = False


            else: # Sequential Execution
                 logger.info(f"Using SEQUENTIAL batter fetch for season {season}.")
                 progress_bar = tqdm(ranges_to_fetch, desc=f"Batter Chunks S{season} (Sequential)")
                 for start, end in progress_bar:
                      try:
                           chunk_success, processed_end_date = process_chunk(start, end)
                           if chunk_success and processed_end_date:
                                successful_end_dates.append(processed_end_date)
                                processed_chunks_count += 1
                           elif not chunk_success:
                                season_fetch_success = False
                            # Update progress bar description
                           progress_bar.set_postfix_str(f"Stored: {season_stored_rows}, Errors: {len(self.failed_batter_fetches)}")
                      except Exception as e:
                           logger.error(f"Critical error processing batter range {start}-{end} sequentially: {e}", exc_info=True)
                           self.failed_batter_fetches.add((season, start, end))
                           season_fetch_success = False


            # --- Update Checkpoint for the Season ---
            if successful_end_dates:
                 # Find the maximum date string among successfully processed chunks
                 try:
                      max_successful_date_str = max(successful_end_dates)
                      self.checkpoint_manager.update_last_processed_batter_date(season, max_successful_date_str)
                      logger.info(f"Updated checkpoint for season {season} to last successful date: {max_successful_date_str}")
                 except Exception as e:
                      logger.error(f"Failed to determine or update max successful date for season {season}: {e}")
                      season_fetch_success = False # Mark as failure if checkpoint update fails
            elif ranges_to_fetch: # If we attempted fetches but none succeeded
                 logger.warning(f"No chunks successfully processed for season {season}. Checkpoint not updated.")
                 season_fetch_success = False # Indicate failure if fetches were attempted but none succeeded

            # Save checkpoint after each season is processed
            self.checkpoint_manager.save_overall_checkpoint()

            total_stored_rows += season_stored_rows
            if not season_fetch_success:
                 overall_success = False # Mark overall run as failed if any season had issues
                 logger.error(f"Season {season} completed with errors.")
            else:
                 logger.info(f"Season {season} batter fetch phase complete. Processed {processed_chunks_count}/{len(ranges_to_fetch)} chunks. Stored {season_stored_rows} new rows this season.")


        logger.info(f"Historical batter fetch completed. Total stored rows across all seasons: {total_stored_rows}.")
        if not overall_success:
             logger.error("Historical batter fetching finished with errors in one or more seasons.")
        return overall_success


    # --- fetch_scraped_pitcher_data (MODIFIED to use new mlb_api.py) ---
    def fetch_scraped_pitcher_data(self):
        """Fetches probable pitcher data using the updated mlb_api module (via API)."""
        if not self.args.mlb_api:
            logger.info("Skipping pitcher API fetch (--mlb-api not set).")
            return True
        if not self.args.date:
             # This check is also in __main__, but good to have here too
            logger.error("--mlb-api requires --date.")
            return False

        target_date_str = self.target_fetch_date_obj.strftime('%Y-%m-%d')
        logger.info(f"Starting probable pitcher fetch via API for: {target_date_str}")
        fetch_key = f"mlb_api_{target_date_str}" # For error tracking

        try:
            # Call the updated function - no longer needs team_mapping_df
            daily_pitcher_data = scrape_probable_pitchers(target_date_str)

            # Check the return type and content
            if daily_pitcher_data is None: # Should not happen based on current mlb_api.py, but check
                logger.warning(f"scrape_probable_pitchers returned None for {target_date_str}. Treating as no data.")
                daily_pitcher_data = []
            elif not isinstance(daily_pitcher_data, list):
                 logger.error(f"scrape_probable_pitchers returned unexpected type: {type(daily_pitcher_data)} for {target_date_str}.")
                 self.failed_mlb_api_fetches.add(target_date_str)
                 return False


            if daily_pitcher_data:
                 # Convert list of dicts to DataFrame
                 pdf = pd.DataFrame(daily_pitcher_data)

                 # Ensure required columns exist from the API response (adjust if needed)
                 expected_cols = ["game_date", "game_pk", "home_team_abbr", "away_team_abbr",
                                  "home_probable_pitcher_name", "home_probable_pitcher_id",
                                  "away_probable_pitcher_name", "away_probable_pitcher_id"]
                 if not all(col in pdf.columns for col in expected_cols):
                      logger.error(f"API response for {target_date_str} missing expected columns. Found: {pdf.columns.tolist()}")
                      self.failed_mlb_api_fetches.add(target_date_str)
                      return False

                 # Select and potentially rename columns to match the target 'mlb_api' table structure if needed
                 # Assuming the keys from scrape_probable_pitchers already match the desired table columns
                 # pdf = pdf[expected_cols] # Select only expected cols in desired order?

                 logger.info(f"Storing {len(pdf)} probable pitcher entries for {target_date_str} (replacing existing table)...")
                 # Replace the table entirely with the day's data
                 success = store_data_to_sql(pdf, 'mlb_api', self.db_path, if_exists='replace')

                 if success:
                     self.checkpoint_manager.add_processed_mlb_api_date(target_date_str)
                     logger.info(f"Successfully stored probable pitcher data for {target_date_str}.")
                     # Save checkpoint immediately after successful API fetch/store
                     self.checkpoint_manager.save_overall_checkpoint()
                     return True
                 else:
                     logger.error(f"Failed to store probable pitcher data for {target_date_str}.")
                     self.failed_mlb_api_fetches.add(target_date_str)
                     return False
            else:
                 logger.info(f"No probable pitchers found via API for {target_date_str}.")
                 # Consider if this scenario should update the checkpoint - yes, means the date was processed.
                 self.checkpoint_manager.add_processed_mlb_api_date(target_date_str)
                 self.checkpoint_manager.save_overall_checkpoint()
                 return True # No data is not necessarily an error

        except Exception as e:
             logger.error(f"Error during probable pitcher fetch/store for {target_date_str}: {e}", exc_info=True)
             self.failed_mlb_api_fetches.add(target_date_str)
             # Save checkpoint even on failure? Maybe not for this specific date.
             # self.checkpoint_manager.save_overall_checkpoint() # Let final save handle it
             return False

    # --- NEW: Method to create database indexes ---
    def create_indexes(self):
         """Creates potentially missing indexes on key tables for performance."""
         logger.info("Attempting to create database indexes (if they don't exist)...")
         indexes_to_create = {
              'statcast_pitchers': [
                   'idx_sp_game_date', 'game_date',
                   'idx_sp_pitcher', 'pitcher',
                   'idx_sp_batter', 'batter',
                   'idx_sp_game_pk', 'game_pk',
                   'idx_sp_season_date', 'season, game_date'
              ],
              'statcast_batters': [
                   'idx_sb_game_date', 'game_date',
                   'idx_sb_pitcher', 'pitcher',
                   'idx_sb_batter', 'batter',
                   'idx_sb_game_pk', 'game_pk',
                   'idx_sb_season_date', 'season, game_date'
              ],
              'mlb_api': [
                   'idx_mlbapi_game_date', 'game_date',
                   'idx_mlbapi_game_pk', 'game_pk'
               ]
         }
         all_successful = True
         try:
              with DBConnection(self.db_path) as conn:
                   cursor = conn.cursor()
                   for table, index_info in indexes_to_create.items():
                        # Check if table exists first
                        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                        if not cursor.fetchone():
                            logger.debug(f"Table '{table}' not found, skipping index creation.")
                            continue

                        # Create indexes in pairs (name, column(s))
                        for i in range(0, len(index_info), 2):
                             index_name = index_info[i]
                             columns = index_info[i+1]
                             sql = f"CREATE INDEX IF NOT EXISTS \"{index_name}\" ON \"{table}\" ({columns})"
                             try:
                                  cursor.execute(sql)
                                  logger.info(f"Ensured index '{index_name}' exists on table '{table}'.")
                             except sqlite3.Error as e:
                                  logger.error(f"Failed to create index '{index_name}' on table '{table}': {e}")
                                  all_successful = False
                   conn.commit()
         except sqlite3.Error as e:
              logger.error(f"SQLite error during index creation: {e}", exc_info=True)
              all_successful = False
         except Exception as e:
              logger.error(f"Unexpected error during index creation: {e}", exc_info=True)
              all_successful = False

         if all_successful: logger.info("Index creation process completed.")
         else: logger.error("Index creation process finished with errors.")
         return all_successful

    # --- run Method (MODIFIED) ---
    def run(self):
        """Runs the main data fetching pipeline based on the mode."""
        logger.info(f"--- Starting Data Fetching Pipeline (Mode: {'MLB API' if self.args.mlb_api else 'Single Date Hist' if self.single_date_historical_mode else 'Historical Backfill'}) ---")
        start_time = time.time()
        pipeline_success = True

        try:
            if self.args.mlb_api:
                # Mode 1: MLB API Scraper Only (Probable Pitchers)
                logger.info("[API Mode - Step 1/1] Fetching Probable Pitcher Data via API...")
                if not self.fetch_scraped_pitcher_data():
                    pipeline_success = False
            elif self.single_date_historical_mode:
                # Mode 2: Single-Date Historical Fetch (Pitcher & Batter Statcast)
                logger.info(f"Running Single-Date Historical Fetch for {self.target_fetch_date_obj.strftime('%Y-%m-%d')}...")

                logger.info("[Single Date - Step 1/2] Fetching Pitcher Statcast Data...")
                pitcher_mapping = self.fetch_pitcher_id_mapping()
                if pitcher_mapping is not None and not pitcher_mapping.empty:
                    if not self.fetch_all_pitchers(pitcher_mapping): # Calls refactored helper internally
                        pipeline_success = False
                else:
                    logger.warning("Skipping single-date pitcher Statcast fetch: mapping failed or empty.")
                    # Consider if this should be a failure
                    # pipeline_success = False

                logger.info("[Single Date - Step 2/2] Fetching Batter Statcast Data...")
                if not self.fetch_batter_data_efficient(): # Calls refactored helper internally
                    pipeline_success = False

            else:
                # Mode 3: Full Historical Backfill (Mapping, Pitcher, Team Batting, Batter Statcast)
                logger.info("Running Full Historical Backfill mode...")

                # Step 1: Load Pitcher Mapping (Essential for fetching pitcher data)
                logger.info("[Historical - Step 1/4] Loading Pitcher ID Mapping...")
                pitcher_mapping = self.fetch_pitcher_id_mapping()
                if pitcher_mapping is None or pitcher_mapping.empty:
                    logger.error("Pitcher mapping failed to load or is empty. Aborting historical pitcher/batter fetch as mapping is required.")
                    pipeline_success = False # Cannot proceed without mapping
                else:
                    logger.info("Pitcher mapping loaded successfully.")

                    # Step 2: Fetch Pitcher Statcast (Historical)
                    logger.info("[Historical - Step 2/4] Fetching Pitcher Statcast Data...")
                    if not self.fetch_all_pitchers(pitcher_mapping): # Calls refactored helper internally
                        pipeline_success = False

                    # Step 3: Fetch Team Batting Data (Historical)
                    logger.info("[Historical - Step 3/4] Fetching Team Batting Data...")
                    if not self.fetch_team_batting_data():
                        pipeline_success = False

                    # Step 4: Fetch Batter Statcast (Historical)
                    logger.info("[Historical - Step 4/4] Fetching Batter Statcast Data...")
                    if not self.fetch_batter_data_efficient(): # Calls refactored helper internally
                        pipeline_success = False

            # Step 5: Create Indexes (Run regardless of mode if successful so far)
            if pipeline_success:
                 logger.info("[Pipeline End Step] Ensuring database indexes exist...")
                 if not self.create_indexes():
                      logger.warning("Index creation step finished with errors.")
                      # Decide if index failure should mark pipeline as failed
                      # pipeline_success = False


        except Exception as e:
            logger.critical(f"Unhandled exception in pipeline execution: {e}", exc_info=True)
            pipeline_success = False
        finally:
            total_time = time.time() - start_time
            logger.info("--- Data Fetching Pipeline Finished ---")
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            # Report any persistent issues tracked during the run
            self._report_issues()
            # Save final checkpoint state
            logger.info("Saving final checkpoint.")
            self.checkpoint_manager.save_overall_checkpoint()
            logger.info(f"Pipeline overall success status: {pipeline_success}")

        return pipeline_success


# --- store_data_to_sql function (MODIFIED to use context manager properly) ---
def store_data_to_sql(df, table_name, db_path, if_exists='append'):
    """Stores DataFrame to SQLite table with dynamic chunksize, robust logging, and context manager."""
    if df is None or df.empty:
        logger.debug(f"Empty DataFrame provided for '{table_name}'. Skipping database save.")
        return True # Nothing to save is considered success

    if table_name == 'statcast_pitchers':
        df = dedup_pitch_df(df)

    db_path_str = str(db_path)
    num_columns = len(df.columns)
    if num_columns == 0:
        logger.warning(f"DataFrame for '{table_name}' has 0 columns. Cannot store.")
        return False # Cannot store a DF with no columns

    # Calculate dynamic chunksize based on SQLite variable limit
    SQLITE_MAX_VARS = 30000 # Max variables often around 32766, use a safer limit
    # Ensure num_columns > 0 (checked above)
    pandas_chunksize = max(1, SQLITE_MAX_VARS // num_columns)
    # Cap chunksize to avoid excessive memory usage per chunk (e.g., 1000 rows max)
    pandas_chunksize = min(pandas_chunksize, 1000)
    variables_per_chunk = num_columns * pandas_chunksize

    logger.info(f"Storing {len(df)} records to table '{table_name}' in database '{db_path_str}' (mode: {if_exists}, chunksize: {pandas_chunksize}, vars/chunk: ~{variables_per_chunk})...")

    try:
        # Use the DBConnection context manager
        with DBConnection(db_path_str) as conn:
            if conn is None:
                # The context manager should ideally raise an error if connection fails
                # but we add a check just in case it returns None silently.
                raise ConnectionError(f"DBConnection failed to establish connection to {db_path_str}")

            # Handle 'replace' logic: Drop table before writing
            if if_exists == 'replace':
                logger.info(f"Attempting to drop table '{table_name}' before replacing...")
                try:
                    cursor = conn.cursor()
                    # Use standard SQL syntax for dropping table if exists
                    cursor.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")
                    conn.commit()
                    logger.info(f"Dropped existing table '{table_name}' (if it existed).")
                except sqlite3.Error as drop_e:
                    # Log warning but proceed; to_sql might still work if table didn't exist
                    logger.warning(f"Could not explicitly drop table '{table_name}': {drop_e}. Continuing with to_sql...")


            # Use pandas to_sql for writing data
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists, # Let pandas handle append/replace logic after potential drop
                index=False,
                chunksize=pandas_chunksize,
                method='multi' # Generally recommended for performance with chunking
            )
            logger.info(f"Finished storing data to '{table_name}'.")
            # Context manager handles commit/close on exit
            return True

    # Specific error handling for SQLite operational errors like "too many SQL variables"
    except sqlite3.OperationalError as oe:
         logger.error(f"SQLite OperationalError storing data to '{table_name}': {oe}", exc_info=True)
         if 'too many SQL variables' in str(oe).lower():
             logger.error(f"DYNAMIC CHUNKSIZE FAILED. Calculated chunksize ({pandas_chunksize}) for {num_columns} columns exceeded SQLite variable limit.")
         elif 'has no column named' in str(oe).lower():
             logger.error(f"Schema mismatch? Error indicates table '{table_name}' is missing an expected column from the DataFrame.")
             logger.error(f"DataFrame columns: {df.columns.tolist()}")
         else:
              logger.error(f"Unhandled SQLite OperationalError: {oe}")
         # Log traceback for detailed debugging
         # logger.error(traceback.format_exc()) # Redundant with exc_info=True
         return False # Indicate failure

    # Catch potential connection errors if DBConnection fails
    except ConnectionError as ce:
        logger.error(f"Database connection error for '{db_path_str}': {ce}", exc_info=True)
        return False

    # Catch other general exceptions during the process
    except Exception as e:
         logger.error(f"General Error storing data to table '{table_name}': {e}", exc_info=True)
         # logger.error(traceback.format_exc()) # Redundant with exc_info=True
         return False # Indicate failure

# --- Argument Parser and Main Execution Block (Unchanged) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Fetch MLB Statcast data OR probable pitchers via API.")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD) for API scrape OR single-date historical fetch.")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Seasons for historical backfill (default: from config or 2019-today).")
    parser.add_argument("--parallel", action="store_true", help="Use parallel fetching for historical pitcher AND batter data.")
    parser.add_argument("--mlb-api", action="store_true", help="ONLY fetch probable pitchers via API for the SINGLE date specified by --date.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers: handler.setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled.")
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers: handler.setLevel(logging.INFO)


    if args.mlb_api and not args.date:
        logger.error("--mlb-api requires --date.")
        sys.exit(1)
    if args.mlb_api and args.seasons:
        logger.warning("--seasons argument is ignored when --mlb-api is used.")
    if args.date and args.seasons and not args.mlb_api:
         logger.warning("--date argument is ignored when --seasons is used (unless in single-date mode without --mlb-api). Effective end date is calculated.")


    ensure_dir(project_root / 'data')
    ensure_dir(project_root / 'data' / '.checkpoints') # Ensure checkpoint dir exists
    ensure_dir(project_root / 'logs')

    if not MODULE_IMPORTS_OK:
        logger.critical("Exiting: Failed crucial module imports during initialization.")
        sys.exit(1)

    logger.info("--- Initializing MLB Data Fetcher ---")
    try:
        fetcher = DataFetcher(args)
        success = fetcher.run()
        if success:
            logger.info("--- Data Fetching Script Finished Successfully ---")
            sys.exit(0)
        else:
            logger.error("--- Data Fetching Script Finished With Errors ---")
            sys.exit(1)
    except Exception as main_err:
        logger.critical(f"Critical error during DataFetcher setup or run: {main_err}", exc_info=True)
        sys.exit(2) # Different exit code for critical failure