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

from .modules.checkpoint_manager import CheckpointManager
from .modules.pitcher_fetcher import fetch_pitcher_single_date, fetch_pitcher_historical
from .modules.batter_fetcher import fetch_batter_single_date, fetch_batter_historical
from .modules.api_fetcher import fetch_probable_pitchers
from .modules.store_utils import store_data_to_sql

# Columns that uniquely identify a pitch in Statcast data
UNIQUE_PITCH_COLS = [
    "game_pk",
    "pitcher",
    "inning",
    "batter",
    "pitch_number",
]


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

    def _build_pitcher_mapping_via_api(self) -> pd.DataFrame:
        """Fallback: build pitcher mapping by scraping MLB Stats API for seasons."""
        logger.info("Attempting to build pitcher mapping via MLB Stats API for seasons: %s", self.seasons_to_fetch)
        mapping: dict[int, str] = {}
        for season in self.seasons_to_fetch:
            start = date(season, 3, 1)
            end = date(season, 11, 30)
            current = start
            while current <= end:
                daily_data = scrape_probable_pitchers(current.strftime("%Y-%m-%d"))
                for game in daily_data:
                    for pid, name in [
                        (game.get("home_probable_pitcher_id"), game.get("home_probable_pitcher_name")),
                        (game.get("away_probable_pitcher_id"), game.get("away_probable_pitcher_name")),
                    ]:
                        if pid and name:
                            mapping[int(pid)] = name
                current += timedelta(days=1)

        if not mapping:
            logger.error("No pitcher info retrieved while building mapping from API")
            return pd.DataFrame()

        df = pd.DataFrame({"pitcher_id": list(mapping.keys()), "name": list(mapping.values())})
        saved = store_data_to_sql(df, "pitcher_mapping", self.db_path, if_exists="replace")
        if saved:
            logger.info("Stored %d pitcher mappings to table 'pitcher_mapping'", len(df))
        else:
            logger.error("Failed to store pitcher mapping to database")
        return df

    def ensure_pitcher_mapping(self) -> pd.DataFrame:
        """Load pitcher mapping, building it if missing."""
        pm = self.fetch_pitcher_id_mapping()
        if pm is not None and not pm.empty:
            return pm
        logger.warning("Pitcher mapping missing or empty; attempting rebuild via API")
        return self._build_pitcher_mapping_via_api()

    # --- fetch_statcast_for_pitcher (REFACTORED into helpers) ---
    # This method is now removed, logic moved to helpers below

    # --- NEW Helper: Fetch Pitcher Statcast (Single Date) ---
    def _fetch_pitcher_statcast_single_date(self, pitcher_id, name, target_date_obj):
        return fetch_pitcher_single_date(pitcher_id, name, target_date_obj, Path(self.db_path), self.fetch_with_retries, self.problematic_pitcher_ids)


    # --- NEW Helper: Fetch Pitcher Statcast (Historical) ---
    def _fetch_pitcher_statcast_historical(self, pitcher_id, name, seasons_list):
        return fetch_pitcher_historical(pitcher_id, name, seasons_list, self.end_date_limit, Path(self.db_path), self.fetch_with_retries, self.checkpoint_manager, self.problematic_pitcher_ids)


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
        return fetch_batter_single_date(target_date_obj, Path(self.db_path), self.fetch_with_retries, self.failed_batter_fetches)

    # --- NEW Helper: Fetch Batter Statcast (Historical - MODIFIED for new checkpoint & parallelism) ---
    def _fetch_batter_statcast_historical(self):
        return fetch_batter_historical(self.seasons_to_fetch, self.end_date_limit, Path(self.db_path), self.fetch_with_retries, self.checkpoint_manager, self.failed_batter_fetches, parallel=self.args.parallel)

    def fetch_scraped_pitcher_data(self):
        if not self.args.mlb_api:
            logger.info("Skipping pitcher API fetch (--mlb-api not set).")
            return True
        if not self.args.date:
            logger.error("--mlb-api requires --date.")
            return False
        return fetch_probable_pitchers(self.target_fetch_date_obj, Path(self.db_path), self.checkpoint_manager, self.failed_mlb_api_fetches)


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
                pitcher_mapping = self.ensure_pitcher_mapping()
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
                pitcher_mapping = self.ensure_pitcher_mapping()
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


