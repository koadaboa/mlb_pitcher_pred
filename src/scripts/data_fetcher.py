# src/scripts/data_fetcher.py

import sqlite3
import pandas as pd
import numpy as np
import pybaseball as pb # Keep for other functions
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
from tqdm import tqdm # Keep for other functions
from concurrent.futures import ThreadPoolExecutor, as_completed # Keep for other functions
import warnings

# Imports for scraper/mapping (requests/bs4 are only needed for mlb_api)
try: import requests; import bs4
except ImportError: pass

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path: sys.path.append(str(project_root))

try:
    from src.config import DBConfig, DataConfig
    # Use DBConnection consistent with utils.py
    from src.data.utils import setup_logger, ensure_dir, DBConnection
    from src.data.mlb_api import scrape_probable_pitchers, load_team_mapping # Keep for mlb_api mode
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Imports failed: {e}"); MODULE_IMPORTS_OK = False


warnings.filterwarnings("ignore", category=FutureWarning)

log_dir = project_root / 'logs'; ensure_dir(log_dir)
logger = setup_logger('data_fetcher', log_file=log_dir/'data_fetcher.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('data_fetcher_fallback')

# --- CheckpointManager Class (No changes needed) ---
class CheckpointManager:
    # --- (Keep the entire CheckpointManager class identical to the previous version) ---
    # Note: We might consider removing 'pitcher_mapping_completed' state later
    # if it's no longer useful, but for now, leave the class structure intact.
    def __init__(self, checkpoint_dir=project_root / 'data/checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir); self.checkpoint_dir.mkdir(exist_ok=True, parents=True); self.current_checkpoint = {}
        try: self.load_overall_checkpoint()
        except Exception as e: logger.error(f"CheckpointManager init error: {e}. Init fresh."); self._initialize_checkpoint(); self.save_overall_checkpoint()
    def load_overall_checkpoint(self):
        f = self.checkpoint_dir / 'overall_progress.json'
        if f.exists():
            try:
                with open(f, 'r') as fp: self.current_checkpoint = json.load(fp); logger.info("Loaded checkpoint.")
            except Exception as e: logger.error(f"Failed load/decode checkpoint ({e}). Init new."); self._initialize_checkpoint()
        else: logger.info("No checkpoint file. Init new."); self._initialize_checkpoint()
        self._ensure_keys()
    def _initialize_checkpoint(self): self.current_checkpoint = {'pitcher_mapping_completed': False, 'processed_pitcher_ids': [], 'team_batting_completed': False, 'processed_seasons_batter_data': {}, 'processed_mlb_api_dates': [], 'last_update': datetime.now().isoformat()}
    def _ensure_keys(self):
        # Keep 'pitcher_mapping_completed' for now for compatibility, though it won't be set by fetch_pitcher_id_mapping anymore
        defaults = {'pitcher_mapping_completed': False, 'processed_pitcher_ids': [], 'team_batting_completed': False, 'processed_seasons_batter_data': {}, 'processed_mlb_api_dates': [], 'last_update': datetime.now().isoformat()}
        updated = False
        for key, default_val in defaults.items():
            current_val = self.current_checkpoint.get(key); correct_type = isinstance(current_val, type(default_val))
            if key not in self.current_checkpoint or current_val is None or not correct_type:
                if key != 'last_update': logger.warning(f"Init/Reset checkpoint key '{key}' (was {type(current_val)})."); self.current_checkpoint[key] = default_val; updated = True
    def save_overall_checkpoint(self):
        f = self.checkpoint_dir / 'overall_progress.json'; self.current_checkpoint['last_update'] = datetime.now().isoformat()
        try:
            for key in ['processed_pitcher_ids', 'processed_mlb_api_dates']:
                 current_list = self.current_checkpoint.get(key, []);
                 if isinstance(current_list, list):
                      try: s_list = [item.item() if hasattr(item, 'item') else item for item in current_list]; self.current_checkpoint[key] = sorted(list(set(s_list)))
                      except TypeError: logger.warning(f"Sort checkpoint list '{key}' failed."); self.current_checkpoint[key] = current_list
            with open(f, 'w') as fp: json.dump(self.current_checkpoint, fp, indent=4); logger.debug("Saved checkpoint")
        except Exception as e: logger.error(f"Failed save checkpoint: {e}")
    # Remove is_completed and mark_completed for 'pitcher_mapping' as they are no longer used by fetch_pitcher_id_mapping
    # def is_completed(self, task): return self.current_checkpoint.get(f"{task}_completed", False) # Can be removed if only used for pitcher_mapping
    # def mark_completed(self, task): self.current_checkpoint[f"{task}_completed"] = True; self.save_overall_checkpoint() # Can be removed if only used for pitcher_mapping
    def add_processed_pitcher(self, p): l=self.current_checkpoint.setdefault('processed_pitcher_ids', []); i=p.item() if hasattr(p,'item') else p; i not in l and l.append(i)
    def is_pitcher_processed(self, p): return p in self.current_checkpoint.get('processed_pitcher_ids', [])
    def add_processed_season_date_range(self, s, dr): sd=self.current_checkpoint.setdefault('processed_seasons_batter_data', {}); rl=sd.setdefault(str(s),[]); dr not in rl and rl.append(dr)
    def is_season_date_range_processed(self, s, dr): pd=self.current_checkpoint.get('processed_seasons_batter_data',{}); return dr in pd.get(str(s),[])
    def get_last_processed_mlb_api_date(self):
        pl = self.current_checkpoint.get('processed_mlb_api_dates', []);
        if not pl: return None
        try: ld=pl[-1]; datetime.strptime(ld,"%Y-%m-%d"); return ld
        except: logger.warning(f"Invalid date in checkpoint: {pl[-1]}."); return None
    def add_processed_mlb_api_date(self, ds):
        pl=self.current_checkpoint.setdefault('processed_mlb_api_dates',[]); ds not in pl and pl.append(ds)
    def is_mlb_api_date_processed(self, ds): return ds in self.current_checkpoint.get('processed_mlb_api_dates', [])


# --- DataFetcher Class ---
class DataFetcher:
    def __init__(self, args):
        if not MODULE_IMPORTS_OK: raise ImportError("Modules not imported.")
        self.args = args; self.db_path = project_root / DBConfig.PATH
        self.single_date_historical_mode = (not args.mlb_api and args.date is not None)

        if self.single_date_historical_mode:
             logger.info("Running in Single-Date Historical Fetch mode.")
             try: self.target_fetch_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
             except ValueError: logger.error(f"Invalid date: {args.date}."); sys.exit(1)
             self.seasons_to_fetch = [self.target_fetch_date_obj.year]
             self.end_date_limit = self.target_fetch_date_obj
        elif args.mlb_api:
             logger.info("Running in MLB API Scraper mode.")
             if not args.date: logger.error("--mlb-api requires --date."); sys.exit(1)
             try: self.target_fetch_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
             except ValueError: logger.error(f"Invalid date: {args.date}."); sys.exit(1)
             self.seasons_to_fetch = [] # No historical seasons needed
             self.end_date_limit = self.target_fetch_date_obj
        else: # Default historical backfill mode
             logger.info("Running in Full Historical Backfill mode.")
             self.seasons_to_fetch = sorted(args.seasons if args.seasons else DataConfig.SEASONS)
             self.end_date_limit = date.today() - timedelta(days=1)
             self.target_fetch_date_obj = self.end_date_limit # Used for determining latest season end date

        self.end_date_limit_str = self.end_date_limit.strftime('%Y-%m-%d')
        logger.info(f"Effective End Date Limit: {self.end_date_limit_str}")
        logger.info(f"Seasons to consider for fetching (if historical): {self.seasons_to_fetch}")

        self.checkpoint_manager = CheckpointManager()
        signal.signal(signal.SIGINT, self.handle_interrupt); signal.signal(signal.SIGTERM, self.handle_interrupt)
        try: pb.cache.enable()
        except Exception as e: logger.warning(f"Pybaseball cache fail: {e}")
        ensure_dir(Path(self.db_path).parent)
        self.team_mapping_df = None
        if args.mlb_api: self.team_mapping_df = load_team_mapping(self.db_path); # Keep for mlb_api mode
        if args.mlb_api and self.team_mapping_df is None: logger.error("Mapping needed for --mlb-api failed load.")


    def handle_interrupt(self, s, f): logger.warning(f"Interrupt {s}. Save checkpoint..."); self.checkpoint_manager.save_overall_checkpoint(); logger.info("Exiting..."); sys.exit(0)
    def fetch_with_retries(self, fn, *a, max_retries=3, retry_delay=5, **kw): # (Identical)
        le=None;
        for i in range(max_retries):
            try: d=(DataConfig.RATE_LIMIT_PAUSE/2)*(1.5**i); time.sleep(d); return fn(*a, **kw)
            except Exception as e: le=e; logger.warning(f"Attempt {i+1}/{max_retries} fail: {e}"); time.sleep(retry_delay*(2**i))
            if i==max_retries-1: logger.error(f"Retries failed for {fn.__name__}"); raise le

    # --- MODIFIED fetch_pitcher_id_mapping ---
    def fetch_pitcher_id_mapping(self):
        """
        Loads pitcher mapping (mlbid, playername) from the existing
        'pitcher_mapping' table in the database.
        Renames columns to 'pitcher_id' and 'name' for internal consistency.
        """
        logger.info("Loading pitcher mapping from database...")
        mapping_table = 'pitcher_mapping'
        required_cols_db = ['mlbid', 'playername']
        final_cols = ['pitcher_id', 'name']

        try:
            # Use DBConnection for consistency
            with DBConnection(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if table exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{mapping_table}'")
                if not cursor.fetchone():
                    logger.error(f"Pitcher mapping table '{mapping_table}' not found in the database.")
                    return pd.DataFrame()

                # Check for required columns
                cursor.execute(f"PRAGMA table_info({mapping_table})")
                available_cols = [info[1] for info in cursor.fetchall()]
                missing_cols = [col for col in required_cols_db if col not in available_cols]
                if missing_cols:
                    logger.error(f"Pitcher mapping table '{mapping_table}' is missing required columns: {missing_cols}. Available: {available_cols}")
                    return pd.DataFrame()

                # Fetch data
                query = f"SELECT {', '.join(required_cols_db)} FROM {mapping_table}"
                pm = pd.read_sql_query(query, conn)

            if pm.empty:
                logger.warning(f"Pitcher mapping table '{mapping_table}' is empty.")
                return pd.DataFrame()

            # Rename columns for internal use
            rename_dict = dict(zip(required_cols_db, final_cols))
            pm = pm.rename(columns=rename_dict)

            # Ensure pitcher_id is numeric (specifically integer)
            pm['pitcher_id'] = pd.to_numeric(pm['pitcher_id'], errors='coerce')
            pm = pm.dropna(subset=['pitcher_id'])
            # Use nullable integer type Int64
            pm['pitcher_id'] = pm['pitcher_id'].astype('Int64')

            # Ensure name is string
            pm['name'] = pm['name'].astype(str)

            # Drop rows where essential info is missing after conversion
            pm = pm.dropna(subset=final_cols)

            if pm.empty:
                 logger.warning(f"Pitcher mapping loaded, but resulted in empty DataFrame after cleaning/conversion.")
                 return pd.DataFrame()

            logger.info(f"Successfully loaded {len(pm)} pitcher mappings from '{mapping_table}'.")
            return pm

        except sqlite3.Error as e:
            logger.error(f"SQLite error loading pitcher mapping from '{mapping_table}': {e}", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error loading pitcher mapping: {e}", exc_info=True)
            return pd.DataFrame()

    # --- fetch_statcast_for_pitcher (No changes needed here) ---
    def fetch_statcast_for_pitcher(self, pitcher_id, name, seasons_list):
        # (Keep identical to previous version)
        """ Fetch Statcast data for a single pitcher. """
        if self.single_date_historical_mode:
            target_date_str = self.target_fetch_date_obj.strftime("%Y-%m-%d")
            target_season = self.target_fetch_date_obj.year
            data_exists = False
            try:
                with DBConnection(self.db_path) as conn:
                    cursor = conn.cursor()
                    query = "SELECT COUNT(*) FROM statcast_pitchers WHERE DATE(game_date) = ? AND pitcher = ?"
                    cursor.execute(query, (target_date_str, pitcher_id))
                    count = cursor.fetchone()[0]
                    if count > 0: data_exists = True
            except Exception as e: logger.warning(f"DB check failed for pitcher {pitcher_id} on {target_date_str}: {e}. Will attempt fetch.")
            if data_exists:
                logger.debug(f" -> Skipping P fetch {name} ({pitcher_id}): Data already exists for {target_date_str}")
                return pd.DataFrame()
            logger.debug(f" -> Fetch P {name} ({pitcher_id}) for single date: {target_date_str}")
            try:
                pd_data = self.fetch_with_retries(pb.statcast_pitcher, target_date_str, target_date_str, pitcher_id)
                if not pd_data.empty:
                    pd_data['pitcher_id'] = pitcher_id
                    pd_data['season'] = target_season
                    logger.debug(f" -> Fetched {len(pd_data)} rows")
                    try:
                        num_cols = ['release_speed','release_spin_rate','launch_speed','launch_angle']
                        for col in num_cols:
                             if col in pd_data.columns: pd_data[col] = pd.to_numeric(pd_data[col], errors='coerce')
                        pd_data = pd_data.dropna(subset=['game_pk', 'pitcher', 'batter', 'pitch_number']) # Use dropna without inplace
                        return pd_data
                    except Exception as e: logger.error(f"Error processing P {name} single date: {e}"); return pd.DataFrame()
                else: return pd.DataFrame()
            except Exception as e: logger.error(f" -> Error fetching P {name} ({pitcher_id}) single date {target_date_str}: {e}"); return pd.DataFrame()
        else: # Original historical backfill logic using checkpoints
            if self.checkpoint_manager.is_pitcher_processed(pitcher_id): return pd.DataFrame()
            all_data = []; logger.debug(f"Hist fetch P: {name} ({pitcher_id})")
            relevant_seasons = [s for s in seasons_list if s <= self.end_date_limit.year]
            for s in relevant_seasons:
                s_dt=date(s,3,1); e_dt=self.end_date_limit if s == self.end_date_limit.year else date(s,11,30)
                if s_dt > e_dt: continue
                s_str=s_dt.strftime("%Y-%m-%d"); e_str=e_dt.strftime("%Y-%m-%d"); logger.debug(f" -> Fetch {name} ({s}): {s_str} to {e_str}")
                try:
                    pd_data = self.fetch_with_retries(pb.statcast_pitcher, s_str, e_str, pitcher_id)
                    if not pd_data.empty: pd_data['pitcher_id']=pitcher_id; pd_data['season']=s; all_data.append(pd_data); logger.debug(f" -> Fetched {len(pd_data)} rows")
                except Exception as e: logger.error(f" -> Error Hist Statcast {name} ({s}): {e}")
            if not all_data: logger.debug(f"No hist Statcast {name}."); return pd.DataFrame()
            try:
                cd = pd.concat(all_data, ignore_index=True); num_cols = ['release_speed', 'release_spin_rate', 'launch_speed', 'launch_angle']
                for col in num_cols:
                     if col in cd.columns: cd[col] = pd.to_numeric(cd[col], errors='coerce')
                cd = cd.dropna(subset=['game_pk', 'pitcher', 'batter', 'pitch_number']) # Use dropna without inplace
                logger.debug(f"Combined {len(cd)} hist rows for {name}.")
                return cd
            except Exception as e: logger.error(f"Error combine Hist Statcast {name}: {e}"); return pd.DataFrame()


    # --- fetch_all_pitchers (No changes needed - relies on output of fetch_pitcher_id_mapping) ---
    def fetch_all_pitchers(self, pitcher_mapping):
        # (Keep identical to previous version)
        """Fetch data for pitchers."""
        if pitcher_mapping is None or pitcher_mapping.empty: logger.error("Mapping empty."); return False
        try:
            # Ensure pitcher_id exists and is of a suitable type after loading
            if 'pitcher_id' not in pitcher_mapping.columns:
                 logger.error("Column 'pitcher_id' not found in loaded mapping.")
                 return False
            # Already converted to Int64 in fetch_pitcher_id_mapping
            # pitcher_mapping['pitcher_id'] = pitcher_mapping['pitcher_id'].astype(int) # Avoid simple int conversion if Int64 is used
        except Exception as e: logger.error(f"Bad pitcher_id type after loading: {e}"); return False

        p_list = list(zip(pitcher_mapping['pitcher_id'], pitcher_mapping['name']))
        if self.single_date_historical_mode:
            logger.info(f"Fetching pitcher Statcast for single date: {self.target_fetch_date_obj.strftime('%Y-%m-%d')}")
            fetch_args = [(pid, name, []) for pid, name in p_list] # Seasons list ignored
            total_to_process = len(fetch_args); logger.info(f"Checking/Processing {total_to_process} pitchers.")
        else: # Historical backfill mode
            fetch_args = [(pid, name, self.seasons_to_fetch) for pid, name in p_list if not self.checkpoint_manager.is_pitcher_processed(pid)]
            total_to_process = len(fetch_args); processed_count_prev = len(p_list) - total_to_process
            logger.info(f"{len(p_list)} mapped. Skipping {processed_count_prev} processed. Fetching {total_to_process} historically.")
            if not fetch_args: logger.info("No new pitchers need historical fetching."); return True
        proc_c = 0; success_flag = True; data_stored_count = 0
        mode_desc = "Pitcher Statcast (Single Date)" if self.single_date_historical_mode else "Pitcher Statcast (Historical)"
        if self.args.parallel:
            workers = min(12, os.cpu_count() or 1); logger.info(f"Using PARALLEL fetch ({workers} workers).")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                f_to_p = {executor.submit(self.fetch_statcast_for_pitcher, pid, name, seasons): (pid, name) for pid, name, seasons in fetch_args}
                for future in tqdm(as_completed(f_to_p), total=total_to_process, desc=f"{mode_desc} (Parallel)"):
                    pid, name = f_to_p[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                             save_mode = 'append'
                             s_ok = store_data_to_sql(data, 'statcast_pitchers', self.db_path, if_exists=save_mode)
                             if s_ok: data_stored_count += len(data); logger.debug(f"Stored {len(data)} for {name}")
                             else: logger.warning(f"Failed store {name}."); success_flag = False
                        if data is not None and not self.single_date_historical_mode: self.checkpoint_manager.add_processed_pitcher(pid)
                        proc_c += 1
                        if not self.single_date_historical_mode and proc_c % 100 == 0: self.checkpoint_manager.save_overall_checkpoint()
                    except Exception as exc: logger.error(f"Pitcher {name} exception: {exc}"); success_flag = False
        else: # Sequential
            logger.info("Using SEQUENTIAL pitcher fetch.")
            for i, (pid, name, seasons) in enumerate(tqdm(fetch_args, desc=f"{mode_desc} (Sequential)")):
                try:
                    data = self.fetch_statcast_for_pitcher(pid, name, seasons)
                    if data is not None and not data.empty:
                        save_mode = 'append'
                        s_ok = store_data_to_sql(data, 'statcast_pitchers', self.db_path, if_exists=save_mode)
                        if s_ok: data_stored_count += len(data); logger.debug(f"Stored {len(data)} for {name}")
                        else: logger.warning(f"Failed store {name}."); success_flag = False
                    if data is not None and not self.single_date_historical_mode: self.checkpoint_manager.add_processed_pitcher(pid)
                    proc_c += 1
                    if not self.single_date_historical_mode and (proc_c % 100 == 0 or proc_c == total_to_process): self.checkpoint_manager.save_overall_checkpoint()
                except Exception as e: logger.error(f"Critical error pitcher {name}: {e}"); logger.error(traceback.format_exc()); success_flag = False
        if not self.single_date_historical_mode: self.checkpoint_manager.save_overall_checkpoint()
        logger.info(f"Pitcher Statcast fetching phase complete. Processed {proc_c}/{total_to_process}. Stored {data_stored_count} new rows.");
        return success_flag

    # --- fetch_team_batting_data (Identical, skips in single-date mode) ---
    def fetch_team_batting_data(self):
        # (Keep identical to previous version)
        """ Fetch team batting data. Skips if in single date mode. """
        if self.single_date_historical_mode:
            logger.info("Skipping team batting fetch in single-date mode.")
            return True
        # Removed checkpoint check for team_batting_completed - let it run if not single date
        logger.info(f"Fetching team batting up to {self.end_date_limit.year}"); ad = []
        seasons_hist = sorted(self.args.seasons if self.args.seasons else DataConfig.SEASONS)
        seasons_to_check = [s for s in seasons_hist if s <= self.end_date_limit.year]
        for s in tqdm(seasons_to_check, desc="Fetching Team Batting"):
            try:
                td = self.fetch_with_retries(pb.team_batting, s, s)
                if not td.empty: td['Season'] = s; ad.append(td)
            except Exception as e: logger.error(f"Error team batting {s}: {e}")
        if not ad: logger.warning("No team batting fetched."); return True # Still success if no data found
        cd = pd.concat(ad, ignore_index=True); logger.info(f"Storing {len(cd)} team batting records...")
        success = store_data_to_sql(cd, 'team_batting', self.db_path, if_exists='replace')
        if success: logger.info("Stored team batting.") # No checkpoint needed here
        else: logger.error("Failed store team batting.")
        return success


    # --- fetch_batter_data_efficient (MODIFIED dropna) ---
    def fetch_batter_data_efficient(self):
        # (Keep identical, except for dropna)
        """Fetch batter Statcast data."""
        if self.single_date_historical_mode:
            target_date_str = self.target_fetch_date_obj.strftime('%Y-%m-%d')
            target_season = self.target_fetch_date_obj.year
            data_exists = False
            try:
                with DBConnection(self.db_path) as conn:
                    cursor = conn.cursor()
                    query = "SELECT COUNT(*) FROM statcast_batters WHERE DATE(game_date) = ?"
                    cursor.execute(query, (target_date_str,))
                    count = cursor.fetchone()[0]
                    if count > 0: data_exists = True
            except Exception as e: logger.warning(f"DB check failed for batters on {target_date_str}: {e}. Will attempt fetch.")
            if data_exists:
                logger.info(f"Skipping batter fetch: Data already exists for {target_date_str}")
                return True
            logger.info(f"Fetching batter Statcast for single date: {target_date_str}")
            try:
                pdata = self.fetch_with_retries(pb.statcast, start_dt=target_date_str, end_dt=target_date_str)
                if pdata.empty: logger.info(f"No batter data found for {target_date_str}."); return True
                pdata['season'] = target_season; num_cols = ['release_speed','launch_speed','launch_angle','woba_value']
                for col in num_cols:
                    if col in pdata.columns: pdata[col]=pd.to_numeric(pdata[col],errors='coerce')
                pdata = pdata.dropna(subset=['batter','pitcher','game_pk']) # Use dropna without inplace
                pr = len(pdata);
                success=store_data_to_sql(pdata,'statcast_batters',self.db_path,if_exists='append')
                if success: logger.info(f"Stored {pr} batter rows for {target_date_str}.")
                else: logger.error(f"Failed store batter data for {target_date_str}.")
                return success
            except Exception as e: logger.error(f"Error fetch/proc single date batter {target_date_str}: {e}"); logger.error(traceback.format_exc()); return False
        else: # Historical
            logger.info(f"Starting historical batter Statcast fetch up to {self.end_date_limit_str}")
            stored_hist = 0
            seasons_hist = sorted(self.args.seasons if self.args.seasons else DataConfig.SEASONS)
            seasons_to_check = [s for s in seasons_hist if s <= self.end_date_limit.year]
            for s in seasons_to_check:
                logger.info(f"Processing batter season {s}"); s_dt=date(s,3,1); e_limit=date(s,11,30)
                s_end_dt=self.end_date_limit if s==self.end_date_limit.year else e_limit
                if s_end_dt < s_dt: continue
                ranges = []; cs_dt = s_dt
                chunk_days = DataConfig.CHUNK_SIZE or 14
                while cs_dt <= s_end_dt: ce_dt=min(cs_dt+timedelta(days=chunk_days-1),s_end_dt); ranges.append((cs_dt.strftime("%Y-%m-%d"),ce_dt.strftime("%Y-%m-%d"))); cs_dt=ce_dt+timedelta(days=1)
                logger.info(f"{len(ranges)} ranges for {s}."); proc_r = 0
                for s_str, e_str in tqdm(ranges, desc=f"Proc {s} Batter Ranges"):
                    rk=f"{s_str}_{e_str}";
                    if self.checkpoint_manager.is_season_date_range_processed(s, rk): continue
                    logger.debug(f"Fetching hist batter: {rk}")
                    try:
                        pdata = self.fetch_with_retries(pb.statcast, start_dt=s_str, end_dt=e_str)
                        if pdata.empty: self.checkpoint_manager.add_processed_season_date_range(s, rk); continue
                        pdata['season'] = s; num_cols = ['release_speed','launch_speed','launch_angle','woba_value']
                        for col in num_cols:
                            if col in pdata.columns: pdata[col]=pd.to_numeric(pdata[col],errors='coerce')
                        pdata = pdata.dropna(subset=['batter','pitcher','game_pk']) # Use dropna without inplace
                        pr = len(pdata); success=store_data_to_sql(pdata,'statcast_batters',self.db_path,if_exists='append')
                        if success: self.checkpoint_manager.add_processed_season_date_range(s,rk); stored_hist+=pr; logger.debug(f"Stored {pr} for {rk}.")
                        else: logger.error(f"Failed store hist range {rk}.")
                        proc_r += 1
                    except Exception as e: logger.error(f"Error fetch/proc hist {rk}: {e}"); logger.error(traceback.format_exc())
                if proc_r > 0: logger.info(f"Saving checkpoint post season {s}."); self.checkpoint_manager.save_overall_checkpoint()
            logger.info(f"Hist batter fetch complete. Stored {stored_hist} new rows."); self.checkpoint_manager.save_overall_checkpoint()
            return True

    # --- fetch_scraped_pitcher_data (Identical) ---
    def fetch_scraped_pitcher_data(self):
        # (Keep identical to previous version)
        """Scrape and store only the seven desired columns, replacing the mlb_api table."""
        if not self.args.mlb_api: logger.info("Skipping pitcher scrape (--mlb-api not set)."); return True
        if not self.args.date: logger.error("--mlb-api requires --date."); return False
        logger.info(f"Starting pitcher scraping for: {self.target_fetch_date_obj.strftime('%Y-%m-%d')}")
        if self.team_mapping_df is None: logger.error("Team mapping needed but not loaded."); return False
        daily_pitcher_data = scrape_probable_pitchers(self.target_fetch_date_obj.strftime('%Y-%m-%d'), self.team_mapping_df)
        if daily_pitcher_data:
            try:
                pdf = pd.DataFrame(daily_pitcher_data)
                if 'game_date' not in pdf.columns or pdf['game_date'].isnull().all(): pdf['game_date'] = self.target_fetch_date_obj.strftime("%Y-%m-%d")
                pdf = pdf.rename(columns={"home_team":"home_team_abbr","away_team":"away_team_abbr","home_pitcher_name":"home_probable_pitcher_name","away_pitcher_name":"away_probable_pitcher_name","home_pitcher_id":"home_probable_pitcher_id","away_pitcher_id":"away_probable_pitcher_id","game_pk":"game_pk"})
                pdf = pdf[["game_date","game_pk","home_team_abbr","away_team_abbr","home_probable_pitcher_name","home_probable_pitcher_id","away_probable_pitcher_name","away_probable_pitcher_id"]]
                logger.info(f"Storing {len(pdf)} scraped entries for {self.target_fetch_date_obj} (replacing)...")
                success = store_data_to_sql(pdf,'mlb_api',self.db_path,if_exists='replace')
                if success: self.checkpoint_manager.add_processed_mlb_api_date(self.target_fetch_date_obj.strftime("%Y-%m-%d")); logger.info(f"Stored scraped data for {self.target_fetch_date_obj}.")
                else: logger.error(f"Failed store scraped data for {self.target_fetch_date_obj}."); return False
            except Exception as e: logger.error(f"Error processing/storing scraped data: {e}"); logger.error(traceback.format_exc()); return False
        else: logger.info(f"No pitchers scraped for {self.target_fetch_date_obj}.")
        logger.info(f"Pitcher scraping finished for {self.target_fetch_date_obj}.")
        self.checkpoint_manager.save_overall_checkpoint()
        return True

    def fetch_daily_lineups_for_date(self, date_obj):
        """Fetch starting lineups for a specific date and store them."""
        date_str = date_obj.strftime("%Y-%m-%d")
        logger.info(f"Fetching daily lineups for {date_str}...")
        try:
            df = scrape_daily_lineups(date_str)
            if df is None or df.empty:
                logger.info(f"No lineup data for {date_str}")
                return True
            success = store_data_to_sql(df, 'daily_lineups', self.db_path, if_exists='append')
            if success:
                logger.info(f"Stored {len(df)} lineup rows for {date_str}")
            else:
                logger.error(f"Failed storing lineup data for {date_str}")
            return success
        except Exception as e:
            logger.error(f"Error fetching/storing lineups for {date_str}: {e}")
            logger.error(traceback.format_exc())
            return False

    def fetch_daily_lineups_range(self, start_date, end_date):
        """Fetch lineups for each date in a range."""
        current = start_date
        overall_success = True
        while current <= end_date:
            if not self.fetch_daily_lineups_for_date(current):
                overall_success = False
            current += timedelta(days=1)
        return overall_success

    # --- run Method (Identical - calls modified fetch methods) ---
    def run(self):
        # (Keep identical to previous version)
        """Run the main data fetching pipeline."""
        logger.info(f"--- Starting Data Fetching Pipeline ---")
        start_time = time.time(); pipeline_success = True
        try:
            if self.args.mlb_api:
                logger.info("Running in MLB API Scraper Only mode...")
                logger.info("[Scraper Step 1/1] Fetching Scraped Probable Pitcher Data...")
                if not self.fetch_scraped_pitcher_data():
                    pipeline_success = False
                if not self.fetch_daily_lineups_for_date(self.target_fetch_date_obj):
                    pipeline_success = False
            elif self.single_date_historical_mode:
                logger.info(f"Running in Single-Date Historical Fetch mode for {self.target_fetch_date_obj.strftime('%Y-%m-%d')}...")
                logger.info("[Single Date Step 1/2] Fetching Pitcher Statcast Data...")
                pitcher_mapping = self.fetch_pitcher_id_mapping() # Calls updated function
                if pitcher_mapping is not None and not pitcher_mapping.empty:
                    if not self.fetch_all_pitchers(pitcher_mapping): pipeline_success = False
                else: logger.warning("Skipping pitcher Statcast fetch: mapping failed/empty.")
                logger.info("[Single Date Step 2/3] Fetching Batter Statcast Data...")
                if not self.fetch_batter_data_efficient():
                    pipeline_success = False
                logger.info("[Single Date Step 3/3] Fetching Daily Lineups...")
                if not self.fetch_daily_lineups_for_date(self.target_fetch_date_obj):
                    pipeline_success = False
            else: # Full Historical Backfill mode
                logger.info("Running in Full Historical Backfill mode...")
                # Step 1: Load Pitcher Mapping (No longer fetches/creates)
                logger.info("[Historical Step 1/4] Loading Pitcher ID Mapping...")
                pitcher_mapping = self.fetch_pitcher_id_mapping() # Calls updated function
                if pitcher_mapping is None or pitcher_mapping.empty:
                     logger.error("Pitcher mapping failed to load or is empty. Aborting historical pitcher/batter fetch.")
                     pipeline_success = False # Can't proceed without mapping
                     # Exit early if mapping is essential and failed
                     total_time = time.time() - start_time
                     logger.info("Final checkpoint save."); self.checkpoint_manager.save_overall_checkpoint()
                     logger.info(f"--- Data Fetching Pipeline Finished (Aborted) in {total_time:.2f} seconds ---")
                     return pipeline_success
                else:
                     logger.info("Pitcher mapping loaded successfully.")

                # Step 2: Fetch Pitcher Statcast
                logger.info("[Historical Step 2/4] Fetching Pitcher Statcast Data...")
                if not self.fetch_all_pitchers(pitcher_mapping): pipeline_success = False # fetch_all_pitchers already checks for empty mapping

                # Step 3: Fetch Team Batting
                logger.info("[Historical Step 3/4] Fetching Team Batting Data...")
                if not self.fetch_team_batting_data(): pipeline_success = False

                # Step 4: Fetch Batter Statcast
                logger.info("[Historical Step 4/4] Fetching Batter Statcast Data...")
                if not self.fetch_batter_data_efficient():
                    pipeline_success = False

                # Step 5: Fetch Daily Lineups for historical range
                start_hist_date = date(min(self.seasons_to_fetch), 3, 1)
                logger.info("[Historical Step 5/5] Fetching Daily Lineups...")
                if not self.fetch_daily_lineups_range(start_hist_date, self.end_date_limit):
                    pipeline_success = False

        except Exception as e: logger.error(f"Unhandled exception in pipeline: {e}"); logger.error(traceback.format_exc()); pipeline_success = False
        finally: total_time = time.time() - start_time;
        logger.info("Final checkpoint save."); self.checkpoint_manager.save_overall_checkpoint()
        logger.info(f"--- Data Fetching Pipeline Finished in {total_time:.2f} seconds ---")
        return pipeline_success

# --- store_data_to_sql function (Identical) ---
def store_data_to_sql(df, table_name, db_path, if_exists='append'):
    # (Keep identical to previous version)
    """Stores DataFrame to SQLite table with dynamic chunksize and robust logging."""
    if df is None or df.empty: logger.debug(f"Empty DataFrame provided for '{table_name}'. Skipping save."); return True
    db_path_str = str(db_path); num_columns = len(df.columns)
    if num_columns == 0: logger.warning(f"DataFrame for '{table_name}' has 0 columns."); return False
    SQLITE_MAX_VARS = 30000; pandas_chunksize = max(1, SQLITE_MAX_VARS // num_columns); pandas_chunksize = min(pandas_chunksize, 1000)
    variables_per_chunk = num_columns * pandas_chunksize
    logger.info(f"Storing {len(df)} records to '{table_name}' (mode: {if_exists}, chunksize: {pandas_chunksize}, vars/chunk: ~{variables_per_chunk})...")
    conn = None; db_conn_context = None
    try:
        db_conn_context = DBConnection(db_path_str); conn = db_conn_context.__enter__()
        if conn is None: raise ConnectionError("DB connection failed.")
        if if_exists == 'replace':
            logger.info(f"Attempting to drop table '{table_name}' before replacing...")
            try: cursor=conn.cursor(); cursor.execute(f"DROP TABLE IF EXISTS \"{table_name}\""); conn.commit(); logger.info(f"Dropped existing table '{table_name}' (if any).")
            except Exception as drop_e: logger.warning(f"Could not explicitly drop table {table_name}: {drop_e}")
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False, chunksize=pandas_chunksize, method='multi')
        logger.info(f"Finished storing data to '{table_name}'."); db_conn_context.__exit__(None, None, None); return True
    except sqlite3.OperationalError as oe:
         logger.error(f"SQLite OperationalError storing to '{table_name}': {oe}", exc_info=True)
         if 'too many SQL variables' in str(oe): logger.error(f"DYNAMIC CHUNKSIZE ({pandas_chunksize}) FAILED for {num_columns} columns.")
         elif 'has no column named' in str(oe): logger.error(f"Schema mismatch? Table '{table_name}'. Check if table exists with correct columns.")
         logger.error(traceback.format_exc());
         if conn and db_conn_context: db_conn_context.__exit__(type(oe), oe, oe.__traceback__);
         return False
    except Exception as e:
         logger.error(f"General Error storing data to '{table_name}': {e}", exc_info=True); logger.error(traceback.format_exc())
         if conn and db_conn_context: db_conn_context.__exit__(type(e), e, e.__traceback__);
         return False

# --- Argument Parser and Main Execution Block (Identical) ---
def parse_args():
    # (Keep identical to previous version)
    parser = argparse.ArgumentParser(description="Fetch MLB data (Statcast, Team) OR scrape probable pitchers for a specific date.")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD) for scrape OR single-date historical fetch OR historical end date limit.")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help=f"Seasons for historical backfill (default: from config {DataConfig.SEASONS})")
    parser.add_argument("--parallel", action="store_true", help="Use parallel fetch for pitcher Statcast.")
    parser.add_argument("--mlb-api", action="store_true", help="ONLY scrape probable pitchers for the SINGLE date specified by --date.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()

if __name__ == "__main__":
    # (Keep identical to previous version)
    args = parse_args()
    if args.debug: logger.setLevel(logging.DEBUG); logger.info("DEBUG logging enabled.")
    else: logger.setLevel(logging.INFO)
    if args.mlb_api and not args.date: logger.error("--mlb-api requires --date."); sys.exit(1)
    ensure_dir(project_root / 'data'); ensure_dir(project_root / 'data' / 'checkpoints'); ensure_dir(project_root / 'logs')
    if not MODULE_IMPORTS_OK: logger.error("Exiting: Failed module imports."); sys.exit(1)
    logger.info("--- Initializing MLB Data Fetcher ---")
    fetcher = DataFetcher(args)
    success = fetcher.run()
    if success: logger.info("--- Data Fetching Script Finished Successfully ---"); sys.exit(0)
    else: logger.error("--- Data Fetching Script Finished With Errors ---"); sys.exit(1)