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
from datetime import datetime, timedelta, date
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Imports for scraper/mapping (same as before)
try: import requests; import bs4
except ImportError: pass

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path: sys.path.append(str(project_root))

try:
    from src.config import DBConfig, DataConfig
    from src.data.utils import setup_logger, ensure_dir, DBConnection
    from src.data.mlb_api import scrape_probable_pitchers, load_team_mapping
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Imports failed: {e}"); MODULE_IMPORTS_OK = False
    # Dummy definitions (same as before)
    def setup_logger(n, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(n)
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
    class DBConnection:
        def __init__(self,p):self.p=p; self.conn=None
        def __enter__(self): import sqlite3; print("WARN: Dummy DB"); self.conn=sqlite3.connect(self.p); return self.conn
        def __exit__(self,t,v,tb):
            if self.conn: self.conn.close()
    class DBConfig: PATH="data/pitcher_stats.db"; BATCH_SIZE=1000
    class DataConfig: SEASONS=[2024]; RATE_LIMIT_PAUSE=1; CHUNK_SIZE = 14
    def scrape_probable_pitchers(tds, tm): return []
    def load_team_mapping(p): return None

warnings.filterwarnings("ignore", category=FutureWarning)

log_dir = project_root / 'logs'; ensure_dir(log_dir)
logger = setup_logger('data_fetcher', log_file=log_dir/'data_fetcher.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('data_fetcher_fallback')

# --- CheckpointManager Class (No changes needed) ---
class CheckpointManager:
    # (Keep the class identical to the previous version)
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
    def is_completed(self, task): return self.current_checkpoint.get(f"{task}_completed", False)
    def mark_completed(self, task): self.current_checkpoint[f"{task}_completed"] = True; self.save_overall_checkpoint()
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
        # *** MODIFIED: Determine mode based on --date presence for historical runs ***
        self.single_date_historical_mode = (not args.mlb_api and args.date is not None)

        if self.single_date_historical_mode:
             logger.info("Running in Single-Date Historical Fetch mode.")
             try: self.target_fetch_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
             except ValueError: logger.error(f"Invalid date: {args.date}."); sys.exit(1)
             self.seasons_to_fetch = [self.target_fetch_date_obj.year] # Only fetch target year
             self.end_date_limit = self.target_fetch_date_obj # End limit is the target date
        elif args.mlb_api:
             logger.info("Running in MLB API Scraper mode.")
             if not args.date: logger.error("--mlb-api requires --date."); sys.exit(1)
             try: self.target_fetch_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
             except ValueError: logger.error(f"Invalid date: {args.date}."); sys.exit(1)
             self.seasons_to_fetch = [] # No historical seasons needed
             self.end_date_limit = self.target_fetch_date_obj
        else: # Default historical backfill mode (no --date or --mlb-api)
             logger.info("Running in Full Historical Backfill mode.")
             self.seasons_to_fetch = sorted(args.seasons if args.seasons else DataConfig.SEASONS)
             self.end_date_limit = date.today() - timedelta(days=1) # Default to yesterday
             self.target_fetch_date_obj = self.end_date_limit # Set target for consistency

        self.end_date_limit_str = self.end_date_limit.strftime('%Y-%m-%d')
        logger.info(f"Effective End Date Limit: {self.end_date_limit_str}")
        logger.info(f"Seasons to consider for fetching: {self.seasons_to_fetch}")

        # Rest of init remains the same
        self.checkpoint_manager = CheckpointManager()
        signal.signal(signal.SIGINT, self.handle_interrupt); signal.signal(signal.SIGTERM, self.handle_interrupt)
        try: pb.cache.enable()
        except Exception as e: logger.warning(f"Pybaseball cache fail: {e}")
        ensure_dir(Path(self.db_path).parent)
        self.team_mapping_df = None
        if args.mlb_api: self.team_mapping_df = load_team_mapping(self.db_path);
        if args.mlb_api and self.team_mapping_df is None: logger.error("Mapping needed for --mlb-api failed load.")


    def handle_interrupt(self, s, f): logger.warning(f"Interrupt {s}. Save checkpoint..."); self.checkpoint_manager.save_overall_checkpoint(); logger.info("Exiting..."); sys.exit(0)
    def fetch_with_retries(self, fn, *a, max_retries=3, retry_delay=5, **kw): # (Identical)
        le=None;
        for i in range(max_retries):
            try: d=(DataConfig.RATE_LIMIT_PAUSE/2)*(1.5**i); time.sleep(d); return fn(*a, **kw)
            except Exception as e: le=e; logger.warning(f"Attempt {i+1}/{max_retries} fail: {e}"); time.sleep(retry_delay*(2**i))
            if i==max_retries-1: logger.error(f"Retries failed for {fn.__name__}"); raise le

    def fetch_pitcher_id_mapping(self): # (Identical)
        # (Keep this function identical to the previous version)
        if self.checkpoint_manager.is_completed('pitcher_mapping'):
            logger.info("Mapping completed. Loading from DB...");
            try:
                with DBConnection(self.db_path) as conn:
                    cursor = conn.cursor(); cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pitcher_mapping'")
                    if cursor.fetchone():
                        pm = pd.read_sql_query("SELECT * FROM pitcher_mapping", conn)
                        if not pm.empty:
                            if 'pitcher_id' not in pm.columns and 'key_mlbam' in pm.columns: logger.warning("Renaming key_mlbam->pitcher_id."); pm.rename(columns={'key_mlbam': 'pitcher_id'}, inplace=True)
                            if 'pitcher_id' in pm.columns and 'name' in pm.columns: logger.info(f"Loaded {len(pm)} mappings."); return pm
                            else: logger.error("Mapping missing columns. Refetching.")
                        else: logger.warning("Mapping table empty. Refetching.")
                    else: logger.warning("Mapping table not found. Refetching.")
            except Exception as e: logger.warning(f"Failed load mapping ({e}). Refetching.")
            self.checkpoint_manager.current_checkpoint['pitcher_mapping_completed'] = False
        logger.info(f"Fetching new pitcher mappings...");
        try: pl = self.fetch_with_retries(pb.chadwick_register)
        except Exception as e: logger.error(f"Chadwick failed: {e}"); return pd.DataFrame()
        ap = []; seasons_for_map = [s for s in (self.args.seasons or DataConfig.SEASONS) if s <= self.end_date_limit.year] # Use full season list for map
        for season in tqdm(seasons_for_map, desc="Mapping pitchers"):
            try:
                ps = self.fetch_with_retries(pb.pitching_stats, season, season, qual=1); g, gs = ('G', 'GS')
                if ps.empty or g not in ps.columns or gs not in ps.columns: continue
                cutoff = (season == self.end_date_limit.year); crit = (ps[gs] >= 1) if cutoff else ((ps[gs] >= 3)|((ps[gs]/ps[g].replace(0,1)>=0.5)&(ps[g]>=5)))
                st = ps[crit].copy(); idc = 'playerid' if 'playerid' in st.columns else 'IDfg'
                if st.empty or idc not in st.columns: continue
                st['is_starter']=1; st['season']=season; st=st.rename(columns={idc:'key_fangraphs'}); ap.append(st[['key_fangraphs','is_starter','season']])
            except Exception as e: logger.error(f"Error mapping {season}: {e}")
        if not ap: logger.error("No starter data."); return pd.DataFrame()
        apdf=pd.concat(ap,ignore_index=True); plf=pl[['key_fangraphs','key_mlbam','name_first','name_last']].dropna(subset=['key_fangraphs','key_mlbam'])
        try:
            apdf['key_fangraphs']=pd.to_numeric(apdf['key_fangraphs'],errors='coerce').astype('Int64'); plf['key_fangraphs']=pd.to_numeric(plf['key_fangraphs'],errors='coerce').astype('Int64'); plf['key_mlbam']=pd.to_numeric(plf['key_mlbam'],errors='coerce').astype('Int64')
            apdf.dropna(subset=['key_fangraphs'],inplace=True); plf.dropna(subset=['key_fangraphs','key_mlbam'],inplace=True); mdf=pd.merge(apdf,plf,on='key_fangraphs',how='inner')
        except Exception as e: logger.error(f"Mapping merge error: {e}"); return pd.DataFrame()
        if mdf.empty: logger.error("Merge empty."); return pd.DataFrame()
        mdf=mdf.sort_values('season',ascending=False).drop_duplicates('key_mlbam'); mdf['name']=mdf['name_first']+' '+mdf['name_last']
        fm=mdf[['key_mlbam','name','key_fangraphs','is_starter']].copy(); fm.rename(columns={'key_mlbam':'pitcher_id'},inplace=True)
        if not fm.empty:
            logger.info(f"Storing {len(fm)} mappings..."); success=store_data_to_sql(fm,'pitcher_mapping',self.db_path,if_exists='replace')
            if success: self.checkpoint_manager.mark_completed('pitcher_mapping'); logger.info("Stored new mapping.")
            else: logger.error("Failed store mapping."); return pd.DataFrame()
        else: logger.error("Final mapping empty."); return pd.DataFrame()
        return fm

    # --- fetch_statcast_for_pitcher (MODIFIED for single date mode) ---
    def fetch_statcast_for_pitcher(self, pitcher_id, name, seasons_list):
        """ Fetch Statcast data for a single pitcher. """
        # In single date mode, always fetch the target date, ignore checkpoint
        if self.single_date_historical_mode:
            s_str = self.target_fetch_date_obj.strftime("%Y-%m-%d")
            e_str = s_str # Fetch only one day
            target_season = self.target_fetch_date_obj.year
            logger.debug(f" -> Fetch P {name} ({pitcher_id}) for single date: {s_str}")
            try:
                pd_data = self.fetch_with_retries(pb.statcast_pitcher, s_str, e_str, pitcher_id)
                if not pd_data.empty:
                    pd_data['pitcher_id'] = pitcher_id
                    pd_data['season'] = target_season
                    logger.debug(f" -> Fetched {len(pd_data)} rows")
                    # Combine logic (simple case for single date)
                    try:
                        num_cols = ['release_speed','release_spin_rate','launch_speed','launch_angle']
                        for col in num_cols:
                             if col in pd_data.columns: pd_data[col] = pd.to_numeric(pd_data[col], errors='coerce')
                        pd_data.dropna(subset=['game_pk', 'pitcher', 'batter', 'pitch_number'], inplace=True)
                        return pd_data
                    except Exception as e: logger.error(f"Error processing P {name} single date: {e}"); return pd.DataFrame()
                else:
                    return pd.DataFrame() # No data for that specific date
            except Exception as e:
                logger.error(f" -> Error fetching P {name} ({pitcher_id}) single date {s_str}: {e}")
                return pd.DataFrame()
        else: # Original historical backfill logic using checkpoints
            if self.checkpoint_manager.is_pitcher_processed(pitcher_id):
                return pd.DataFrame() # Skip if already processed historically

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
                cd.dropna(subset=['game_pk', 'pitcher', 'batter', 'pitch_number'], inplace=True); logger.debug(f"Combined {len(cd)} hist rows for {name}.")
                return cd
            except Exception as e: logger.error(f"Error combine Hist Statcast {name}: {e}"); return pd.DataFrame()

    # --- fetch_all_pitchers (MODIFIED for single date mode) ---
    def fetch_all_pitchers(self, pitcher_mapping):
        """Fetch data for pitchers."""
        if pitcher_mapping is None or pitcher_mapping.empty: logger.error("Mapping empty."); return False
        try: pitcher_mapping['pitcher_id'] = pitcher_mapping['pitcher_id'].astype(int)
        except Exception as e: logger.error(f"Bad pitcher_id: {e}"); return False

        p_list = list(zip(pitcher_mapping['pitcher_id'], pitcher_mapping['name']))

        if self.single_date_historical_mode:
            logger.info(f"Fetching pitcher Statcast for single date: {self.target_fetch_date_obj.strftime('%Y-%m-%d')}")
            fetch_args = [(pid, name, []) for pid, name in p_list] # Seasons list is ignored in single date mode
            total_to_process = len(fetch_args)
            logger.info(f"Processing {total_to_process} pitchers for the specified date.")
        else: # Historical backfill mode
            fetch_args = [(pid, name, self.seasons_to_fetch) for pid, name in p_list if not self.checkpoint_manager.is_pitcher_processed(pid)]
            total_to_process = len(fetch_args);
            processed_count_prev = len(p_list) - total_to_process
            logger.info(f"{len(p_list)} mapped. Skipping {processed_count_prev} processed. Fetching {total_to_process} historically.")
            if not fetch_args: logger.info("No new pitchers need historical fetching."); return True

        proc_c = 0; success_flag = True; data_stored_count = 0
        mode_desc = "Pitcher Statcast (Single Date)" if self.single_date_historical_mode else "Pitcher Statcast (Historical)"

        # --- Parallel/Sequential Execution Logic ---
        # This part remains largely the same, but the checkpoint logic inside differs
        if self.args.parallel:
            workers = min(12, os.cpu_count() or 1); logger.info(f"Using PARALLEL fetch ({workers} workers).")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                f_to_p = {executor.submit(self.fetch_statcast_for_pitcher, pid, name, seasons): (pid, name) for pid, name, seasons in fetch_args}
                for future in tqdm(as_completed(f_to_p), total=total_to_process, desc=f"{mode_desc} (Parallel)"):
                    pid, name = f_to_p[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                             # Use append for single date mode, could lead to duplicates if run many times
                             # A better approach might involve checking if the specific game_pk already exists
                             save_mode = 'append' # Append works for both modes here
                             s_ok = store_data_to_sql(data, 'statcast_pitchers', self.db_path, if_exists=save_mode)
                             if s_ok:
                                 data_stored_count += len(data)
                                 # Only checkpoint in historical backfill mode
                                 if not self.single_date_historical_mode: self.checkpoint_manager.add_processed_pitcher(pid)
                                 logger.debug(f"Stored {len(data)} for {name}")
                             else: logger.warning(f"Failed store {name}."); success_flag = False
                        elif data is not None and not self.single_date_historical_mode:
                             self.checkpoint_manager.add_processed_pitcher(pid); logger.debug(f"No hist data {name}. Marked processed.")
                        proc_c += 1
                        # Save checkpoint more frequently in historical mode
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
                        if s_ok:
                            data_stored_count += len(data)
                            if not self.single_date_historical_mode: self.checkpoint_manager.add_processed_pitcher(pid)
                            logger.debug(f"Stored {len(data)} for {name}")
                        else: logger.warning(f"Failed store {name}."); success_flag = False
                    elif data is not None and not self.single_date_historical_mode:
                        self.checkpoint_manager.add_processed_pitcher(pid)
                    proc_c += 1
                    if not self.single_date_historical_mode and (proc_c % 100 == 0 or proc_c == total_to_process): self.checkpoint_manager.save_overall_checkpoint()
                except Exception as e: logger.error(f"Critical error pitcher {name}: {e}"); logger.error(traceback.format_exc()); success_flag = False
        # Final checkpoint save for historical mode
        if not self.single_date_historical_mode: self.checkpoint_manager.save_overall_checkpoint()
        logger.info(f"Pitcher Statcast fetching phase complete. Processed {proc_c}/{total_to_process}. Stored {data_stored_count} new rows.");
        return success_flag

    # --- fetch_team_batting_data (MODIFIED for single date mode) ---
    def fetch_team_batting_data(self):
        """ Fetch team batting data. Skips if in single date mode. """
        if self.single_date_historical_mode:
            logger.info("Skipping team batting fetch in single-date mode.")
            return True
        # Original historical backfill logic
        if self.checkpoint_manager.is_completed('team_batting'): logger.info("Team batting done, skipping."); return True
        logger.info(f"Fetching team batting up to {self.end_date_limit.year}"); ad = []
        # Use the originally configured seasons for historical backfill
        seasons_hist = sorted(self.args.seasons if self.args.seasons else DataConfig.SEASONS)
        seasons_to_check = [s for s in seasons_hist if s <= self.end_date_limit.year]
        for s in tqdm(seasons_to_check, desc="Fetching Team Batting"):
            try:
                td = self.fetch_with_retries(pb.team_batting, s, s)
                if not td.empty:
                    if 'Season' not in td.columns: td['Season'] = s
                    ad.append(td)
            except Exception as e: logger.error(f"Error team batting {s}: {e}")
        if not ad: logger.warning("No team batting fetched."); self.checkpoint_manager.mark_completed('team_batting'); return True
        cd = pd.concat(ad, ignore_index=True); logger.info(f"Storing {len(cd)} team batting records...")
        success = store_data_to_sql(cd, 'team_batting', self.db_path, if_exists='replace')
        if success: self.checkpoint_manager.mark_completed('team_batting'); logger.info("Stored team batting.")
        else: logger.error("Failed store team batting.")
        return success

    # --- fetch_batter_data_efficient (MODIFIED for single date mode) ---
    def fetch_batter_data_efficient(self):
        """Fetch batter Statcast data."""
        if self.single_date_historical_mode:
            # Fetch only the single specified date
            target_date_str = self.target_fetch_date_obj.strftime('%Y-%m-%d')
            target_season = self.target_fetch_date_obj.year
            logger.info(f"Fetching batter Statcast for single date: {target_date_str}")
            try:
                pdata = self.fetch_with_retries(pb.statcast, start_dt=target_date_str, end_dt=target_date_str)
                if pdata.empty: logger.info(f"No batter data found for {target_date_str}."); return True
                pdata['season'] = target_season; num_cols = ['release_speed','launch_speed','launch_angle','woba_value']
                for col in num_cols:
                    if col in pdata.columns: pdata[col]=pd.to_numeric(pdata[col],errors='coerce')
                pdata.dropna(subset=['batter','pitcher','game_pk'],inplace=True)
                pr = len(pdata);
                # Use append - could lead to duplicates if run multiple times for same date without clearing table first
                success=store_data_to_sql(pdata,'statcast_batters',self.db_path,if_exists='append')
                if success: logger.info(f"Stored {pr} batter rows for {target_date_str}.")
                else: logger.error(f"Failed store batter data for {target_date_str}.")
                return success
            except Exception as e: logger.error(f"Error fetch/proc single date batter {target_date_str}: {e}"); logger.error(traceback.format_exc()); return False
        else:
            # Original historical backfill logic using checkpoints
            logger.info(f"Starting historical batter Statcast fetch up to {self.end_date_limit_str}")
            stored_hist = 0
            # Use the originally configured seasons for historical backfill
            seasons_hist = sorted(self.args.seasons if self.args.seasons else DataConfig.SEASONS)
            seasons_to_check = [s for s in seasons_hist if s <= self.end_date_limit.year]
            for s in seasons_to_check:
                logger.info(f"Processing batter season {s}"); s_dt=date(s,3,1); e_limit=date(s,11,30)
                s_end_dt=self.end_date_limit if s==self.end_date_limit.year else e_limit
                if s_end_dt < s_dt: continue
                ranges = []; cs_dt = s_dt
                chunk_days = DataConfig.CHUNK_SIZE or 14 # Get chunk size from config
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
                        pdata.dropna(subset=['batter','pitcher','game_pk'],inplace=True)
                        pr = len(pdata); success=store_data_to_sql(pdata,'statcast_batters',self.db_path,if_exists='append')
                        if success: self.checkpoint_manager.add_processed_season_date_range(s,rk); stored_hist+=pr; logger.debug(f"Stored {pr} for {rk}.")
                        else: logger.error(f"Failed store hist range {rk}.")
                        proc_r += 1
                    except Exception as e: logger.error(f"Error fetch/proc hist {rk}: {e}"); logger.error(traceback.format_exc())
                if proc_r > 0: logger.info(f"Saving checkpoint post season {s}."); self.checkpoint_manager.save_overall_checkpoint()
            logger.info(f"Hist batter fetch complete. Stored {stored_hist} new rows."); self.checkpoint_manager.save_overall_checkpoint()
            return True

    # --- fetch_scraped_pitcher_data (No changes needed) ---
    def fetch_scraped_pitcher_data(self):
        # (Keep this function identical to the previous version)
        if not self.args.mlb_api: logger.info("Skipping pitcher scrape (--mlb-api not set)."); return True
        if not self.args.date: logger.error("--mlb-api requires --date."); return False
        if not MODULE_IMPORTS_OK or 'scrape_probable_pitchers' not in globals() or 'load_team_mapping' not in globals(): logger.error("Scraper/mapping fn not available."); return False
        if 'requests' not in sys.modules or 'bs4' not in sys.modules: logger.error("'requests'/'bs4' required."); return False
        target_date_str = self.target_fetch_date_obj.strftime("%Y-%m-%d"); logger.info(f"Starting pitcher scraping for: {target_date_str}")
        if self.team_mapping_df is None: logger.error("Team mapping needed but not loaded."); return False
        daily_pitcher_data = scrape_probable_pitchers(target_date_str, self.team_mapping_df)
        if daily_pitcher_data:
            try:
                pdf = pd.DataFrame(daily_pitcher_data)
                if 'game_date' not in pdf.columns or pdf['game_date'].isnull().all(): pdf['game_date'] = target_date_str
                exp_cols = ['gamePk','game_date','home_team_id','home_team_name','home_team_abbr','away_team_id','away_team_name','away_team_abbr','home_probable_pitcher_id','home_probable_pitcher_name','away_probable_pitcher_id','away_probable_pitcher_name']
                for col in exp_cols:
                     if col not in pdf.columns: pdf[col] = pd.NA
                for col_id in ['gamePk','home_team_id','away_team_id','home_probable_pitcher_id','away_probable_pitcher_id']: pdf[col_id] = pd.to_numeric(pdf[col_id], errors='coerce').astype('Int64')
                pdf = pdf[exp_cols]; logger.info(f"Storing {len(pdf)} scraped entries for {target_date_str} (replacing)...")
                success = store_data_to_sql(pdf, 'mlb_api', self.db_path, if_exists='replace')
                if success: self.checkpoint_manager.add_processed_mlb_api_date(target_date_str); logger.info(f"Stored scraped data for {target_date_str}.")
                else: logger.error(f"Failed store scraped data for {target_date_str}."); return False
            except Exception as e: logger.error(f"Error proc/store scraped {target_date_str}: {e}"); logger.error(traceback.format_exc()); return False
        elif daily_pitcher_data is None: logger.error(f"Scraping failed critically for {target_date_str}."); return False
        else: logger.info(f"No pitchers scraped for {target_date_str}.")
        logger.info(f"Pitcher scraping finished for {target_date_str}."); self.checkpoint_manager.save_overall_checkpoint()
        return True

    # --- run Method (MODIFIED to use single_date_historical_mode flag) ---
    def run(self):
        """Run the main data fetching pipeline."""
        logger.info(f"--- Starting Data Fetching Pipeline ---")
        start_time = time.time(); pipeline_success = True
        try:
            if self.args.mlb_api:
                logger.info("Running in MLB API Scraper Only mode...")
                logger.info("[Scraper Step 1/1] Fetching Scraped Probable Pitcher Data...")
                if not self.fetch_scraped_pitcher_data(): pipeline_success = False
            elif self.single_date_historical_mode:
                logger.info(f"Running in Single-Date Historical Fetch mode for {self.target_fetch_date_obj.strftime('%Y-%m-%d')}...")
                # Steps for single date historical fetch (skip mapping?, skip team batting?)
                logger.info("[Single Date Step 1/2] Fetching Pitcher Statcast Data...")
                pitcher_mapping = self.fetch_pitcher_id_mapping() # Still need mapping to know WHO to fetch
                if pitcher_mapping is not None and not pitcher_mapping.empty:
                    if not self.fetch_all_pitchers(pitcher_mapping): pipeline_success = False
                else: logger.warning("Skipping pitcher Statcast fetch: mapping failed/empty.")
                logger.info("[Single Date Step 2/2] Fetching Batter Statcast Data...")
                if not self.fetch_batter_data_efficient(): pipeline_success = False
                # Skipping team batting data fetch in single date mode
            else: # Full Historical Backfill mode
                logger.info("Running in Full Historical Backfill mode...")
                logger.info("[Historical Step 1/4] Fetching Pitcher ID Mapping...")
                pitcher_mapping = self.fetch_pitcher_id_mapping()
                if pitcher_mapping is None or pitcher_mapping.empty: logger.warning("Pitcher mapping failed/empty.")
                logger.info("[Historical Step 2/4] Fetching Pitcher Statcast Data...")
                if pitcher_mapping is not None and not pitcher_mapping.empty:
                    if not self.fetch_all_pitchers(pitcher_mapping): pipeline_success = False
                else: logger.warning("Skipping pitcher Statcast: requires mapping.")
                logger.info("[Historical Step 3/4] Fetching Team Batting Data...")
                if not self.fetch_team_batting_data(): pipeline_success = False
                logger.info("[Historical Step 4/4] Fetching Batter Statcast Data...")
                if not self.fetch_batter_data_efficient(): pipeline_success = False
        except Exception as e:
            logger.error(f"Unhandled exception in pipeline: {e}"); logger.error(traceback.format_exc()); pipeline_success = False
        finally:
            total_time = time.time() - start_time; logger.info("Final checkpoint save."); self.checkpoint_manager.save_overall_checkpoint()
            logger.info(f"--- Data Fetching Pipeline Finished in {total_time:.2f} seconds ---")
        return pipeline_success

# --- store_data_to_sql function (Keep improved logging version) ---
def store_data_to_sql(df, table_name, db_path, if_exists='append'):
    """Stores DataFrame to SQLite table with dynamic chunksize and robust logging."""
    if df is None or df.empty: logger.debug(f"Empty DataFrame provided for '{table_name}'. Skipping save."); return True # Return True if empty DF
    db_path_str = str(db_path); num_columns = len(df.columns)
    if num_columns == 0: logger.warning(f"DataFrame for '{table_name}' has 0 columns."); return False # Cannot save df with 0 columns
    SQLITE_MAX_VARS = 30000; pandas_chunksize = max(1, SQLITE_MAX_VARS // num_columns); pandas_chunksize = min(pandas_chunksize, 1000)
    variables_per_chunk = num_columns * pandas_chunksize
    logger.info(f"Storing {len(df)} records to '{table_name}' (mode: {if_exists}, chunksize: {pandas_chunksize}, vars/chunk: ~{variables_per_chunk})...")
    conn = None # Initialize conn outside try
    try:
        # Establish connection using context manager principles but allowing explicit close
        db_conn_context = DBConnection(db_path_str)
        conn = db_conn_context.__enter__() # Manually enter context

        if conn is None: raise ConnectionError("DB connection failed.")

        if if_exists == 'replace':
            logger.info(f"Attempting to drop table '{table_name}' before replacing...")
            try:
                cursor=conn.cursor(); cursor.execute(f"DROP TABLE IF EXISTS \"{table_name}\""); conn.commit()
                logger.info(f"Dropped existing table '{table_name}' (if any).")
            except Exception as drop_e:
                # Log warning but proceed, to_sql with 'replace' might handle it
                logger.warning(f"Could not explicitly drop table {table_name}: {drop_e}")

        # The core save operation
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False, chunksize=pandas_chunksize, method='multi')

        logger.info(f"Finished storing data to '{table_name}'.")
        db_conn_context.__exit__(None, None, None) # Manually exit context
        return True # Explicit success return

    except sqlite3.OperationalError as oe:
         # Specific SQLite errors
         logger.error(f"SQLite OperationalError storing to '{table_name}': {oe}", exc_info=True) # Add exc_info
         if 'too many SQL variables' in str(oe): logger.error(f"DYNAMIC CHUNKSIZE ({pandas_chunksize}) FAILED for {num_columns} columns.")
         elif 'has no column named' in str(oe): logger.error(f"Schema mismatch? Table '{table_name}'. Check if table exists with correct columns.")
         # Log traceback here as well
         logger.error(traceback.format_exc())
         if conn: db_conn_context.__exit__(type(oe), oe, oe.__traceback__) # Ensure connection close on error
         return False # Explicit failure return
    except Exception as e:
         # Catch any other exceptions
         logger.error(f"General Error storing data to '{table_name}': {e}", exc_info=True) # Add exc_info
         logger.error(traceback.format_exc()) # Log the full traceback
         if conn: db_conn_context.__exit__(type(e), e, e.__traceback__) # Ensure connection close on error
         return False # Explicit failure return
# --- Argument Parser and Main Execution Block (Identical) ---
def parse_args():
    # (Keep identical to previous version)
    parser = argparse.ArgumentParser(description="Fetch MLB data (Statcast, Team) OR scrape probable pitchers for a specific date.")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD) for scrape OR single-date historical fetch OR historical end date limit.") # Updated help
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help=f"Seasons for historical backfill (default: from config {DataConfig.SEASONS})") # Default None
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