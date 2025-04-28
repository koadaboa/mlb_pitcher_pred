# src/data/aggregate_statcast.py

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging
import time
import gc
from datetime import datetime
from tqdm import tqdm # Import tqdm

# --- Setup Project Root & Logging ---
try:
    project_root = Path(__file__).resolve().parents[2]
    import sys
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.config import DBConfig, LogConfig
    # Ensure DBConnection is correctly imported from utils
    from src.data.utils import setup_logger, DBConnection
    MODULE_IMPORTS_OK = True
except ImportError as e:
    MODULE_IMPORTS_OK = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('aggregate_statcast_fallback')
    logger.warning(f"Could not import project modules: {e}. Using basic logging.")
    # Define DBConfig directly if import fails
    class DBConfig:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        DATA_DIR = PROJECT_ROOT / 'data'
        DB_DIR = DATA_DIR / 'db'
        DB_FILE = DB_DIR / 'mlb_data.db'
        DB_DIR.mkdir(parents=True, exist_ok=True)
    # Fallback DBConnection if needed (basic functionality)
    class DBConnection:
        def __init__(self): # No argument here
            self.db_path = DBConfig.DB_FILE
            self.conn = None
            # Add store_data method directly to fallback if needed
            self.store_data = self._store_data_fallback

        def __enter__(self):
            logger.debug(f"Connecting to database: {self.db_path}")
            self.conn = sqlite3.connect(self.db_path)
            return self.conn
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.conn:
                logger.debug("Closing database connection.")
                self.conn.close()
        def _store_data_fallback(self, df, table_name, if_exists='replace', index=False):
             with self as conn: # Uses the __enter__/__exit__
                 logger.info(f"Storing {len(df)} rows to table '{table_name}' (using fallback connection).")
                 conn.execute(f"DROP TABLE IF EXISTS {table_name}") # Explicit drop
                 df.to_sql(table_name, conn, if_exists=if_exists, index=index)
                 conn.commit()

# --- Setup Logger ---
if MODULE_IMPORTS_OK:
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Use the corrected path structure
    logger = setup_logger('aggregate_statcast', LogConfig.LOG_DIR / 'aggregate_statcast.log')
else:
    logger.info("Using fallback logger setup.")

# --- Constants ---
PITCHER_METRICS_RAW = [
    'pitcher', 'game_pk', 'game_date', 'player_name', 'p_throws', 'home_team', 'away_team',
    'inning', 'inning_topbot', 'events', 'description', 'zone',
    'balls', 'strikes', 'outs_when_up', 'pitch_number', 'pitch_name',
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'effective_speed', 'release_extension', 'release_pos_x', 'release_pos_z', 'release_pos_y',
    'sz_top', 'sz_bot', 'estimated_woba_using_speedangle', 'woba_value', 'woba_denom',
    'babip_value', 'iso_value', 'at_bat_number'
]
BATTER_METRICS_RAW = [
    'batter', 'game_pk', 'game_date', 'player_name', 'stand', 'home_team', 'away_team',
    'inning', 'inning_topbot', 'events', 'description',
    'woba_value', 'woba_denom', 'babip_value', 'iso_value',
    'launch_speed', 'launch_angle', 'hit_distance_sc', 'bb_type', 'balls', 'strikes', 'at_bat_number', # Added at_bat_number
    'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle'
]

# --- Helper Functions ---

def calculate_additional_pitcher_stats(df):
    """Calculates additional stats from raw events."""
    logger.debug("Calculating additional pitcher stats...")
    df['strikeout'] = df['events'].apply(lambda x: 1 if x == 'strikeout' else 0)
    df['walk'] = df['events'].apply(lambda x: 1 if x == 'walk' else 0)
    df['hit'] = df['events'].apply(lambda x: 1 if x in ['single', 'double', 'triple', 'home_run'] else 0)
    df['home_run'] = df['events'].apply(lambda x: 1 if x == 'home_run' else 0)

    if 'game_pk' in df.columns and 'pitcher' in df.columns and 'at_bat_number' in df.columns:
        pa_count = df.groupby(['game_pk', 'pitcher'])['at_bat_number'].nunique().reset_index()
        pa_count = pa_count.rename(columns={'at_bat_number': 'batters_faced'})
        df = pd.merge(df, pa_count, on=['game_pk', 'pitcher'], how='left', suffixes=(None, '_calc'))
        if 'batters_faced_calc' in df.columns:
            df['batters_faced'] = df['batters_faced_calc']
            df = df.drop(columns=['batters_faced_calc'])
    else:
        logger.warning("Missing game_pk, pitcher, or at_bat_number. Cannot calculate batters_faced.")
        df['batters_faced'] = np.nan # Assign NaN if calculation fails

    out_events = ['strikeout', 'field_out', 'force_out', 'grounded_into_double_play',
                  'double_play', 'sac_fly', 'sac_bunt', 'fielders_choice_out',
                  'triple_play']
    df['outs_recorded'] = df['events'].apply(lambda x: 1 if x in out_events else 0)
    df['outs_recorded'] += df['events'].apply(lambda x: 1 if x in ['grounded_into_double_play', 'double_play'] else 0)
    df['outs_recorded'] += df['events'].apply(lambda x: 2 if x == 'triple_play' else 0)

    if 'description' in df.columns:
        df['is_swinging_strike'] = df['description'].apply(lambda x: 1 if x in ['swinging_strike', 'swinging_strike_blocked'] else 0)
        df['is_called_strike'] = df['description'].apply(lambda x: 1 if x == 'called_strike' else 0)
    else:
        logger.warning("'description' column not found. Skipping swinging/called strike calculations.")
        df['is_swinging_strike'] = 0
        df['is_called_strike'] = 0

    if 'zone' in df.columns:
        df['is_in_zone'] = df['zone'].apply(lambda x: 1 if pd.notna(x) and 1 <= x <= 9 else 0)
    else:
        logger.warning("'zone' column not found. Skipping zone calculations.")
        df['is_in_zone'] = 0

    def categorize_pitch(name):
        if pd.isna(name): return 'Unknown'
        name_lower = str(name).lower()
        if any(p in name_lower for p in ['fastball', 'cutter', 'sinker', 'four-seam', 'two-seam']): return 'Fastball'
        elif any(p in name_lower for p in ['slider', 'curve', 'sweeper', 'slurve', 'knuckle-curve']): return 'Breaking'
        elif any(p in name_lower for p in ['changeup', 'splitter', 'forkball', 'knuckleball', 'screwball']): return 'Offspeed'
        else: return 'Unknown'
    df['pitch_category'] = df['pitch_name'].apply(categorize_pitch) if 'pitch_name' in df.columns else 'Unknown'
    df['is_fastball'] = df['pitch_category'].apply(lambda x: 1 if x == 'Fastball' else 0)
    df['is_breaking'] = df['pitch_category'].apply(lambda x: 1 if x == 'Breaking' else 0)
    df['is_offspeed'] = df['pitch_category'].apply(lambda x: 1 if x == 'Offspeed' else 0)

    if 'woba_denom' in df.columns and 'woba_value' in df.columns:
        df['woba_denom'] = pd.to_numeric(df['woba_denom'], errors='coerce').fillna(0)
        df['woba_value'] = pd.to_numeric(df['woba_value'], errors='coerce').fillna(0)
        df['woba_points'] = df['woba_value'] * df['woba_denom']
    else:
        logger.warning("'woba_denom' or 'woba_value' column not found. Skipping woba_points.")
        df['woba_points'] = 0
        if 'woba_denom' not in df.columns: df['woba_denom'] = 0

    if 'babip_value' in df.columns:
         df['babip_value'] = pd.to_numeric(df['babip_value'], errors='coerce').fillna(0)
         df['babip_points'] = df['babip_value'] * df['woba_denom']
    else:
        logger.warning("'babip_value' column not found. Skipping babip_points.")
        df['babip_points'] = 0

    if 'iso_value' in df.columns:
        df['iso_value'] = pd.to_numeric(df['iso_value'], errors='coerce').fillna(0)
        df['iso_points'] = df['iso_value'] * df['woba_denom']
    else:
        logger.warning("'iso_value' column not found. Skipping iso_points.")
        df['iso_points'] = 0

    logger.debug("Finished calculating additional pitcher stats.")
    return df


def aggregate_pitcher_data(df):
    """Aggregates pitcher data to game-pitcher level."""
    logger.info(f"Aggregating pitcher data for {df['game_pk'].nunique()} games, {df['pitcher'].nunique()} pitchers...")
    if df.empty: return pd.DataFrame()

    numeric_cols = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'effective_speed', 'release_extension', 'release_pos_x', 'release_pos_z', 'release_pos_y',
        'sz_top', 'sz_bot', 'strikeout', 'walk', 'hit', 'home_run', 'batters_faced', 'outs_recorded',
        'is_swinging_strike', 'is_called_strike', 'is_in_zone', 'is_fastball', 'is_breaking', 'is_offspeed',
        'woba_points', 'woba_denom', 'babip_points', 'iso_points'
    ]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: logger.warning(f"Numeric column {col} not found for pitcher aggregation.")

    agg_dict = {
        'pitch_number': ('pitch_number', 'count'), # Use tuple format for modern pandas
        'strikeout': ('strikeout', 'sum'), 'walk': ('walk', 'sum'), 'hit': ('hit', 'sum'), 'home_run': ('home_run', 'sum'),
        'batters_faced': ('batters_faced', 'first'), 'outs_recorded': ('outs_recorded', 'sum'),
        'release_speed_mean': ('release_speed', 'mean'), 'release_speed_max': ('release_speed', 'max'),
        'effective_speed': ('effective_speed', 'mean'), 'release_spin_rate': ('release_spin_rate', 'mean'),
        'release_extension': ('release_extension', 'mean'), 'pfx_x': ('pfx_x', 'mean'), 'pfx_z': ('pfx_z', 'mean'),
        'spin_axis': ('spin_axis', 'mean'), 'is_swinging_strike': ('is_swinging_strike', 'sum'),
        'is_called_strike': ('is_called_strike', 'sum'), 'is_in_zone': ('is_in_zone', 'sum'),
        'is_fastball': ('is_fastball', 'sum'), 'is_breaking': ('is_breaking', 'sum'), 'is_offspeed': ('is_offspeed', 'sum'),
        'woba_points': ('woba_points', 'sum'), 'woba_denom': ('woba_denom', 'sum'),
        'babip_points': ('babip_points', 'sum'), 'iso_points': ('iso_points', 'sum'),
        'player_name': ('player_name', 'first'), 'p_throws': ('p_throws', 'first'),
        'home_team': ('home_team', 'first'), 'away_team': ('away_team', 'first'),
        'inning_topbot': ('inning_topbot', 'first') # Keep for team assignment later
    }

    agg_dict_filtered = {k: v for k, v in agg_dict.items() if v[0] in df.columns}
    logger.debug(f"Aggregating pitchers with spec: {agg_dict_filtered}")
    if not agg_dict_filtered: logger.error("No columns found for pitcher aggregation."); return pd.DataFrame()

    group_cols = ['game_pk', 'pitcher', 'game_date']
    if any(c not in df.columns for c in group_cols): logger.error(f"Missing grouping columns for pitcher agg: {group_cols}"); return pd.DataFrame()

    game_pitcher_stats = df.groupby(group_cols, observed=True, dropna=False).agg(**agg_dict_filtered).reset_index()
    # Rename columns after aggregation if needed (agg already does this with tuple format keys)
    game_pitcher_stats = game_pitcher_stats.rename(columns={
        'pitch_number': 'total_pitches',
        'release_speed_mean': 'avg_velocity', 'release_speed_max': 'max_velocity',
        'release_spin_rate': 'avg_spin_rate', 'pfx_x': 'avg_pfx_x', 'pfx_z': 'avg_pfx_z'
        # batters_faced is already correct name
    })

    # --- Derived Stats Calculation ---
    logger.debug("Calculating derived pitcher stats...")
    gp_stats = game_pitcher_stats
    outs_divisor = gp_stats.get('outs_recorded', 0).replace(0, np.nan)
    gp_stats['k_per_9'] = (gp_stats.get('strikeout', 0) / outs_divisor * 27).fillna(0)
    gp_stats['bb_per_9'] = (gp_stats.get('walk', 0) / outs_divisor * 27).fillna(0)
    gp_stats['hr_per_9'] = (gp_stats.get('home_run', 0) / outs_divisor * 27).fillna(0)

    bf_divisor = gp_stats.get('batters_faced', 0).replace(0, np.nan)
    gp_stats['k_percent'] = (gp_stats.get('strikeout', 0) / bf_divisor).fillna(0)
    gp_stats['bb_percent'] = (gp_stats.get('walk', 0) / bf_divisor).fillna(0)

    pitches_divisor = gp_stats.get('total_pitches', 0).replace(0, np.nan)
    gp_stats['swinging_strike_percent'] = (gp_stats.get('is_swinging_strike', 0) / pitches_divisor).fillna(0)
    gp_stats['called_strike_percent'] = (gp_stats.get('is_called_strike', 0) / pitches_divisor).fillna(0)
    gp_stats['zone_percent'] = (gp_stats.get('is_in_zone', 0) / pitches_divisor).fillna(0)
    gp_stats['fastball_percent'] = (gp_stats.get('is_fastball', 0) / pitches_divisor).fillna(0)
    gp_stats['breaking_percent'] = (gp_stats.get('is_breaking', 0) / pitches_divisor).fillna(0)
    gp_stats['offspeed_percent'] = (gp_stats.get('is_offspeed', 0) / pitches_divisor).fillna(0)

    woba_den_divisor = gp_stats.get('woba_denom', 0).replace(0, np.nan)
    gp_stats['woba'] = (gp_stats.get('woba_points', 0) / woba_den_divisor).fillna(0)
    gp_stats['babip'] = (gp_stats.get('babip_points', 0) / woba_den_divisor).fillna(0)
    gp_stats['iso'] = (gp_stats.get('iso_points', 0) / woba_den_divisor).fillna(0)

    # Rename sum cols etc. for final output if desired (or keep names from agg spec)
    gp_stats = gp_stats.rename(columns={
        'strikeout': 'strikeouts', 'walk': 'walks', 'hit': 'hits', 'home_run': 'home_runs',
        'is_swinging_strike': 'total_swinging_strikes', 'is_called_strike': 'total_called_strikes',
        'is_fastball': 'total_fastballs', 'is_breaking': 'total_breaking', 'is_offspeed': 'total_offspeed',
        'is_in_zone': 'total_in_zone' # Renamed for clarity
    })
    gp_stats['innings_pitched'] = (gp_stats.get('outs_recorded', 0) / 3.0).fillna(0)

    logger.info(f"Finished aggregating pitcher data. Shape: {gp_stats.shape}")
    return gp_stats


def calculate_additional_batter_stats(df):
    """Calculates additional stats for batters."""
    logger.debug("Calculating additional batter stats...")
    df['strikeout_bat'] = df['events'].apply(lambda x: 1 if x == 'strikeout' else 0)
    df['walk_bat'] = df['events'].apply(lambda x: 1 if x == 'walk' else 0)
    df['hit_bat'] = df['events'].apply(lambda x: 1 if x in ['single', 'double', 'triple', 'home_run'] else 0)
    df['home_run_bat'] = df['events'].apply(lambda x: 1 if x == 'home_run' else 0)
    df['single'] = df['events'].apply(lambda x: 1 if x == 'single' else 0)
    df['double'] = df['events'].apply(lambda x: 1 if x == 'double' else 0)
    df['triple'] = df['events'].apply(lambda x: 1 if x == 'triple' else 0)

    # Use unique at_bat_number for PA approximation
    if 'game_pk' in df.columns and 'batter' in df.columns and 'at_bat_number' in df.columns:
        pa_count_bat = df.groupby(['game_pk', 'batter'])['at_bat_number'].nunique().reset_index()
        pa_count_bat = pa_count_bat.rename(columns={'at_bat_number': 'pa'})
        df = pd.merge(df, pa_count_bat, on=['game_pk', 'batter'], how='left', suffixes=(None, '_calc'))
        if 'pa_calc' in df.columns:
            df['pa'] = df['pa_calc']
            df = df.drop(columns=['pa_calc'])
        elif 'pa' not in df.columns: df['pa'] = np.nan # Handle case where 'pa' column didn't exist initially
    else:
        logger.warning("Missing game_pk, batter, or at_bat_number. Cannot calculate batters' PA.")
        df['pa'] = np.nan

    if 'woba_denom' in df.columns and 'woba_value' in df.columns:
        df['woba_denom'] = pd.to_numeric(df['woba_denom'], errors='coerce').fillna(0)
        df['woba_value'] = pd.to_numeric(df['woba_value'], errors='coerce').fillna(0)
        df['woba_points_bat'] = df['woba_value'] * df['woba_denom']
    else:
        logger.warning("'woba_denom'/'woba_value' missing for batters. Skipping woba_points_bat.")
        df['woba_points_bat'] = 0
        if 'woba_denom' not in df.columns: df['woba_denom'] = 0

    if 'babip_value' in df.columns:
         df['babip_value'] = pd.to_numeric(df['babip_value'], errors='coerce').fillna(0)
         df['babip_points_bat'] = df['babip_value'] * df['woba_denom']
    else:
        logger.warning("'babip_value' missing for batters. Skipping babip_points_bat.")
        df['babip_points_bat'] = 0

    if 'iso_value' in df.columns:
        df['iso_value'] = pd.to_numeric(df['iso_value'], errors='coerce').fillna(0)
        df['iso_points_bat'] = df['iso_value'] * df['woba_denom']
    else:
        logger.warning("'iso_value' missing for batters. Skipping iso_points_bat.")
        df['iso_points_bat'] = 0

    df['launch_speed'] = pd.to_numeric(df.get('launch_speed'), errors='coerce')
    df['launch_angle'] = pd.to_numeric(df.get('launch_angle'), errors='coerce')
    df['is_hard_hit'] = (df['launch_speed'] >= 95).astype(int).fillna(0)
    df['is_barrel'] = ((df['launch_angle'] >= 26) & \
                       (df['launch_angle'] <= 30) & \
                       (df['launch_speed'] >= 98)).astype(int).fillna(0)

    logger.debug("Finished calculating additional batter stats.")
    return df

def aggregate_batter_data(df):
    """Aggregates batter data to game-batter level."""
    logger.info(f"Aggregating batter data for {df['game_pk'].nunique()} games, {df['batter'].nunique()} batters...")
    if df.empty: return pd.DataFrame()

    numeric_cols = [
        'launch_speed', 'launch_angle', 'hit_distance_sc',
        'strikeout_bat', 'walk_bat', 'hit_bat', 'home_run_bat',
        'single', 'double', 'triple', 'pa',
        'woba_points_bat', 'woba_denom', 'babip_points_bat', 'iso_points_bat',
        'is_hard_hit', 'is_barrel'
    ]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: logger.warning(f"Numeric column {col} not found for batter aggregation.")

    agg_dict = {
        'strikeout_bat_sum': ('strikeout_bat', 'sum'), 'walk_bat_sum': ('walk_bat', 'sum'),
        'hit_bat_sum': ('hit_bat', 'sum'), 'home_run_bat_sum': ('home_run_bat', 'sum'),
        'single_sum': ('single', 'sum'), 'double_sum': ('double', 'sum'), 'triple_sum': ('triple', 'sum'),
        'pa_sum': ('pa', 'first'), # Get the pre-calculated PA count per batter-game
        'launch_speed_mean': ('launch_speed', 'mean'), 'launch_speed_max': ('launch_speed', 'max'),
        'launch_angle_mean': ('launch_angle', 'mean'), 'hit_distance_sc_mean': ('hit_distance_sc', 'mean'),
        'hit_distance_sc_max': ('hit_distance_sc', 'max'), 'woba_points_bat_sum': ('woba_points_bat', 'sum'),
        'woba_denom_sum': ('woba_denom', 'sum'), 'babip_points_bat_sum': ('babip_points_bat', 'sum'),
        'iso_points_bat_sum': ('iso_points_bat', 'sum'), 'is_hard_hit_sum': ('is_hard_hit', 'sum'),
        'is_barrel_sum': ('is_barrel', 'sum'), 'player_name': ('player_name', 'first'),
        'stand': ('stand', 'first'), 'home_team': ('home_team', 'first'), 'away_team': ('away_team', 'first'),
        'inning_topbot': ('inning_topbot', 'first') # Keep for team assignment
    }

    agg_dict_filtered = {k: v for k, v in agg_dict.items() if v[0] in df.columns}
    logger.debug(f"Aggregating batters with spec: {agg_dict_filtered}")
    if not agg_dict_filtered: logger.error("No columns found for batter aggregation."); return pd.DataFrame()

    group_cols = ['game_pk', 'batter', 'game_date']
    if any(c not in df.columns for c in group_cols): logger.error(f"Missing grouping columns for batter agg: {group_cols}"); return pd.DataFrame()

    game_batter_stats = df.groupby(group_cols, observed=True, dropna=False).agg(**agg_dict_filtered).reset_index()
    game_batter_stats = game_batter_stats.rename(columns={
        'launch_speed_mean': 'avg_launch_speed', 'launch_speed_max': 'max_launch_speed',
        'launch_angle_mean': 'avg_launch_angle', 'hit_distance_sc_mean': 'avg_hit_distance',
        'hit_distance_sc_max': 'max_hit_distance', 'pa_sum': 'pa' # Rename aggregated PA count
    })

    # --- Derived Batter Stats ---
    logger.debug("Calculating derived batter stats...")
    gb_stats = game_batter_stats
    pa_divisor = gb_stats.get('pa', 0).replace(0, np.nan)
    gb_stats['k_percent_bat'] = (gb_stats.get('strikeout_bat_sum', 0) / pa_divisor).fillna(0)
    gb_stats['bb_percent_bat'] = (gb_stats.get('walk_bat_sum', 0) / pa_divisor).fillna(0)
    gb_stats['hr_per_pa'] = (gb_stats.get('home_run_bat_sum', 0) / pa_divisor).fillna(0)
    gb_stats['hard_hit_percent'] = (gb_stats.get('is_hard_hit_sum', 0) / pa_divisor).fillna(0)
    gb_stats['barrel_percent'] = (gb_stats.get('is_barrel_sum', 0) / pa_divisor).fillna(0)

    woba_den_divisor = gb_stats.get('woba_denom_sum', 0).replace(0, np.nan)
    gb_stats['woba_bat'] = (gb_stats.get('woba_points_bat_sum', 0) / woba_den_divisor).fillna(0)
    gb_stats['babip_bat'] = (gb_stats.get('babip_points_bat_sum', 0) / woba_den_divisor).fillna(0)
    gb_stats['iso_bat'] = (gb_stats.get('iso_points_bat_sum', 0) / woba_den_divisor).fillna(0)

    gb_stats['at_bats_approx'] = (gb_stats.get('pa', 0) - gb_stats.get('walk_bat_sum', 0)).clip(lower=0)
    ab_divisor = gb_stats.get('at_bats_approx', 0).replace(0, np.nan)
    gb_stats['batting_avg'] = (gb_stats.get('hit_bat_sum', 0) / ab_divisor).fillna(0)
    total_bases = gb_stats.get('single_sum', 0) + (gb_stats.get('double_sum', 0) * 2) + \
                  (gb_stats.get('triple_sum', 0) * 3) + (gb_stats.get('home_run_bat_sum', 0) * 4)
    gb_stats['slugging_pct'] = (total_bases / ab_divisor).fillna(0)
    gb_stats['ops_approx'] = gb_stats['batting_avg'] + gb_stats['slugging_pct']

    logger.info(f"Finished aggregating batter data. Shape: {gb_stats.shape}")
    return gb_stats


# ================================================================
# == REVISED aggregate_team_data FUNCTION (BULK PROCESSING) ==
# ================================================================
def aggregate_team_data(pitcher_game_stats, batter_game_stats):
    """
    Aggregates pitcher and batter stats to the game-team level using bulk operations.
    """
    logger.info("Aggregating team-level data (Revised Bulk Method)...")
    if pitcher_game_stats.empty and batter_game_stats.empty:
        logger.warning("Both pitcher and batter aggregated stats are empty. Cannot create team stats.")
        return pd.DataFrame()

    # Combine game_pks from both sources safely
    game_pks_pitcher_game = pitcher_game_stats['game_pk'].unique() if not pitcher_game_stats.empty and 'game_pk' in pitcher_game_stats.columns else np.array([])
    game_pks_batter_game = batter_game_stats['game_pk'].unique() if not batter_game_stats.empty and 'game_pk' in batter_game_stats.columns else np.array([])

    game_pks = np.union1d(game_pks_pitcher_game, game_pks_batter_game)

    if len(game_pks) == 0:
        logger.warning("No game_pk values found in aggregated stats. Cannot determine team stats.")
        return pd.DataFrame()

    logger.info(f"Processing {len(game_pks)} unique games for team aggregation.")

    all_team_stats = pd.DataFrame()

    try:
        # *** Corrected: No argument for DBConnection ***
        with DBConnection() as conn:
            # Fetch minimal pitch-level data ONCE for all relevant games
            logger.info("Fetching pitch-level data for team assignment...")
            game_pks_str = ','.join(map(str, game_pks))
            cols = ["game_pk", "pitcher", "batter", "inning_topbot", "home_team", "away_team", "game_date"] # Added game_date
            # Check if statcast_pitchers exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='statcast_pitchers';")
            if cursor.fetchone() is None:
                logger.error("Source table 'statcast_pitchers' not found for team assignment. Aborting team aggregation.")
                return pd.DataFrame()

            q_pitch_data = f'SELECT {", ".join(cols)} FROM statcast_pitchers WHERE game_pk IN ({game_pks_str})'
            pitch_level_data = pd.read_sql(q_pitch_data, conn)
            logger.info(f"Fetched {len(pitch_level_data)} pitch-level rows for team assignment.")

            if pitch_level_data.empty:
                 logger.warning("No pitch-level data found for the relevant games. Cannot determine team stats.")
                 return pd.DataFrame()

            # Ensure types before processing
            pitch_level_data['game_pk'] = pd.to_numeric(pitch_level_data['game_pk'], errors='coerce')
            pitch_level_data['pitcher'] = pd.to_numeric(pitch_level_data['pitcher'], errors='coerce')
            pitch_level_data['batter'] = pd.to_numeric(pitch_level_data['batter'], errors='coerce')
            pitch_level_data = pitch_level_data.dropna(subset=['game_pk', 'pitcher', 'batter', 'inning_topbot'])


            # --- Determine Team Affiliation in Bulk ---
            pitch_level_data['pitching_team'] = np.where(
                pitch_level_data['inning_topbot'] == 'Top', pitch_level_data['home_team'], pitch_level_data['away_team']
            )
            pitch_level_data['batting_team'] = np.where(
                pitch_level_data['inning_topbot'] == 'Top', pitch_level_data['away_team'], pitch_level_data['home_team']
            )

            # Create unique mappings for pitcher/batter to their team *in that game*
            pitcher_team_map = pitch_level_data[['game_pk', 'pitcher', 'pitching_team']].drop_duplicates().rename(columns={'pitching_team': 'team'})
            batter_team_map = pitch_level_data[['game_pk', 'batter', 'batting_team']].drop_duplicates().rename(columns={'batting_team': 'team'})

            # Ensure correct dtypes for merging in aggregated dataframes
            if not pitcher_game_stats.empty:
                # Use pitcher_id if renamed, otherwise pitcher
                pitcher_id_col = 'pitcher_id' if 'pitcher_id' in pitcher_game_stats.columns else 'pitcher'
                pitcher_game_stats[pitcher_id_col] = pd.to_numeric(pitcher_game_stats[pitcher_id_col], errors='coerce')
                pitcher_game_stats['game_pk'] = pd.to_numeric(pitcher_game_stats['game_pk'], errors='coerce')
            if not batter_game_stats.empty:
                batter_game_stats['batter'] = pd.to_numeric(batter_game_stats['batter'], errors='coerce')
                batter_game_stats['game_pk'] = pd.to_numeric(batter_game_stats['game_pk'], errors='coerce')

            # --- Merge Team Info with Aggregated Stats ---
            logger.info("Merging team information with aggregated stats...")
            pgs_merged = pd.DataFrame()
            if not pitcher_game_stats.empty:
                 pitcher_id_col = 'pitcher_id' if 'pitcher_id' in pitcher_game_stats.columns else 'pitcher'
                 # Rename map key to match aggregated table ('pitcher_id' or 'pitcher')
                 pitcher_team_map_renamed = pitcher_team_map.rename(columns={'pitcher': pitcher_id_col})
                 pgs_merged = pd.merge(pitcher_game_stats, pitcher_team_map_renamed, on=['game_pk', pitcher_id_col], how='left')
                 missing_teams_p = pgs_merged['team'].isnull().sum()
                 if missing_teams_p > 0: logger.warning(f"Could not determine team for {missing_teams_p} pitcher-game entries.")

            bgs_merged = pd.DataFrame()
            if not batter_game_stats.empty:
                bgs_merged = pd.merge(batter_game_stats, batter_team_map, on=['game_pk', 'batter'], how='left')
                missing_teams_b = bgs_merged['team'].isnull().sum()
                if missing_teams_b > 0: logger.warning(f"Could not determine team for {missing_teams_b} batter-game entries.")

            # --- Aggregate by Game and Team ---
            logger.info("Aggregating stats by game and team...")
            team_pitching_agg = pd.DataFrame()
            if not pgs_merged.empty and 'team' in pgs_merged.columns:
                 pitch_agg_spec = {
                     # Use renamed columns from aggregate_pitcher_data output
                     'strikeouts': ('strikeouts', 'sum'), 'walks': ('walks', 'sum'), 'hits': ('hits', 'sum'), 'home_runs': ('home_runs', 'sum'),
                     'batters_faced': ('batters_faced', 'sum'), 'outs_recorded': ('outs_recorded', 'sum'), 'total_pitches': ('total_pitches', 'sum'),
                     'total_swinging_strikes': ('total_swinging_strikes', 'sum'), 'total_called_strikes': ('total_called_strikes', 'sum'),
                     'total_fastballs': ('total_fastballs', 'sum'), 'total_breaking': ('total_breaking', 'sum'), 'total_offspeed': ('total_offspeed', 'sum'),
                     'woba_points': ('woba_points', 'sum'), 'woba_denom': ('woba_denom', 'sum'),
                     'babip_points': ('babip_points', 'sum'), 'iso_points': ('iso_points', 'sum'),
                     'avg_velocity': ('avg_velocity', 'mean'), 'max_velocity': ('max_velocity', 'max')
                 }
                 pitch_agg_spec_filtered = {k: v for k, v in pitch_agg_spec.items() if v[0] in pgs_merged.columns}
                 logger.debug(f"Pitching team aggregation spec: {pitch_agg_spec_filtered}")
                 if pitch_agg_spec_filtered:
                      team_pitching_agg = pgs_merged.groupby(['game_pk', 'team']).agg(**pitch_agg_spec_filtered).reset_index()
                 else: logger.warning("No valid columns found for pitching team aggregation.")

            team_batting_agg = pd.DataFrame()
            if not bgs_merged.empty and 'team' in bgs_merged.columns:
                 bat_agg_spec = {
                     # Use columns from aggregate_batter_data output
                     'strikeouts_bat': ('strikeout_bat_sum', 'sum'), 'walks_bat': ('walk_bat_sum', 'sum'),
                     'hits_bat': ('hit_bat_sum', 'sum'), 'home_runs_bat': ('home_run_bat_sum', 'sum'),
                     'pa_bat': ('pa', 'sum'), # Use 'pa' as source
                     'hard_hits_bat': ('is_hard_hit_sum', 'sum'), 'barrels_bat': ('is_barrel_sum', 'sum'),
                     'woba_points_bat': ('woba_points_bat_sum', 'sum'), 'woba_denom_bat': ('woba_denom_sum', 'sum'),
                     'babip_points_bat': ('babip_points_bat_sum', 'sum'), 'iso_points_bat': ('iso_points_bat_sum', 'sum'),
                     'avg_ls_bat': ('avg_launch_speed', 'mean'), 'avg_la_bat': ('avg_launch_angle', 'mean')
                 }
                 bat_agg_spec_filtered = {k: v for k, v in bat_agg_spec.items() if v[0] in bgs_merged.columns}
                 # Ensure 'pa' source column exists before including 'pa_bat' target
                 if 'pa' not in bgs_merged.columns:
                     logger.warning("Source column 'pa' not found in batter stats, cannot aggregate 'pa_bat'.")
                     bat_agg_spec_filtered.pop('pa_bat', None)

                 logger.debug(f"Batting team aggregation spec: {bat_agg_spec_filtered}")
                 if bat_agg_spec_filtered:
                     team_batting_agg = bgs_merged.groupby(['game_pk', 'team']).agg(**bat_agg_spec_filtered).reset_index()
                 else: logger.warning("No valid columns remaining for batting team aggregation.")


            # --- Merge Pitching and Batting Aggregates ---
            logger.info("Merging team pitching and batting aggregates...")
            if not team_pitching_agg.empty and not team_batting_agg.empty:
                 all_team_stats = pd.merge(team_pitching_agg, team_batting_agg, on=['game_pk', 'team'], how='outer')
            elif not team_pitching_agg.empty: all_team_stats = team_pitching_agg
            elif not team_batting_agg.empty: all_team_stats = team_batting_agg
            else:
                logger.warning("Both team pitching and batting aggregates are empty. Cannot create final team stats.")
                return pd.DataFrame()

            # --- Add Game Metadata ---
            game_metadata = pitch_level_data[['game_pk', 'game_date', 'home_team', 'away_team']].drop_duplicates(subset=['game_pk'])
            all_team_stats = pd.merge(all_team_stats, game_metadata, on='game_pk', how='left')
            all_team_stats['is_home_team'] = (all_team_stats['team'] == all_team_stats['home_team']).astype(int)

            # --- Recalculate Rate Stats at Team Level ---
            logger.info("Recalculating rate stats at team level...")
            # Pitching
            outs_divisor = all_team_stats.get('outs_recorded', 0).replace(0, np.nan)
            all_team_stats['hr_per_9'] = (all_team_stats.get('home_runs', 0) / outs_divisor * 27).fillna(0)
            bf_divisor = all_team_stats.get('batters_faced', 0).replace(0, np.nan)
            all_team_stats['k_percent'] = (all_team_stats.get('strikeouts', 0) / bf_divisor).fillna(0)
            all_team_stats['bb_percent'] = (all_team_stats.get('walks', 0) / bf_divisor).fillna(0)
            pitches_divisor = all_team_stats.get('total_pitches', 0).replace(0, np.nan)
            all_team_stats['swinging_strike_percent'] = (all_team_stats.get('total_swinging_strikes', 0) / pitches_divisor).fillna(0)
            all_team_stats['called_strike_percent'] = (all_team_stats.get('total_called_strikes', 0) / pitches_divisor).fillna(0)
            all_team_stats['fastball_percent'] = (all_team_stats.get('total_fastballs', 0) / pitches_divisor).fillna(0)
            all_team_stats['breaking_percent'] = (all_team_stats.get('total_breaking', 0) / pitches_divisor).fillna(0)
            all_team_stats['offspeed_percent'] = (all_team_stats.get('total_offspeed', 0) / pitches_divisor).fillna(0)
            woba_den_divisor = all_team_stats.get('woba_denom', 0).replace(0, np.nan)
            all_team_stats['woba'] = (all_team_stats.get('woba_points', 0) / woba_den_divisor).fillna(0)
            all_team_stats['babip'] = (all_team_stats.get('babip_points', 0) / woba_den_divisor).fillna(0)
            all_team_stats['iso'] = (all_team_stats.get('iso_points', 0) / woba_den_divisor).fillna(0)

            # Batting
            pa_divisor = all_team_stats.get('pa_bat', 0).replace(0, np.nan)
            all_team_stats['k_percent_bat'] = (all_team_stats.get('strikeouts_bat', 0) / pa_divisor).fillna(0)
            all_team_stats['bb_percent_bat'] = (all_team_stats.get('walks_bat', 0) / pa_divisor).fillna(0)
            all_team_stats['hr_per_pa'] = (all_team_stats.get('home_runs_bat', 0) / pa_divisor).fillna(0)
            all_team_stats['hard_hit_percent'] = (all_team_stats.get('hard_hits_bat', 0) / pa_divisor).fillna(0)
            all_team_stats['barrel_percent'] = (all_team_stats.get('barrels_bat', 0) / pa_divisor).fillna(0)
            woba_den_bat_divisor = all_team_stats.get('woba_denom_bat', 0).replace(0, np.nan)
            all_team_stats['woba_bat'] = (all_team_stats.get('woba_points_bat', 0) / woba_den_bat_divisor).fillna(0)
            all_team_stats['babip_bat'] = (all_team_stats.get('babip_points_bat', 0) / woba_den_bat_divisor).fillna(0)
            all_team_stats['iso_bat'] = (all_team_stats.get('iso_points_bat', 0) / woba_den_bat_divisor).fillna(0)

    except sqlite3.Error as e:
        logger.error(f"SQLite error during bulk team aggregation: {e}", exc_info=True)
        return pd.DataFrame()
    except KeyError as e:
        logger.error(f"KeyError during bulk team aggregation, likely missing/misnamed column: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error during bulk team aggregation: {e}", exc_info=True)
        return pd.DataFrame()

    logger.info(f"Finished aggregating team data (Revised Bulk Method). Shape: {all_team_stats.shape}")
    required_cols = ['game_pk', 'team', 'game_date', 'home_team', 'away_team']
    missing_req = [c for c in required_cols if c not in all_team_stats.columns]
    if missing_req:
        logger.error(f"Final team stats DF is missing required columns: {missing_req}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Convert game_date to string 'YYYY-MM-DD' format if it's datetime
    if 'game_date' in all_team_stats.columns and pd.api.types.is_datetime64_any_dtype(all_team_stats['game_date']):
        logger.debug("Converting game_date to string format in team stats.")
        all_team_stats['game_date'] = all_team_stats['game_date'].dt.strftime('%Y-%m-%d')


    return all_team_stats
# ================================================================
# == END REVISED aggregate_team_data FUNCTION ==
# ================================================================


# --- Main Aggregation Functions ---

def aggregate_statcast_pitchers_sql(target_date=None):
    start_time = time.time()
    logger.info(f"Starting pitcher aggregation.")
    table_name = 'statcast_pitchers'
    output_table = 'game_level_pitchers'

    try:
        with DBConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if cursor.fetchone() is None: logger.error(f"Source table '{table_name}' does not exist."); return

            cursor.execute(f"PRAGMA table_info({table_name})")
            available_cols = [info[1] for info in cursor.fetchall()]
            cols_to_select_list = [c for c in PITCHER_METRICS_RAW if c in available_cols]
            missing_metrics = [c for c in PITCHER_METRICS_RAW if c not in available_cols]
            if missing_metrics: logger.warning(f"Columns missing from {table_name}: {missing_metrics}")
            for essential_col in ['game_pk', 'pitcher', 'game_date', 'at_bat_number', 'events', 'inning_topbot', 'home_team', 'away_team']:
                 if essential_col not in cols_to_select_list and essential_col in available_cols: cols_to_select_list.append(essential_col)

            cols_to_select = ", ".join(f'"{c}"' for c in cols_to_select_list)
            if not cols_to_select: logger.error(f"No selectable columns found for {table_name}"); return

            query = f"SELECT {cols_to_select} FROM {table_name}"
            if target_date:
                if 'game_date' not in cols_to_select_list: logger.error("Cannot filter by target_date, 'game_date' column not selected/available."); return
                query += f" WHERE DATE(game_date) <= '{target_date}'" # Use DATE() for safety

            logger.info(f"Executing pitcher query: {query[:200]}...")
            pitcher_data = pd.read_sql_query(query, conn)
            logger.info(f"Read {len(pitcher_data)} rows from {table_name}.")

            if pitcher_data.empty:
                logger.warning(f"No pitcher data found for query.")
                with DBConnection() as conn_store: conn_store.execute(f"DROP TABLE IF EXISTS {output_table}"); conn_store.commit()
                logger.info(f"Dropped potentially existing table {output_table}.")
                return

            pitcher_data = calculate_additional_pitcher_stats(pitcher_data)
            game_pitcher_stats = aggregate_pitcher_data(pitcher_data)

            if not game_pitcher_stats.empty:
                 # *** RENAME 'pitcher' column TO 'pitcher_id' ***
                 if 'pitcher' in game_pitcher_stats.columns:
                     logger.debug(f"Renaming 'pitcher' column to 'pitcher_id' in {output_table}")
                     game_pitcher_stats = game_pitcher_stats.rename(columns={'pitcher': 'pitcher_id'})
                 else: logger.warning(f"Column 'pitcher' not found before saving {output_table}, cannot rename to 'pitcher_id'.")

                 # Convert game_date to string 'YYYY-MM-DD' format if it's datetime
                 if 'game_date' in game_pitcher_stats.columns and pd.api.types.is_datetime64_any_dtype(game_pitcher_stats['game_date']):
                     logger.debug("Converting game_date to string format in pitcher stats.")
                     game_pitcher_stats['game_date'] = game_pitcher_stats['game_date'].dt.strftime('%Y-%m-%d')

                 logger.info(f"Storing {len(game_pitcher_stats)} aggregated pitcher rows to {output_table}...")
                 with DBConnection() as conn_store:
                     conn_store.execute(f"DROP TABLE IF EXISTS {output_table}")
                     game_pitcher_stats.to_sql(output_table, conn_store, if_exists='replace', index=False)
                     conn_store.commit()
                 logger.info(f"Successfully stored aggregated data to {output_table}.")
            else:
                 logger.warning(f"No pitcher game stats to store.")
                 with DBConnection() as conn_store: conn_store.execute(f"DROP TABLE IF EXISTS {output_table}"); conn_store.commit()

    except sqlite3.Error as e: logger.error(f"SQLite error during pitcher aggregation: {e}", exc_info=True)
    except KeyError as e: logger.error(f"KeyError during pitcher aggregation: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error during pitcher aggregation: {e}", exc_info=True)
    finally: gc.collect(); logger.info(f"Pitcher aggregation finished in {time.time() - start_time:.2f} seconds.")


def aggregate_statcast_batters_sql(target_date=None):
    start_time = time.time()
    logger.info(f"Starting batter aggregation.")
    table_name = 'statcast_batters'
    output_table = 'game_level_batters' # Use user's preferred name

    try:
        with DBConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if cursor.fetchone() is None: logger.error(f"Source table '{table_name}' does not exist."); return

            cursor.execute(f"PRAGMA table_info({table_name})")
            available_cols = [info[1] for info in cursor.fetchall()]
            cols_to_select_list = [c for c in BATTER_METRICS_RAW if c in available_cols]
            missing_metrics = [c for c in BATTER_METRICS_RAW if c not in available_cols]
            if missing_metrics: logger.warning(f"Columns missing from {table_name}: {missing_metrics}")
            for essential_col in ['game_pk', 'batter', 'game_date', 'at_bat_number', 'events', 'inning_topbot', 'home_team', 'away_team']:
                 if essential_col not in cols_to_select_list and essential_col in available_cols: cols_to_select_list.append(essential_col)

            cols_to_select = ", ".join(f'"{c}"' for c in cols_to_select_list)
            if not cols_to_select: logger.error(f"No selectable columns found for {table_name}"); return

            query = f"SELECT {cols_to_select} FROM {table_name}"
            if target_date:
                if 'game_date' not in cols_to_select_list: logger.error("Cannot filter by target_date, 'game_date' column not selected/available."); return
                query += f" WHERE DATE(game_date) <= '{target_date}'"

            logger.info(f"Executing batter query: {query[:200]}...")
            batter_data = pd.read_sql_query(query, conn)
            logger.info(f"Read {len(batter_data)} rows from {table_name}.")

            if batter_data.empty:
                logger.warning(f"No batter data found for query.")
                with DBConnection() as conn_store: conn_store.execute(f"DROP TABLE IF EXISTS {output_table}"); conn_store.commit()
                logger.info(f"Dropped potentially existing table {output_table}.")
                return

            batter_data = calculate_additional_batter_stats(batter_data)
            game_batter_stats = aggregate_batter_data(batter_data)

            if not game_batter_stats.empty:
                 # Convert game_date to string 'YYYY-MM-DD' format if it's datetime
                 if 'game_date' in game_batter_stats.columns and pd.api.types.is_datetime64_any_dtype(game_batter_stats['game_date']):
                     logger.debug("Converting game_date to string format in batter stats.")
                     game_batter_stats['game_date'] = game_batter_stats['game_date'].dt.strftime('%Y-%m-%d')

                 logger.info(f"Storing {len(game_batter_stats)} aggregated batter rows to {output_table}...")
                 with DBConnection() as conn_store:
                     conn_store.execute(f"DROP TABLE IF EXISTS {output_table}")
                     game_batter_stats.to_sql(output_table, conn_store, if_exists='replace', index=False)
                     conn_store.commit()
                 logger.info(f"Successfully stored aggregated data to {output_table}.")
            else:
                logger.warning(f"No batter game stats to store.")
                with DBConnection() as conn_store: conn_store.execute(f"DROP TABLE IF EXISTS {output_table}"); conn_store.commit()

    except sqlite3.Error as e: logger.error(f"SQLite error during batter aggregation: {e}", exc_info=True)
    except KeyError as e: logger.error(f"KeyError during batter aggregation: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error during batter aggregation: {e}", exc_info=True)
    finally: gc.collect(); logger.info(f"Batter aggregation finished in {time.time() - start_time:.2f} seconds.")

def aggregate_game_level_data():
    start_time = time.time()
    logger.info("Starting game-level team and starter aggregation.")
    pitcher_stats_table = 'game_level_pitchers' # Use user's preferred name
    batter_stats_table = 'game_level_batters' # Use user's preferred name
    team_stats_output_table = 'game_level_team_stats' # Use user's preferred name
    starter_stats_output_table = 'game_level_starter_stats'
    pitcher_mapping_table = 'pitcher_mapping'

    try:
        with DBConnection() as conn:
            logger.info(f"Loading data from {pitcher_stats_table} and {batter_stats_table}...")
            pitcher_game_stats = pd.DataFrame()
            batter_game_stats = pd.DataFrame()
            try: pitcher_game_stats = pd.read_sql_query(f"SELECT * FROM {pitcher_stats_table}", conn)
            except Exception as e: logger.error(f"Failed loading {pitcher_stats_table}: {e}")
            try: batter_game_stats = pd.read_sql_query(f"SELECT * FROM {batter_stats_table}", conn)
            except Exception as e: logger.error(f"Failed loading {batter_stats_table}: {e}")

            if pitcher_game_stats.empty and batter_game_stats.empty:
                logger.error("Aggregated pitcher and batter data empty/missing. Cannot proceed.")
                return

            # --- Team Level Aggregation (using revised bulk function) ---
            game_team_stats = aggregate_team_data(pitcher_game_stats, batter_game_stats)
            if not game_team_stats.empty:
                logger.info(f"Storing {len(game_team_stats)} aggregated team rows to {team_stats_output_table}...")
                with DBConnection() as conn_store:
                     conn_store.execute(f"DROP TABLE IF EXISTS {team_stats_output_table}")
                     game_team_stats.to_sql(team_stats_output_table, conn_store, if_exists='replace', index=False)
                     conn_store.commit()
                logger.info(f"Successfully stored aggregated team data to {team_stats_output_table}.")
            else:
                logger.warning("No team stats generated to store.")
                with DBConnection() as conn_store: conn_store.execute(f"DROP TABLE IF EXISTS {team_stats_output_table}"); conn_store.commit()

            # --- Starting Pitcher Identification ---
            logger.info("Identifying starting pitchers...")
            if pitcher_game_stats.empty: logger.warning("Pitcher game stats are empty, cannot identify starters.")
            else:
                try:
                    pitcher_mapping = pd.read_sql_query(f"SELECT pitcher_id, is_starter FROM {pitcher_mapping_table}", conn)
                    pitcher_mapping = pitcher_mapping.rename(columns={'pitcher_id': 'pitcher'}) # Use 'pitcher' internally to match source table
                    pitcher_mapping['pitcher'] = pd.to_numeric(pitcher_mapping['pitcher'], errors='coerce')
                    pitcher_mapping['is_starter'] = pd.to_numeric(pitcher_mapping['is_starter'], errors='coerce')
                    starter_pitcher_ids = pitcher_mapping.dropna(subset=['pitcher', 'is_starter'])
                    starter_pitcher_ids = starter_pitcher_ids[starter_pitcher_ids['is_starter'] == 1][['pitcher']].drop_duplicates()

                    # Use correct column name ('pitcher_id') from loaded pitcher_game_stats
                    pitcher_id_col_starter = 'pitcher_id' if 'pitcher_id' in pitcher_game_stats.columns else 'pitcher'
                    if pitcher_id_col_starter == 'pitcher': logger.warning("Using 'pitcher' column for starter merge; expected 'pitcher_id'.")

                    # Rename map to match pitcher_id_col_starter for merge
                    starter_pitcher_ids_renamed = starter_pitcher_ids.rename(columns={'pitcher': pitcher_id_col_starter})
                    starter_stats = pd.merge(pitcher_game_stats, starter_pitcher_ids_renamed, on=pitcher_id_col_starter, how='inner')

                    # Assign team and opponent using already aggregated columns
                    starter_stats['team'] = np.where(starter_stats.get('inning_topbot', 'Unknown') == 'Top',
                                                    starter_stats.get('home_team'), starter_stats.get('away_team'))
                    starter_stats['team'] = starter_stats.groupby(['game_pk', pitcher_id_col_starter])['team'].transform('first')
                    starter_stats['opponent_team'] = np.where(starter_stats['team'] == starter_stats.get('home_team'),
                                                             starter_stats.get('away_team'), starter_stats.get('home_team'))
                    starter_stats = starter_stats.drop_duplicates(subset=['game_pk', 'team'], keep='first')

                    # Select and rename columns for the final starter table
                    # Ensure these column names match output of aggregate_pitcher_data
                    starter_cols = [
                        'game_pk', 'game_date', pitcher_id_col_starter, # Use actual column name
                        'player_name', 'p_throws', 'team', 'opponent_team', 'home_team', 'away_team',
                        'strikeouts', 'walks', 'batters_faced', 'outs_recorded', 'innings_pitched',
                        'total_pitches', 'k_percent', 'bb_percent' # Add more derived stats if needed
                    ]
                    final_starter_cols = [col for col in starter_cols if col in starter_stats.columns]
                    final_starter_stats = starter_stats[final_starter_cols].copy()

                    # *** RENAME pitcher identifier column TO 'pitcher_id' for final output ***
                    if pitcher_id_col_starter in final_starter_stats.columns and pitcher_id_col_starter != 'pitcher_id':
                        logger.debug(f"Renaming '{pitcher_id_col_starter}' column to 'pitcher_id' in {starter_stats_output_table}")
                        final_starter_stats = final_starter_stats.rename(columns={pitcher_id_col_starter: 'pitcher_id'})
                    elif 'pitcher_id' not in final_starter_stats.columns:
                         logger.warning(f"Could not find or create 'pitcher_id' column in {starter_stats_output_table}.")

                    # Rename other columns for clarity if needed (player_name etc. should be correct from agg)

                    # Convert game_date to string 'YYYY-MM-DD' format if it's datetime
                    if 'game_date' in final_starter_stats.columns and pd.api.types.is_datetime64_any_dtype(final_starter_stats['game_date']):
                        logger.debug("Converting game_date to string format in starter stats.")
                        final_starter_stats['game_date'] = final_starter_stats['game_date'].dt.strftime('%Y-%m-%d')


                    if not final_starter_stats.empty:
                        logger.info(f"Storing {len(final_starter_stats)} starter rows to {starter_stats_output_table}...")
                        with DBConnection() as conn_store:
                             conn_store.execute(f"DROP TABLE IF EXISTS {starter_stats_output_table}")
                             final_starter_stats.to_sql(starter_stats_output_table, conn_store, if_exists='replace', index=False)
                             conn_store.commit()
                        logger.info(f"Successfully stored starter data to {starter_stats_output_table}.")
                    else:
                         logger.warning("No starter stats generated to store.")
                         with DBConnection() as conn_store: conn_store.execute(f"DROP TABLE IF EXISTS {starter_stats_output_table}"); conn_store.commit()

                except Exception as e: logger.error(f"Error identifying starters: {e}", exc_info=True)

    except sqlite3.Error as e: logger.error(f"SQLite error during game-level aggregation: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error during game-level aggregation: {e}", exc_info=True)
    finally: gc.collect(); logger.info(f"Game-level aggregation finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    if not MODULE_IMPORTS_OK:
        logger.error("Required modules not found. Exiting.")
        sys.exit(1) # Exit if essential modules fail
    else:
        logger.info("Starting Statcast Aggregation Script")
        # Step 1: Aggregate Pitcher Game-Level Stats
        logger.info("--- Running Pitcher Game Level Aggregation ---")
        aggregate_statcast_pitchers_sql()
        gc.collect()
        # Step 2: Aggregate Batter Game-Level Stats
        logger.info("--- Running Batter Game Level Aggregation ---")
        aggregate_statcast_batters_sql()
        gc.collect()
        # Step 3: Aggregate Team-Level Stats and Identify Starters
        logger.info("--- Running Team and Starter Level Aggregation ---")
        aggregate_game_level_data()
        gc.collect()
        logger.info("Statcast Aggregation Script Completed")