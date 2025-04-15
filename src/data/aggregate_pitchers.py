# src/data/aggregate_pitchers.py (Rewritten using Pandas)
import sqlite3
import logging
import pandas as pd
import numpy as np
import time
import traceback
from pathlib import Path
import sys

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.data.utils import setup_logger, DBConnection
    from src.config import DBConfig, DataConfig # Import DataConfig if chunksize needed
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed import in aggregate_pitchers: {e}")
    MODULE_IMPORTS_OK = False
    # Dummy definitions
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, p=None): self.p=p or "dummy.db"; self.conn = None
        def __enter__(self): import sqlite3; print("WARN: Dummy DB"); self.conn = sqlite3.connect(self.p); return self.conn
        def __exit__(self,t,v,tb):
             if self.conn: self.conn.close()
    class DBConfig: PATH = "data/pitcher_stats.db"
    class DataConfig: CHUNK_SIZE = 1000000 # Example chunk size

logger = setup_logger('aggregate_pitchers') if MODULE_IMPORTS_OK else logging.getLogger('aggregate_pitchers_fallback')


def aggregate_pitchers_to_game_level():
    """
    Aggregates pitch-level data from statcast_pitchers to game-level for pitchers using pandas,
    and stores the result in the game_level_pitchers table, replacing any existing table.

    Returns:
        bool: Success status
    """
    start_time = time.time()
    if not MODULE_IMPORTS_OK: logger.error("Exiting: Module imports failed."); return False
    db_path = project_root / DBConfig.PATH
    logger.info("Starting game-level pitcher aggregation using pandas...")

    required_cols = [
        'pitcher', 'player_name', 'game_date', 'game_pk', 'home_team', 'away_team',
        'p_throws', 'season', 'events', 'at_bat_number', 'description',
        'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'zone', 'pitch_type'
        # Add 'batter' if needed for specific metrics like batters_faced based on unique batters
    ]
    numeric_cols = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'zone']

    try:
        # --- Load Data ---
        logger.info(f"Loading required columns from statcast_pitchers...")
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            # Check available columns
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(statcast_pitchers)")
            available_columns = {info[1] for info in cursor.fetchall()}
            cols_to_select = [col for col in required_cols if col in available_columns]
            missing_req = [col for col in required_cols if col not in available_columns]
            if missing_req: logger.warning(f"Required columns missing from statcast_pitchers: {missing_req}. Aggregation might be incomplete.")
            if not cols_to_select: logger.error("No required columns found in statcast_pitchers."); return False

            logger.info(f"Selected columns: {cols_to_select}")
            query = f"SELECT {', '.join(cols_to_select)} FROM statcast_pitchers"

            # Load in chunks if table is large
            chunk_size = getattr(DataConfig, 'CHUNK_SIZE', 1000000)
            chunks = []
            logger.info(f"Loading data in chunks of {chunk_size}...")
            for i, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=chunk_size)):
                 chunks.append(chunk)
                 if i % 5 == 0: logger.info(f"Loaded chunk {i+1}...")
            if not chunks: logger.error("No data loaded from statcast_pitchers."); return False
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Data loading complete. Shape: {df.shape}")

        # --- Preprocessing ---
        logger.info("Preprocessing data...")
        df['game_date'] = pd.to_datetime(df['game_date']).dt.date
        # Convert numeric columns, coercing errors
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Define grouping keys - ensure types are consistent
        group_keys = ['pitcher', 'player_name', 'game_date', 'game_pk', 'home_team', 'away_team', 'p_throws', 'season']
        key_check = [key for key in group_keys if key not in df.columns]
        if key_check: logger.error(f"Missing grouping key columns: {key_check}"); return False
        df['pitcher'] = pd.to_numeric(df['pitcher'], errors='coerce').astype('Int64')
        df['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce').astype('Int64')
        df.dropna(subset=['pitcher', 'game_pk', 'game_date'], inplace=True) # Drop rows where key IDs are missing

        # --- Calculate Intermediate Flags ---
        df['is_strikeout'] = df['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)
        # Outs approximation - adjust based on available 'events' data accuracy
        out_events = ['field_out', 'strikeout', 'grounded_into_double_play', 'force_out', 'sac_fly', 'sac_bunt', 'double_play']
        df['outs_recorded'] = df['events'].apply(lambda x: 1 if x in out_events else 2 if x in ['grounded_into_double_play', 'double_play'] else 0).astype(int)
        df['is_swinging_strike'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)
        df['is_zone'] = df['zone'].between(1, 9, inclusive='both') if 'zone' in df.columns else pd.Series(False, index=df.index)

        # Pitch type categories (can be refined)
        df['is_fastball'] = df['pitch_type'].isin(['FF', 'FT', 'FC', 'SI']) if 'pitch_type' in df.columns else pd.Series(False, index=df.index)
        df['is_breaking'] = df['pitch_type'].isin(['SL', 'CU', 'KC', 'KN']) if 'pitch_type' in df.columns else pd.Series(False, index=df.index)
        df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS', 'FO', 'SC']) if 'pitch_type' in df.columns else pd.Series(False, index=df.index)

        # --- Aggregation ---
        logger.info("Performing aggregations...")
        agg_dict = {
            'strikeouts': ('is_strikeout', 'sum'),
            'batters_faced': ('at_bat_number', 'nunique'), # Count unique ABs faced
            'total_pitches': ('pitcher', 'count'), # Count all pitches
            'total_outs_recorded': ('outs_recorded', 'sum'),
            'zone_pitches': ('is_zone', 'sum'),
            'swinging_strikes': ('is_swinging_strike', 'sum'),
            'fastball_count': ('is_fastball', 'sum'),
            'breaking_count': ('is_breaking', 'sum'),
            'offspeed_count': ('is_offspeed', 'sum'),
        }
        # Add numeric aggregations only if columns exist
        if 'release_speed' in df.columns:
            agg_dict['avg_velocity'] = ('release_speed', 'mean')
            agg_dict['max_velocity'] = ('release_speed', 'max')
        if 'release_spin_rate' in df.columns:
            agg_dict['avg_spin_rate'] = ('release_spin_rate', 'mean')
        if 'pfx_x' in df.columns:
            agg_dict['avg_horizontal_break'] = ('pfx_x', 'mean')
        if 'pfx_z' in df.columns:
            agg_dict['avg_vertical_break'] = ('pfx_z', 'mean')

        game_level_df = df.groupby(group_keys, observed=False).agg(**agg_dict).reset_index()
        logger.info(f"Aggregation complete. Shape: {game_level_df.shape}")

        # --- Calculate Derived Metrics ---
        logger.info("Calculating derived metrics...")
        # Innings Pitched (handle potential 0 outs)
        game_level_df['innings_pitched'] = game_level_df['total_outs_recorded'] / 3.0
        # K/9 (handle 0 IP)
        game_level_df['k_per_9'] = (game_level_df['strikeouts'] * 9.0 / game_level_df['innings_pitched'].replace(0, np.nan)).fillna(0)
        # K% (handle 0 BF)
        game_level_df['k_percent'] = (game_level_df['strikeouts'] / game_level_df['batters_faced'].replace(0, np.nan)).fillna(0)
        # Zone% (handle 0 pitches)
        game_level_df['zone_percent'] = (game_level_df['zone_pitches'] / game_level_df['total_pitches'].replace(0, np.nan)).fillna(0)
        # SwStr% (handle 0 pitches)
        game_level_df['swinging_strike_percent'] = (game_level_df['swinging_strikes'] / game_level_df['total_pitches'].replace(0, np.nan)).fillna(0)
        # Pitch Mix % (handle 0 pitches)
        game_level_df['fastball_percent'] = (game_level_df['fastball_count'] / game_level_df['total_pitches'].replace(0, np.nan)).fillna(0)
        game_level_df['breaking_percent'] = (game_level_df['breaking_count'] / game_level_df['total_pitches'].replace(0, np.nan)).fillna(0)
        game_level_df['offspeed_percent'] = (game_level_df['offspeed_count'] / game_level_df['total_pitches'].replace(0, np.nan)).fillna(0)

        # Rename pitcher -> pitcher_id for consistency if needed (depends on downstream usage)
        game_level_df.rename(columns={'pitcher': 'pitcher_id'}, inplace=True)

        # Select and order final columns (optional, but good practice)
        final_cols = [
            'pitcher_id', 'player_name', 'game_date', 'game_pk', 'home_team', 'away_team', 'p_throws', 'season',
            'strikeouts', 'batters_faced', 'total_pitches', 'avg_velocity', 'max_velocity', 'avg_spin_rate',
            'avg_horizontal_break', 'avg_vertical_break', 'zone_percent', 'swinging_strike_percent',
            'innings_pitched', 'k_per_9', 'k_percent', 'fastball_percent', 'breaking_percent', 'offspeed_percent'
        ]
        # Keep only columns that actually exist in the dataframe
        final_cols_exist = [col for col in final_cols if col in game_level_df.columns]
        game_level_df = game_level_df[final_cols_exist]

        # --- Save to Database ---
        logger.info(f"Saving {len(game_level_df)} aggregated pitcher records to game_level_pitchers...")
        # Use the imported store_data_to_sql for robust saving
        from src.scripts.data_fetcher import store_data_to_sql # Ensure imported
        success = store_data_to_sql(game_level_df, 'game_level_pitchers', db_path, if_exists='replace')

        if success:
            total_time = time.time() - start_time
            logger.info(f"Successfully aggregated pitcher data using pandas in {total_time:.2f} seconds")
            return True
        else:
            logger.error("Failed to save aggregated pitcher data.")
            return False

    except Exception as e:
        logger.error(f"Error aggregating pitcher data using pandas: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # This can be called directly for testing
    logger.info("Running pitcher aggregation directly for testing...")
    aggregate_pitchers_to_game_level()
    logger.info("Test run finished.")