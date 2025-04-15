# src/features/batter_features.py (Rewritten - Replace table, import traceback)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import time
import traceback # <--- Added import

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.data.utils import setup_logger, DBConnection
    from src.config import StrikeoutModelConfig, DBConfig
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed import in batter_features: {e}")
    MODULE_IMPORTS_OK = False
    # Dummy definitions... (same as before)
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, p=None): self.p=p or "dummy.db"; self.conn = None
        def __enter__(self): import sqlite3; print("WARN: Dummy DB"); self.conn = sqlite3.connect(self.p); return self.conn
        def __exit__(self,t,v,tb):
            if self.conn: self.conn.close()
    class StrikeoutModelConfig: WINDOW_SIZES = [3, 5, 10]
    class DBConfig: PATH = "data/pitcher_stats.db"

logger = setup_logger('batter_features') if MODULE_IMPORTS_OK else logging.getLogger('batter_features_fallback')

def create_batter_features(df=None, dataset_type="all"):
    """Create batter features from pre-aggregated game-level data.

    Args:
        df (pandas.DataFrame, optional): Game-level data from aggregate_batters.
            If None, load from game_level_batters DB table.
        dataset_type (str): Split type ('train', 'test', 'predict', 'all').

    Returns:
        pandas.DataFrame: DataFrame with batter features including rolling metrics.
    """
    start_time = time.time()
    if not MODULE_IMPORTS_OK: logger.error("Exiting: Module imports failed."); return pd.DataFrame()
    db_path = project_root / DBConfig.PATH

    try:
        # Load game-level batter data if not provided
        if df is None:
            logger.info(f"Loading game_level_batters data for {dataset_type} set...")
            table_to_load = 'game_level_batters'
            with DBConnection(db_path) as conn:
                 if conn is None: raise ConnectionError("DB Connection failed.")
                 cursor = conn.cursor()
                 cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_to_load}'")
                 if not cursor.fetchone():
                     logger.error(f"Source table '{table_to_load}' not found. Run aggregate_batters.py first."); return pd.DataFrame()
                 query = f"SELECT * FROM {table_to_load}"
                 df = pd.read_sql_query(query, conn)
            if df.empty: logger.error(f"No data found in {table_to_load}."); return pd.DataFrame()
            logger.info(f"Loaded {len(df)} rows from {table_to_load}")

        # --- Preprocessing ---
        if 'game_date' not in df.columns: logger.error("Missing 'game_date'."); return pd.DataFrame()
        if 'batter_id' not in df.columns: logger.error("Missing 'batter_id'."); return pd.DataFrame()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['batter_id', 'game_date']).reset_index(drop=True)

        # --- Feature Calculation ---
        logger.info(f"Calculating rolling features for {dataset_type} batter set...")
        count_categories = ['ahead', 'behind', 'even', '0-0', '3-2']
        count_metrics = ['k_pct', 'woba', 'pa']
        base_metrics = [
            'k_percent', 'woba', 'swinging_strike_percent', 'chase_percent',
            'zone_swing_percent', 'contact_percent', 'zone_contact_percent',
            'fastball_whiff_pct', 'breaking_whiff_pct', 'offspeed_whiff_pct']
        split_metrics = ['k_pct', 'woba', 'pa']
        split_hands = ['R', 'L']
        metrics_to_roll = list(base_metrics)
        for metric in count_metrics:
            for cat in count_categories: metrics_to_roll.append(f'{metric}_{cat}')
        for metric in split_metrics:
            for hand in split_hands: metrics_to_roll.append(f'{metric}_vs_{hand}')

        available_metrics = [m for m in metrics_to_roll if m in df.columns]
        missing_metrics = [m for m in metrics_to_roll if m not in df.columns]
        if missing_metrics: logger.warning(f"Metrics missing from input data (will not be rolled): {missing_metrics}")
        if not available_metrics: logger.error("No available metrics for rolling."); return df

        logger.info(f"Rolling features for: {available_metrics}")
        window_sizes = StrikeoutModelConfig.WINDOW_SIZES
        grouped_batter = df.groupby('batter_id')
        rolling_features_dict = {}

        for metric in available_metrics:
            shifted_metric = grouped_batter[metric].shift(1)
            for window in window_sizes:
                roll_mean_col = f'rolling_{window}g_{metric}'
                rolling_features_dict[roll_mean_col] = shifted_metric.rolling(window, min_periods=max(1, window // 2)).mean()
                roll_std_col = f'rolling_{window}g_{metric}_std'
                rolling_features_dict[roll_std_col] = shifted_metric.rolling(window, min_periods=max(2, window // 2)).std()

        rolling_features_df = pd.DataFrame(rolling_features_dict, index=df.index)
        df = pd.concat([df, rolling_features_df], axis=1)
        logger.info("Rolling features calculation and concatenation complete.")

        # --- Fill Missing Values (for Rolling Features) ---
        newly_created_cols = list(rolling_features_dict.keys())
        logger.info(f"Filling NaNs in {len(newly_created_cols)} rolling feature columns...")
        na_fill_start = time.time()
        for col in newly_created_cols:
            if df[col].isnull().any(): df[col].fillna(df[col].median(), inplace=True)
        logger.info(f"NaN filling took {time.time() - na_fill_start:.2f}s")

        # --- Save to Database ---
        output_table_name = "batter_features"
        logger.info(f"Preparing to save {len(df)} features to '{output_table_name}'...")
        df['split'] = dataset_type if dataset_type in ("train", "test", "predict") else "all"

        if 'umpire' in df.columns:
            initial_rows = len(df); df = df[df['umpire'].notnull()]; rows_dropped = initial_rows - len(df)
            if rows_dropped > 0: logger.info(f"Dropped {rows_dropped} rows with null umpire values.")
        else: logger.warning("Column 'umpire' not found. Cannot filter for null umpires.")

        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed before saving.")
            # *** FIX: Use 'replace' instead of 'append' ***
            # Drop table explicitly first for robustness with different DB types/versions
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {output_table_name}")
                conn.commit()
                logger.info(f"Dropped existing {output_table_name} table (if any).")
            except Exception as drop_e:
                 logger.warning(f"Could not drop {output_table_name} table: {drop_e}")

            df.to_sql(output_table_name, conn, if_exists='replace', index=False, chunksize=5000)
            logger.info(f"Saved {len(df)} batter features to '{output_table_name}' table (replaced).")

        total_time = time.time() - start_time
        logger.info(f"Batter feature creation completed in {total_time:.2f} seconds")
        return df

    except Exception as e:
        # *** FIX: Use imported traceback ***
        logger.error(f"Error creating batter features for {dataset_type}: {e}", exc_info=True)
        logger.error(traceback.format_exc()) # Print full traceback
        return pd.DataFrame()

if __name__ == "__main__":
     if not MODULE_IMPORTS_OK: sys.exit("Exiting: Module imports failed.")
     logger.info("Running batter feature creation directly for testing (using all data)...")
     test_features = create_batter_features(dataset_type="all")
     if test_features is not None and not test_features.empty:
         logger.info("Test run completed successfully.")
     else: logger.error("Test run failed or produced empty DataFrame.")