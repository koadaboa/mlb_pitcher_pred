# src/features/umpire_features.py
import pandas as pd
import numpy as np
import logging
import sqlite3
from typing import Callable, List, Dict, Tuple
from pathlib import Path
import sys

# Assuming the calling script (generate_features.py) handles path setup
# OR the project is installed correctly.
try:
    from src.data.utils import DBConnection
    from src.config import DBConfig
except ImportError as e:
    # Log import error here, but don't necessarily stop execution yet.
    # Functions below will fail if imports didn't work.
    logging.getLogger(__name__).error(f"Failed to import DBConnection or DBConfig: {e}. Umpire features may fail.")
    # Set DBConnection/DBConfig to None or placeholders if needed,
    # though subsequent code will likely raise exceptions.
    DBConnection = None
    DBConfig = None


logger = logging.getLogger(__name__)


def load_team_mapping_from_db() -> pd.DataFrame:
    """Loads the team mapping from the SQLite 'team_mapping' table."""
    # Check if imports succeeded
    if DBConnection is None or DBConfig is None:
         logger.error("DBConnection or DBConfig not imported. Cannot load team mapping from DB.")
         return pd.DataFrame()

    # Get DB path directly from config
    try:
        db_path = DBConfig.PATH
        if not db_path or not Path(db_path).exists():
             logger.error(f"Database path not found or invalid in DBConfig: {db_path}")
             return pd.DataFrame()
    except AttributeError:
         logger.error("DBConfig.DB_FILE not defined.")
         return pd.DataFrame()
    except Exception as e:
         logger.error(f"Error accessing DBConfig.DB_FILE: {e}")
         return pd.DataFrame()


    mapping_table = 'team_mapping'
    required_cols = ['team_abbr', 'team_name']
    logger.info(f"Loading team mapping from database table: {mapping_table}")

    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{mapping_table}'")
            if not cursor.fetchone():
                logger.error(f"Team mapping table '{mapping_table}' not found in the database: {db_path}")
                return pd.DataFrame()

            cursor.execute(f"PRAGMA table_info({mapping_table})")
            available_cols = [info[1] for info in cursor.fetchall()]
            missing_cols = [col for col in required_cols if col not in available_cols]
            if missing_cols:
                logger.error(f"Team mapping table '{mapping_table}' is missing required columns: {missing_cols}. Available: {available_cols}")
                return pd.DataFrame()

            query = f"SELECT {', '.join(required_cols)} FROM {mapping_table}"
            team_map_df = pd.read_sql_query(query, conn)

        if team_map_df.empty:
            logger.warning(f"Team mapping table '{mapping_table}' is empty.")
            return pd.DataFrame()

        team_map_df = team_map_df.drop_duplicates(subset=['team_name'], keep='first')
        logger.info(f"Loaded {len(team_map_df)} unique team name mappings from DB table '{mapping_table}'.")
        return team_map_df

    except sqlite3.Error as e:
        logger.error(f"SQLite error loading team mapping from table '{mapping_table}': {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading team mapping from DB: {e}", exc_info=True)
        return pd.DataFrame()


# --- Modified function ---
def calculate_umpire_rolling_features(
    pitcher_hist_df: pd.DataFrame,
    umpire_hist_df: pd.DataFrame,
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Calculates rolling features for home plate umpires based on pitcher stats
    in games they officiated. Includes team name standardization using the
    'team_mapping' table from the SQLite database. Assumes imports work.
    """
    # Remove the initial check for MODULE_IMPORTS_OK / DB_PATH
    # Rely on load_team_mapping_from_db to handle DB setup issues

    if pitcher_hist_df is None or pitcher_hist_df.empty or umpire_hist_df is None or umpire_hist_df.empty:
        logger.warning("Input DataFrames for umpire rolling features are invalid or empty.")
        return pd.DataFrame(index=pitcher_hist_df.index if pitcher_hist_df is not None else None), {}

    # --- Load Team Mapping from DB ---
    team_map_df = load_team_mapping_from_db() # Use updated DB loading function
    if team_map_df.empty:
        # Error already logged by load_team_mapping_from_db
        logger.error("Failed to load team mapping from DB. Cannot calculate umpire features.")
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    # --- Check Required Columns ---
    required_pitcher_cols = [date_col, 'home_team', 'away_team'] + metrics
    required_umpire_cols = [date_col, 'home_team', 'away_team', group_col]
    # ... (rest of the column checks remain the same) ...
    missing_pitcher_cols = [col for col in required_pitcher_cols if col not in pitcher_hist_df.columns]
    missing_umpire_cols = [col for col in required_umpire_cols if col not in umpire_hist_df.columns]

    if missing_pitcher_cols:
        logger.error(f"Missing required columns in pitcher_hist_df for umpire features: {missing_pitcher_cols}")
        return pd.DataFrame(index=pitcher_hist_df.index), {}
    if missing_umpire_cols:
        logger.error(f"Missing required columns in umpire_hist_df for umpire features: {missing_umpire_cols}")
        return pd.DataFrame(index=pitcher_hist_df.index), {}


    logger.info(f"Calculating umpire rolling features for '{group_col}' (Windows: {windows}). Standardizing teams using DB mapping...")

    # --- Prepare DataFrames for Merge ---
    # ... (The standardization and merge logic remains identical to the previous version) ...
    try:
        # 1. Prepare Pitcher Data
        pitcher_hist_df_copy = pitcher_hist_df[required_pitcher_cols].copy()
        if not pd.api.types.is_datetime64_any_dtype(pitcher_hist_df_copy[date_col]):
             pitcher_hist_df_copy[date_col] = pd.to_datetime(pitcher_hist_df_copy[date_col], errors='coerce')
        pitcher_hist_df_copy = pitcher_hist_df_copy.dropna(subset=[date_col])
        pitcher_hist_df_copy = pitcher_hist_df_copy.rename(columns={'home_team': 'home_team_abbr', 'away_team': 'away_team_abbr'})

        # 2. Prepare Umpire Data
        umpire_merge_df = umpire_hist_df[required_umpire_cols].drop_duplicates().copy()
        if not pd.api.types.is_datetime64_any_dtype(umpire_merge_df[date_col]):
            umpire_merge_df[date_col] = pd.to_datetime(umpire_merge_df[date_col], errors='coerce')
        umpire_merge_df = umpire_merge_df.dropna(subset=[date_col])
        umpire_merge_df = umpire_merge_df.rename(columns={'home_team': 'home_team_name', 'away_team': 'away_team_name'})

        # Merge mapping for home team
        umpire_merge_df = pd.merge( umpire_merge_df, team_map_df[['team_name', 'team_abbr']], left_on='home_team_name', right_on='team_name', how='left')
        umpire_merge_df = umpire_merge_df.assign(home_team_abbr=umpire_merge_df['team_abbr'])
        umpire_merge_df = umpire_merge_df.drop(columns=['team_name', 'team_abbr'])

        # Merge mapping for away team
        umpire_merge_df = pd.merge( umpire_merge_df, team_map_df[['team_name', 'team_abbr']], left_on='away_team_name', right_on='team_name', how='left')
        umpire_merge_df = umpire_merge_df.assign(away_team_abbr=umpire_merge_df['team_abbr'])
        umpire_merge_df = umpire_merge_df.drop(columns=['team_name', 'team_abbr'])

        # Check for teams that failed to map
        missing_home_map = umpire_merge_df['home_team_abbr'].isnull().sum()
        missing_away_map = umpire_merge_df['away_team_abbr'].isnull().sum()
        if missing_home_map > 0:
             unmapped_home = umpire_merge_df.loc[umpire_merge_df['home_team_abbr'].isnull(), 'home_team_name'].unique()
             logger.warning(f"Could not map {missing_home_map} umpire home teams to standard abbreviations. Unmapped names: {unmapped_home[:10]}...")
        if missing_away_map > 0:
             unmapped_away = umpire_merge_df.loc[umpire_merge_df['away_team_abbr'].isnull(), 'away_team_name'].unique()
             logger.warning(f"Could not map {missing_away_map} umpire away teams to standard abbreviations. Unmapped names: {unmapped_away[:10]}...")

        # Select final columns and drop rows where mapping failed
        final_umpire_cols = [date_col, 'home_team_abbr', 'away_team_abbr', group_col]
        umpire_merge_df = umpire_merge_df[final_umpire_cols].dropna(subset=['home_team_abbr', 'away_team_abbr'])
        umpire_merge_df = umpire_merge_df.drop_duplicates()

        if umpire_merge_df.empty:
            logger.error("Umpire data is empty after attempting to standardize team names via DB mapping. Cannot merge.")
            return pd.DataFrame(index=pitcher_hist_df.index), {}

    except Exception as e:
        logger.error(f"Failed during data preparation or team standardization: {e}", exc_info=True)
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    # --- Merge umpire name onto pitcher history ---
    logger.info("Merging pitcher history with standardized umpire data...")
    merged_df = pd.merge(
        pitcher_hist_df_copy,
        umpire_merge_df,
        on=[date_col, 'home_team_abbr', 'away_team_abbr'],
        how='left'
    )
    merged_df = merged_df.set_index(pitcher_hist_df.index) # Align index

    # --- Check Merge Success & Calculate Rolling Features ---
    # ... (Rest of the function: checking merge results, calculating rolling features, renaming, reindexing remains the same) ...
    missing_ump_count = merged_df[group_col].isnull().sum()
    total_rows = len(merged_df)
    successful_merge_count = total_rows - missing_ump_count
    logger.info(f"Merge Result: Found umpires for {successful_merge_count} / {total_rows} pitcher appearances.")

    if successful_merge_count == 0:
         logger.error(f"Merge failed completely: Could not find home plate umpire for any of the {total_rows} pitcher appearances after standardization. Check DB team mapping and date alignment.")
         return pd.DataFrame(index=pitcher_hist_df.index), {}
    elif missing_ump_count > 0:
        logger.warning(f"Could not find home plate umpire for {missing_ump_count} pitcher appearances after standardization (check dates/teams/mapping).")

    available_metrics = [m for m in metrics if m in merged_df.columns]
    if not available_metrics:
        logger.warning("No specified umpire metrics found in the merged DataFrame.")
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    valid_umpire_merged_df = merged_df.dropna(subset=[group_col])
    if valid_umpire_merged_df.empty:
        logger.warning(f"No rows with valid umpire names ({group_col}) remained after merge. Cannot calculate umpire features.")
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    logger.info(f"Calculating rolling features for {len(valid_umpire_merged_df[group_col].unique())} umpires on {len(valid_umpire_merged_df)} rows.")
    umpire_rolling_calc = calculate_multi_window_rolling(
        df=valid_umpire_merged_df,
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods
    )

    rename_map = { f"{m}_roll{w}g": f"ump_roll{w}g_{m}" for w in windows for m in available_metrics if f"{m}_roll{w}g" in umpire_rolling_calc.columns }
    umpire_rolling_renamed = umpire_rolling_calc.rename(columns=rename_map)
    umpire_rolling_final = umpire_rolling_renamed.reindex(pitcher_hist_df.index)

    logger.info(f"Finished calculating umpire rolling features. Found {len(umpire_rolling_final.columns)} features.")
    return umpire_rolling_final, rename_map