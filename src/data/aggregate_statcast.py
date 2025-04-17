# src/data/aggregate_statcast.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sqlite3
from pathlib import Path
import logging
import time
import gc
from datetime import datetime # Ensure datetime is imported

# Assuming script is run via engineer_features or standalone with similar setup
try:
    # Added FileConfig assumption if needed for checkpoints (though removed)
    from src.config import DBConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection
    MODULE_IMPORTS_OK = True
except ImportError:
    MODULE_IMPORTS_OK = False
    # Basic fallbacks if config/utils are not found
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('aggregate_statcast_fallback')
    DB_PATH = Path("./data/mlb_data.db") # Default path
    class DBConnection: # Basic context manager
        def __init__(self, db_path=DB_PATH): self.db_path = db_path
        def __enter__(self): self.conn = sqlite3.connect(self.db_path); return self.conn
        def __exit__(self,et,ev,tb): self.conn.close()
else:
    # Setup logger using imported function
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('aggregate_statcast', LogConfig.LOG_DIR / 'aggregate_statcast.log', level=logging.DEBUG) # Use DEBUG level
    DB_PATH = Path(DBConfig.PATH)

# --- Constants ---
ID_COLS_TO_EXCLUDE_FROM_DOWNCAST = ['pitcher', 'batter', 'game_pk', 'pitcher_id']

# --- Memory Optimization Helper ---
def optimize_dtypes(df):
    """Attempt to reduce memory usage by downcasting dtypes, excluding specified ID columns."""
    logger.debug("Optimizing dtypes...")
    df_copy = df.copy() # Work on a copy
    cols_to_exclude = [col for col in ID_COLS_TO_EXCLUDE_FROM_DOWNCAST if col in df_copy.columns]
    logger.debug(f" Excluding ID columns from downcasting: {cols_to_exclude}")
    for col in df_copy.select_dtypes(include=['int64']).columns:
        if col not in cols_to_exclude: # Check if column should be excluded
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
            except Exception as e:
                 logger.warning(f"Could not downcast integer column '{col}': {e}")
    for col in df_copy.select_dtypes(include=['float64']).columns:
         try:
            df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
         except Exception as e:
            logger.warning(f"Could not downcast float column '{col}': {e}")
    logger.debug("Dtype optimization attempt complete.")
    return df_copy


# --- Imputation Functions ---

def smart_impute(df, group_col, target_cols):
    """Imputes missing values using group median first, then global median."""
    logger.info(f"Performing smart imputation for {group_col} missing values...")
    df_copy = df.copy()
    for col in target_cols:
        if col in df_copy.columns and df_copy[col].isnull().any(): # Check col exists
            logger.info(f"Imputing {df_copy[col].isnull().sum()} NaNs in '{col}' using {group_col}/global median...")
            # Calculate group median (ensure numeric type for median)
            numeric_groups = df_copy.dropna(subset=[col])
            if pd.api.types.is_numeric_dtype(numeric_groups[col]):
                 # Use apply with lambda to handle potential empty groups gracefully
                 group_median_map = numeric_groups.groupby(group_col)[col].median()
                 df_copy[col] = df_copy[col].fillna(df_copy[group_col].map(group_median_map))

                 # Apply global median for remaining NaNs (groups with all NaNs or new NaNs from mapping)
                 global_median = df_copy[col].median()
                 if pd.notna(global_median):
                      df_copy[col] = df_copy[col].fillna(global_median)
                 else: # Handle case where global median might also be NaN (all values were NaN)
                      logger.warning(f"   Global median for '{col}' is NaN. Filling remaining NaNs with 0.")
                      df_copy[col] = df_copy[col].fillna(0)

                 logger.debug(f"   '{col}' imputation complete. Remaining NaNs: {df_copy[col].isnull().sum()}")
            else:
                 logger.warning(f"   Column '{col}' is not numeric, skipping median imputation.")
        elif col in df_copy.columns:
             logger.debug(f"   Column '{col}' has no NaNs to impute.")
    return df_copy

def knn_impute_complex_features(df, target_cols, n_neighbors=5):
    """Performs KNN imputation on specified columns."""
    logger.info(f"Performing KNN imputation for remaining complex features: {target_cols}...")
    df_copy = df.copy()
    # Select only numeric columns for KNN, including target_cols and potential predictors
    numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    valid_target_cols = []
    for col in target_cols:
        if col in df_copy.columns:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                 logger.warning(f"KNN Imputation: Column '{col}' is not numeric, attempting conversion.")
                 df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                 valid_target_cols.append(col)
            else:
                 logger.warning(f"KNN Imputation: Column '{col}' could not be converted to numeric, skipping.")
        else:
             logger.warning(f"KNN Imputation: Target column '{col}' not found in DataFrame.")

    cols_to_impute = [col for col in valid_target_cols if df_copy[col].isnull().any()]
    if not cols_to_impute:
        logger.info("No NaNs found in specified valid KNN target columns.")
        return df_copy

    potential_predictors = [c for c in numeric_cols if c not in cols_to_impute and df_copy[c].isnull().sum() < len(df_copy) * 0.5]
    if not potential_predictors:
         logger.error("KNN Imputation: No suitable predictor columns found. Skipping KNN imputation.")
         return df_copy

    impute_df_subset = df_copy[potential_predictors + cols_to_impute]
    all_nan_cols = impute_df_subset.columns[impute_df_subset.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning(f"KNN Imputation: Columns {all_nan_cols} contain all NaNs. Cannot be imputed. Skipping KNN.")
        return df_copy # Or impute with median/zero first

    try:
        # Impute NaNs in predictors first (using median) before KNN
        for p_col in potential_predictors:
             if impute_df_subset[p_col].isnull().any():
                  impute_df_subset[p_col] = impute_df_subset[p_col].fillna(impute_df_subset[p_col].median())

        imputer = KNNImputer(n_neighbors=n_neighbors)
        # Operate on the subset for imputation
        imputed_data = imputer.fit_transform(impute_df_subset)
        imputed_df = pd.DataFrame(imputed_data, columns=impute_df_subset.columns, index=impute_df_subset.index)

        # Update original dataframe copy using the results from the imputed subset
        df_copy.update(imputed_df[cols_to_impute])

        logger.info(f"KNN imputation complete for columns: {cols_to_impute}.")
    except Exception as e:
        logger.error(f"KNN imputation failed: {e}", exc_info=True)

    return df_copy

# --- PITCHER AGGREGATION ---

def aggregate_statcast_pitchers_sql(start_date=None, end_date=None):
    """
    Aggregates raw Statcast pitcher data to game level using SQL.
    Filters results based on games present in master_schedule.
    """
    logger.info("Starting PITCHER Statcast aggregation using SQL...")
    start_time = time.time()

    # Define the SQL query for aggregation (No change needed in the aggregation logic itself)
    # *** NOTE: is_home calculation assumes inning_topbot exists and is accurate ***
    sql = """
    WITH GamePitcherStats AS (
        SELECT
            s.pitcher AS pitcher_id,
            s.game_pk,
            DATE(s.game_date) AS game_date,
            MAX(s.player_name) AS player_name,
            MAX(s.home_team) AS home_team,
            MAX(s.away_team) AS away_team,
            MAX(s.p_throws) AS p_throws,
            MAX(CASE WHEN s.inning_topbot = 'Top' THEN 1 ELSE 0 END) AS is_home,
            SUM(CASE WHEN s.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts,
            COUNT(DISTINCT s.batter) AS batters_faced,
            COUNT(*) AS total_pitches,
            SUM(CASE WHEN s.type = 'S' AND s.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS total_swinging_strikes,
            SUM(CASE WHEN s.type = 'S' AND s.description = 'called_strike' THEN 1 ELSE 0 END) AS total_called_strikes,
            SUM(CASE WHEN s.pitch_type IN ('FF', 'SI', 'FT', 'FC') THEN 1 ELSE 0 END) AS total_fastballs,
            SUM(CASE WHEN s.pitch_type IN ('SL', 'CU', 'KC', 'CS', 'ST', 'SV') THEN 1 ELSE 0 END) AS total_breaking,
            SUM(CASE WHEN s.pitch_type IN ('CH', 'FS', 'KN', 'EP', 'SC') THEN 1 ELSE 0 END) AS total_offspeed,
            SUM(CASE WHEN s.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS total_in_zone,
            AVG(s.release_speed) AS avg_velocity,
            MAX(s.release_speed) AS max_velocity,
            AVG(s.release_spin_rate) AS avg_spin_rate,
            AVG(s.pfx_x) AS avg_horizontal_break,
            AVG(s.pfx_z) AS avg_vertical_break
        FROM statcast_pitchers s
        GROUP BY s.pitcher, s.game_pk, DATE(s.game_date)
    ),
    GameInnings AS (
        SELECT
            pitcher AS pitcher_id,
            game_pk,
            COUNT(DISTINCT inning) + MAX(CASE WHEN outs_when_up = 1 THEN 0.1 WHEN outs_when_up = 2 THEN 0.2 ELSE 0 END) as innings_pitched_approx
        FROM statcast_pitchers
        GROUP BY pitcher, game_pk
    )
    SELECT
        gps.*,
        STRFTIME('%Y', gps.game_date) AS season,
        gi.innings_pitched_approx AS innings_pitched,
        CAST(gps.strikeouts AS REAL) / NULLIF(gps.batters_faced, 0) AS k_percent,
        (CAST(gps.strikeouts AS REAL) * 9.0) / NULLIF(gi.innings_pitched_approx, 0) AS k_per_9,
        CAST(gps.total_swinging_strikes AS REAL) / NULLIF(gps.total_pitches, 0) AS swinging_strike_percent,
        CAST(gps.total_called_strikes AS REAL) / NULLIF(gps.total_pitches, 0) AS called_strike_percent,
        CAST(gps.total_fastballs AS REAL) / NULLIF(gps.total_pitches, 0) AS fastball_percent,
        CAST(gps.total_breaking AS REAL) / NULLIF(gps.total_pitches, 0) AS breaking_percent,
        CAST(gps.total_offspeed AS REAL) / NULLIF(gps.total_pitches, 0) AS offspeed_percent,
        CAST(gps.total_in_zone AS REAL) / NULLIF(gps.total_pitches, 0) AS zone_percent,
        CASE
            WHEN gps.is_home = 1 THEN gps.away_team
            ELSE gps.home_team
        END AS opponent_team
    FROM GamePitcherStats gps
    LEFT JOIN GameInnings gi ON gps.pitcher_id = gi.pitcher_id AND gps.game_pk = gi.game_pk
    ORDER BY gps.game_date, gps.game_pk, gps.pitcher_id
    """

    agg_df = pd.DataFrame()
    try:
        logger.info("Executing SQL query for PITCHER game-level aggregation...")
        with DBConnection(DB_PATH) as conn:
            # Load identifying columns from master_schedule (CORRECTED TABLE)
            logger.info("Loading game identifiers from master_schedule...")
            schedule_games_df = pd.read_sql_query("SELECT DISTINCT game_date, home_team, away_team FROM master_schedule", conn)
            logger.debug(f"Loaded {len(schedule_games_df)} distinct games from master_schedule.")
            if not schedule_games_df.empty:
                 # Create composite key for schedule data
                 schedule_games_df['game_date'] = pd.to_datetime(schedule_games_df['game_date']).dt.strftime('%Y-%m-%d')
                 schedule_games_df['home_team'] = schedule_games_df['home_team'].str.strip()
                 schedule_games_df['away_team'] = schedule_games_df['away_team'].str.strip()
                 schedule_game_ids = set(schedule_games_df['game_date'] + '_' + schedule_games_df['away_team'] + '_' + schedule_games_df['home_team'])
                 logger.debug(f"Created {len(schedule_game_ids)} unique game identifiers from master_schedule.")
            else:
                 logger.warning("Master schedule table is empty or query failed. Cannot filter based on schedule data.")
                 schedule_game_ids = set()

            # Execute main aggregation query
            agg_df = pd.read_sql_query(sql, conn)
            logger.info(f"SQL aggregation returned {len(agg_df)} PITCHER game-level records")

            # Debug date range before filtering
            if not agg_df.empty:
                 agg_df['game_date'] = pd.to_datetime(agg_df['game_date']).dt.strftime('%Y-%m-%d')
                 min_date_agg = agg_df['game_date'].min(); max_date_agg = agg_df['game_date'].max()
                 logger.debug(f"Date range in aggregated PITCHER data BEFORE schedule filter: {min_date_agg} to {max_date_agg}")

        # Filter results based on games that have schedule data available
        if not schedule_game_ids:
             logger.warning("Skipping filtering based on schedule data as no schedule game identifiers were loaded.")
        elif not agg_df.empty:
            logger.info("Filtering PITCHER results based on available schedule data...")
            original_count = len(agg_df)
            try:
                 # Ensure team columns exist and are strings before stripping/combining
                 if 'home_team' not in agg_df.columns or 'away_team' not in agg_df.columns:
                     logger.error("Missing 'home_team' or 'away_team' in aggregated pitcher data. Cannot filter.")
                 else:
                     agg_df['home_team'] = agg_df['home_team'].astype(str).str.strip()
                     agg_df['away_team'] = agg_df['away_team'].astype(str).str.strip()
                     agg_df['composite_game_id'] = agg_df['game_date'] + '_' + agg_df['away_team'] + '_' + agg_df['home_team']
                     # Keep rows where composite_game_id is in the set from schedule_data
                     agg_df_filtered = agg_df[agg_df['composite_game_id'].isin(schedule_game_ids)].copy()
                     # Use drop on the filtered copy, not the original slice
                     agg_df_filtered = agg_df_filtered.drop(columns=['composite_game_id'])
                     agg_df = agg_df_filtered # Assign back to agg_df
                     logger.info(f"Filtered PITCHER records to {len(agg_df)} with schedule data (from {original_count})")
            except KeyError as ke:
                 logger.error(f"Missing key column for schedule filtering in aggregated data: {ke}. Skipping filter.")
            except Exception as fe:
                 logger.error(f"Error during schedule data filtering: {fe}. Skipping filter.")

            # Debug date range after filtering
            if not agg_df.empty:
                 min_date_filt = agg_df['game_date'].min(); max_date_filt = agg_df['game_date'].max()
                 logger.debug(f"Date range in aggregated PITCHER data AFTER schedule filter: {min_date_filt} to {max_date_filt}")
        else:
             logger.info("Aggregated PITCHER DataFrame is empty, skipping schedule filtering.")

        # Imputation for missing values
        if not agg_df.empty:
            pitcher_impute_cols = ['avg_velocity', 'max_velocity', 'avg_spin_rate', 'avg_horizontal_break', 'avg_vertical_break']
            agg_df = smart_impute(agg_df, 'pitcher_id', pitcher_impute_cols)
            knn_target_cols = [col for col in pitcher_impute_cols if col in agg_df.columns and agg_df[col].isnull().any()]
            if knn_target_cols:
                 agg_df = knn_impute_complex_features(agg_df, knn_target_cols)
            logger.info(f"PITCHER NaN counts after imputation:\n{agg_df.isnull().sum()[agg_df.isnull().sum() > 0]}")
        else:
             logger.info("Skipping imputation as PITCHER DataFrame is empty.")


        # Save to database
        logger.info("Saving final aggregated PITCHER data to database table 'game_level_pitchers'...")
        with DBConnection(DB_PATH) as conn:
            agg_df.to_sql('game_level_pitchers', conn, if_exists='replace', index=False)
        logger.info(f"Saved {len(agg_df)} PITCHER records to game_level_pitchers table")

    except sqlite3.OperationalError as oe:
         if "no such table: master_schedule" in str(oe).lower():
              logger.error(f"SQL Operational Error: The required 'master_schedule' table does not exist in the database '{DB_PATH}'. Please ensure it is created and populated.")
         elif "no such table: statcast_pitchers" in str(oe).lower():
              logger.error(f"SQL Operational Error: The required 'statcast_pitchers' table does not exist in the database '{DB_PATH}'. Please ensure it is created and populated.")
         else:
              logger.error(f"SQL Operational Error during PITCHER aggregation: {oe}. Check query syntax and table/column names.")
         return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error during PITCHER SQL aggregation: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

    elapsed = time.time() - start_time
    logger.info(f"PITCHER SQL aggregation completed in {elapsed:.2f}s")
    return agg_df


# --- TEAM/BATTER AGGREGATION ---

def aggregate_statcast_batters_sql(start_date=None, end_date=None):
    """
    Aggregates raw Statcast data to team game level using SQL.
    Filters results based on games present in master_schedule.
    """
    logger.info("Starting BATTER/TEAM Statcast aggregation using SQL...")
    start_time = time.time()

    # Define the SQL query for team game-level aggregation (No change needed here)
    # *** is_home calculation assumes inning_topbot exists and is accurate ***
    sql = """
        WITH TeamGamePitches AS (
            SELECT
                game_pk,
                DATE(game_date) AS game_date,
                CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS team,
                CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS opponent,
                home_team,
                away_team,
                description,
                type,
                zone,
                events,
                balls,
                strikes,
                launch_speed,
                launch_angle,
                estimated_woba_using_speedangle AS est_woba,
                woba_value,
                woba_denom,
                babip_value,
                iso_value,
                at_bat_number,
                inning_topbot
            FROM statcast_batters -- Ensure this table exists
        )
        SELECT
            t.game_pk,
            t.game_date,
            t.team,
            t.opponent,
            t.home_team,
            t.away_team,
            MAX(CASE WHEN t.inning_topbot = 'Bot' THEN 1 ELSE 0 END) AS is_home,
            COUNT(DISTINCT CASE WHEN t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END) AS pa,
            SUM(CASE WHEN t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts,
            SUM(CASE WHEN t.events = 'walk' THEN 1 ELSE 0 END) AS walks,
            SUM(CASE WHEN t.events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) AS hits,
            COUNT(*) AS pitches_faced,
            SUM(CASE WHEN t.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS swings,
            SUM(CASE WHEN t.description IN ('foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS contact,
            SUM(CASE WHEN t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS swinging_strikes,
            SUM(CASE WHEN t.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS zone_pitches,
            SUM(CASE WHEN t.zone BETWEEN 11 AND 14 THEN 1 ELSE 0 END) AS chases,
            SUM(CASE WHEN t.zone BETWEEN 1 AND 9 AND t.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS zone_swings,
            SUM(CASE WHEN t.zone BETWEEN 1 AND 9 AND t.description IN ('foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS zone_contact,
            STRFTIME('%Y', t.game_date) AS season,
            CAST(SUM(CASE WHEN t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(DISTINCT CASE WHEN t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END), 0) AS k_percent,
            CAST(SUM(CASE WHEN t.events = 'walk' THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(DISTINCT CASE WHEN t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END), 0) AS bb_percent,
            CAST(SUM(CASE WHEN t.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(*), 0) AS swing_percent,
            CAST(SUM(CASE WHEN t.description IN ('foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS REAL) / NULLIF(SUM(CASE WHEN t.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END), 0) AS contact_percent,
            CAST(SUM(CASE WHEN t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(*), 0) AS swinging_strike_percent,
            CAST(SUM(CASE WHEN t.zone BETWEEN 11 AND 14 AND t.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(SUM(CASE WHEN t.zone BETWEEN 11 AND 14 THEN 1 ELSE 0 END), 0) AS chase_percent,
            CAST(SUM(CASE WHEN t.zone BETWEEN 1 AND 9 AND t.description IN ('foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(SUM(CASE WHEN t.zone BETWEEN 1 AND 9 AND t.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score') THEN 1 ELSE 0 END), 0) AS zone_contact_percent
        FROM TeamGamePitches t
        GROUP BY t.game_pk, t.game_date, t.team, t.opponent, t.home_team, t.away_team
        ORDER BY t.game_date, t.game_pk, t.team
    """

    agg_df = pd.DataFrame()
    try:
        logger.info("Executing SQL query for BATTER/TEAM game-level aggregation...")
        with DBConnection(DB_PATH) as conn:
             # Load identifying columns from master_schedule (CORRECTED TABLE)
             logger.info("Loading game identifiers from master_schedule...")
             schedule_games_df = pd.read_sql_query("SELECT DISTINCT game_date, home_team, away_team FROM master_schedule", conn)
             logger.debug(f"Loaded {len(schedule_games_df)} distinct games from master_schedule.")
             if not schedule_games_df.empty:
                  # Create composite key for schedule data
                  schedule_games_df['game_date'] = pd.to_datetime(schedule_games_df['game_date']).dt.strftime('%Y-%m-%d')
                  schedule_games_df['home_team'] = schedule_games_df['home_team'].str.strip()
                  schedule_games_df['away_team'] = schedule_games_df['away_team'].str.strip()
                  schedule_game_ids = set(schedule_games_df['game_date'] + '_' + schedule_games_df['away_team'] + '_' + schedule_games_df['home_team'])
                  logger.debug(f"Created {len(schedule_game_ids)} unique game identifiers from master_schedule.")
             else:
                  logger.warning("Master schedule table is empty or query failed. Cannot filter based on schedule data.")
                  schedule_game_ids = set()

             # Execute main aggregation query
             agg_df = pd.read_sql_query(sql, conn)
             logger.info(f"SQL aggregation returned {len(agg_df)} TEAM game-level records")

             # Debug date range before filtering
             if not agg_df.empty:
                  agg_df['game_date'] = pd.to_datetime(agg_df['game_date']).dt.strftime('%Y-%m-%d')
                  min_date_agg = agg_df['game_date'].min(); max_date_agg = agg_df['game_date'].max()
                  logger.debug(f"Date range in aggregated TEAM data BEFORE schedule filter: {min_date_agg} to {max_date_agg}")

        # Filter results based on games that have schedule data available
        if not schedule_game_ids:
             logger.warning("Skipping filtering based on schedule data as no schedule game identifiers were loaded.")
        elif not agg_df.empty:
            logger.info("Filtering TEAM results based on available schedule data...")
            original_count = len(agg_df)
            try:
                 # Ensure team columns exist and are strings before stripping/combining
                 if 'home_team' not in agg_df.columns or 'away_team' not in agg_df.columns:
                      logger.error("Missing 'home_team' or 'away_team' in aggregated TEAM data. Cannot filter.")
                 else:
                      agg_df['home_team'] = agg_df['home_team'].astype(str).str.strip()
                      agg_df['away_team'] = agg_df['away_team'].astype(str).str.strip()
                      agg_df['composite_game_id'] = agg_df['game_date'] + '_' + agg_df['away_team'] + '_' + agg_df['home_team']
                      # Keep rows where composite_game_id is in the set from schedule_data
                      agg_df_filtered = agg_df[agg_df['composite_game_id'].isin(schedule_game_ids)].copy()
                      # Use drop on the filtered copy
                      agg_df_filtered = agg_df_filtered.drop(columns=['composite_game_id'])
                      agg_df = agg_df_filtered # Assign back
                      logger.info(f"Filtered TEAM records to {len(agg_df)} with schedule data (from {original_count})")
            except KeyError as ke:
                 logger.error(f"Missing key column for schedule filtering in aggregated data: {ke}. Skipping filter.")
            except Exception as fe:
                 logger.error(f"Error during schedule data filtering: {fe}. Skipping filter.")

            # Debug date range after filtering
            if not agg_df.empty:
                 min_date_filt = agg_df['game_date'].min(); max_date_filt = agg_df['game_date'].max()
                 logger.debug(f"Date range in aggregated TEAM data AFTER schedule filter: {min_date_filt} to {max_date_filt}")
        else:
             logger.info("Aggregated TEAM DataFrame is empty, skipping schedule filtering.")


        # Imputation for missing values
        if not agg_df.empty:
            team_impute_cols = ['contact_percent', 'chase_percent', 'zone_contact_percent'] # Add others as needed
            agg_df = smart_impute(agg_df, 'team', team_impute_cols)
            logger.info(f"TEAM NaN counts after imputation:\n{agg_df.isnull().sum()[agg_df.isnull().sum() > 0]}")
        else:
             logger.info("Skipping imputation as TEAM DataFrame is empty.")


        # Save to database
        logger.info("Saving final aggregated TEAM data to database table 'game_level_team_stats'...")
        with DBConnection(DB_PATH) as conn:
            agg_df.to_sql('game_level_team_stats', conn, if_exists='replace', index=False)
        logger.info(f"Saved {len(agg_df)} records to game_level_team_stats table")

    except sqlite3.OperationalError as oe:
         if "no such table: master_schedule" in str(oe).lower():
              logger.error(f"SQL Operational Error: The required 'master_schedule' table does not exist in the database '{DB_PATH}'. Please ensure it is created and populated.")
         elif "no such table: statcast_batters" in str(oe).lower():
              logger.error(f"SQL Operational Error: The required 'statcast_batters' table does not exist in the database '{DB_PATH}'. Please ensure it is created and populated.")
         else:
              logger.error(f"SQL Operational Error during TEAM aggregation: {oe}. Check query syntax and table/column names.")
         return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error during TEAM SQL aggregation: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

    elapsed = time.time() - start_time
    logger.info(f"BATTER/TEAM SQL aggregation completed in {elapsed:.2f}s")
    return agg_df


# --- Main Execution (Example - usually called from engineer_features) ---
if __name__ == "__main__":
    logger.info("Running aggregate_statcast.py directly (for debugging/testing)...")
    # Example: Run for full history
    logger.info("Running PITCHER aggregation...")
    pitcher_agg = aggregate_statcast_pitchers_sql()
    if pitcher_agg is not None and not pitcher_agg.empty:
         logger.info(f"Pitcher aggregation returned {len(pitcher_agg)} rows.")
         logger.info(f"Pitcher date range: {pitcher_agg['game_date'].min()} to {pitcher_agg['game_date'].max()}")
    elif pitcher_agg is not None:
         logger.info("Pitcher aggregation returned an empty DataFrame.")
    else: # Should not happen with current return logic, but safety check
         logger.error("Pitcher aggregation returned None.")


    logger.info("Running TEAM aggregation...")
    team_agg = aggregate_statcast_batters_sql()
    if team_agg is not None and not team_agg.empty:
        logger.info(f"Team aggregation returned {len(team_agg)} rows.")
        logger.info(f"Team date range: {team_agg['game_date'].min()} to {team_agg['game_date'].max()}")
    elif team_agg is not None:
         logger.info("Team aggregation returned an empty DataFrame.")
    else:
         logger.error("Team aggregation returned None.")

    logger.info("Direct execution finished.")