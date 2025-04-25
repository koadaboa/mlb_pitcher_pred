# src/data/aggregate_statcast.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sqlite3
from pathlib import Path
import logging
import time
import gc
from datetime import datetime

# --- Setup Project Root & Logging ---
try:
    from src.config import DBConfig, LogConfig
    from src.data.utils import setup_logger, DBConnection # Ensure DBConnection is imported
    MODULE_IMPORTS_OK = True
except ImportError as e:
    MODULE_IMPORTS_OK = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('aggregate_statcast_fallback')
    DB_PATH = Path("./data/mlb_data.db")
    class DBConnection: # Basic fallback
        def __init__(self, db_path=None): self.db_path = db_path if db_path else Path("./data/mlb_data.db")
        def __enter__(self): self.conn = sqlite3.connect(self.db_path); return self.conn
        def __exit__(self,et,ev,tb):
            if self.conn:
                if et: self.conn.rollback()
                else: self.conn.commit()
                self.conn.close()
else:
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('aggregate_statcast', LogConfig.LOG_DIR / 'aggregate_statcast.log', level=logging.DEBUG)
    DB_PATH = Path(DBConfig.PATH)

# --- Constants ---
ID_COLS_TO_EXCLUDE_FROM_DOWNCAST = ['pitcher', 'batter', 'game_pk', 'pitcher_id']

# --- Helper Functions --- (optimize_dtypes, smart_impute, knn_impute_complex_features remain unchanged)
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

def smart_impute(df, group_col, target_cols):
    """Imputes missing values using group median first, then global median."""
    logger.info(f"Performing smart imputation for {group_col} missing values...")
    df_copy = df.copy()
    for col in target_cols:
        if col in df_copy.columns and df_copy[col].isnull().any(): # Check col exists
            nan_count_start = df_copy[col].isnull().sum()
            logger.info(f"Imputing {nan_count_start} NaNs in '{col}' using {group_col}/global median...")
            # Calculate group median (ensure numeric type for median)
            numeric_groups = df_copy.dropna(subset=[col])
            if pd.api.types.is_numeric_dtype(numeric_groups[col]):
                 # Use apply with lambda to handle potential empty groups gracefully
                 group_median_map = numeric_groups.groupby(group_col)[col].median()
                 df_copy[col] = df_copy[col].fillna(df_copy[group_col].map(group_median_map))
                 nan_count_after_group = df_copy[col].isnull().sum()
                 logger.debug(f"   NaNs remaining after group median for '{col}': {nan_count_after_group}")

                 # Apply global median for remaining NaNs (groups with all NaNs or new NaNs from mapping)
                 if nan_count_after_group > 0:
                     global_median = df_copy[col].median()
                     if pd.notna(global_median):
                          df_copy[col] = df_copy[col].fillna(global_median)
                          logger.debug(f"   Filled remaining NaNs in '{col}' with global median ({global_median:.3f}).")
                     else: # Handle case where global median might also be NaN (all values were NaN)
                          logger.warning(f"   Global median for '{col}' is NaN. Filling remaining NaNs with 0.")
                          df_copy[col] = df_copy[col].fillna(0)

                 final_nan_count = df_copy[col].isnull().sum()
                 logger.debug(f"   '{col}' imputation complete. Remaining NaNs: {final_nan_count}")
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
                  median_val = impute_df_subset[p_col].median()
                  fill_val = median_val if pd.notna(median_val) else 0 # Fallback to 0 if median is NaN
                  impute_df_subset[p_col] = impute_df_subset[p_col].fillna(fill_val)

        imputer = KNNImputer(n_neighbors=n_neighbors)
        # Operate on the subset for imputation
        imputed_data = imputer.fit_transform(impute_df_subset)
        imputed_df = pd.DataFrame(imputed_data, columns=impute_df_subset.columns, index=impute_df_subset.index)

        # Update original dataframe copy using the results from the imputed subset
        for col in cols_to_impute:
            df_copy[col] = imputed_df[col]

        logger.info(f"KNN imputation complete for columns: {cols_to_impute}.")
    except Exception as e:
        logger.error(f"KNN imputation failed: {e}", exc_info=True)

    return df_copy


# --- PITCHER AGGREGATION ---
def aggregate_statcast_pitchers_sql(target_date=None):
    """
    Aggregates pitcher data, handles DELETE/INSERT manually.
    Includes platoon splits using the 'stand' column.
    """
    # ... (Mode determination and SQL definition remain the same) ...
    if target_date:
        logger.info(f"Starting PITCHER Statcast aggregation (incl. platoon, manual replace) using SQL for DATE: {target_date}...")
        date_filter_sql = f"WHERE DATE(s.game_date) = '{target_date}'"
        where_clause_load = f"WHERE DATE(game_date) = '{target_date}'"
    else:
        logger.info("Starting PITCHER Statcast aggregation (incl. platoon, manual replace) using SQL for ALL DATES...")
        date_filter_sql = ""
        where_clause_load = ""

    start_time = time.time()
    # SQL Definition using s.stand (keep as before)
    sql = f"""
    WITH GamePitcherStats AS (
        SELECT
            s.pitcher AS pitcher_id, s.game_pk, DATE(s.game_date) AS game_date,
            MAX(s.player_name) AS player_name, MAX(s.home_team) AS home_team, MAX(s.away_team) AS away_team,
            MAX(s.p_throws) AS p_throws, MAX(CASE WHEN s.inning_topbot = 'Top' THEN 1 ELSE 0 END) AS is_home,
            SUM(CASE WHEN s.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts,
            COUNT(DISTINCT s.batter) AS batters_faced, COUNT(*) AS total_pitches,
            SUM(CASE WHEN s.type = 'S' AND s.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS total_swinging_strikes,
            SUM(CASE WHEN s.type = 'S' AND s.description = 'called_strike' THEN 1 ELSE 0 END) AS total_called_strikes,
            SUM(CASE WHEN s.stand = 'L' AND s.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts_vs_lhb,
            COUNT(DISTINCT CASE WHEN s.stand = 'L' THEN s.at_bat_number || '_' || s.game_pk ELSE NULL END) AS pa_vs_lhb,
            SUM(CASE WHEN s.stand = 'L' THEN 1 ELSE 0 END) AS pitches_vs_lhb,
            SUM(CASE WHEN s.stand = 'L' AND s.type = 'S' AND s.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS swinging_strikes_vs_lhb,
            SUM(CASE WHEN s.stand = 'R' AND s.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts_vs_rhb,
            COUNT(DISTINCT CASE WHEN s.stand = 'R' THEN s.at_bat_number || '_' || s.game_pk ELSE NULL END) AS pa_vs_rhb,
            SUM(CASE WHEN s.stand = 'R' THEN 1 ELSE 0 END) AS pitches_vs_rhb,
            SUM(CASE WHEN s.stand = 'R' AND s.type = 'S' AND s.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS swinging_strikes_vs_rhb,
            SUM(CASE WHEN s.pitch_type IN ('FF', 'SI', 'FT', 'FC') THEN 1 ELSE 0 END) AS total_fastballs,
            SUM(CASE WHEN s.pitch_type IN ('SL', 'CU', 'KC', 'CS', 'ST', 'SV') THEN 1 ELSE 0 END) AS total_breaking,
            SUM(CASE WHEN s.pitch_type IN ('CH', 'FS', 'KN', 'EP', 'SC') THEN 1 ELSE 0 END) AS total_offspeed,
            SUM(CASE WHEN s.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS total_in_zone,
            AVG(s.release_speed) AS avg_velocity, MAX(s.release_speed) AS max_velocity,
            AVG(s.release_spin_rate) AS avg_spin_rate, AVG(s.pfx_x) AS avg_horizontal_break, AVG(s.pfx_z) AS avg_vertical_break
        FROM statcast_pitchers s {date_filter_sql} GROUP BY s.pitcher, s.game_pk, DATE(s.game_date)
    ), GameInnings AS (
        SELECT pitcher AS pitcher_id, game_pk, COUNT(DISTINCT inning) + MAX(CASE WHEN outs_when_up = 1 THEN 0.1 WHEN outs_when_up = 2 THEN 0.2 ELSE 0 END) as innings_pitched_approx
        FROM statcast_pitchers s {date_filter_sql} GROUP BY pitcher, game_pk
    )
    SELECT gps.*, STRFTIME('%Y', gps.game_date) AS season, gi.innings_pitched_approx AS innings_pitched, CASE WHEN gps.is_home = 1 THEN gps.away_team ELSE gps.home_team END AS opponent_team,
           CAST(gps.strikeouts AS REAL) / NULLIF(gps.pa_vs_lhb + gps.pa_vs_rhb, 0) AS k_percent,
           (CAST(gps.strikeouts AS REAL) * 9.0) / NULLIF(gi.innings_pitched_approx, 0) AS k_per_9,
           CAST(gps.total_swinging_strikes AS REAL) / NULLIF(gps.total_pitches, 0) AS swinging_strike_percent,
           CAST(gps.total_called_strikes AS REAL) / NULLIF(gps.total_pitches, 0) AS called_strike_percent,
           CAST(gps.strikeouts_vs_lhb AS REAL) / NULLIF(gps.pa_vs_lhb, 0) AS k_percent_vs_lhb,
           CAST(gps.swinging_strikes_vs_lhb AS REAL) / NULLIF(gps.pitches_vs_lhb, 0) AS swinging_strike_percent_vs_lhb,
           CAST(gps.strikeouts_vs_rhb AS REAL) / NULLIF(gps.pa_vs_rhb, 0) AS k_percent_vs_rhb,
           CAST(gps.swinging_strikes_vs_rhb AS REAL) / NULLIF(gps.pitches_vs_rhb, 0) AS swinging_strike_percent_vs_rhb,
           CAST(gps.total_fastballs AS REAL) / NULLIF(gps.total_pitches, 0) AS fastball_percent,
           CAST(gps.total_breaking AS REAL) / NULLIF(gps.total_pitches, 0) AS breaking_percent,
           CAST(gps.total_offspeed AS REAL) / NULLIF(gps.total_pitches, 0) AS offspeed_percent,
           CAST(gps.total_in_zone AS REAL) / NULLIF(gps.total_pitches, 0) AS zone_percent
    FROM GamePitcherStats gps LEFT JOIN GameInnings gi ON gps.pitcher_id = gi.pitcher_id AND gps.game_pk = gi.game_pk
    ORDER BY gps.game_date, gps.game_pk, gps.pitcher_id
    """

    agg_df = pd.DataFrame()
    schedule_game_ids = set()
    output_table_name = 'game_level_pitchers'

    try:
        with DBConnection(DB_PATH) as conn:
            # ... (keep column check, schedule loading, main query execution, filtering as before) ...
            logger.info("Executing SQL query for PITCHER game-level aggregation (incl. platoon)...")
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT stand FROM statcast_pitchers LIMIT 1")
                logger.info("'stand' column found in statcast_pitchers.")
            except sqlite3.OperationalError as oe:
                # ... (handle missing 'stand' column) ...
                if "no such column: stand" in str(oe).lower():
                     logger.error("FATAL: Required 'stand' column missing from 'statcast_pitchers' table.")
                     logger.error("Cannot calculate platoon splits. Please ensure 'stand' is included during data ingestion.")
                     return None
                else:
                     logger.error(f"SQL Operational Error checking for 'stand' column: {oe}")
                     raise
            # Load schedule data
            logger.info("Loading game identifiers from historical_umpire_data...")
            schedule_query = f"SELECT DISTINCT game_date, home_team, away_team FROM historical_umpire_data {where_clause_load}"
            try:
                schedule_games_df = pd.read_sql_query(schedule_query, conn)
                if not schedule_games_df.empty:
                    # ... (build schedule_game_ids) ...
                    schedule_games_df['game_date'] = pd.to_datetime(schedule_games_df['game_date']).dt.strftime('%Y-%m-%d')
                    schedule_games_df['home_team'] = schedule_games_df['home_team'].str.strip()
                    schedule_games_df['away_team'] = schedule_games_df['away_team'].str.strip()
                    schedule_game_ids = set(schedule_games_df['game_date'] + '_' + schedule_games_df['away_team'] + '_' + schedule_games_df['home_team'])
                    logger.debug(f"Created {len(schedule_game_ids)} unique game identifiers from historical_umpire_data for filter.")
                else:
                     logger.warning(f"Master schedule data for filter condition ('{where_clause_load}') is empty or query failed.")
            except Exception as e_sched:
                logger.error(f"Error loading schedule data: {e_sched}")
                logger.warning("Proceeding without schedule filter due to error.")
                schedule_game_ids = set()

            # Execute main aggregation query
            try:
                agg_df = pd.read_sql_query(sql, conn)
                logger.info(f"SQL aggregation returned {len(agg_df)} PITCHER game-level records for date '{target_date if target_date else 'ALL'}'")
            except Exception as e_main:
                 logger.error(f"Error during main PITCHER aggregation: {e_main}")
                 return None

            # Filter results
            if not schedule_game_ids:
                 logger.warning("Skipping filtering based on schedule data.")
            elif not agg_df.empty:
                 # ... (filtering logic) ...
                logger.info("Filtering PITCHER results based on available schedule data...")
                original_count = len(agg_df)
                try:
                     agg_df['game_date'] = pd.to_datetime(agg_df['game_date']).dt.strftime('%Y-%m-%d')
                     agg_df['home_team'] = agg_df['home_team'].astype(str).str.strip()
                     agg_df['away_team'] = agg_df['away_team'].astype(str).str.strip()
                     agg_df['composite_game_id'] = agg_df['game_date'] + '_' + agg_df['away_team'] + '_' + agg_df['home_team']
                     agg_df = agg_df[agg_df['composite_game_id'].isin(schedule_game_ids)].drop(columns=['composite_game_id'])
                     logger.info(f"Filtered PITCHER records to {len(agg_df)} (from {original_count})")
                except Exception as fe:
                     logger.error(f"Error during schedule filtering: {fe}.")
                     logger.warning("Proceeding with unfiltered data due to filtering error.")
            else:
                 logger.info("Aggregated PITCHER DataFrame is empty, skipping schedule filtering.")

            # Imputation (logic remains the same)
            if not agg_df.empty:
                # ... (imputation logic as before) ...
                pitcher_basic_impute_cols = ['avg_velocity', 'max_velocity', 'avg_spin_rate', 'avg_horizontal_break', 'avg_vertical_break']
                pitcher_rate_impute_cols = [
                    'k_percent', 'swinging_strike_percent', 'called_strike_percent',
                    'k_percent_vs_lhb', 'swinging_strike_percent_vs_lhb',
                    'k_percent_vs_rhb', 'swinging_strike_percent_vs_rhb',
                    'fastball_percent', 'breaking_percent', 'offspeed_percent', 'zone_percent'
                ]
                logger.info("Imputing basic pitcher physical metrics...")
                agg_df = smart_impute(agg_df, 'pitcher_id', pitcher_basic_impute_cols)
                knn_target_cols_basic = [col for col in pitcher_basic_impute_cols if col in agg_df.columns and agg_df[col].isnull().any()]
                if knn_target_cols_basic:
                    agg_df = knn_impute_complex_features(agg_df, knn_target_cols_basic)
                logger.info("Imputing pitcher rate metrics (including platoon)...")
                for col in pitcher_rate_impute_cols:
                     if col in agg_df.columns and agg_df[col].isnull().any():
                        nan_count = agg_df[col].isnull().sum()
                        logger.debug(f"   Imputing {nan_count} NaNs in '{col}' with 0.0.")
                        agg_df[col] = agg_df[col].fillna(0.0)


            # *** MODIFIED SAVE LOGIC: Manual Drop/Append ***
            if not agg_df.empty:
                cursor = conn.cursor() # Get cursor from the connection
                if target_date: # Incremental update (DELETE specific date, then append)
                    logger.info(f"Deleting existing PITCHER records for date {target_date} from '{output_table_name}'...")
                    delete_sql = f"DELETE FROM {output_table_name} WHERE DATE(game_date) = ?"
                    try:
                        cursor.execute(delete_sql, (target_date,))
                        deleted_count = cursor.rowcount
                        logger.info(f"Deleted {deleted_count} existing PITCHER records for {target_date}.")
                        # Append new data
                        logger.info(f"Appending {len(agg_df)} new PITCHER records for {target_date} to '{output_table_name}'...")
                        agg_df.to_sql(output_table_name, conn, if_exists='append', index=False)
                        logger.info(f"Appended {len(agg_df)} records for {target_date}.")
                    except sqlite3.Error as e_del_app:
                        logger.error(f"Error during DELETE/APPEND for PITCHER {target_date}: {e_del_app}")
                        # Rollback will happen automatically via context manager's __exit__
                        raise # Re-raise the error to prevent incorrect success message

                else: # Full run mode (DROP then append)
                    logger.info(f"Dropping existing PITCHER table '{output_table_name}' (if exists)...")
                    try:
                        # Manually drop the table first
                        cursor.execute(f"DROP TABLE IF EXISTS {output_table_name}")
                        logger.info(f"Table '{output_table_name}' dropped successfully (or did not exist).")
                        # Now append data (which creates the table if dropped)
                        logger.info(f"Writing {len(agg_df)} new PITCHER records to '{output_table_name}'...")
                        agg_df.to_sql(output_table_name, conn, if_exists='append', index=False)
                        logger.info(f"Finished writing records to {output_table_name}.")
                    except sqlite3.Error as e_drop_app:
                        logger.error(f"Error during DROP/APPEND for PITCHER (full run): {e_drop_app}")
                        # Rollback will happen automatically via context manager's __exit__
                        raise # Re-raise the error to prevent incorrect success message
            else:
                logger.info(f"No PITCHER data processed to save for date '{target_date if target_date else 'ALL'}'.")
                # If running for ALL dates and agg_df is empty, should we still drop the table?
                # Let's add logic to drop even if empty on a full run to ensure schema consistency if needed.
                if not target_date:
                    logger.info(f"Dropping existing PITCHER table '{output_table_name}' (if exists) even though no new data was generated...")
                    try:
                        cursor = conn.cursor()
                        cursor.execute(f"DROP TABLE IF EXISTS {output_table_name}")
                        logger.info(f"Table '{output_table_name}' dropped successfully (or did not exist).")
                    except sqlite3.Error as e_drop_empty:
                         logger.error(f"Error during DROP TABLE for PITCHER (empty full run): {e_drop_empty}")
                         raise


        # Context manager handles commit/rollback and close

    except sqlite3.Error as db_e:
        logger.error(f"Database error during PITCHER aggregation for '{target_date if target_date else 'ALL'}': {db_e}", exc_info=False) # Keep traceback clean
        return None
    except Exception as e:
        logger.error(f"Unexpected error during PITCHER aggregation for '{target_date if target_date else 'ALL'}': {e}", exc_info=True)
        return None

    elapsed = time.time() - start_time
    logger.info(f"PITCHER SQL aggregation (incl. platoon, manual replace) for '{target_date if target_date else 'ALL'}' completed in {elapsed:.2f}s")
    return agg_df


# --- TEAM/BATTER AGGREGATION (MODIFIED FOR VS PITCHER HAND) ---
def aggregate_statcast_batters_sql(target_date=None):
    """
    Aggregates raw Statcast data to team game level using SQL.
    Includes stats broken down by opposing pitcher hand (LHP/RHP).
    Uses manual drop/append for saving.
    **Requires 'p_throws' column in 'statcast_batters'.**
    """
    # Determine mode based on target_date
    if target_date:
        logger.info(f"Starting TEAM Statcast aggregation (vs P-Hand, manual replace) using SQL for DATE: {target_date}...")
        date_filter_sql = f"WHERE DATE(s.game_date) = '{target_date}'" # Alias as 's'
        where_clause_load = f"WHERE DATE(game_date) = '{target_date}'"
    else:
        logger.info("Starting TEAM Statcast aggregation (vs P-Hand, manual replace) using SQL for ALL DATES...")
        date_filter_sql = ""
        where_clause_load = ""

    start_time = time.time()

    # Define the SQL query for team game-level aggregation
    # *** MODIFIED: Added aggregations conditional on p_throws ***
    # *** ASSUMPTION: 'statcast_batters' table MUST have 'p_throws' ('L' or 'R') ***
    sql = f"""
    WITH TeamGamePitches AS (
        SELECT
            s.game_pk, DATE(s.game_date) AS game_date,
            CASE WHEN s.inning_topbot = 'Top' THEN s.away_team ELSE s.home_team END AS team,
            CASE WHEN s.inning_topbot = 'Top' THEN s.home_team ELSE s.away_team END AS opponent,
            s.home_team, s.away_team, s.description, s.type, s.zone, s.events, s.balls, s.strikes,
            s.launch_speed, s.launch_angle, s.estimated_woba_using_speedangle AS est_woba,
            s.woba_value, s.woba_denom, s.babip_value, s.iso_value, s.at_bat_number, s.inning_topbot,
            s.p_throws -- Need pitcher hand for conditional aggregation
        FROM statcast_batters s -- Alias as 's'
        {date_filter_sql} -- Apply date filter here
    )
    SELECT
        t.game_pk, t.game_date, t.team, t.opponent, t.home_team, t.away_team,
        MAX(CASE WHEN t.inning_topbot = 'Bot' THEN 1 ELSE 0 END) AS is_home,
        STRFTIME('%Y', t.game_date) AS season,

        -- Overall Aggregations
        COUNT(DISTINCT CASE WHEN t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END) AS pa,
        SUM(CASE WHEN t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts,
        COUNT(*) AS pitches_faced,
        SUM(CASE WHEN t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS swinging_strikes,

        -- Aggregations vs LHP
        COUNT(DISTINCT CASE WHEN t.p_throws = 'L' AND t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END) AS pa_vs_LHP,
        SUM(CASE WHEN t.p_throws = 'L' AND t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts_vs_LHP,
        SUM(CASE WHEN t.p_throws = 'L' THEN 1 ELSE 0 END) AS pitches_faced_vs_LHP,
        SUM(CASE WHEN t.p_throws = 'L' AND t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS swinging_strikes_vs_LHP,

        -- Aggregations vs RHP
        COUNT(DISTINCT CASE WHEN t.p_throws = 'R' AND t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END) AS pa_vs_RHP,
        SUM(CASE WHEN t.p_throws = 'R' AND t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS strikeouts_vs_RHP,
        SUM(CASE WHEN t.p_throws = 'R' THEN 1 ELSE 0 END) AS pitches_faced_vs_RHP,
        SUM(CASE WHEN t.p_throws = 'R' AND t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS swinging_strikes_vs_RHP,

        -- Calculate Overall Rates
        CAST(SUM(CASE WHEN t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(COUNT(DISTINCT CASE WHEN t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END), 0) AS k_percent,
        CAST(SUM(CASE WHEN t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(COUNT(*), 0) AS swinging_strike_percent,

        -- Calculate Rates vs LHP
        CAST(SUM(CASE WHEN t.p_throws = 'L' AND t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(COUNT(DISTINCT CASE WHEN t.p_throws = 'L' AND t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END), 0) AS k_percent_vs_LHP,
        CAST(SUM(CASE WHEN t.p_throws = 'L' AND t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(SUM(CASE WHEN t.p_throws = 'L' THEN 1 ELSE 0 END), 0) AS swinging_strike_percent_vs_LHP,

        -- Calculate Rates vs RHP
        CAST(SUM(CASE WHEN t.p_throws = 'R' AND t.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(COUNT(DISTINCT CASE WHEN t.p_throws = 'R' AND t.events IS NOT NULL THEN t.at_bat_number || '_' || t.game_pk ELSE NULL END), 0) AS k_percent_vs_RHP,
        CAST(SUM(CASE WHEN t.p_throws = 'R' AND t.type = 'S' AND t.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(SUM(CASE WHEN t.p_throws = 'R' THEN 1 ELSE 0 END), 0) AS swinging_strike_percent_vs_RHP

        -- Add other metrics as needed (e.g., walks, contact%, chase%) broken down similarly

    FROM TeamGamePitches t
    GROUP BY t.game_pk, t.game_date, t.team, t.opponent, t.home_team, t.away_team
    ORDER BY t.game_date, t.game_pk, t.team
    """

    agg_df = pd.DataFrame()
    schedule_game_ids = set()
    output_table_name = 'game_level_team_stats' # Keep same table name

    try:
        with DBConnection(DB_PATH) as conn:
            logger.info("Executing SQL query for TEAM game-level aggregation (vs P-Hand)...")

            # Check if 'p_throws' column exists in statcast_batters
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT p_throws FROM statcast_batters LIMIT 1")
                logger.info("'p_throws' column found in statcast_batters.")
            except sqlite3.OperationalError as oe:
                if "no such column: p_throws" in str(oe).lower():
                     logger.error("FATAL: Required 'p_throws' column missing from 'statcast_batters' table.")
                     logger.error("Cannot calculate opponent stats vs pitcher hand. Please ensure 'p_throws' is included during data ingestion.")
                     return None # Exit function if required column is missing
                else:
                     logger.error(f"SQL Operational Error checking for 'p_throws' column: {oe}")
                     raise # Re-raise other operational errors

            # ... (keep schedule loading, main query execution, filtering as before) ...
            # Load schedule data
            logger.info("Loading game identifiers from historical_umpire_data...")
            schedule_query = f"SELECT DISTINCT game_date, home_team, away_team FROM historical_umpire_data {where_clause_load}"
            try:
                schedule_games_df = pd.read_sql_query(schedule_query, conn)
                if not schedule_games_df.empty:
                    schedule_games_df['game_date'] = pd.to_datetime(schedule_games_df['game_date']).dt.strftime('%Y-%m-%d')
                    schedule_games_df['home_team'] = schedule_games_df['home_team'].str.strip()
                    schedule_games_df['away_team'] = schedule_games_df['away_team'].str.strip()
                    schedule_game_ids = set(schedule_games_df['game_date'] + '_' + schedule_games_df['away_team'] + '_' + schedule_games_df['home_team'])
                    logger.debug(f"Created {len(schedule_game_ids)} unique game identifiers from historical_umpire_data for filter.")
                else:
                     logger.warning(f"Master schedule data for filter condition ('{where_clause_load}') is empty or query failed.")
            except Exception as e_sched:
                logger.error(f"Error loading schedule data: {e_sched}")
                logger.warning("Proceeding without schedule filter due to error.")
                schedule_game_ids = set()

            # Execute main aggregation query
            try:
                agg_df = pd.read_sql_query(sql, conn)
                logger.info(f"SQL aggregation returned {len(agg_df)} TEAM game-level records for date '{target_date if target_date else 'ALL'}'")
            except Exception as e_main:
                 logger.error(f"Error during main TEAM aggregation: {e_main}")
                 return None

             # Filter results
            if not schedule_game_ids:
                 logger.warning("Skipping filtering based on schedule data.")
            elif not agg_df.empty:
                logger.info("Filtering TEAM results based on available schedule data...")
                original_count = len(agg_df)
                try:
                     agg_df['game_date'] = pd.to_datetime(agg_df['game_date']).dt.strftime('%Y-%m-%d')
                     agg_df['home_team'] = agg_df['home_team'].astype(str).str.strip()
                     agg_df['away_team'] = agg_df['away_team'].astype(str).str.strip()
                     agg_df['composite_game_id'] = agg_df['game_date'] + '_' + agg_df['away_team'] + '_' + agg_df['home_team']
                     agg_df = agg_df[agg_df['composite_game_id'].isin(schedule_game_ids)].drop(columns=['composite_game_id'])
                     logger.info(f"Filtered TEAM records to {len(agg_df)} (from {original_count})")
                except Exception as fe:
                     logger.error(f"Error during schedule filtering: {fe}.")
                     logger.warning("Proceeding with unfiltered data due to filtering error.")
            else:
                 logger.info("Aggregated TEAM DataFrame is empty, skipping schedule filtering.")


            # Imputation (add new columns to imputation list, fill NaNs with 0.0)
            if not agg_df.empty:
                team_rate_impute_cols = [
                    'k_percent', 'swinging_strike_percent',
                    'k_percent_vs_LHP', 'swinging_strike_percent_vs_LHP',
                    'k_percent_vs_RHP', 'swinging_strike_percent_vs_RHP',
                    # Add others like 'bb_percent', 'contact_percent', etc. if calculated
                ]
                logger.info("Imputing team rate metrics (including vs P-Hand)...")
                # Optional: Could try smart_impute first, but 0.0 fill is often safe for rates
                # agg_df = smart_impute(agg_df, 'team', team_rate_impute_cols)
                for col in team_rate_impute_cols:
                     if col in agg_df.columns and agg_df[col].isnull().any():
                        nan_count = agg_df[col].isnull().sum()
                        logger.debug(f"   Imputing {nan_count} NaNs in '{col}' with 0.0.")
                        agg_df[col] = agg_df[col].fillna(0.0)

            # --- Manual Drop/Append Save Logic ---
            if not agg_df.empty:
                cursor = conn.cursor()
                if target_date: # Incremental update
                    logger.info(f"Deleting existing TEAM records for date {target_date} from '{output_table_name}'...")
                    delete_sql = f"DELETE FROM {output_table_name} WHERE DATE(game_date) = ?"
                    try:
                        cursor.execute(delete_sql, (target_date,))
                        logger.info(f"Deleted {cursor.rowcount} existing TEAM records for {target_date}.")
                        logger.info(f"Appending {len(agg_df)} new TEAM records for {target_date} to '{output_table_name}'...")
                        agg_df.to_sql(output_table_name, conn, if_exists='append', index=False)
                        logger.info(f"Appended {len(agg_df)} records for {target_date}.")
                    except sqlite3.Error as e_del_app:
                        logger.error(f"Error during DELETE/APPEND for TEAM {target_date}: {e_del_app}")
                        raise
                else: # Full run mode
                    logger.info(f"Dropping existing TEAM table '{output_table_name}' (if exists)...")
                    try:
                        cursor.execute(f"DROP TABLE IF EXISTS {output_table_name}")
                        logger.info(f"Table '{output_table_name}' dropped successfully (or did not exist).")
                        logger.info(f"Writing {len(agg_df)} new TEAM records to '{output_table_name}'...")
                        agg_df.to_sql(output_table_name, conn, if_exists='append', index=False)
                        logger.info(f"Finished writing records to {output_table_name}.")
                    except sqlite3.Error as e_drop_app:
                        logger.error(f"Error during DROP/APPEND for TEAM (full run): {e_drop_app}")
                        raise
            else:
                logger.info(f"No TEAM data processed to save for date '{target_date if target_date else 'ALL'}'.")
                if not target_date:
                     logger.info(f"Dropping existing TEAM table '{output_table_name}' (if exists) even though no new data was generated...")
                     try:
                         cursor = conn.cursor()
                         cursor.execute(f"DROP TABLE IF EXISTS {output_table_name}")
                         logger.info(f"Table '{output_table_name}' dropped successfully (or did not exist).")
                     except sqlite3.Error as e_drop_empty:
                          logger.error(f"Error during DROP TABLE for TEAM (empty full run): {e_drop_empty}")
                          raise

    except sqlite3.Error as db_e:
        logger.error(f"Database error during TEAM aggregation for '{target_date if target_date else 'ALL'}': {db_e}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during TEAM aggregation for '{target_date if target_date else 'ALL'}': {e}", exc_info=True)
        return None

    elapsed = time.time() - start_time
    logger.info(f"TEAM SQL aggregation (vs P-Hand, manual replace) for '{target_date if target_date else 'ALL'}' completed in {elapsed:.2f}s")
    return agg_df


# --- Main Execution Block (Keep as is for testing) ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK:
        print("ERROR: Failed to import necessary modules.")
        print("Ensure 'src.config' and 'src.data.utils' are accessible from the script's location.")
        print("Attempting to proceed with fallback settings...")
    else:
        print("Module imports successful.")

    logger.info("Running aggregate_statcast.py directly (for debugging/testing)...")
    test_date = None # For full run

    logger.info(f"Running PITCHER aggregation (incl. platoon, manual replace) for {test_date if test_date else 'ALL DATES'}...")
    pitcher_agg_result = aggregate_statcast_pitchers_sql(target_date=test_date)
    if pitcher_agg_result is not None:
        logger.info(f"Pitcher aggregation completed successfully. Returned {len(pitcher_agg_result)} rows.")
    else:
        logger.error("Pitcher aggregation failed.")

    logger.info(f"Running TEAM aggregation (manual replace) for {test_date if test_date else 'ALL DATES'}...")
    team_agg_result = aggregate_statcast_batters_sql(target_date=test_date)
    if team_agg_result is not None:
        logger.info(f"Team aggregation completed successfully. Returned {len(team_agg_result)} rows.")
    else:
        logger.error("Team aggregation failed.")

    logger.info("Direct execution finished.")