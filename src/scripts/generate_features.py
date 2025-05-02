# src/scripts/generate_features.py
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
import gc
import warnings
from itertools import product # For looping through windows/metrics

# --- Setup Project Root ---
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config import DBConfig, LogConfig, StrikeoutModelConfig # Import config
    # Ensure correct DBConnection path
    from src.data.utils import setup_logger, DBConnection
    # Import the aggregation functions needed for Step 0
    from src.data.aggregate_statcast import (
        aggregate_statcast_pitchers_sql,
        aggregate_statcast_batters_sql,
        aggregate_game_level_data
    )
    # Import feature modules
    from src.features.pitcher_features import calculate_pitcher_rolling_features, calculate_pitcher_rest_days
    from src.features.opponent_features import (
        calculate_opponent_rolling_features,
        merge_opponent_features_historical,
        merge_opponent_features_prediction
    )
    from src.features.ballpark_features import (
        calculate_ballpark_rolling_features,
        merge_ballpark_features_historical,
        merge_ballpark_features_prediction
    )
    from src.features.umpire_features import calculate_umpire_rolling_features
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    MODULE_IMPORTS_OK = False
    # Fallback logger if setup fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('generate_features_fallback')
else:
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('generate_features', LogConfig.LOG_DIR / 'generate_features.log')


# --- Configuration ---
PITCHER_METRICS_FOR_ROLLING = [
    'k_percent', 'swinging_strike_percent', 'avg_velocity',
    # 'k_percent_vs_lhb', 'swinging_strike_percent_vs_lhb', # These might not exist in starter stats
    # 'k_percent_vs_rhb', 'swinging_strike_percent_vs_rhb', # Need to confirm columns in game_level_starter_stats
    'fastball_percent', 'breaking_percent', 'offspeed_percent',
    'bb_percent', 'woba', 'iso', 'babip' # Added more potentially available stats
]
OPPONENT_METRICS_FOR_ROLLING = [
    # These come from game_level_team_stats (batting side)
    'k_percent_bat', 'swinging_strike_percent', # Assuming SwStr% is opponent pitching stat
    'woba_bat', 'iso_bat', 'babip_bat', 'hard_hit_percent', 'barrel_percent'
    # Add _vs_LHP/_vs_RHP if available in game_level_team_stats
]
BALLPARK_METRICS_FOR_ROLLING = ['k_percent'] # Based on pitcher (starter) k_percent
UMPIRE_METRICS_FOR_ROLLING = ['k_percent'] # Based on pitcher (starter) k_percent

LEAGUE_AVERAGE_METRICS = ['k_percent', 'bb_percent', 'woba'] # Calculate based on starters

ROLLING_WINDOWS = StrikeoutModelConfig.WINDOW_SIZES
MIN_ROLLING_PERIODS = 2

# --- Helper Functions (load_data_from_db, calculate_multi_window_rolling) ---
# No changes needed in helper functions
def load_data_from_db(query: str, db_path: Path, optimize: bool = True) -> pd.DataFrame:
    """Loads data from the database using a given query."""
    logger.info(f"Executing query: {query[:100]}...")
    start_time = datetime.now()
    df = pd.DataFrame() # Initialize empty df
    try:
        # Use DBConnection context manager from utils.py
        # *** Corrected: No argument for DBConnection ***
        with DBConnection() as conn:
            df = pd.read_sql_query(query, conn)
        duration = datetime.now() - start_time
        logger.info(f"Loaded {len(df)} rows in {duration.total_seconds():.2f}s.")
        if optimize and not df.empty:
            # Optimize memory usage - apply carefully
            for col in df.select_dtypes(include=['int64']).columns: df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64']).columns: df[col] = pd.to_numeric(df[col], downcast='float')
    except Exception as e:
        logger.error(f"Failed to load data with query: {query[:100]}... Error: {e}", exc_info=True)
    return df

def calculate_multi_window_rolling(df, group_col, date_col, metrics, windows, min_periods, shift_periods=1):
    """
    Calculates rolling features for multiple windows efficiently.
    Uses shift() to prevent data leakage. Returns results indexed like input df.
    """
    if df is None or df.empty or not metrics or not windows:
        logger.warning(f"Input invalid for multi-window rolling on {group_col}.")
        return pd.DataFrame(index=df.index if df is not None else None)

    logger.info(f"Calculating multi-window {windows} rolling for group '{group_col}' on {len(metrics)} metrics...")
    df_internal = df.copy() # Work on copy

    # Ensure date_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_internal[date_col]):
         try: df_internal[date_col] = pd.to_datetime(df_internal[date_col])
         except Exception as e: logger.error(f"Failed to convert {date_col} to datetime: {e}"); return pd.DataFrame(index=df.index)

    sort_cols = [group_col, date_col]
    df_sorted = df_internal.sort_values(by=sort_cols, na_position='first')
    # Use observed=True for performance with Categorical group_col if applicable
    grouped = df_sorted.groupby(group_col, observed=True, dropna=False)
    results_dict = {}

    for metric in metrics:
        if metric not in df_sorted.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe for rolling calculation. Skipping.")
            continue
        # Ensure metric is numeric before rolling
        metric_series = pd.to_numeric(df_sorted[metric], errors='coerce')
        if metric_series.isnull().all():
            logger.warning(f"Metric '{metric}' is entirely non-numeric or NaN after coercion. Skipping rolling.")
            continue
        # Assign coerced series back for rolling (important!)
        df_sorted[metric] = metric_series


        for window in windows:
            roll_col_name = f"{metric}_roll{window}g"
            # Ensure min_periods is valid
            current_min_periods = max(1, min(min_periods, window)) # Allow min_periods=window
            try:
                # Use transform for alignment, apply rolling within lambda
                rolling_result = grouped[metric].transform(
                    lambda x: x.shift(shift_periods)
                               .rolling(window=window, min_periods=current_min_periods)
                               .mean()
                )
                results_dict[roll_col_name] = rolling_result
            except Exception as e:
                 logger.error(f"Error calculating rolling for {metric} window {window}: {e}", exc_info=True)
                 results_dict[roll_col_name] = pd.Series(np.nan, index=df_sorted.index)

    if not results_dict: return pd.DataFrame(index=df.index)
    results_df = pd.DataFrame(results_dict, index=df_sorted.index)
    return results_df.reindex(df.index) # Reindex back to original


# --- Main Feature Generation Logic ---
def generate_features(prediction_date_str: str | None,
                        train_years: list[int] | None = None,
                        test_years: list[int] | None = None):
    """
    Generates features using modular functions for pitcher, opponent, ballpark, and umpire stats.
    Includes daily league average features.
    MODIFIED TO USE STARTER STATS AS BASE.
    """
    mode = "PREDICTION" if prediction_date_str else "HISTORICAL"
    logger.info(f"--- Starting Feature Generation [{mode} Mode] (Using Starters) ---")
    # Use DBConfig.PATH directly from config module
    db_path = DBConfig.PATH
    prediction_date = None
    max_hist_date_str = '9999-12-31' # Default for historical
    if prediction_date_str:
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
            max_hist_date_str = (prediction_date - timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"Prediction Date: {prediction_date_str}. Loading history up to {max_hist_date_str}.")
        except ValueError: logger.error(f"Invalid prediction date format: {prediction_date_str}. Use YYYY-MM-DD."); return
    else:
        logger.info("Running for all historical data.")

    # --- STEP 0: Run Aggregations ---
    # Aggregations must complete successfully before loading data
    logger.info("STEP 0: Running Statcast Aggregations (Full History)...")
    try:
        aggregate_statcast_pitchers_sql(target_date=None)
        gc.collect()
        aggregate_statcast_batters_sql(target_date=None)
        gc.collect()
        # --- Call the team/starter aggregation ---
        aggregate_game_level_data()
        # -----------------------------------------
        gc.collect()
        logger.info("Aggregations completed.")
    except NameError as ne: logger.error(f"Aggregation function not found (check imports?): {ne}", exc_info=True); return
    except Exception as agg_e: logger.error(f"Error during aggregation: {agg_e}", exc_info=True); return
    gc.collect()

    # --- STEP 1: Load Data ---
    logger.info("STEP 1: Loading necessary data (Base: Starters)...")
    start_load_time = datetime.now()
    db_connection_successful = False
    try:
        with DBConnection() as conn: db_connection_successful = True # Test connection
    except Exception as db_err: logger.error(f"Failed initial DB connection test: {db_err}"); return

    if not db_connection_successful: return # Stop if initial connection fails

    # Load Team Mapping
    team_map_query = "SELECT team_abbr, ballpark FROM team_mapping"
    team_map_df = load_data_from_db(team_map_query, db_path, optimize=False)
    team_to_ballpark_map = team_map_df.set_index('team_abbr')['ballpark'].to_dict() if not team_map_df.empty else {}
    if not team_to_ballpark_map: logger.warning("Team mapping empty.")

    # --- Load Pitcher History FROM STARTERS ---
    pitcher_base_table = 'game_level_starter_stats' # <<< CHANGED TABLE NAME
    logger.info(f"Loading base pitcher data from: {pitcher_base_table}")
    try:
        with DBConnection() as conn: starter_cols_avail = pd.read_sql_query(f"SELECT * FROM {pitcher_base_table} LIMIT 1", conn).columns.tolist()
    except Exception as e: logger.error(f"Cannot read columns from {pitcher_base_table}: {e}"); return

    # Define columns needed from starter table
    pitcher_metrics_needed = set(PITCHER_METRICS_FOR_ROLLING) | set(BALLPARK_METRICS_FOR_ROLLING) | set(UMPIRE_METRICS_FOR_ROLLING) | set(LEAGUE_AVERAGE_METRICS)
    pitcher_base_cols = ['pitcher_id', 'game_date', 'game_pk', 'p_throws', 'team', 'opponent_team', 'home_team', 'away_team']
    # Add target variables if they exist in starter table
    target_cols = ['strikeouts', 'batters_faced']
    pitcher_cols_to_load = list(set(pitcher_base_cols) | pitcher_metrics_needed | set(target_cols))
    pitcher_cols_to_load_str = ', '.join([f'"{c}"' for c in pitcher_cols_to_load if c in starter_cols_avail])
    if 'pitcher_id' not in pitcher_cols_to_load_str: logger.error(f"'pitcher_id' missing from available columns in {pitcher_base_table}"); return
    if not pitcher_cols_to_load_str: logger.error(f"No pitcher columns could be identified for loading from {pitcher_base_table}."); return

    pitcher_hist_query = f"SELECT {pitcher_cols_to_load_str} FROM {pitcher_base_table} WHERE DATE(game_date) <= '{max_hist_date_str}'"
    pitcher_hist_df = load_data_from_db(pitcher_hist_query, db_path)
    # <<< END PITCHER HISTORY LOAD (FROM STARTERS) ---

    # --- Load Team History (No Change Here) ---
    team_stats_table = 'game_level_team_stats'
    try:
        with DBConnection() as conn: team_cols_avail = pd.read_sql_query(f"SELECT * FROM {team_stats_table} LIMIT 1", conn).columns.tolist()
    except Exception as e: logger.warning(f"Cannot read columns from {team_stats_table}: {e}"); team_cols_avail = []

    # Update opponent metrics based on what's likely in game_level_team_stats
    opponent_metrics_likely = ['k_percent_bat', 'swinging_strike_percent', 'woba_bat', 'iso_bat', 'babip_bat', 'hard_hit_percent', 'barrel_percent', 'bb_percent_bat', 'hr_per_pa']
    opponent_metrics_to_use = [m for m in opponent_metrics_likely if m in team_cols_avail]
    if not opponent_metrics_to_use: logger.warning(f"No opponent batting metrics found in {team_stats_table}")

    team_base_cols = ['team', 'game_date', 'game_pk', 'home_team']
    team_cols_to_load = list(set(team_base_cols) | set(opponent_metrics_to_use))
    team_cols_to_load_str = ', '.join([f'"{c}"' for c in team_cols_to_load if c in team_cols_avail])
    team_hist_df = load_data_from_db(f"SELECT {team_cols_to_load_str} FROM {team_stats_table} WHERE DATE(game_date) <= '{max_hist_date_str}'", db_path) if team_cols_to_load_str else pd.DataFrame()
    # <<< END TEAM HISTORY LOAD ---

    # --- Load Historical Umpire Data (No Change Here) ---
    umpire_table = 'historical_umpire_data'
    umpire_cols_to_load = ['game_date', 'home_plate_umpire', 'home_team', 'away_team', 'game_pk']
    umpire_cols_to_load_str = ', '.join([f'"{c}"' for c in umpire_cols_to_load])
    umpire_hist_query = f"SELECT {umpire_cols_to_load_str} FROM {umpire_table} WHERE DATE(game_date) <= '{max_hist_date_str}'"
    umpire_hist_df = load_data_from_db(umpire_hist_query, db_path, optimize=False)
    if umpire_hist_df.empty: logger.warning(f"Historical umpire data table '{umpire_table}' is empty.")
    # <<< END UMPIRE HISTORY LOAD ---

    # --- Data Validation and Prep ---
    if pitcher_hist_df.empty: logger.error(f"Starter pitcher history empty (loaded from {pitcher_base_table}). Cannot proceed."); return
    if team_hist_df.empty: logger.warning(f"Team history empty (loaded from {team_stats_table}). Opponent features will be limited.")

    # Convert dates AFTER loading all data
    logger.info("Converting date columns...")
    pitcher_hist_df['game_date'] = pd.to_datetime(pitcher_hist_df['game_date'])
    if not team_hist_df.empty: team_hist_df['game_date'] = pd.to_datetime(team_hist_df['game_date'])
    if not umpire_hist_df.empty: umpire_hist_df['game_date'] = pd.to_datetime(umpire_hist_df['game_date'])

    # Add 'is_home' derived column to pitcher_hist_df (STARTER BASED)
    if 'team' in pitcher_hist_df.columns and 'home_team' in pitcher_hist_df.columns:
        pitcher_hist_df['is_home'] = (pitcher_hist_df['team'] == pitcher_hist_df['home_team']).astype(int)
        logger.info("Derived 'is_home' column for starters.")
    else:
        logger.error("'team' or 'home_team' column missing from starter history. Cannot derive 'is_home'.")
        pitcher_hist_df['is_home'] = np.nan # Add as NaN if columns missing

    # Add 'ballpark' derived column
    if 'home_team' in pitcher_hist_df.columns:
        pitcher_hist_df['ballpark'] = pitcher_hist_df['home_team'].map(team_to_ballpark_map).fillna("Unknown Park")
        logger.info("Derived 'ballpark' column for starters.")
    else:
        logger.warning("Missing 'home_team' in starter history, cannot map ballparks accurately.")
        pitcher_hist_df['ballpark'] = "Unknown Park"

    # Ensure necessary columns for features exist before proceeding
    required_for_features = ['pitcher_id', 'game_date', 'team', 'opponent_team', 'ballpark', 'home_team', 'away_team']
    missing_req_cols = [c for c in required_for_features if c not in pitcher_hist_df.columns]
    if missing_req_cols:
        logger.error(f"Starter history DF is missing essential columns: {missing_req_cols}. Aborting feature calculation.")
        return

    # --- Calculate and Merge Daily League Averages (Now based on Starters) ---
    logger.info("Calculating daily league averages (based on starters)...")
    league_avg_cols = {}
    if not pitcher_hist_df.empty and LEAGUE_AVERAGE_METRICS:
        metrics_for_league_avg = [m for m in LEAGUE_AVERAGE_METRICS if m in pitcher_hist_df.columns]
        if not metrics_for_league_avg:
             logger.warning("None of the specified LEAGUE_AVERAGE_METRICS found in starter data.")
        else:
             # Ensure metrics are numeric
             for metric in metrics_for_league_avg:
                 pitcher_hist_df.loc[:, metric] = pd.to_numeric(pitcher_hist_df[metric], errors='coerce')

             daily_league_stats = pitcher_hist_df.groupby('game_date')[metrics_for_league_avg].agg(['mean', 'std'])
             daily_league_stats.columns = ['_'.join(col).strip() + '_league_daily' for col in daily_league_stats.columns.values]
             daily_league_stats = daily_league_stats.reset_index()

             original_index = pitcher_hist_df.index
             pitcher_hist_df = pd.merge(pitcher_hist_df.reset_index(), daily_league_stats, on='game_date', how='left').set_index('index')
             pitcher_hist_df.index.name = None
             league_avg_cols = {col: col for col in daily_league_stats.columns if col != 'game_date'}
             logger.info(f"Added daily league average columns (starter-based): {list(league_avg_cols.keys())}")

             # Simple imputation
             for col in league_avg_cols:
                 if pitcher_hist_df[col].isnull().any():
                      fill_value = 0.0 if 'std' in col else pitcher_hist_df[col].mean()
                      logger.warning(f"Imputing {pitcher_hist_df[col].isnull().sum()} NaNs in league avg column '{col}' with {fill_value:.4f}")
                      pitcher_hist_df.loc[:, col] = pitcher_hist_df[col].fillna(fill_value)
    else: logger.warning("Starter history empty or no league average metrics defined.")
    # --- End League Average Calculation ---

    logger.info(f"Data loading & prep finished in {(datetime.now() - start_load_time).total_seconds():.2f}s.")
    gc.collect()

    # --- STEP 2 & 3: Calculate Historical Rolling Features ---
    logger.info(f"STEP 2&3: Calculating historical rolling features (Windows: {ROLLING_WINDOWS}) (Base: Starters)...")
    calc_start_time = datetime.now()
    all_rolling_features = {}
    all_rename_maps = {}

    # Pitcher Rolling Features (Now based on Starters only)
    # Ensure metrics exist in the loaded starter data
    pitcher_metrics_avail = [m for m in PITCHER_METRICS_FOR_ROLLING if m in pitcher_hist_df.columns]
    if not pitcher_metrics_avail: logger.warning("No pitcher metrics found in starter data for rolling calculations.")
    pitcher_rolling_df = calculate_pitcher_rolling_features(
        df=pitcher_hist_df, group_col='pitcher_id', date_col='game_date',
        metrics=pitcher_metrics_avail, # Use only available metrics
        windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS,
        calculate_multi_window_rolling=calculate_multi_window_rolling
    )
    all_rolling_features['pitcher'] = pitcher_rolling_df
    all_rename_maps['pitcher'] = {col: col for col in pitcher_rolling_df.columns} # Keep original names for now

    # Pitcher Days Rest (Based on Starters only)
    pitcher_hist_df['p_days_rest'] = calculate_pitcher_rest_days(pitcher_hist_df)

    # Opponent/Team Rolling Features (Still based on game_level_team_stats)
    if not team_hist_df.empty:
        opponent_rolling_df, opp_rename_map = calculate_opponent_rolling_features(
            team_hist_df=team_hist_df, group_col='team', date_col='game_date',
            metrics=opponent_metrics_to_use, # Use metrics found in team_stats table
            windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS,
            calculate_multi_window_rolling=calculate_multi_window_rolling
        )
        all_rolling_features['team'] = opponent_rolling_df
        all_rename_maps['opponent'] = opp_rename_map
    else: logger.warning("Skipping opponent rolling features calculation as team history is empty.")


    # Ballpark Rolling Features (Now based on Starters only)
    ballpark_metrics_avail = [m for m in BALLPARK_METRICS_FOR_ROLLING if m in pitcher_hist_df.columns]
    if not ballpark_metrics_avail: logger.warning("No ballpark metrics found in starter data for rolling calculations.")
    ballpark_rolling_df, bpark_rename_map = calculate_ballpark_rolling_features(
        pitcher_hist_df=pitcher_hist_df, # Pass starter data
        group_col='ballpark', date_col='game_date',
        metrics=ballpark_metrics_avail, # Use available metrics
        windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS,
        calculate_multi_window_rolling=calculate_multi_window_rolling
    )
    all_rolling_features['ballpark'] = ballpark_rolling_df
    all_rename_maps['ballpark'] = bpark_rename_map

    # Umpire Rolling Features (Now based on Starters only)
    umpire_metrics_avail = [m for m in UMPIRE_METRICS_FOR_ROLLING if m in pitcher_hist_df.columns]
    if not umpire_metrics_avail: logger.warning("No umpire metrics found in starter data for rolling calculations.")
    umpire_rolling_df, ump_rename_map = calculate_umpire_rolling_features(
        pitcher_hist_df=pitcher_hist_df, # Pass starter data
        umpire_hist_df=umpire_hist_df,
        group_col='home_plate_umpire', date_col='game_date',
        metrics=umpire_metrics_avail, # Use available metrics
        windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS,
        calculate_multi_window_rolling=calculate_multi_window_rolling
    )
    all_rolling_features['umpire'] = umpire_rolling_df
    all_rename_maps['umpire'] = ump_rename_map


    logger.info(f"Feature calculation finished in {(datetime.now() - calc_start_time).total_seconds():.2f}s.")
    gc.collect()

    # --- STEP 4: Prepare Final DataFrame based on Mode ---
    final_features_df = pd.DataFrame()

    if mode == "HISTORICAL":
        logger.info("STEP 4 [HISTORICAL]: Merging features (Base: Starters)...")
        # Base DataFrame: Now comes from starter pitcher history
        base_cols = ['game_pk', 'game_date', 'pitcher_id', 'team', 'opponent_team',
                     'is_home', 'ballpark', 'p_throws', 'home_team', 'away_team']
        if 'p_days_rest' in pitcher_hist_df.columns: base_cols.append('p_days_rest')
        base_cols.extend(league_avg_cols.keys()) # Add league avg cols

        # Add target vars if they exist
        if 'strikeouts' in pitcher_hist_df.columns: base_cols.append('strikeouts')
        if 'batters_faced' in pitcher_hist_df.columns: base_cols.append('batters_faced')

        # Ensure columns exist before selecting
        present_base_cols = [col for col in base_cols if col in pitcher_hist_df.columns]
        missing_base_cols = [col for col in base_cols if col not in present_base_cols]
        if missing_base_cols: logger.warning(f"Missing base columns from pitcher_hist_df: {missing_base_cols}")
        final_features_df = pitcher_hist_df[present_base_cols].copy()

        # Add home plate umpire name
        if not umpire_hist_df.empty and 'home_plate_umpire' in umpire_hist_df.columns:
             umpire_lookup = umpire_hist_df[['game_date', 'home_team', 'away_team', 'home_plate_umpire']].drop_duplicates()
             if not pd.api.types.is_datetime64_any_dtype(umpire_lookup['game_date']): umpire_lookup['game_date'] = pd.to_datetime(umpire_lookup['game_date'])
             if not pd.api.types.is_datetime64_any_dtype(final_features_df['game_date']): final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date'])

             # Check for required merge keys before merging
             merge_keys_ump = ['game_date', 'home_team', 'away_team']
             if all(key in final_features_df.columns for key in merge_keys_ump):
                  final_features_df = pd.merge(final_features_df, umpire_lookup, on=merge_keys_ump, how='left')
                  logger.debug(f"Added home_plate_umpire column. {final_features_df['home_plate_umpire'].isnull().sum()} nulls.")
             else:
                 logger.warning(f"Missing keys {merge_keys_ump} for umpire name merge. Skipping.")
                 final_features_df['home_plate_umpire'] = np.nan
        else: final_features_df['home_plate_umpire'] = np.nan

        # Merge Pitcher features (index aligned)
        if 'pitcher' in all_rolling_features and not all_rolling_features['pitcher'].empty:
            final_features_df = pd.concat([final_features_df, all_rolling_features['pitcher']], axis=1)
            logger.debug("Merged pitcher rolling features (starter-based).")

        # Merge Opponent features (derived from team_stats)
        if 'team' in all_rolling_features and 'opponent' in all_rename_maps and not all_rolling_features['team'].empty:
            final_features_df = merge_opponent_features_historical(
                final_features_df=final_features_df,
                opponent_rolling_df=all_rolling_features['team'],
                opp_rename_map=all_rename_maps['opponent']
            ) # merge_opponent_features_historical needs 'opponent_team' column in final_features_df

        # Merge Ballpark features (derived from starter pitcher stats)
        if 'ballpark' in all_rolling_features and 'ballpark' in all_rename_maps and not all_rolling_features['ballpark'].empty:
             final_features_df = merge_ballpark_features_historical(
                 final_features_df=final_features_df,
                 ballpark_rolling_df=all_rolling_features['ballpark'],
                 bpark_rename_map=all_rename_maps['ballpark']
             ) # merge_ballpark_features_historical needs 'ballpark' column

        # Merge Umpire features (derived from starter pitcher stats & umpire hist)
        if 'umpire' in all_rolling_features and not all_rolling_features['umpire'].empty:
             final_features_df = pd.concat([final_features_df, all_rolling_features['umpire']], axis=1)
             logger.debug("Merged umpire rolling features (starter-based).")
        else:
             logger.warning("No umpire rolling features calculated, skipping historical merge.")
             if 'umpire' in all_rename_maps:
                 for col in all_rename_maps['umpire'].values(): final_features_df[col] = np.nan

        # Add season
        if 'game_date' in final_features_df.columns:
             final_features_df['season'] = pd.to_datetime(final_features_df['game_date']).dt.year


    elif mode == "PREDICTION":
        logger.info("STEP 4 [PREDICTION]: Merging latest features onto prediction baseline (Base: Starters)...")
        # Load schedule, build baseline for probable starters
        schedule_query = f"SELECT * FROM mlb_api WHERE DATE(game_date) = '{prediction_date_str}'"
        schedule_df = load_data_from_db(schedule_query, db_path, optimize=False)
        if schedule_df.empty: logger.error("Prediction schedule missing."); return
        baseline_data = []
        for _, game in schedule_df.iterrows():
            game_date_dt = pd.to_datetime(game['game_date'])
            # Ensure IDs are numeric, handle potential errors
            home_pid = pd.to_numeric(game.get('home_probable_pitcher_id'), errors='coerce')
            away_pid = pd.to_numeric(game.get('away_probable_pitcher_id'), errors='coerce')
            home_team_abbr = game.get('home_team_abbr')
            away_team_abbr = game.get('away_team_abbr')
            ballpark = team_to_ballpark_map.get(home_team_abbr, "Unknown Park")
            game_pk = game.get('game_pk')
            # Only add if pitcher ID is valid
            if pd.notna(home_pid): baseline_data.append({'pitcher_id': int(home_pid), 'game_date': game_date_dt, 'game_pk': game_pk, 'opponent_team': away_team_abbr, 'is_home': 1, 'ballpark': ballpark, 'home_team': home_team_abbr, 'away_team': away_team_abbr})
            if pd.notna(away_pid): baseline_data.append({'pitcher_id': int(away_pid), 'game_date': game_date_dt, 'game_pk': game_pk, 'opponent_team': home_team_abbr, 'is_home': 0, 'ballpark': ballpark, 'home_team': home_team_abbr, 'away_team': away_team_abbr})
        if not baseline_data: logger.error("No valid probable pitchers found in schedule."); return
        final_features_df = pd.DataFrame(baseline_data)

        # Extract latest rolling values indices FROM STARTER HISTORY
        latest_pitcher_indices = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['pitcher_id'], keep='last').index
        # Team history indices remain the same
        latest_team_indices = team_hist_df.sort_values('game_date').drop_duplicates(subset=['team'], keep='last').index if not team_hist_df.empty else pd.Index([])
        # Ballpark indices now derived from starter history
        latest_ballpark_indices = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['ballpark'], keep='last').index
        # Umpire indices now derived from starter history + umpire mapping
        latest_umpire_indices = pd.Index([])
        temp_ump_df = pd.DataFrame() # To hold umpire names aligned with pitcher_hist_df index
        if not umpire_hist_df.empty and 'home_plate_umpire' in umpire_hist_df.columns:
            temp_ump_lookup = umpire_hist_df[['game_date', 'home_team', 'away_team', 'home_plate_umpire']].drop_duplicates()
            if not pd.api.types.is_datetime64_any_dtype(temp_ump_lookup['game_date']): temp_ump_lookup['game_date'] = pd.to_datetime(temp_ump_lookup['game_date'])

            # Merge onto starter history index
            merge_keys_ump_pred = ['game_date', 'home_team', 'away_team']
            if all(key in pitcher_hist_df.columns for key in merge_keys_ump_pred):
                  temp_ump_df = pd.merge(pitcher_hist_df[merge_keys_ump_pred].reset_index(), temp_ump_lookup, on=merge_keys_ump_pred, how='left').set_index('index')
                  temp_ump_df.index.name = None
            else: logger.warning(f"Missing keys {merge_keys_ump_pred} in starter history for umpire mapping.")

            if 'home_plate_umpire' in temp_ump_df.columns:
                 temp_ump_df_for_sort = pd.concat([temp_ump_df[['home_plate_umpire']], pitcher_hist_df[['game_date']]], axis=1)
                 latest_umpire_indices = temp_ump_df_for_sort.dropna(subset=['home_plate_umpire'])\
                                                             .sort_values('game_date')\
                                                             .drop_duplicates(subset=['home_plate_umpire'], keep='last').index


        # Get latest rolling features DataFrames using indices
        latest_pitcher_rolling = all_rolling_features.get('pitcher', pd.DataFrame()).loc[latest_pitcher_indices] if 'pitcher' in all_rolling_features else pd.DataFrame()
        latest_opponent_rolling = all_rolling_features.get('team', pd.DataFrame()).loc[latest_team_indices] if ('team' in all_rolling_features and not latest_team_indices.empty) else pd.DataFrame()
        latest_ballpark_rolling = all_rolling_features.get('ballpark', pd.DataFrame()).loc[latest_ballpark_indices] if 'ballpark' in all_rolling_features else pd.DataFrame()
        latest_umpire_rolling = all_rolling_features.get('umpire', pd.DataFrame()).loc[latest_umpire_indices] if ('umpire' in all_rolling_features and not latest_umpire_indices.empty) else pd.DataFrame()


        # Add keys back for merging
        if not latest_pitcher_rolling.empty: latest_pitcher_rolling['pitcher_id'] = pitcher_hist_df.loc[latest_pitcher_rolling.index, 'pitcher_id']
        if not latest_opponent_rolling.empty: latest_opponent_rolling['team'] = team_hist_df.loc[latest_opponent_rolling.index, 'team']
        if not latest_ballpark_rolling.empty: latest_ballpark_rolling['ballpark'] = pitcher_hist_df.loc[latest_ballpark_rolling.index, 'ballpark']
        if not latest_umpire_rolling.empty and not temp_ump_df.empty and 'home_plate_umpire' in temp_ump_df.columns:
             latest_umpire_rolling['home_plate_umpire'] = temp_ump_df.loc[latest_umpire_rolling.index, 'home_plate_umpire']


        # --- Merge league averages for prediction date (based on starters) ---
        if league_avg_cols and not daily_league_stats.empty:
             latest_league_avg_date = daily_league_stats['game_date'].max()
             logger.info(f"Using latest available daily league averages (starter-based) from: {latest_league_avg_date.strftime('%Y-%m-%d')}")
             latest_league_avgs = daily_league_stats[daily_league_stats['game_date'] == latest_league_avg_date]
             if not latest_league_avgs.empty:
                 latest_league_avg_dict = latest_league_avgs.iloc[0].to_dict()
                 latest_league_avg_dict.pop('game_date', None)
                 for col_name, value in latest_league_avg_dict.items():
                     final_features_df[col_name] = value
                     logger.debug(f"Added prediction league avg column '{col_name}' with value {value}")
             else: logger.warning(f"No league average data found for date {latest_league_avg_date}.")
        else: logger.warning("No league average data available to merge for prediction.")


        # Calculate Days Rest (based on starter history)
        last_game_dates = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['pitcher_id'], keep='last')[['pitcher_id', 'game_date']]
        final_features_df = pd.merge(final_features_df, last_game_dates.rename(columns={'game_date':'last_game_date'}), on='pitcher_id', how='left')
        final_features_df['last_game_date'] = pd.to_datetime(final_features_df['last_game_date'])
        if not pd.api.types.is_datetime64_any_dtype(final_features_df['game_date']): final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date'])
        final_features_df['p_days_rest'] = (final_features_df['game_date'] - final_features_df['last_game_date']).dt.days
        final_features_df = final_features_df.drop(columns=['last_game_date'])

        # Get Pitcher Handedness (based on starter history)
        pitcher_throws = pitcher_hist_df.dropna(subset=['p_throws']).drop_duplicates(subset=['pitcher_id'], keep='last')[['pitcher_id', 'p_throws']]
        final_features_df = pd.merge(final_features_df, pitcher_throws, on='pitcher_id', how='left')
        final_features_df['p_throws'] = final_features_df['p_throws'].fillna('R') # Assume R if unknown

        # Get Umpire for Prediction Date Games
        umpire_pred_lookup = load_data_from_db(
            f"SELECT game_pk, home_plate_umpire FROM {umpire_table} WHERE DATE(game_date) = '{prediction_date_str}'",
            db_path, optimize=False
        )
        if not umpire_pred_lookup.empty:
            final_features_df = pd.merge(final_features_df, umpire_pred_lookup, on='game_pk', how='left')
            logger.info(f"Looked up umpires for prediction date. Found assignments for {len(final_features_df) - final_features_df['home_plate_umpire'].isnull().sum()} games.")
        else:
            logger.warning(f"No umpire data found in {umpire_table} for prediction date {prediction_date_str}.")
            final_features_df['home_plate_umpire'] = np.nan


        # --- Merge Latest Features ---
        # Pitcher Features (from starter history)
        if 'pitcher' in all_rename_maps and not latest_pitcher_rolling.empty:
            p_rename_map = all_rename_maps['pitcher']
            p_cols_to_merge = ['pitcher_id'] + list(p_rename_map.keys())
            p_cols_to_merge = [col for col in p_cols_to_merge if col in latest_pitcher_rolling.columns]
            if len(p_cols_to_merge) > 1: final_features_df = pd.merge(final_features_df, latest_pitcher_rolling[p_cols_to_merge], on='pitcher_id', how='left')

        # Opponent Features (from team history)
        if 'opponent' in all_rename_maps and not latest_opponent_rolling.empty:
            final_features_df = merge_opponent_features_prediction(
                final_features_df=final_features_df, latest_opponent_rolling=latest_opponent_rolling,
                opp_rename_map=all_rename_maps['opponent'], rolling_windows=ROLLING_WINDOWS
            )

        # Ballpark Features (from starter history)
        if 'ballpark' in all_rename_maps and not latest_ballpark_rolling.empty:
            final_features_df = merge_ballpark_features_prediction(
                final_features_df=final_features_df, latest_ballpark_rolling=latest_ballpark_rolling,
                bpark_rename_map=all_rename_maps['ballpark']
            )

        # Umpire Features (from starter history)
        if 'umpire' in all_rename_maps and not latest_umpire_rolling.empty:
             ump_rename_map = all_rename_maps['umpire']
             ump_cols_to_merge = ['home_plate_umpire'] + list(ump_rename_map.values())
             ump_cols_to_merge = [col for col in ump_cols_to_merge if col in latest_umpire_rolling.columns]
             if len(ump_cols_to_merge) > 1 and 'home_plate_umpire' in final_features_df.columns:
                  final_features_df = pd.merge(final_features_df, latest_umpire_rolling[ump_cols_to_merge], on='home_plate_umpire', how='left')
                  logger.debug("Merged latest umpire features (starter-based).")
             else: logger.warning("Could not merge latest umpire features (missing key or data).")
        else: logger.warning("Umpire rename map or latest umpire rolling data missing.")


        # Format date back to string for saving
        final_features_df['game_date'] = final_features_df['game_date'].dt.strftime('%Y-%m-%d')


    # --- STEP 5: Final Cleanup & Define Expected Columns ---
    logger.info("STEP 5: Defining expected columns and cleaning up (Base: Starters)...")
    if final_features_df.empty: logger.error("Features empty before final cleanup."); return

    # Define expected column groups (based on starter context)
    expected_base_cols = ['game_pk', 'game_date', 'pitcher_id', 'team', 'opponent_team', 'is_home', 'ballpark', 'p_throws', 'p_days_rest', 'home_plate_umpire']
    expected_league_avg_cols = list(league_avg_cols.keys())
    expected_target_cols = ['strikeouts', 'batters_faced', 'season'] if mode == "HISTORICAL" else []

    # Get expected rolling columns from rename maps
    expected_p_roll_cols = list(all_rename_maps.get('pitcher', {}).keys()) # Use keys if no rename map
    expected_opp_roll_cols_base = list(all_rename_maps.get('opponent', {}).values())
    expected_bp_roll_cols = list(all_rename_maps.get('ballpark', {}).values())
    expected_ump_roll_cols = list(all_rename_maps.get('umpire', {}).values())

    # Adjust opponent cols based on prediction mode logic (no change needed here)
    if mode == 'PREDICTION':
         base_opp_metrics = [m for m in opponent_metrics_likely if '_vs_' not in m] # Use base metrics
         expected_opp_roll_cols = [col for col in expected_opp_roll_cols_base if '_vs_' not in col] # Base rolling
         # Add vs_pitcher cols if they were created by merge_opponent_features_prediction
         expected_opp_roll_cols += [f'opp_roll{w}g_{m}_vs_pitcher' for w in ROLLING_WINDOWS for m in base_opp_metrics]
         # Remove duplicates just in case
         expected_opp_roll_cols = sorted(list(set(expected_opp_roll_cols)))
    else: expected_opp_roll_cols = expected_opp_roll_cols_base

    # Combine all expected columns
    expected_cols = list(set(
        expected_base_cols + expected_league_avg_cols + expected_target_cols +
        expected_p_roll_cols + expected_opp_roll_cols + expected_bp_roll_cols + expected_ump_roll_cols
    ))

    # Add missing columns as NaN and select/reorder
    present_cols = final_features_df.columns.tolist()
    final_cols_ordered = []
    # Prioritize base/target/league avg
    priority_cols = expected_base_cols + expected_league_avg_cols + expected_target_cols
    for col in priority_cols:
         if col in present_cols: final_cols_ordered.append(col)
         elif col in expected_cols:
              logger.warning(f"Adding missing expected base/target/league column '{col}' as NaN.")
              final_features_df[col] = np.nan; final_cols_ordered.append(col)
    # Add rolling
    for col_list in [expected_p_roll_cols, expected_opp_roll_cols, expected_bp_roll_cols, expected_ump_roll_cols]:
         for col in col_list:
              if col in expected_cols:
                   if col not in present_cols:
                        logger.warning(f"Adding missing expected rolling column '{col}' as NaN.")
                        final_features_df[col] = np.nan
                   if col not in final_cols_ordered: final_cols_ordered.append(col)

    final_cols_ordered = [col for col in final_cols_ordered if col in final_features_df.columns]
    final_features_df = final_features_df[final_cols_ordered]

    logger.info(f"Final DataFrame shape after cleanup: {final_features_df.shape}")


    # --- STEP 6: Save Results ---
    output_table_train = "train_features"
    output_table_test = "test_features"
    output_table_pred = "prediction_features" # Use consistent name

    logger.info(f"STEP 6: Saving final features (Base: Starters)...")
    try:
        with DBConnection() as conn:
            if mode == "HISTORICAL":
                logger.info(f"Splitting historical data into train ({train_years}) and test ({test_years})...")
                if 'season' not in final_features_df.columns: logger.error("'season' column missing, cannot split by year.")
                else:
                    train_df = final_features_df[final_features_df['season'].isin(train_years)].copy()
                    test_df = final_features_df[final_features_df['season'].isin(test_years)].copy()
                    logger.info(f"Saving {len(train_df)} training rows to '{output_table_train}'...")
                    conn.execute(f"DROP TABLE IF EXISTS {output_table_train}")
                    train_df.to_sql(output_table_train, conn, if_exists='replace', index=False)
                    conn.commit()
                    logger.info(f"Saving {len(test_df)} test rows to '{output_table_test}'...")
                    conn.execute(f"DROP TABLE IF EXISTS {output_table_test}")
                    test_df.to_sql(output_table_test, conn, if_exists='replace', index=False)
                    conn.commit()
            elif mode == "PREDICTION":
                logger.info(f"Saving {len(final_features_df)} prediction rows with {len(final_features_df.columns)} columns to '{output_table_pred}'...")
                conn.execute(f"DROP TABLE IF EXISTS {output_table_pred}")
                final_features_df.to_sql(output_table_pred, conn, if_exists='replace', index=False)
                conn.commit()
                logger.debug(f"Final prediction columns: {final_features_df.columns.tolist()}")
    except Exception as e: logger.error(f"Failed to save features to database: {e}", exc_info=True)

    gc.collect()
    logger.info(f"--- Feature Generation [{mode} Mode] (Base: Starters) Completed ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    parser = argparse.ArgumentParser(description="Generate MLB Features (Base: Starters) for Training/Prediction.")
    parser.add_argument("--prediction-date", type=str, default=None, help="Generate features for specific date (YYYY-MM-DD). Default: full historical.")
    parser.add_argument("--train-years", type=int, nargs='+', default=None, help="Years for training set. Overrides config.")
    parser.add_argument("--test-years", type=int, nargs='+', default=None, help="Years for test set. Overrides config.")
    args = parser.parse_args()

    train_years_to_use = args.train_years if args.train_years else StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_years_to_use = args.test_years if args.test_years else StrikeoutModelConfig.DEFAULT_TEST_YEARS

    generate_features(
        prediction_date_str=args.prediction_date,
        train_years=train_years_to_use if not args.prediction_date else None,
        test_years=test_years_to_use if not args.prediction_date else None
    )