# src/scripts/generate_basic_features.py

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
    from src.data.utils import setup_logger, DBConnection
    # Assuming aggregate_statcast functions are updated for opponent platoons
    from src.data.aggregate_statcast import aggregate_statcast_pitchers_sql, aggregate_statcast_batters_sql
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    MODULE_IMPORTS_OK = False
else:
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('generate_basic_features', LogConfig.LOG_DIR / 'generate_basic_features.log')

# --- Configuration (Using WINDOW_SIZES from config) ---
# Define base metrics for different groups
PITCHER_METRICS_FOR_ROLLING = [
    'k_percent', 'swinging_strike_percent', 'avg_velocity',
    'k_percent_vs_lhb', 'swinging_strike_percent_vs_lhb',
    'k_percent_vs_rhb', 'swinging_strike_percent_vs_rhb'
]
OPPONENT_METRICS_FOR_ROLLING = [
    'k_percent', 'swinging_strike_percent', # Overall team K% / SwStr%
    'k_percent_vs_LHP', 'swinging_strike_percent_vs_LHP', # Team stats when facing LHP
    'k_percent_vs_RHP', 'swinging_strike_percent_vs_RHP'  # Team stats when facing RHP
]
BALLPARK_METRICS_FOR_ROLLING = ['k_percent'] # Using pitcher K% in that park

# Use WINDOW_SIZES from config
ROLLING_WINDOWS = StrikeoutModelConfig.WINDOW_SIZES # e.g., [3, 5, 10]
MIN_ROLLING_PERIODS = 2 # Lower min periods for smaller windows? Or make dynamic?

# --- Helper Functions --- (load_data_from_db, calculate_multi_window_rolling remain the same)
def load_data_from_db(query: str, db_path: Path, optimize: bool = True) -> pd.DataFrame:
    """Loads data from the database using a given query."""
    logger.info(f"Executing query: {query[:100]}...")
    start_time = datetime.now()
    df = pd.DataFrame() # Initialize empty df
    try:
        with DBConnection(db_path) as conn:
            df = pd.read_sql_query(query, conn)
        duration = datetime.now() - start_time
        logger.info(f"Loaded {len(df)} rows in {duration.total_seconds():.2f}s.")
        if optimize and not df.empty:
            # Basic type optimization
            for col in df.select_dtypes(include=['int64']).columns: df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64']).columns: df[col] = pd.to_numeric(df[col], downcast='float')
    except Exception as e:
        logger.error(f"Failed to load data with query: {query[:100]}... Error: {e}", exc_info=True)
    return df

def calculate_multi_window_rolling(df, group_col, date_col, metrics, windows, min_periods, shift_periods=1):
    """
    Calculates rolling features for multiple windows efficiently.
    Groups data once and calculates all windows for each metric.
    Uses shift() to prevent data leakage. Returns results indexed like input df.
    """
    if df is None or df.empty or not metrics or not windows:
        logger.warning(f"Input invalid for multi-window rolling on {group_col}.")
        return pd.DataFrame(index=df.index if df is not None else None)

    logger.info(f"Calculating multi-window {windows} rolling for group '{group_col}' on {len(metrics)} metrics...")
    # Sort once for all calculations
    df_sorted = df.sort_values(by=[group_col, date_col])
    # Grouping outside the loop for potential efficiency, but transform handles alignment
    grouped = df_sorted.groupby(group_col, observed=True)
    results_dict = {} # Store results keyed by new column name

    for metric in metrics:
        if metric not in df_sorted.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe for rolling calculation. Skipping.")
            continue

        # Ensure metric is numeric first before attempting shift/roll
        metric_series = pd.to_numeric(df_sorted[metric], errors='coerce')
        if metric_series.isnull().all():
            logger.warning(f"Metric '{metric}' is entirely non-numeric or NaN. Skipping.")
            continue

        # Calculate rolling mean for all windows for this metric using transform for alignment
        for window in windows:
            roll_col_name = f"{metric}_roll{window}g"
            current_min_periods = max(1, min(min_periods, window - 1))

            try:
                # Apply rolling calculation directly using transform
                # Shift happens inside the lambda to prevent leakage
                results_dict[roll_col_name] = grouped[metric].transform(
                    lambda x: x.shift(shift_periods)
                               .rolling(window=window, min_periods=current_min_periods)
                               .mean()
                )
                logger.debug(f"Calculated rolling feature: {roll_col_name}")
            except Exception as e:
                 logger.error(f"Error calculating rolling for {metric} window {window}: {e}", exc_info=True)
                 # Assign NaN series with the correct index if calculation fails
                 results_dict[roll_col_name] = pd.Series(np.nan, index=df_sorted.index)

    # Combine results and reindex to match original df index
    if not results_dict:
         return pd.DataFrame(index=df.index)

    results_df = pd.DataFrame(results_dict, index=df_sorted.index)
    return results_df.reindex(df.index)


# --- Main Feature Generation Logic ---
def generate_features(prediction_date_str: str | None,
                        # output_table_pred, output_table_hist removed as args
                        train_years: list[int] | None = None,
                        test_years: list[int] | None = None):
    """
    Generates features including pitcher/opp platoon splits and multiple rolling windows.
    Operates in prediction or historical mode. Includes aggregation calls.
    Saves to fixed table names: 'train_features', 'test_features', 'prediction_features'.
    """
    mode = "PREDICTION" if prediction_date_str else "HISTORICAL"
    logger.info(f"--- Starting Feature Generation [{mode} Mode] (Multi-Window, Platoons) ---")
    db_path = Path(DBConfig.PATH)
    prediction_date = None
    max_hist_date_str = '9999-12-31'
    if prediction_date_str:
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
            max_hist_date_str = (prediction_date - timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"Prediction Date: {prediction_date_str}")
        except ValueError: logger.error(f"Invalid prediction date format: {prediction_date_str}. Use YYYY-MM-DD."); return
    else:
        logger.info("Running for all historical data.")


    # --- STEP 0: Run Aggregations ---
    logger.info("STEP 0: Running Statcast Aggregations (Full History)...")
    try:
        # Ensure these functions were modified previously for opponent platoons
        aggregate_statcast_pitchers_sql(target_date=None)
        aggregate_statcast_batters_sql(target_date=None)
        logger.info("Aggregations completed.")
    except Exception as agg_e: logger.error(f"Error during aggregation: {agg_e}", exc_info=True); return
    gc.collect()

    # --- STEP 1: Load Data ---
    logger.info("STEP 1: Loading necessary data...")
    start_load_time = datetime.now()

    # Load Team Mapping (remains the same)
    team_map_query = "SELECT team_abbr, ballpark FROM team_mapping"
    team_map_df = load_data_from_db(team_map_query, db_path, optimize=False)
    team_to_ballpark_map = team_map_df.set_index('team_abbr')['ballpark'].to_dict() if not team_map_df.empty else {}
    if not team_to_ballpark_map: logger.warning("Team mapping empty.")

    # Load Pitcher History (select all relevant metric cols)
    try:
        with DBConnection(db_path) as conn: pitcher_cols = pd.read_sql_query("SELECT * FROM game_level_pitchers LIMIT 1", conn).columns.tolist()
    except Exception as e: logger.error(f"Cannot read columns from game_level_pitchers: {e}"); return
    pitcher_metrics_needed = set(PITCHER_METRICS_FOR_ROLLING) | set(BALLPARK_METRICS_FOR_ROLLING)
    pitcher_base_cols = ['pitcher_id', 'game_date', 'game_pk', 'p_throws', 'opponent_team', 'home_team', 'away_team', 'is_home', 'strikeouts', 'batters_faced']
    pitcher_cols_to_load = list(set(pitcher_base_cols) | pitcher_metrics_needed)
    pitcher_cols_to_load_str = ', '.join([f'"{c}"' for c in pitcher_cols_to_load if c in pitcher_cols]) # Quote cols
    if not pitcher_cols_to_load_str: logger.error("No pitcher columns could be identified for loading."); return
    pitcher_hist_query = f"SELECT {pitcher_cols_to_load_str} FROM game_level_pitchers WHERE DATE(game_date) <= '{max_hist_date_str}'"
    pitcher_hist_df = load_data_from_db(pitcher_hist_query, db_path)

    # Load Team History (select all relevant metric cols, including new ones)
    try:
        with DBConnection(db_path) as conn: team_cols = pd.read_sql_query("SELECT * FROM game_level_team_stats LIMIT 1", conn).columns.tolist()
    except Exception as e: logger.warning(f"Cannot read columns from game_level_team_stats: {e}"); team_cols = []
    team_metrics_needed = set(OPPONENT_METRICS_FOR_ROLLING)
    team_base_cols = ['team', 'game_date', 'game_pk', 'home_team']
    team_cols_to_load = list(set(team_base_cols) | team_metrics_needed)
    team_cols_to_load_str = ', '.join([f'"{c}"' for c in team_cols_to_load if c in team_cols]) # Quote cols
    team_hist_df = load_data_from_db(f"SELECT {team_cols_to_load_str} FROM game_level_team_stats WHERE DATE(game_date) <= '{max_hist_date_str}'", db_path) if team_cols_to_load_str else pd.DataFrame()


    if pitcher_hist_df.empty: logger.error("Pitcher history empty."); return
    if team_hist_df.empty: logger.warning("Team history empty.")

    # Add derived 'team', 'ballpark' to pitcher_hist_df (as before)
    try: pitcher_hist_df['team'] = np.where(pitcher_hist_df['is_home'] == 1, pitcher_hist_df['home_team'], pitcher_hist_df['away_team'])
    except KeyError: logger.error("Cannot derive pitcher 'team'."); pitcher_hist_df['team'] = 'UNK'
    if 'home_team' in pitcher_hist_df.columns: pitcher_hist_df['ballpark'] = pitcher_hist_df['home_team'].map(team_to_ballpark_map).fillna("Unknown Park")
    else: logger.warning("Missing 'home_team', cannot map ballparks."); pitcher_hist_df['ballpark'] = "Unknown Park"

    # Convert dates
    pitcher_hist_df['game_date'] = pd.to_datetime(pitcher_hist_df['game_date'])
    if not team_hist_df.empty: team_hist_df['game_date'] = pd.to_datetime(team_hist_df['game_date'])

    logger.info(f"Data loading finished in {(datetime.now() - start_load_time).total_seconds():.2f}s.")
    gc.collect()

    # --- STEP 2 & 3: Calculate Historical Rolling Features ---
    logger.info(f"STEP 2&3: Calculating historical rolling features (Windows: {ROLLING_WINDOWS})...")
    calc_start_time = datetime.now()
    all_rolling_features = {} # Dictionary to store DFs for each group

    # Pitcher Rolling
    available_pitcher_metrics = [m for m in PITCHER_METRICS_FOR_ROLLING if m in pitcher_hist_df.columns]
    pitcher_rolling_df = calculate_multi_window_rolling(
        df=pitcher_hist_df, group_col='pitcher_id', date_col='game_date',
        metrics=available_pitcher_metrics, windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS
    )
    p_rename_map = {f"{m}_roll{w}g": f"p_roll{w}g_{m}" for w in ROLLING_WINDOWS for m in available_pitcher_metrics if f"{m}_roll{w}g" in pitcher_rolling_df.columns}
    all_rolling_features['pitcher'] = pitcher_rolling_df.rename(columns=p_rename_map)
    logger.info("Pitcher rolling features calculated.")

    # Team/Opponent Rolling
    if not team_hist_df.empty:
        available_team_metrics = [m for m in OPPONENT_METRICS_FOR_ROLLING if m in team_hist_df.columns]
        team_rolling_calc = calculate_multi_window_rolling(
            df=team_hist_df, group_col='team', date_col='game_date',
            metrics=available_team_metrics, windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS
        )
        opp_rename_map = {f"{m}_roll{w}g": f"opp_roll{w}g_{m}" for w in ROLLING_WINDOWS for m in available_team_metrics if f"{m}_roll{w}g" in team_rolling_calc.columns}
        team_rolling_df = team_rolling_calc.rename(columns=opp_rename_map)
        team_rolling_df[['team', 'game_date']] = team_hist_df[['team', 'game_date']] # Add keys back
        all_rolling_features['team'] = team_rolling_df
        logger.info("Team/Opponent rolling features calculated.")

    # Ballpark Rolling
    available_bpark_metrics = [m for m in BALLPARK_METRICS_FOR_ROLLING if m in pitcher_hist_df.columns]
    if 'ballpark' in pitcher_hist_df.columns and available_bpark_metrics:
         ballpark_rolling_calc = calculate_multi_window_rolling(
             df=pitcher_hist_df, group_col='ballpark', date_col='game_date',
             metrics=available_bpark_metrics, windows=ROLLING_WINDOWS, min_periods=MIN_ROLLING_PERIODS
         )
         bpark_rename_map = {f"{m}_roll{w}g": f"bp_roll{w}g_{m}" for w in ROLLING_WINDOWS for m in available_bpark_metrics if f"{m}_roll{w}g" in ballpark_rolling_calc.columns}
         ballpark_rolling_df = ballpark_rolling_calc.rename(columns=bpark_rename_map)
         ballpark_rolling_df[['ballpark', 'game_date']] = pitcher_hist_df[['ballpark', 'game_date']] # Add keys back
         all_rolling_features['ballpark'] = ballpark_rolling_df
         logger.info("Ballpark rolling features calculated.")

    # Pitcher Days Rest
    logger.info("Calculating pitcher days rest...")
    pitcher_hist_df_sorted = pitcher_hist_df.sort_values(by=['pitcher_id', 'game_date'])
    pitcher_hist_df_sorted['p_days_rest'] = pitcher_hist_df_sorted.groupby('pitcher_id')['game_date'].diff().dt.days
    pitcher_hist_df['p_days_rest'] = pitcher_hist_df_sorted['p_days_rest'] # Merge back

    logger.info(f"Feature calculation finished in {(datetime.now() - calc_start_time).total_seconds():.2f}s.")
    gc.collect()

    # --- STEP 4: Prepare Final DataFrame based on Mode ---
    final_features_df = pd.DataFrame()

    if mode == "HISTORICAL":
        logger.info("STEP 4 [HISTORICAL]: Merging features...")
        base_cols = ['game_pk', 'game_date', 'pitcher_id', 'team', 'opponent_team',
                     'is_home', 'ballpark', 'p_throws', 'p_days_rest',
                     'strikeouts', 'batters_faced']
        final_features_df = pitcher_hist_df[[col for col in base_cols if col in pitcher_hist_df.columns]].copy()

        # Merge Pitcher features (index aligned)
        if 'pitcher' in all_rolling_features:
            final_features_df = pd.concat([final_features_df, all_rolling_features['pitcher']], axis=1)
            logger.debug("Merged pitcher rolling features.")

        # Merge Opponent features (time-aligned)
        if 'team' in all_rolling_features:
            team_rolling_df = all_rolling_features['team']
            opp_roll_cols_to_merge = [col for col in opp_rename_map.values() if col in team_rolling_df.columns]
            if opp_roll_cols_to_merge:
                 final_features_df['merge_key_opponent'] = final_features_df['opponent_team'].astype(str)
                 team_rolling_df['merge_key_team'] = team_rolling_df['team'].astype(str)
                 right_merge_cols = ['merge_key_team', 'game_date'] + opp_roll_cols_to_merge
                 final_features_df = pd.merge_asof(
                     final_features_df.sort_values('game_date'),
                     team_rolling_df[right_merge_cols].sort_values('game_date'),
                     on='game_date', left_by='merge_key_opponent', right_by='merge_key_team',
                     direction='backward', allow_exact_matches=False
                 ).drop(columns=['merge_key_opponent'], errors='ignore')
                 logger.debug("Merged opponent rolling features.")

        # Merge Ballpark features (time-aligned)
        if 'ballpark' in all_rolling_features:
             ballpark_rolling_df = all_rolling_features['ballpark']
             bpark_roll_cols_to_merge = [col for col in bpark_rename_map.values() if col in ballpark_rolling_df.columns]
             if bpark_roll_cols_to_merge:
                  final_features_df['merge_key_ballpark_left'] = final_features_df['ballpark'].astype(str)
                  ballpark_rolling_df['merge_key_ballpark_right'] = ballpark_rolling_df['ballpark'].astype(str)
                  right_merge_cols_bp = ['merge_key_ballpark_right', 'game_date'] + bpark_roll_cols_to_merge
                  final_features_df = pd.merge_asof(
                      final_features_df.sort_values('game_date'),
                      ballpark_rolling_df[right_merge_cols_bp].sort_values('game_date'),
                      on='game_date', left_by='merge_key_ballpark_left', right_by='merge_key_ballpark_right',
                      direction='backward', allow_exact_matches=False
                  ).drop(columns=['merge_key_ballpark_left'], errors='ignore')
                  logger.debug("Merged ballpark rolling features.")

        # Add season
        final_features_df['season'] = pd.to_datetime(final_features_df['game_date']).dt.year

    elif mode == "PREDICTION":
        logger.info("STEP 4 [PREDICTION]: Merging latest features onto prediction baseline...")
        # Load schedule, build baseline (as before)
        schedule_query = f"SELECT * FROM mlb_api WHERE DATE(game_date) = '{prediction_date_str}'"
        schedule_df = load_data_from_db(schedule_query, db_path, optimize=False)
        if schedule_df.empty: logger.error("Prediction schedule missing."); return
        baseline_data = []
        for _, game in schedule_df.iterrows():
            game_date_str_pred = pd.to_datetime(game['game_date']).strftime('%Y-%m-%d')
            home_pid = pd.to_numeric(game.get('home_probable_pitcher_id'), errors='coerce')
            away_pid = pd.to_numeric(game.get('away_probable_pitcher_id'), errors='coerce')
            home_team_abbr = game.get('home_team_abbr')
            away_team_abbr = game.get('away_team_abbr')
            ballpark = team_to_ballpark_map.get(home_team_abbr, "Unknown Park")
            if pd.notna(home_pid): baseline_data.append({'pitcher_id': int(home_pid), 'game_date': game_date_str_pred, 'game_pk': game.get('game_pk'), 'opponent_team': away_team_abbr, 'is_home': 1, 'ballpark': ballpark})
            if pd.notna(away_pid): baseline_data.append({'pitcher_id': int(away_pid), 'game_date': game_date_str_pred, 'game_pk': game.get('game_pk'), 'opponent_team': home_team_abbr, 'is_home': 0, 'ballpark': ballpark})
        if not baseline_data: logger.error("No probable pitchers."); return
        final_features_df = pd.DataFrame(baseline_data)
        final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date'])

        # Extract latest rolling values (needs adjustment for multi-window structure)
        # We need the latest row for each group from the original history DFs' index
        latest_pitcher_indices = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['pitcher_id'], keep='last').index
        latest_team_indices = team_hist_df.sort_values('game_date').drop_duplicates(subset=['team'], keep='last').index if not team_hist_df.empty else []
        latest_ballpark_indices = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['ballpark'], keep='last').index

        latest_pitcher_rolling = all_rolling_features.get('pitcher', pd.DataFrame()).loc[latest_pitcher_indices] if 'pitcher' in all_rolling_features else pd.DataFrame()
        latest_team_rolling = all_rolling_features.get('team', pd.DataFrame()).loc[latest_team_indices] if ('team' in all_rolling_features and not latest_team_indices.empty) else pd.DataFrame()
        latest_ballpark_rolling = all_rolling_features.get('ballpark', pd.DataFrame()).loc[latest_ballpark_indices] if 'ballpark' in all_rolling_features else pd.DataFrame()

        # Add keys back for merging
        if not latest_pitcher_rolling.empty: latest_pitcher_rolling['pitcher_id'] = pitcher_hist_df.loc[latest_pitcher_rolling.index, 'pitcher_id']
        if not latest_team_rolling.empty: latest_team_rolling['team'] = team_hist_df.loc[latest_team_rolling.index, 'team']
        if not latest_ballpark_rolling.empty: latest_ballpark_rolling['ballpark'] = pitcher_hist_df.loc[latest_ballpark_rolling.index, 'ballpark']

        # Calculate Days Rest (as before)
        last_game_dates = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['pitcher_id'], keep='last')[['pitcher_id', 'game_date']]
        final_features_df = pd.merge(final_features_df, last_game_dates.rename(columns={'game_date':'last_game_date'}), on='pitcher_id', how='left')
        final_features_df['p_days_rest'] = (final_features_df['game_date'] - pd.to_datetime(final_features_df['last_game_date'])).dt.days
        final_features_df = final_features_df.drop(columns=['last_game_date'])

        # Get Pitcher Handedness (as before)
        pitcher_throws = pitcher_hist_df.dropna(subset=['p_throws']).drop_duplicates(subset=['pitcher_id'], keep='last')[['pitcher_id', 'p_throws']]
        final_features_df = pd.merge(final_features_df, pitcher_throws, on='pitcher_id', how='left')
        final_features_df['p_throws'] = final_features_df['p_throws'].fillna('R')

        # --- Merge Latest Features ---
        # Pitcher Features
        p_cols_to_merge = ['pitcher_id'] + [col for col in p_rename_map.values() if col in latest_pitcher_rolling.columns]
        if len(p_cols_to_merge) > 1: final_features_df = pd.merge(final_features_df, latest_pitcher_rolling[p_cols_to_merge], on='pitcher_id', how='left')

        # Opponent Features (Select correct platoon based on pitcher hand)
        if not latest_team_rolling.empty:
             opp_merge_cols = ['team'] + [col for col in opp_rename_map.values() if col in latest_team_rolling.columns]
             if len(opp_merge_cols) > 1:
                  final_features_df = pd.merge(final_features_df, latest_team_rolling[opp_merge_cols], left_on='opponent_team', right_on='team', how='left')
                  # Select Correct Opponent Platoon Feature for each window
                  for w in ROLLING_WINDOWS:
                      for metric_base in ['k_percent', 'swinging_strike_percent']:
                          opp_met_vs_p = f'opp_roll{w}g_{metric_base}_vs_pitcher'
                          opp_met_vs_L = f'opp_roll{w}g_{metric_base}_vs_LHP'
                          opp_met_vs_R = f'opp_roll{w}g_{metric_base}_vs_RHP'
                          final_features_df[opp_met_vs_p] = np.where(
                              final_features_df['p_throws'] == 'L',
                              final_features_df.get(opp_met_vs_L, np.nan),
                              final_features_df.get(opp_met_vs_R, np.nan) )
                          # Drop original L/R specific columns
                          final_features_df = final_features_df.drop(columns=[opp_met_vs_L, opp_met_vs_R], errors='ignore')
                  final_features_df = final_features_df.drop(columns=['team'], errors='ignore')

        # Ballpark Features
        bp_cols_to_merge = ['ballpark'] + [col for col in bpark_rename_map.values() if col in latest_ballpark_rolling.columns]
        if len(bp_cols_to_merge) > 1: final_features_df = pd.merge(final_features_df, latest_ballpark_rolling[bp_cols_to_merge], on='ballpark', how='left')

        final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date']).dt.strftime('%Y-%m-%d')


    # --- STEP 5: Final Cleanup & Impute ---
    logger.info("STEP 5: Cleaning up and imputing missing values...")
    if final_features_df.empty: logger.error("Features empty before imputation."); return

    # --- Define expected column groups --- <<< ADD THIS SECTION >>>
    expected_base_cols = [
        'game_pk', 'game_date', 'pitcher_id', 'opponent_team',
        'is_home', 'ballpark', 'p_throws', 'p_days_rest'
        ]
    expected_target_cols = ['strikeouts', 'batters_faced', 'season'] if mode == "HISTORICAL" else []
    # --- End added section ---

    # Dynamically define all expected rolling columns based on maps generated earlier
    expected_p_roll_cols = list(p_rename_map.values()) if 'pitcher' in all_rolling_features else []
    expected_opp_roll_cols = list(opp_rename_map.values()) if 'team' in all_rolling_features else [] # Includes LHP/RHP before selection
    expected_bp_roll_cols = list(bpark_rename_map.values()) if 'ballpark' in all_rolling_features else []

    # Adjust opponent cols based on prediction mode logic
    if mode == 'PREDICTION':
         base_opp = ['k_percent', 'swinging_strike_percent']
         expected_opp_roll_cols = [f'opp_roll{w}g_{m}' for w in ROLLING_WINDOWS for m in base_opp]
         expected_opp_roll_cols += [f'opp_roll{w}g_{m}_vs_pitcher' for w in ROLLING_WINDOWS for m in base_opp]
         # Remove the LHP/RHP specific columns if they existed in the map from historical run
         expected_opp_roll_cols = [c for c in expected_opp_roll_cols if '_vs_LHP' not in c and '_vs_RHP' not in c]


    # --- Combine all expected columns --- <<< This line now uses the defined lists >>>
    expected_cols = list(set(expected_base_cols + expected_target_cols + expected_p_roll_cols + expected_opp_roll_cols + expected_bp_roll_cols))

    # Add missing & select (as before)
    for col in expected_cols:
        if col not in final_features_df.columns:
            logger.warning(f"Adding missing expected column '{col}' as NaN.")
            final_features_df[col] = np.nan
    final_cols_order = [col for col in expected_cols if col in final_features_df.columns] # Keep only existing cols
    final_features_df = final_features_df[final_cols_order]

    # --- STEP 6: Save Results (Using FIXED table names) ---
    output_table_train = "train_features"
    output_table_test = "test_features"
    output_table_pred_fixed = "prediction_features"

    logger.info(f"STEP 6: Saving final features...")
    try:
        if mode == "HISTORICAL":
            logger.info(f"Splitting historical data into train ({train_years}) and test ({test_years})...")
            if 'season' not in final_features_df.columns:
                logger.error("'season' column missing, cannot split by year.")
                # Optionally save unsplit data if needed?
                # with DBConnection(db_path) as conn: final_features_df.to_sql("all_historical_features", conn, if_exists='replace', index=False)
            else:
                train_df = final_features_df[final_features_df['season'].isin(train_years)].copy()
                test_df = final_features_df[final_features_df['season'].isin(test_years)].copy()
                with DBConnection(db_path) as conn:
                    train_df.to_sql(output_table_train, conn, if_exists='replace', index=False)
                    logger.info(f"Saved {len(train_df)} training rows to '{output_table_train}'.")
                    test_df.to_sql(output_table_test, conn, if_exists='replace', index=False)
                    logger.info(f"Saved {len(test_df)} test rows to '{output_table_test}'.")
        elif mode == "PREDICTION":
            output_table_name = output_table_pred_fixed # Use fixed prediction table name
            with DBConnection(db_path) as conn:
                final_features_df.to_sql(output_table_name, conn, if_exists='replace', index=False)
            logger.info(f"Successfully saved {len(final_features_df)} rows with {len(final_features_df.columns)} columns to '{output_table_name}'.")
            logger.debug(f"Final columns: {final_features_df.columns.tolist()}")

    except Exception as e: logger.error(f"Failed to save features to database: {e}", exc_info=True)

    gc.collect()
    logger.info(f"--- Feature Generation [{mode} Mode] (Multi-Window, Platoons) Completed ---")


# --- Main Execution Block (MODIFIED to remove table name args) ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    parser = argparse.ArgumentParser(description="Generate MLB Features (Multi-Window, Platoons) for Training/Prediction.")
    parser.add_argument("--prediction-date", type=str, default=None,
                        help="Generate features for specific date (YYYY-MM-DD). Default: full historical.")
    # Removed --output-table-pred and --output-table-hist args
    parser.add_argument("--train-years", type=int, nargs='+', default=None, help="Years for training set. Overrides config.")
    parser.add_argument("--test-years", type=int, nargs='+', default=None, help="Years for test set. Overrides config.")
    args = parser.parse_args()

    train_years_to_use = args.train_years if args.train_years else StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_years_to_use = args.test_years if args.test_years else StrikeoutModelConfig.DEFAULT_TEST_YEARS

    generate_features(
        prediction_date_str=args.prediction_date,
        # No longer passing output table names
        train_years=train_years_to_use if not args.prediction_date else None,
        test_years=test_years_to_use if not args.prediction_date else None
    )