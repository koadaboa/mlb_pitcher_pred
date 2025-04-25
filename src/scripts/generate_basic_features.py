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

# --- Setup Project Root ---
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config import DBConfig, LogConfig, StrikeoutModelConfig
    from src.data.utils import setup_logger, DBConnection
    from src.data.aggregate_statcast import aggregate_statcast_pitchers_sql, aggregate_statcast_batters_sql
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    MODULE_IMPORTS_OK = False
else:
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('generate_basic_features', LogConfig.LOG_DIR / 'generate_basic_features.log')

# --- Configuration ---
# *** MODIFIED: Removed 'woba' from TEAM_ROLLING_METRICS ***
PITCHER_ROLLING_METRICS = ['k_percent', 'swinging_strike_percent', 'avg_velocity']
TEAM_ROLLING_METRICS = ['k_percent', 'swinging_strike_percent'] # Removed 'woba'
BALLPARK_ROLLING_METRIC = 'k_percent'

ROLLING_WINDOW = 10
MIN_ROLLING_PERIODS = 5

# --- Helper Functions --- (Keep safe_division, load_data_from_db, calculate_rolling_features as before)
def safe_division(numerator, denominator, default=0.0):
    """Performs division, returning default value for division by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.where(denominator != 0, numerator / denominator, default)
    return result

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
            for col in df.select_dtypes(include=['int64']).columns:
                # Avoid downcasting potential large IDs if necessary
                # if col not in ['pitcher_id', 'game_pk']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
    except Exception as e:
        logger.error(f"Failed to load data with query: {query[:100]}... Error: {e}", exc_info=True)
    return df

def calculate_rolling_features(df, group_col, date_col, metrics, window, min_periods, shift_periods=1):
    """
    Calculates rolling features on a dataframe, grouped and sorted.
    Uses shift() to prevent data leakage. Returns results indexed like input df.
    """
    if df is None or df.empty:
        logger.warning(f"Input df is empty for rolling calculation on {group_col}.")
        return pd.DataFrame(index=df.index if df is not None else None) # Return empty df with index if possible

    logger.info(f"Calculating rolling features (window={window}, min_periods={min_periods}) for group '{group_col}' on metrics: {metrics}")
    # Sort inplace for efficiency if df is large, ensure it's not a slice
    df.sort_values(by=[group_col, date_col], inplace=True)

    results_df = pd.DataFrame(index=df.index) # Preserve original index

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe for rolling calculation.")
            results_df[f"{metric}_roll{window}g"] = np.nan # Add NaN column
            continue

        roll_col_name = f"{metric}_roll{window}g"
        # --- DATA LEAKAGE PREVENTION: shift(1) before rolling ---
        # Use transform to align results back to the original DataFrame's index
        # Handle potential non-numeric data before transform if necessary
        numeric_metric_series = pd.to_numeric(df[metric], errors='coerce')
        if numeric_metric_series.isnull().all():
             logger.warning(f"Metric '{metric}' is entirely non-numeric or NaN. Skipping rolling calculation.")
             results_df[roll_col_name] = np.nan
             continue

        results_df[roll_col_name] = df.groupby(group_col, observed=True)[metric].transform( # Use observed=True
            lambda x: x.shift(shift_periods).rolling(window=window, min_periods=min_periods).mean()
        )
        logger.debug(f"Calculated rolling feature: {roll_col_name}")

    return results_df


# --- Main Feature Generation Logic ---
def generate_features(prediction_date_str: str | None,
                        output_table_pred: str,
                        output_table_hist: str,
                        train_years: list[int] | None = None,
                        test_years: list[int] | None = None):
    """
    Generates basic features. Operates in prediction or historical mode.
    Includes calls to aggregation functions first.
    """
    mode = "PREDICTION" if prediction_date_str else "HISTORICAL"
    logger.info(f"--- Starting Basic Feature Generation [{mode} Mode] ---")
    if prediction_date_str:
        logger.info(f"Prediction Date: {prediction_date_str}")
    else:
        logger.info("Running for all historical data.")

    db_path = Path(DBConfig.PATH)
    prediction_date = None
    max_hist_date_str = '9999-12-31' # Default for historical runs

    if prediction_date_str:
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
            max_hist_date_str = (prediction_date - timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid prediction date format: {prediction_date_str}. Use YYYY-MM-DD.")
            return

    # --- STEP 0: Run Aggregations ---
    logger.info("STEP 0: Running Statcast Aggregations (Full History)...")
    try:
        aggregate_statcast_pitchers_sql(target_date=None)
        aggregate_statcast_batters_sql(target_date=None)
        logger.info("Aggregations completed.")
    except Exception as agg_e:
        logger.error(f"Error occurred during aggregation calls: {agg_e}", exc_info=True)
        return
    gc.collect()

    # --- STEP 1: Load Data ---
    logger.info("STEP 1: Loading necessary data...")
    start_load_time = datetime.now()

    # Load Team Mapping
    team_map_query = "SELECT team_abbr, ballpark FROM team_mapping"
    team_map_df = load_data_from_db(team_map_query, db_path, optimize=False)
    if not team_map_df.empty:
        team_to_ballpark_map = pd.Series(team_map_df.ballpark.values, index=team_map_df.team_abbr).to_dict()
    else:
        logger.warning("Team mapping table empty or not found. Ballpark features will be limited.")
        team_to_ballpark_map = {}

    # *** MODIFIED pitcher_hist_query: Removed 'team', added 'home_team', 'away_team' ***
    pitcher_hist_cols = ['pitcher_id', 'game_date', 'game_pk', 'p_throws', 'opponent_team', 'home_team', 'away_team', 'is_home', 'k_percent', 'swinging_strike_percent', 'avg_velocity', 'strikeouts', 'batters_faced']
    pitcher_hist_query = f"SELECT {', '.join(pitcher_hist_cols)} FROM game_level_pitchers WHERE DATE(game_date) <= '{max_hist_date_str}'"
    pitcher_hist_df = load_data_from_db(pitcher_hist_query, db_path)

    # *** MODIFIED team_hist_query: Removed 'woba' ***
    team_hist_cols = ['team', 'game_date', 'game_pk', 'home_team', 'k_percent', 'swinging_strike_percent'] # Removed woba
    team_hist_query = f"SELECT {', '.join(team_hist_cols)} FROM game_level_team_stats WHERE DATE(game_date) <= '{max_hist_date_str}'"
    team_hist_df = load_data_from_db(team_hist_query, db_path)

    if pitcher_hist_df.empty:
        logger.error("Historical pitcher stats are empty. Cannot proceed.")
        return
    if team_hist_df.empty:
        logger.warning("Historical team stats are empty. Opponent features will be limited.")

    # --- Add derived 'team' column to pitcher_hist_df ---
    # Needed for mapping pitcher to ballpark consistently
    try:
         pitcher_hist_df['team'] = np.where(pitcher_hist_df['is_home'] == 1, pitcher_hist_df['home_team'], pitcher_hist_df['away_team'])
         logger.info("Derived 'team' column for pitchers.")
    except KeyError as e:
         logger.error(f"Cannot derive pitcher 'team'. Missing required columns (is_home, home_team, away_team): {e}")
         # Decide how to handle - maybe skip ballpark features or exit?
         pitcher_hist_df['team'] = 'UNK' # Assign unknown as fallback


    # Add ballpark column to pitcher_hist_df
    if 'home_team' in pitcher_hist_df.columns: # Use home_team of the game for park context
        pitcher_hist_df['ballpark'] = pitcher_hist_df['home_team'].map(team_to_ballpark_map).fillna("Unknown Park")
    else:
        logger.warning("Missing 'home_team' in pitcher history, cannot map ballparks accurately.")
        pitcher_hist_df['ballpark'] = "Unknown Park"

    # Convert date columns to datetime
    pitcher_hist_df['game_date'] = pd.to_datetime(pitcher_hist_df['game_date'])
    if not team_hist_df.empty:
        team_hist_df['game_date'] = pd.to_datetime(team_hist_df['game_date'])

    logger.info(f"Data loading finished in {(datetime.now() - start_load_time).total_seconds():.2f}s.")
    gc.collect()

    # --- STEP 2 & 3: Calculate Historical Rolling Features ---
    logger.info("STEP 2&3: Calculating historical rolling features...")
    calc_start_time = datetime.now()

    # --- Pitcher Rolling ---
    pitcher_rolling_df = calculate_rolling_features(
        df=pitcher_hist_df, group_col='pitcher_id', date_col='game_date',
        metrics=PITCHER_ROLLING_METRICS, window=ROLLING_WINDOW, min_periods=MIN_ROLLING_PERIODS
    )
    p_roll_rename_dict = {f"{m}_roll{ROLLING_WINDOW}g": f"p_roll{ROLLING_WINDOW}g_{m}" for m in PITCHER_ROLLING_METRICS}
    pitcher_rolling_df = pitcher_rolling_df.rename(columns=p_roll_rename_dict)

    # --- Team Rolling ---
    team_rolling_df = pd.DataFrame(index=team_hist_df.index)
    t_roll_rename_dict = {f"{m}_roll{ROLLING_WINDOW}g": f"opp_roll{ROLLING_WINDOW}g_{m}" for m in TEAM_ROLLING_METRICS} # Use modified list
    if not team_hist_df.empty:
        team_rolling_calc = calculate_rolling_features(
            df=team_hist_df, group_col='team', date_col='game_date',
            metrics=TEAM_ROLLING_METRICS, window=ROLLING_WINDOW, min_periods=MIN_ROLLING_PERIODS # Use modified list
        )
        team_rolling_df = team_rolling_calc.rename(columns=t_roll_rename_dict)
        # Add 'team' and 'game_date' back for merging
        team_rolling_df[['team', 'game_date']] = team_hist_df[['team', 'game_date']]

    # --- Ballpark Rolling ---
    ballpark_rolling_df = pd.DataFrame(index=pitcher_hist_df.index)
    bp_roll_rename_dict = {f"{BALLPARK_ROLLING_METRIC}_roll{ROLLING_WINDOW}g": f"bp_roll{ROLLING_WINDOW}g_{BALLPARK_ROLLING_METRIC}"}
    if 'ballpark' in pitcher_hist_df.columns and BALLPARK_ROLLING_METRIC in pitcher_hist_df.columns: # Use pitcher K% aggregated by park
         ballpark_rolling_calc = calculate_rolling_features(
             df=pitcher_hist_df, group_col='ballpark', date_col='game_date',
             metrics=[BALLPARK_ROLLING_METRIC], window=ROLLING_WINDOW, min_periods=MIN_ROLLING_PERIODS
         )
         ballpark_rolling_calc = ballpark_rolling_calc.rename(columns=bp_roll_rename_dict)
         # Add the calculated column to ballpark_rolling_df (aligning by index)
         ballpark_rolling_df[list(bp_roll_rename_dict.values())[0]] = ballpark_rolling_calc[list(bp_roll_rename_dict.values())[0]]
         # Add 'ballpark' and 'game_date' back for merging
         ballpark_rolling_df[['ballpark', 'game_date']] = pitcher_hist_df[['ballpark', 'game_date']]
    else:
         logger.warning("Cannot calculate ballpark features, 'ballpark' or metric column missing.")


    # --- Pitcher Days Rest ---
    logger.info("Calculating pitcher days rest...")
    pitcher_hist_df = pitcher_hist_df.sort_values(by=['pitcher_id', 'game_date'])
    pitcher_hist_df['p_days_rest'] = pitcher_hist_df.groupby('pitcher_id')['game_date'].diff().dt.days

    logger.info(f"Feature calculation finished in {(datetime.now() - calc_start_time).total_seconds():.2f}s.")
    gc.collect()


    # --- STEP 4: Prepare Final DataFrame based on Mode ---
    final_features_df = pd.DataFrame()
    output_table_name = ""

    if mode == "HISTORICAL":
        logger.info("STEP 4 [HISTORICAL]: Merging features for all historical data...")
        output_table_name = output_table_hist

        # Start with the original pitcher history
        base_cols = ['game_pk', 'game_date', 'pitcher_id', 'team', 'opponent_team', 'is_home', 'ballpark', 'p_throws', 'p_days_rest', 'strikeouts', 'batters_faced'] # Use derived team
        final_features_df = pitcher_hist_df[[col for col in base_cols if col in pitcher_hist_df.columns]].copy()

        # Merge Pitcher Rolling Features (index already aligned)
        final_features_df = pd.concat([final_features_df, pitcher_rolling_df], axis=1)

        # Merge Opponent Rolling Features (Time-Aligned)
        if not team_rolling_df.empty:
            logger.info("Performing time-aligned merge for opponent features...")
            final_features_df = final_features_df.sort_values(by='game_date')
            team_rolling_df = team_rolling_df.sort_values(by='game_date')
            # Ensure keys are correct type before merge
            final_features_df['opponent_team'] = final_features_df['opponent_team'].astype(str)
            team_rolling_df['team'] = team_rolling_df['team'].astype(str)
            final_features_df = pd.merge_asof(
                final_features_df,
                team_rolling_df, # Includes renamed opp_roll features
                on='game_date',
                left_by='opponent_team',
                right_by='team',
                direction='backward', allow_exact_matches=False,
                suffixes=('', '_opp_roll')
            )
            final_features_df = final_features_df.drop(columns=['team_opp_roll'], errors='ignore')
            logger.info("Opponent feature merge complete.")
        else:
             for col in t_roll_rename_dict.values(): final_features_df[col] = np.nan

        # Merge Ballpark Rolling Features (Time-Aligned)
        if not ballpark_rolling_df.empty and list(bp_roll_rename_dict.values())[0] in ballpark_rolling_df.columns:
            logger.info("Performing time-aligned merge for ballpark features...")
            final_features_df = final_features_df.sort_values(by='game_date')
            ballpark_rolling_df = ballpark_rolling_df.sort_values(by='game_date')
            # Ensure keys are correct type before merge
            final_features_df['ballpark'] = final_features_df['ballpark'].astype(str)
            ballpark_rolling_df['ballpark'] = ballpark_rolling_df['ballpark'].astype(str)
            final_features_df = pd.merge_asof(
                final_features_df,
                ballpark_rolling_df,
                on='game_date', by='ballpark',
                direction='backward', allow_exact_matches=False,
                suffixes=('', '_bp_roll')
            )
            logger.info("Ballpark feature merge complete.")
        else:
            for col in bp_roll_rename_dict.values(): final_features_df[col] = np.nan

        # Add season column for potential splitting
        final_features_df['season'] = pd.to_datetime(final_features_df['game_date']).dt.year


    elif mode == "PREDICTION":
        logger.info("STEP 4 [PREDICTION]: Merging latest features onto prediction baseline...")
        output_table_name = output_table_pred

        # Load schedule
        schedule_query = f"SELECT * FROM mlb_api WHERE DATE(game_date) = '{prediction_date_str}'"
        schedule_df = load_data_from_db(schedule_query, db_path, optimize=False)
        if schedule_df.empty: logger.error("Prediction schedule missing."); return

        baseline_data = []
        # Rebuild baseline for prediction date
        for _, game in schedule_df.iterrows():
            game_date_str_pred = pd.to_datetime(game['game_date']).strftime('%Y-%m-%d')
            home_pid = pd.to_numeric(game.get('home_probable_pitcher_id'), errors='coerce')
            away_pid = pd.to_numeric(game.get('away_probable_pitcher_id'), errors='coerce')
            home_team_abbr = game.get('home_team_abbr')
            away_team_abbr = game.get('away_team_abbr')
            ballpark = team_to_ballpark_map.get(home_team_abbr, "Unknown Park")

            # Need pitcher's actual team derived for mapping later if needed
            # This baseline only needs pitcher_id, opponent, ballpark, is_home for merging
            if pd.notna(home_pid):
                baseline_data.append({
                    'pitcher_id': int(home_pid), 'game_date': game_date_str_pred,
                    'game_pk': game.get('game_pk'), #'team': home_team_abbr, # Team not strictly needed here
                    'opponent_team': away_team_abbr, 'is_home': 1, 'ballpark': ballpark
                })
            if pd.notna(away_pid):
                baseline_data.append({
                    'pitcher_id': int(away_pid), 'game_date': game_date_str_pred,
                    'game_pk': game.get('game_pk'), #'team': away_team_abbr,
                    'opponent_team': home_team_abbr, 'is_home': 0, 'ballpark': ballpark
                })
        if not baseline_data: logger.error("No probable pitchers for prediction baseline."); return
        final_features_df = pd.DataFrame(baseline_data)
        final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date'])

        # Extract latest values from rolling calculations
        latest_pitcher_rolling = pitcher_rolling_df.sort_values('game_date').drop_duplicates(subset=['pitcher_id'], keep='last')
        latest_team_rolling = team_rolling_df.sort_values('game_date').drop_duplicates(subset=['team'], keep='last')
        latest_ballpark_rolling = ballpark_rolling_df.sort_values('game_date').drop_duplicates(subset=['ballpark'], keep='last')

        # Calculate Days Rest relative to prediction date
        last_game_dates = pitcher_hist_df.sort_values('game_date').drop_duplicates(subset=['pitcher_id'], keep='last')[['pitcher_id', 'game_date']]
        final_features_df = pd.merge(final_features_df, last_game_dates.rename(columns={'game_date':'last_game_date'}), on='pitcher_id', how='left')
        final_features_df['last_game_date'] = pd.to_datetime(final_features_df['last_game_date'])
        final_features_df['p_days_rest'] = (final_features_df['game_date'] - final_features_df['last_game_date']).dt.days
        final_features_df = final_features_df.drop(columns=['last_game_date'])

        # Get Pitcher Handedness
        pitcher_throws = pitcher_hist_df.dropna(subset=['p_throws']).drop_duplicates(subset=['pitcher_id'], keep='last')[['pitcher_id', 'p_throws']]

        # --- Merge Latest Features ---
        p_cols_to_merge = ['pitcher_id'] + list(p_roll_rename_dict.values())
        if len(p_cols_to_merge) > 1:
            final_features_df = pd.merge(final_features_df, latest_pitcher_rolling[p_cols_to_merge], on='pitcher_id', how='left')
        if not pitcher_throws.empty:
            final_features_df = pd.merge(final_features_df, pitcher_throws, on='pitcher_id', how='left')
        else: final_features_df['p_throws'] = 'R'

        if not latest_team_rolling.empty and list(t_roll_rename_dict.values()):
            opp_cols_to_merge = ['team'] + list(t_roll_rename_dict.values()) # Use RENAMED cols
            final_features_df = pd.merge(
                final_features_df, latest_team_rolling[opp_cols_to_merge],
                left_on='opponent_team', right_on='team', how='left', suffixes=('', '_opp')
            )
            final_features_df = final_features_df.drop(columns=['team'], errors='ignore') # Drop joined 'team' col
        else:
            for col in t_roll_rename_dict.values(): final_features_df[col] = np.nan

        if not latest_ballpark_rolling.empty and list(bp_roll_rename_dict.values()):
            bp_cols_to_merge = ['ballpark'] + list(bp_roll_rename_dict.values()) # Use RENAMED cols
            final_features_df = pd.merge(
                final_features_df, latest_ballpark_rolling[bp_cols_to_merge],
                on='ballpark', how='left'
            )
        else:
            for col in bp_roll_rename_dict.values(): final_features_df[col] = np.nan

        # Convert game_date back to string for saving
        final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date']).dt.strftime('%Y-%m-%d')


    # --- STEP 5: Final Cleanup & Impute Missing Values ---
    logger.info("STEP 5: Cleaning up and imputing missing values...")
    if final_features_df.empty:
        logger.error("Feature DataFrame is empty before imputation.")
        return

    # Define expected columns (adjust based on features actually created)
    expected_cols = [
         'game_pk', 'game_date', 'pitcher_id', 'opponent_team', 'is_home', 'ballpark', 'p_throws', 'p_days_rest'
    ]
    if mode == "HISTORICAL": expected_cols.extend(['strikeouts', 'batters_faced', 'season'])
    expected_cols.extend(list(p_roll_rename_dict.values()))
    expected_cols.extend(list(t_roll_rename_dict.values()))
    expected_cols.extend(list(bp_roll_rename_dict.values()))

    # Add missing expected columns with NaN
    for col in expected_cols:
        if col not in final_features_df.columns:
            final_features_df[col] = np.nan

    # Select only expected columns in desired order (handle missing ones added above)
    final_features_df = final_features_df[[col for col in expected_cols if col in final_features_df.columns]]


    # Impute numeric columns
    numeric_cols = final_features_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if final_features_df[col].isnull().any():
            median_val = final_features_df[col].median()
            fill_val = median_val if pd.notna(median_val) else 0
            nan_count = final_features_df[col].isnull().sum()
            logger.info(f"Imputing {nan_count} NaNs in '{col}' with median/0 ({fill_val:.3f}).")
            final_features_df[col] = final_features_df[col].fillna(fill_val)

    # Fill categorical NaNs
    if 'p_throws' in final_features_df.columns: final_features_df['p_throws'] = final_features_df['p_throws'].fillna('R')
    if 'ballpark' in final_features_df.columns: final_features_df['ballpark'] = final_features_df['ballpark'].fillna('Unknown Park')

    # Ensure correct types before saving
    final_features_df['game_date'] = pd.to_datetime(final_features_df['game_date']).dt.strftime('%Y-%m-%d')
    final_features_df['pitcher_id'] = pd.to_numeric(final_features_df['pitcher_id'], errors='coerce').astype('Int64')
    if 'game_pk' in final_features_df.columns: final_features_df['game_pk'] = pd.to_numeric(final_features_df['game_pk'], errors='coerce').astype('Int64')
    if 'is_home' in final_features_df.columns: final_features_df['is_home'] = pd.to_numeric(final_features_df['is_home'], errors='coerce').astype('Int8')
    if 'season' in final_features_df.columns: final_features_df['season'] = pd.to_numeric(final_features_df['season'], errors='coerce').astype('Int64')


    # --- STEP 6: Save Results ---
    logger.info(f"STEP 6: Saving features to table '{output_table_name}'...")
    try:
        if mode == "HISTORICAL" and train_years and test_years:
            logger.info(f"Splitting historical data into train ({train_years}) and test ({test_years})...")
            if 'season' not in final_features_df.columns:
                logger.warning("'season' column missing, cannot split by year. Saving all historical data.")
                with DBConnection(db_path) as conn:
                     final_features_df.to_sql(output_table_name, conn, if_exists='replace', index=False)
                logger.info(f"Successfully saved {len(final_features_df)} rows to '{output_table_name}'.")
            else:
                train_df = final_features_df[final_features_df['season'].isin(train_years)].copy()
                test_df = final_features_df[final_features_df['season'].isin(test_years)].copy()
                output_table_train = output_table_hist + "_train"
                output_table_test = output_table_hist + "_test"
                with DBConnection(db_path) as conn:
                    train_df.to_sql(output_table_train, conn, if_exists='replace', index=False)
                    logger.info(f"Saved {len(train_df)} training rows to '{output_table_train}'.")
                    test_df.to_sql(output_table_test, conn, if_exists='replace', index=False)
                    logger.info(f"Saved {len(test_df)} test rows to '{output_table_test}'.")
        else:
            with DBConnection(db_path) as conn:
                final_features_df.to_sql(output_table_name, conn, if_exists='replace', index=False)
            logger.info(f"Successfully saved {len(final_features_df)} rows with {len(final_features_df.columns)} columns to '{output_table_name}'.")
            logger.debug(f"Final columns: {final_features_df.columns.tolist()}")

    except Exception as e:
        logger.error(f"Failed to save features to database: {e}", exc_info=True)

    gc.collect()
    logger.info(f"--- Basic Feature Generation [{mode} Mode] Completed ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK:
        sys.exit("Exiting: Failed required module imports.")

    parser = argparse.ArgumentParser(description="Generate Basic MLB Features for Training/Prediction.")
    parser.add_argument("--prediction-date", type=str, default=None,
                        help="Generate features for a specific prediction date (YYYY-MM-DD). If omitted, generates full historical features.")
    parser.add_argument("--output-table-pred", type=str, default="prediction_features_basic",
                        help="Name of table to save prediction features to.")
    parser.add_argument("--output-table-hist", type=str, default="historical_features_basic",
                        help="Name of table to save historical features to (or prefix if splitting).")
    parser.add_argument("--train-years", type=int, nargs='+', default=None,
                        help="List of years for training set (e.g., 2021 2022). Overrides config.")
    parser.add_argument("--test-years", type=int, nargs='+', default=None,
                        help="List of years for test set (e.g., 2023). Overrides config.")
    args = parser.parse_args()

    train_years_to_use = args.train_years if args.train_years else StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_years_to_use = args.test_years if args.test_years else StrikeoutModelConfig.DEFAULT_TEST_YEARS

    generate_features(
        prediction_date_str=args.prediction_date,
        output_table_pred=args.output_table_pred,
        output_table_hist=args.output_table_hist,
        train_years=train_years_to_use if not args.prediction_date else None,
        test_years=test_years_to_use if not args.prediction_date else None
    )