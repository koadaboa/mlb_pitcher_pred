# src/scripts/generate_features.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import argparse

# --- Project Setup ---
# Assuming the script is in src/scripts, and project root is two levels up
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Imports from project modules ---
try:
    from src import config
    from src.data.utils import DBConnection, setup_logger, ensure_dir
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
    from src.features.umpire_features import calculate_umpire_rolling_features # Updated version
    from src.features.predictive_umpire_assignment import predict_home_plate_umpire
    # For fetching today's games for prediction (if applicable)
    from src.data.mlb_api import scrape_probable_pitchers
except ImportError as e:
    print(f"ERROR: Failed to import one or more project modules: {e}")
    print("Ensure your PYTHONPATH is set correctly or run from the project root.")
    sys.exit(1)

# --- Logger Setup ---
log_dir = Path(config.LogConfig.LOG_DIR) if hasattr(config, 'LogConfig') else project_root / "logs"
ensure_dir(log_dir)
logger = setup_logger('generate_features', log_file=log_dir / 'generate_features.log')

# --- Configuration Constants (examples, adjust from your config.py or define here) ---
# These would ideally come from config.StrikeoutModelConfig or similar
WINDOW_SIZES = getattr(config.StrikeoutModelConfig, 'WINDOW_SIZES', [5, 10, 25, 50])
MIN_PERIODS_DEFAULT = getattr(config.StrikeoutModelConfig, 'MIN_PERIODS_DEFAULT', 3) # Example
TARGET_VARIABLE = getattr(config.StrikeoutModelConfig, 'TARGET_VARIABLE', 'strikeouts_recorded') # from pitcher_game_stats

# Define metrics for rolling features (examples, customize as needed)
# These are the raw metrics from game_level_aggregates or game_level_team_stats
PITCHER_METRICS_FOR_ROLLING = [
    'k_percent', 'bb_percent', 'woba_conceded', 'iso_conceded', 'babip_conceded',
    'avg_release_speed', 'swinging_strike_percent', 'csw_percent', 'fps_percent',
    'fastball_percent', 'slider_percent', 'curveball_percent', 'changeup_percent' # Add if available
]
OPPONENT_BATTING_METRICS_FOR_ROLLING = [ # Stats for the opposing team (as batters)
    'k_percent_bat', 'bb_percent_bat', 'woba_bat', 'iso_bat', 'babip_bat',
    'swinging_strike_percent_bat', 'csw_percent_bat',
    # Platoon splits should be pre-calculated in game_level_team_stats if used directly for rolling
    # e.g., 'k_percent_bat_vs_LHP', 'k_percent_bat_vs_RHP'
]
BALLPARK_METRICS_FOR_ROLLING = [ # Metrics observed in games at a specific ballpark
    'k_percent', 'woba_conceded', 'iso_conceded', 'runs_scored_per_game' # Example
]
UMPIRE_METRICS_FOR_ROLLING = [ # Pitcher metrics observed in games officiated by a specific umpire
    'k_percent', 'bb_percent', 'called_strike_plus_whiff_rate_ump' # Example
]

# Output configuration
FEATURES_OUTPUT_DIR = project_root / "data" / "features"
ensure_dir(FEATURES_OUTPUT_DIR)
HISTORICAL_FEATURES_FILE = FEATURES_OUTPUT_DIR / "historical_features.parquet"
PREDICTION_FEATURES_FILE = FEATURES_OUTPUT_DIR / "prediction_features.parquet"


# --- Helper Function: Multi-Window Rolling Calculation ---
def calculate_multi_window_rolling_expanded(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int = 1,
    lag: int = 1 # Lag features by 1 to prevent data leakage (use previous games' data)
) -> pd.DataFrame:
    """
    Calculates rolling means for multiple metrics over multiple windows,
    grouped by `group_col` and sorted by `date_col`. Lags results.

    Args:
        df: Input DataFrame.
        group_col: Column to group by (e.g., 'pitcher_id', 'team', 'ballpark').
        date_col: Date column for sorting.
        metrics: List of metric columns to calculate rolling stats for.
        windows: List of window sizes (e.g., [5, 10, 25]).
        min_periods: Minimum number of observations in window required.
        lag: Number of periods to shift the rolling features. Default is 1.

    Returns:
        DataFrame with new rolling feature columns. Original index is preserved.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Sort by group and date to ensure correct rolling calculation and shift
    df_copy = df_copy.sort_values(by=[group_col, date_col])

    all_rolling_features = []

    for metric in metrics:
        if metric not in df_copy.columns:
            logger.warning(f"Metric '{metric}' not found in DataFrame for rolling calculation. Skipping.")
            continue
        for window in windows:
            roll_col_name = f"{metric}_roll{window}g"
            try:
                # Calculate rolling mean within each group
                grouped = df_copy.groupby(group_col, observed=True)[metric]
                rolling_mean = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=min_periods).mean()
                )
                # Lag the feature
                lagged_rolling_mean = rolling_mean.groupby(df_copy[group_col], observed=True).shift(lag)
                
                df_copy[roll_col_name] = lagged_rolling_mean
                all_rolling_features.append(roll_col_name)
            except Exception as e:
                logger.error(f"Error calculating rolling feature for {metric} with window {window}: {e}", exc_info=True)
                df_copy[roll_col_name] = np.nan # Add NaN column on error

    # Return only the newly created rolling columns, aligned with the original DataFrame's index
    if not all_rolling_features: # No features created
        return pd.DataFrame(index=df.index)
        
    return df_copy[all_rolling_features].reindex(df.index)


# --- Data Loading Functions ---
def load_game_level_aggregates() -> pd.DataFrame:
    """Loads the main enriched game-level data."""
    table_name = config.PITCHER_GAME_STATS_TABLE # This table now contains enriched game data
    logger.info(f"Loading game-level aggregates from '{table_name}'...")
    try:
        with DBConnection() as conn:
            # Check if table exists
            query_exists = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
            if pd.read_sql_query(query_exists, conn).empty:
                logger.error(f"Table '{table_name}' does not exist in the database.")
                return pd.DataFrame()
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        logger.info(f"Loaded {len(df)} rows from '{table_name}'.")
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        return df
    except Exception as e:
        logger.error(f"Error loading data from '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

def load_team_game_stats() -> pd.DataFrame:
    """Loads game-level team stats (primarily for opponent features)."""
    # Assuming a table like 'game_level_team_stats' exists or will be created.
    # This table should contain team batting performance per game.
    table_name = getattr(config, 'TEAM_GAME_STATS_TABLE', 'game_level_team_stats') # Example name
    logger.info(f"Loading team game stats from '{table_name}'...")
    try:
        with DBConnection() as conn:
            query_exists = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
            if pd.read_sql_query(query_exists, conn).empty:
                logger.warning(f"Team game stats table '{table_name}' does not exist or is empty. Opponent features might be limited.")
                return pd.DataFrame()
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        logger.info(f"Loaded {len(df)} rows from '{table_name}'.")
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        # Ensure a 'team' column exists for grouping (e.g., team abbreviation)
        if 'team' not in df.columns and 'team_abbr' in df.columns: # Common case
            df = df.rename(columns={'team_abbr': 'team'})
        elif 'team' not in df.columns:
            logger.error(f"Missing 'team' or 'team_abbr' column in '{table_name}' for opponent features.")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error loading data from '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

def load_historical_umpire_assignments() -> pd.DataFrame:
    """Loads historical full umpire crew assignments from mlb_boxscores."""
    table_name = config.MLB_BOXSCORES_TABLE
    logger.info(f"Loading historical umpire assignments from '{table_name}' for prediction...")
    cols_needed = ['game_pk', 'game_date', 'home_team', 'away_team',
                   'hp_umpire', '1b_umpire', '2b_umpire', '3b_umpire']
    try:
        with DBConnection() as conn:
            query_exists = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
            if pd.read_sql_query(query_exists, conn).empty:
                logger.error(f"Table '{table_name}' does not exist. Cannot load umpire assignments.")
                return pd.DataFrame()
            # Check for required columns
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            available_cols = [info[1] for info in cursor.fetchall()]
            if not all(col in available_cols for col in cols_needed):
                logger.error(f"Table '{table_name}' is missing some required umpire columns. Needed: {cols_needed}, Available: {available_cols}")
                return pd.DataFrame()

            df = pd.read_sql_query(f"SELECT {', '.join(cols_needed)} FROM {table_name}", conn)
        logger.info(f"Loaded {len(df)} rows from '{table_name}' for umpire assignments.")
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        return df
    except Exception as e:
        logger.error(f"Error loading data from '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

# --- Main Historical Feature Generation Function ---
def generate_historical_features(start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> pd.DataFrame:
    """
    Generates historical features for model training.
    """
    logger.info("Starting generation of historical features...")

    game_data_df = load_game_level_aggregates()
    if game_data_df.empty:
        logger.error("Game level aggregates data is empty. Cannot generate historical features.")
        return pd.DataFrame()

    team_game_stats_df = load_team_game_stats()
    # Opponent features can still be generated if team_game_stats_df is empty, they just won't merge.

    # Filter by date range if provided
    if start_date_str:
        game_data_df = game_data_df[game_data_df['game_date'] >= pd.to_datetime(start_date_str)]
    if end_date_str:
        game_data_df = game_data_df[game_data_df['game_date'] <= pd.to_datetime(end_date_str)]

    if game_data_df.empty:
        logger.error("Game level aggregates data is empty after date filtering.")
        return pd.DataFrame()

    # Ensure essential columns are present
    # 'pitcher' (ID), 'game_date', 'ballpark', 'home_plate_umpire', 'opponent_team', 'p_throws'
    # The target variable also needs to be present, e.g., 'strikeouts_recorded'
    required_base_cols = ['game_pk', 'pitcher', 'game_date', 'ballpark', 'home_plate_umpire', 
                            'opponent_team', 'p_throws', TARGET_VARIABLE]
    missing_base_cols = [col for col in required_base_cols if col not in game_data_df.columns]
    if missing_base_cols:
        logger.error(f"Missing essential base columns in game_data_df: {missing_base_cols}. Cannot proceed.")
        return pd.DataFrame()

    # Sort data for rolling calculations and consistent merging
    game_data_df = game_data_df.sort_values(by=['pitcher', 'game_date']).reset_index(drop=True)
    
    # Initialize final features DataFrame with essential ID and date columns + target
    # Also include all raw boxscore features that are already in game_data_df
    # Identify boxscore feature columns (this is an example list, adjust based on boxscore_features.py output)
    boxscore_feature_cols = [
        'temperature', 'is_night_game', 'park_elevation', 'weather_condition_simplified',
        'is_precipitation', 'is_dome_weather', 'wind_speed_mph', 'wind_speed_category',
        'is_dome_wind', 'wind_blowing_out', 'wind_blowing_in',
        'wind_blowing_across_L_to_R', 'wind_blowing_across_R_to_L',
        'wind_direction_varies_or_unknown'
    ]
    # Select only existing boxscore columns and ensure no duplicates with required_base_cols
    actual_boxscore_cols = [col for col in boxscore_feature_cols if col in game_data_df.columns]
    initial_cols_to_keep = list(set(required_base_cols + actual_boxscore_cols))
    final_features_df = game_data_df[initial_cols_to_keep].copy()


    # 1. Pitcher Features
    logger.info("Calculating pitcher features...")
    # Filter PITCHER_METRICS_FOR_ROLLING to those present in game_data_df
    valid_pitcher_metrics = [m for m in PITCHER_METRICS_FOR_ROLLING if m in game_data_df.columns]
    if valid_pitcher_metrics:
        pitcher_roll_df = calculate_pitcher_rolling_features(
            df=game_data_df, # game_data_df contains pitcher stats per game
            group_col='pitcher',
            date_col='game_date',
            metrics=valid_pitcher_metrics,
            windows=WINDOW_SIZES,
            min_periods=MIN_PERIODS_DEFAULT,
            calculate_multi_window_rolling=calculate_multi_window_rolling_expanded
        )
        if not pitcher_roll_df.empty:
            final_features_df = pd.concat([final_features_df, pitcher_roll_df], axis=1)

    pitcher_rest_s = calculate_pitcher_rest_days(game_data_df) # Uses 'pitcher', 'game_date'
    if not pitcher_rest_s.empty:
        final_features_df['days_rest'] = pitcher_rest_s
    
    # 2. Opponent Features
    logger.info("Calculating opponent features...")
    if not team_game_stats_df.empty:
        valid_opponent_metrics = [m for m in OPPONENT_BATTING_METRICS_FOR_ROLLING if m in team_game_stats_df.columns]
        if 'team' not in team_game_stats_df.columns:
             logger.warning("Skipping opponent features: 'team' column missing in team_game_stats_df.")
        elif valid_opponent_metrics:
            opponent_rolling_df, opp_rename_map = calculate_opponent_rolling_features(
                team_hist_df=team_game_stats_df, # This df has team batting stats
                group_col='team', # Group by team
                date_col='game_date',
                metrics=valid_opponent_metrics,
                windows=WINDOW_SIZES,
                min_periods=MIN_PERIODS_DEFAULT,
                calculate_multi_window_rolling=calculate_multi_window_rolling_expanded
            )
            # Merge opponent features using historical merge (merge_asof)
            # This requires 'opponent_team' in final_features_df and 'team', 'game_date' in opponent_rolling_df
            if not opponent_rolling_df.empty:
                final_features_df = merge_opponent_features_historical(
                    final_features_df, opponent_rolling_df, opp_rename_map
                )
        else:
            logger.warning("No valid opponent metrics found in team_game_stats_df.")
    else:
        logger.warning("Team game stats DataFrame is empty. Skipping opponent features.")

    # 3. Ballpark Features
    logger.info("Calculating ballpark features...")
    # Ballpark features are rolled based on metrics observed in that park from game_data_df
    valid_ballpark_metrics = [m for m in BALLPARK_METRICS_FOR_ROLLING if m in game_data_df.columns]
    if 'ballpark' not in game_data_df.columns:
        logger.warning("Skipping ballpark features: 'ballpark' column missing in game_data_df.")
    elif valid_ballpark_metrics:
        ballpark_rolling_df, bpark_rename_map = calculate_ballpark_rolling_features(
            pitcher_hist_df=game_data_df, # Use main game data
            group_col='ballpark',
            date_col='game_date',
            metrics=valid_ballpark_metrics,
            windows=WINDOW_SIZES,
            min_periods=MIN_PERIODS_DEFAULT,
            calculate_multi_window_rolling=calculate_multi_window_rolling_expanded
        )
        if not ballpark_rolling_df.empty:
            final_features_df = merge_ballpark_features_historical(
                final_features_df, ballpark_rolling_df, bpark_rename_map
            )
    else:
        logger.warning("No valid ballpark metrics found in game_data_df.")

    # 4. Umpire Features
    logger.info("Calculating umpire features...")
    # Umpire features are rolled based on metrics observed with that umpire from game_data_df
    valid_umpire_metrics = [m for m in UMPIRE_METRICS_FOR_ROLLING if m in game_data_df.columns]
    if 'home_plate_umpire' not in game_data_df.columns:
        logger.warning("Skipping umpire features: 'home_plate_umpire' column missing in game_data_df.")
    elif valid_umpire_metrics:
        umpire_rolling_df, ump_rename_map = calculate_umpire_rolling_features(
            main_game_df=game_data_df, # Use main game data
            group_col='home_plate_umpire',
            date_col='game_date',
            metrics=valid_umpire_metrics,
            windows=WINDOW_SIZES,
            min_periods=MIN_PERIODS_DEFAULT,
            calculate_multi_window_rolling=calculate_multi_window_rolling_expanded
        )
        if not umpire_rolling_df.empty:
            # Umpire features are calculated per game, so a direct concat/merge on index should work
            # if calculate_umpire_rolling_features returns index-aligned features
            final_features_df = pd.concat([final_features_df, umpire_rolling_df], axis=1)
    else:
        logger.warning("No valid umpire metrics found in game_data_df.")

    # Final processing
    # Drop rows where target is NaN (cannot be used for training)
    final_features_df = final_features_df.dropna(subset=[TARGET_VARIABLE])
    # Drop rows with too many NaNs in features (optional, based on strategy)
    # Example: final_features_df = final_features_df.dropna(thresh=len(final_features_df.columns) - 10) 

    logger.info(f"Finished generating historical features. Shape: {final_features_df.shape}")
    
    # Save features
    try:
        final_features_df.to_parquet(HISTORICAL_FEATURES_FILE, index=False)
        logger.info(f"Historical features saved to {HISTORICAL_FEATURES_FILE}")
    except Exception as e:
        logger.error(f"Error saving historical features to {HISTORICAL_FEATURES_FILE}: {e}", exc_info=True)
        
    return final_features_df


# --- Prediction Feature Generation Function ---
def generate_prediction_baseline_features(prediction_date_str: str) -> pd.DataFrame:
    """
    Generates features for making predictions on a specific date.
    """
    logger.info(f"Starting generation of features for prediction date: {prediction_date_str}")
    prediction_date = pd.to_datetime(prediction_date_str)

    # 1. Load all historical engineered features (or derive latest stats differently)
    # For simplicity, we load the full historical set and will get latest values from it.
    # Alternatively, save latest rolling stats per entity during historical generation.
    if not HISTORICAL_FEATURES_FILE.exists():
        logger.error(f"Historical features file not found: {HISTORICAL_FEATURES_FILE}. Run historical generation first.")
        return pd.DataFrame()
    all_historical_features_df = pd.read_parquet(HISTORICAL_FEATURES_FILE)
    all_historical_features_df['game_date'] = pd.to_datetime(all_historical_features_df['game_date'])

    # 2. Fetch today's games and probable starters
    # This uses scrape_probable_pitchers from mlb_api.py
    # It returns: game_date, game_pk, home_team_abbr, away_team_abbr,
    #            home_probable_pitcher_name, home_probable_pitcher_id,
    #            away_probable_pitcher_name, away_probable_pitcher_id
    logger.info(f"Fetching probable pitchers for prediction date: {prediction_date_str} using mlb_api...")
    todays_games_raw = scrape_probable_pitchers(prediction_date_str) # From mlb_api.py
    if not todays_games_raw:
        logger.warning(f"No games with probable pitchers found for {prediction_date_str}.")
        return pd.DataFrame()
    
    todays_games_df = pd.DataFrame(todays_games_raw)
    # We need to create one row per *starting pitcher* we are predicting for
    home_starters = todays_games_df[['game_pk', 'game_date', 'home_team_abbr', 'away_team_abbr', 'home_probable_pitcher_id', 'home_probable_pitcher_name']].copy()
    home_starters = home_starters.rename(columns={
        'home_team_abbr': 'team', 'away_team_abbr': 'opponent_team',
        'home_probable_pitcher_id': 'pitcher', 'home_probable_pitcher_name': 'pitcher_name'
    })
    home_starters['is_home_pitcher'] = 1

    away_starters = todays_games_df[['game_pk', 'game_date', 'away_team_abbr', 'home_team_abbr', 'away_probable_pitcher_id', 'away_probable_pitcher_name']].copy()
    away_starters = away_starters.rename(columns={
        'away_team_abbr': 'team', 'home_team_abbr': 'opponent_team',
        'away_probable_pitcher_id': 'pitcher', 'away_probable_pitcher_name': 'pitcher_name'
    })
    away_starters['is_home_pitcher'] = 0
    
    prediction_baseline_df = pd.concat([home_starters, away_starters], ignore_index=True)
    prediction_baseline_df['game_date'] = pd.to_datetime(prediction_baseline_df['game_date'])
    prediction_baseline_df = prediction_baseline_df.dropna(subset=['pitcher']) # Ensure pitcher ID is present
    prediction_baseline_df['pitcher'] = prediction_baseline_df['pitcher'].astype(int)


    # 3. Fetch today's game conditions (weather, wind, temp, ballpark, **umpire**)
    # This is the challenging part for live data.
    # For now, we'll assume these need to be fetched/predicted and added.
    # We'll load the full mlb_boxscores to get historical umpire assignments for prediction.
    historical_umpire_assignments_df = load_historical_umpire_assignments()

    # For each game in prediction_baseline_df, try to get/predict umpire and other conditions
    # This part would involve calling a weather API, and our new umpire predictor.
    # For simplicity in this example, we'll add placeholder columns.
    # In a real scenario, you'd populate these from live sources or predictions.
    
    temp_boxscore_features_for_today = []
    for idx, row in prediction_baseline_df.iterrows():
        game_info_for_ump_pred = {
            'game_pk': row['game_pk'],
            'game_date': row['game_date'], # pd.Timestamp
            'home_team': row['team'] if row['is_home_pitcher'] else row['opponent_team'],
            'away_team': row['opponent_team'] if row['is_home_pitcher'] else row['team']
        }
        
        # Predict HP Umpire
        predicted_hp_ump = None
        if not historical_umpire_assignments_df.empty:
            predicted_hp_ump = predict_home_plate_umpire(game_info_for_ump_pred, historical_umpire_assignments_df)
        
        # Fetch other boxscore info (e.g., from a pre-populated daily file or API)
        # This is a placeholder for actual data fetching for today's conditions
        # For now, we'll use some defaults or NaNs.
        # In a real pipeline, you'd query your `mlb_boxscores` table for today's game_pk
        # if it gets updated pre-game, or use another source.
        # Let's assume `fetch_live_boxscore_info(game_pk)` exists.
        # live_conditions = fetch_live_boxscore_info(row['game_pk']) # Hypothetical
        
        temp_data = {
            'game_pk': row['game_pk'],
            'home_plate_umpire': predicted_hp_ump, # From our predictor
            'ballpark': config.TEAM_BALLPARK_MAP.get(game_info_for_ump_pred['home_team'], "Unknown Park"), # Needs TEAM_BALLPARK_MAP in config
            'p_throws': config.PITCHER_HAND_MAP.get(row['pitcher'], "R"), # Needs PITCHER_HAND_MAP or fetch live
            # Add other boxscore features - these would ideally come from a live source for today
            'temperature': 70, 'is_night_game': 1, 'park_elevation': 500,
            'weather_condition_simplified': "Clear", 'is_precipitation': 0, 'is_dome_weather': 0,
            'wind_speed_mph': 5, 'wind_speed_category': "Light Breeze", 'is_dome_wind': 0,
            'wind_blowing_out': 0, 'wind_blowing_in': 0, 'wind_blowing_across_L_to_R': 0,
            'wind_blowing_across_R_to_L': 0, 'wind_direction_varies_or_unknown': 1
        }
        temp_boxscore_features_for_today.append(temp_data)

    if not temp_boxscore_features_for_today:
        logger.warning("Could not generate any boxscore context for today's games.")
        return pd.DataFrame()
        
    today_boxscore_context_df = pd.DataFrame(temp_boxscore_features_for_today)
    prediction_baseline_df = pd.merge(prediction_baseline_df, today_boxscore_context_df, on='game_pk', how='left')

    # 4. Get Latest Rolling Stats for each entity
    # Pitcher stats
    latest_pitcher_stats = all_historical_features_df[
        all_historical_features_df['game_date'] < prediction_date
    ].sort_values('game_date').groupby('pitcher', observed=True).last()
    # Select only pitcher rolling cols (prefixed with p_) and days_rest
    pitcher_cols_to_merge = ['days_rest'] + [col for col in latest_pitcher_stats.columns if col.startswith('p_roll')]
    prediction_baseline_df = pd.merge(
        prediction_baseline_df,
        latest_pitcher_stats[pitcher_cols_to_merge].reset_index(), # Reset index to merge on 'pitcher'
        on='pitcher',
        how='left'
    )
    # For days_rest, it's for the pitcher's last game. For today, it's game_date - last_game_date.
    # This requires pitcher's last game_date.
    # Simpler: if 'days_rest' is NaN after merge, it means new pitcher or no recent games.
    # A more accurate 'days_rest' for today would be calculated based on their last start from historical_df.
    # For now, the merged one is "rest before their last historical game". This needs refinement for prediction.
    # Let's calculate rest for today:
    last_pitch_date_map = all_historical_features_df[all_historical_features_df['game_date'] < prediction_date].groupby('pitcher')['game_date'].max()
    prediction_baseline_df = prediction_baseline_df.merge(last_pitch_date_map.rename('last_game_date_pitcher'), on='pitcher', how='left')
    prediction_baseline_df['days_rest'] = (prediction_baseline_df['game_date'] - prediction_baseline_df['last_game_date_pitcher']).dt.days
    prediction_baseline_df = prediction_baseline_df.drop(columns=['last_game_date_pitcher'], errors='ignore')


    # Opponent Stats
    if 'opponent_team' in prediction_baseline_df.columns:
        latest_opponent_stats_source = all_historical_features_df[
            all_historical_features_df['game_date'] < prediction_date
        ].sort_values('game_date').groupby('team_for_opp_stats', observed=True).last() # Assuming 'team_for_opp_stats' column exists from historical build
        # This needs careful handling of opponent team ID.
        # For now, let's assume `merge_opponent_features_prediction` handles fetching latest.
        # We need the `game_level_team_stats` for this.
        team_game_stats_df = load_team_game_stats()
        if not team_game_stats_df.empty:
            # Calculate latest opponent rolling from their historical stats
            valid_opponent_metrics = [m for m in OPPONENT_BATTING_METRICS_FOR_ROLLING if m in team_game_stats_df.columns]
            if 'team' in team_game_stats_df.columns and valid_opponent_metrics:
                latest_opp_rolling_df, opp_rename_map = calculate_opponent_rolling_features(
                    team_hist_df=team_game_stats_df[team_game_stats_df['game_date'] < prediction_date],
                    group_col='team', date_col='game_date', metrics=valid_opponent_metrics,
                    windows=WINDOW_SIZES, min_periods=MIN_PERIODS_DEFAULT,
                    calculate_multi_window_rolling=calculate_multi_window_rolling_expanded
                )
                # Get the very last row for each team to represent "latest"
                latest_opp_rolling_df = latest_opp_rolling_df.sort_values('game_date').groupby('team', observed=True).last().reset_index()

                prediction_baseline_df = merge_opponent_features_prediction(
                     prediction_baseline_df, latest_opp_rolling_df, opp_rename_map, WINDOW_SIZES
                ) # Requires 'p_throws' in prediction_baseline_df

    # Ballpark Stats
    if 'ballpark' in prediction_baseline_df.columns:
        latest_ballpark_stats_source = all_historical_features_df[
            all_historical_features_df['game_date'] < prediction_date
        ].sort_values('game_date').groupby('ballpark', observed=True).last()
        bpark_cols_to_merge = [col for col in latest_ballpark_stats_source.columns if col.startswith('bp_roll')]
        # The merge_ballpark_features_prediction needs a map, let's create a dummy one if not available
        # Or, ensure calculate_ballpark_rolling_features is called to get the map.
        # For simplicity, assume the columns are already correctly named in latest_ballpark_stats_source
        dummy_bpark_rename_map = {col: col for col in bpark_cols_to_merge} # This is not ideal
        prediction_baseline_df = merge_ballpark_features_prediction(
            prediction_baseline_df, latest_ballpark_stats_source[bpark_cols_to_merge].reset_index(), dummy_bpark_rename_map
        )


    # Umpire Stats
    if 'home_plate_umpire' in prediction_baseline_df.columns:
        latest_umpire_stats_source = all_historical_features_df[
            all_historical_features_df['game_date'] < prediction_date
        ].sort_values('game_date').groupby('home_plate_umpire', observed=True).last()
        ump_cols_to_merge = [col for col in latest_umpire_stats_source.columns if col.startswith('ump_roll')]
        prediction_baseline_df = pd.merge(
            prediction_baseline_df,
            latest_umpire_stats_source[ump_cols_to_merge].reset_index(),
            on='home_plate_umpire',
            how='left'
        )
        
    logger.info(f"Finished generating prediction features. Shape: {prediction_baseline_df.shape}")
    
    # Save features
    try:
        prediction_baseline_df.to_parquet(PREDICTION_FEATURES_FILE, index=False)
        logger.info(f"Prediction features saved to {PREDICTION_FEATURES_FILE}")
    except Exception as e:
        logger.error(f"Error saving prediction features to {PREDICTION_FEATURES_FILE}: {e}", exc_info=True)

    return prediction_baseline_df


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features for MLB pitcher strikeout prediction.")
    parser.add_argument(
        "--mode",
        choices=['historical', 'prediction'],
        required=True,
        help="Mode of operation: 'historical' to generate features for past data, 'prediction' for a specific future date."
    )
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD) for historical mode."
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD) for historical mode."
    )
    parser.add_argument(
        "--prediction-date",
        help="Date (YYYY-MM-DD) for prediction mode. Defaults to today if not provided."
    )

    args = parser.parse_args()

    if args.mode == 'historical':
        if not args.start_date or not args.end_date:
            logger.warning("For historical mode, --start-date and --end-date are recommended. Running for all available data.")
        generate_historical_features(args.start_date, args.end_date)
    elif args.mode == 'prediction':
        pred_date = args.prediction_date if args.prediction_date else datetime.now().strftime("%Y-%m-%d")
        generate_prediction_baseline_features(pred_date)

    logger.info("Feature generation process finished.")

