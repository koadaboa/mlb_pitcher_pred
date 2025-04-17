# src/scripts/create_advanced_features.py

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
import gc
from tqdm.auto import tqdm # Progress bar
import warnings
from datetime import datetime, timedelta # Ensure datetime is imported
import sqlite3 # Import sqlite3 for specific error catching

# Suppress specific warnings if needed
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore mean of empty slice warnings if desired

# Assuming script is run via python -m src.scripts.create_advanced_features
try:
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1)

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger('create_advanced_features', LogConfig.LOG_DIR / 'create_advanced_features.log', level=logging.INFO)

# --- Ballpark Mapping (Copied from engineer_features for self-containment if needed) ---
TEAM_TO_BALLPARK = {
    'ARI': 'Chase Field', 'ATL': 'Truist Park', 'BAL': 'Oriole Park at Camden Yards',
    'BOS': 'Fenway Park', 'CHC': 'Wrigley Field', 'CWS': 'Guaranteed Rate Field',
    'CIN': 'Great American Ball Park', 'CLE': 'Progressive Field', 'COL': 'Coors Field',
    'DET': 'Comerica Park', 'HOU': 'Minute Maid Park', 'KC': 'Kauffman Stadium',
    'LAA': 'Angel Stadium', 'LAD': 'Dodger Stadium', 'MIA': 'LoanDepot Park',
    'MIL': 'American Family Field', 'MIN': 'Target Field', 'NYM': 'Citi Field',
    'NYY': 'Yankee Stadium', 'OAK': 'RingCentral Coliseum', # Updated
    'PHI': 'Citizens Bank Park', 'PIT': 'PNC Park', 'SD': 'Petco Park', 'SF': 'Oracle Park',
    'SEA': 'T-Mobile Park', 'STL': 'Busch Stadium', 'TB': 'Tropicana Field',
    'TEX': 'Globe Life Field', 'TOR': 'Rogers Centre', 'WSH': 'Nationals Park'
}
DEFAULT_BALLPARK = 'Unknown Park' # Fallback for unmapped teams

# --- Constants ---
ROLLING_PITCHER_WINDOW = '90d' # Rolling window for pitcher pitch-level stats
ROLLING_OPPONENT_WINDOW = 20 # Rolling window (games) for opponent team stats
MIN_PITCHES_FOR_ROLLING = 50 # Min pitches in window for pitcher stats
MIN_GAMES_FOR_ROLLING = 5 # Min games in window for team stats
ID_COLS_TO_EXCLUDE_FROM_DOWNCAST = ['pitcher', 'batter', 'game_pk', 'pitcher_id']

# --- Memory Optimization Helper ---
def optimize_dtypes(df):
    """Attempt to reduce memory usage by downcasting dtypes, excluding specified ID columns."""
    logger.debug("Optimizing dtypes...")
    if df is None or df.empty: return df
    df_copy = df.copy() # Work on a copy
    cols_to_exclude = [col for col in ID_COLS_TO_EXCLUDE_FROM_DOWNCAST if col in df_copy.columns]
    logger.debug(f" Excluding ID columns from downcasting: {cols_to_exclude}")
    for col in df_copy.select_dtypes(include=['int64']).columns:
        if col not in cols_to_exclude: # Check if column should be excluded
            try: df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
            except Exception as e: logger.warning(f"Could not downcast integer column '{col}': {e}")
    for col in df_copy.select_dtypes(include=['float64']).columns:
         try: df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
         except Exception as e: logger.warning(f"Could not downcast float column '{col}': {e}")
    logger.debug("Dtype optimization attempt complete.")
    return df_copy

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Create Advanced Features for Training or Prediction.")
    parser.add_argument("--prediction-date", type=str, default=None,
                        help="If specified, generate features only for this date (YYYY-MM-DD) for prediction.")
    return parser.parse_args()


# --- Feature Calculation Functions (Self-contained within this script) ---

def calculate_dynamic_park_factors(df, historical_game_df):
    """Calculates dynamic park factors based on historical game data."""
    logger.info("Calculating dynamic park factors...")
    df_copy = df.copy() # Work on copy
    park_factor_col = 'park_factor_k_dynamic' # Name of the resulting column

    if historical_game_df is None or historical_game_df.empty:
        logger.warning("Historical game data empty. Returning default park factor (1.0).")
        df_copy[park_factor_col] = 1.0
        return df_copy

    try:
        hist_game_df_copy = historical_game_df.copy() # Work on copy
        hist_game_df_copy['game_date'] = pd.to_datetime(hist_game_df_copy['game_date'])
        hist_game_df_copy['season'] = hist_game_df_copy['game_date'].dt.year
    except Exception as e:
        logger.error(f"Error processing dates in historical_game_df: {e}")
        df_copy[park_factor_col] = 1.0
        return df_copy

    stat_col = 'k_percent'
    if stat_col not in hist_game_df_copy.columns:
         logger.error(f"Required column '{stat_col}' not found. Returning default park factor (1.0).")
         df_copy[park_factor_col] = 1.0
         return df_copy

    if 'home_team' in hist_game_df_copy.columns:
        hist_game_df_copy['ballpark'] = hist_game_df_copy['home_team'].map(TEAM_TO_BALLPARK).fillna(DEFAULT_BALLPARK)
    else:
        logger.warning("'home_team' missing. Cannot map ballparks accurately.")
        hist_game_df_copy['ballpark'] = DEFAULT_BALLPARK

    league_avg_per_season = hist_game_df_copy.groupby('season', observed=False)[stat_col].mean().reset_index() # Use observed=False
    league_avg_per_season = league_avg_per_season.rename(columns={stat_col: 'league_avg_stat'})

    park_avg_per_season = hist_game_df_copy.groupby(['season', 'ballpark'], observed=False)[stat_col].mean().reset_index() # Use observed=False
    park_avg_per_season = park_avg_per_season.rename(columns={stat_col: 'park_avg_stat'})

    park_factors = pd.merge(park_avg_per_season, league_avg_per_season, on='season', how='left')
    with np.errstate(divide='ignore', invalid='ignore'):
        park_factors[park_factor_col] = (park_factors['park_avg_stat'] / park_factors['league_avg_stat'])
    park_factors[park_factor_col] = park_factors[park_factor_col].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    park_factors = park_factors[['season', 'ballpark', park_factor_col]]

    try:
        df_copy['game_date_dt'] = pd.to_datetime(df_copy['game_date'], errors='coerce')
        df_copy['season'] = df_copy['game_date_dt'].dt.year.fillna(datetime.now().year).astype(int)
    except Exception as e:
         logger.error(f"Error processing game_date in main df: {e}")
         df_copy['season'] = datetime.now().year

    if 'ballpark' not in df_copy.columns:
         if 'home_team' in df_copy.columns:
              df_copy['ballpark'] = df_copy['home_team'].map(TEAM_TO_BALLPARK).fillna(DEFAULT_BALLPARK)
         else:
             logger.warning("'home_team' missing, cannot determine ballpark.")
             df_copy['ballpark'] = DEFAULT_BALLPARK

    original_len = len(df_copy)
    df_merged = pd.merge(df_copy, park_factors, on=['season', 'ballpark'], how='left')
    df_merged[park_factor_col] = df_merged[park_factor_col].fillna(1.0)
    if len(df_merged) != original_len:
        logger.warning("Park factor merge changed row count. Check merge keys.")
    df_merged = df_merged.drop(columns=['game_date_dt'], errors='ignore')

    logger.info("Dynamic park factors calculation complete.")
    return df_merged

def calculate_rolling_pitcher_stats(df, historical_pitch_df):
    """Calculates rolling pitcher stats using historical pitch data."""
    logger.info("Calculating rolling pitcher stats...")
    df_copy = df.copy().reset_index(drop=True) # Ensure clean index

    # Add default columns first
    roll_k_pct_col = 'roll_pitcher_k_pct'; roll_whiff_col = 'roll_pitcher_whiff_pct'; roll_fb_col = 'roll_pitcher_fastball_pct'
    df_copy[roll_k_pct_col] = np.nan; df_copy[roll_whiff_col] = np.nan; df_copy[roll_fb_col] = np.nan

    if historical_pitch_df is None or historical_pitch_df.empty:
        logger.warning("Historical pitch data empty. Returning NaNs for rolling pitcher stats.")
        return df_copy

    required_cols = ['pitcher', 'game_date', 'events', 'description', 'type', 'pitch_type', 'game_pk', 'at_bat_number', 'pitch_number']
    if not all(col in historical_pitch_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in historical_pitch_df.columns]
        logger.error(f"Missing required columns in historical_pitch_df: {missing}. Returning NaNs for rolling pitcher stats.")
        return df_copy

    try:
        hist_pitch_df_copy = historical_pitch_df.copy() # Work on copy
        hist_pitch_df_copy['game_date'] = pd.to_datetime(hist_pitch_df_copy['game_date'])
        hist_pitch_df_copy = optimize_dtypes(hist_pitch_df_copy)
        # Ensure 'pitcher' column is suitable type for grouping
        if 'pitcher' in hist_pitch_df_copy.columns:
             hist_pitch_df_copy['pitcher'] = hist_pitch_df_copy['pitcher'].astype(int) # Or appropriate type
        else:
             logger.error("'pitcher' column missing from historical pitch data.")
             return df_copy
        # Sort for temporal calculations
        hist_pitch_df_copy = hist_pitch_df_copy.sort_values(by=['pitcher', 'game_date', 'at_bat_number', 'pitch_number'])
    except Exception as e:
        logger.error(f"Error during historical pitch data prep: {e}")
        return df_copy # Return df with defaults

    # Create Flags/Metrics per Pitch
    hist_pitch_df_copy = hist_pitch_df_copy.assign(
        is_k = hist_pitch_df_copy['events'].isin(['strikeout', 'strikeout_double_play']).astype(np.int8),
        is_swstr = ((hist_pitch_df_copy['type'] == 'S') & (hist_pitch_df_copy['description'].isin(['swinging_strike', 'swinging_strike_blocked']))).astype(np.int8),
        is_swing = hist_pitch_df_copy['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score']).astype(np.int8),
        is_fastball = hist_pitch_df_copy['pitch_type'].isin(['FF', 'SI', 'FT', 'FC']).astype(np.int8)
    )

    # --- Robust Rolling Calculation ---
    all_rolling_results = []
    group_col = 'pitcher'; date_col = 'game_date'
    grouped_pitcher_hist = hist_pitch_df_copy.groupby(group_col, group_keys=True)
    logger.info(f"Calculating rolling stats for {grouped_pitcher_hist.ngroups} pitchers...")

    for pitcher_id, group_df in tqdm(grouped_pitcher_hist, desc="Rolling Pitcher Stats"):
        group_df_sorted = group_df.sort_values(by=[date_col, 'at_bat_number', 'pitch_number'])
        group_indexed = group_df_sorted.set_index(date_col)

        if not group_indexed.index.is_monotonic_increasing:
            logger.warning(f"Index not monotonic for pitcher {pitcher_id}. Deduplicating index.")
            group_indexed = group_indexed[~group_indexed.index.duplicated(keep='last')]
            if not group_indexed.index.is_monotonic_increasing:
                 logger.error(f"Index STILL not monotonic for pitcher {pitcher_id} after dedup. Skipping rolling.")
                 continue

        # Calculate rolling sums/counts for this group
        roll_k_sum = group_indexed['is_k'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum()
        roll_swstr_sum = group_indexed['is_swstr'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum()
        roll_swing_sum = group_indexed['is_swing'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum()
        roll_fastball_sum = group_indexed['is_fastball'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum()
        roll_pitches_count = group_indexed['pitcher'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').count()

        # Calculate rates safely
        with np.errstate(divide='ignore', invalid='ignore'):
             roll_k_pct = (roll_k_sum / roll_pitches_count).astype(np.float32)
             roll_whiff_pct = (roll_swstr_sum / roll_swing_sum).astype(np.float32)
             roll_fb_pct = (roll_fastball_sum / roll_pitches_count).astype(np.float32)

        group_results = pd.DataFrame({
            roll_k_pct_col: roll_k_pct, roll_whiff_col: roll_whiff_pct, roll_fb_col: roll_fb_pct,
            'game_pk': group_indexed['game_pk'] # Include game_pk to get last value per game
        }).reset_index() # Get game_date back

        # Aggregate to game level by taking the value from the LAST pitch of each game for that pitcher
        game_level_group_stats = group_results.sort_values(by=date_col) \
                                      .drop_duplicates(subset=['game_pk'], keep='last') \
                                      .drop(columns=['game_pk']) # Drop game_pk after use
        game_level_group_stats[group_col] = pitcher_id # Add pitcher_id back
        all_rolling_results.append(game_level_group_stats)

    if not all_rolling_results:
         logger.warning("No rolling pitcher stats generated.")
         # Keep NaNs already initialized in df_copy
         return df_copy

    game_level_rolling_stats = pd.concat(all_rolling_results, ignore_index=True)
    del hist_pitch_df_copy, grouped_pitcher_hist, all_rolling_results; gc.collect()

    # --- Merge onto main df ---
    try:
        df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
        game_level_rolling_stats['game_date'] = pd.to_datetime(game_level_rolling_stats['game_date'])

        # Ensure pitcher_id column exists and types match for merge
        if 'pitcher_id' not in df_copy.columns:
            logger.error("'pitcher_id' column missing from main df. Cannot merge rolling stats.")
            return df_copy # Return df with NaNs
        df_copy['pitcher_id'] = df_copy['pitcher_id'].astype(int) # Ensure consistent type
        game_level_rolling_stats['pitcher'] = game_level_rolling_stats['pitcher'].astype(int)

        # Sort both DFs by date for merge_asof
        df_copy = df_copy.sort_values(by='game_date')
        game_level_rolling_stats = game_level_rolling_stats.sort_values(by='game_date')
        if not game_level_rolling_stats['game_date'].is_monotonic_increasing:
             logger.error("FATAL: Rolling stats 'game_date' column is NOT monotonic increasing!")
             return df_copy

        logger.info("Merging game-level rolling pitcher stats using merge_asof...")
        original_len = len(df_copy)
        df_merged = pd.merge_asof(df_copy, game_level_rolling_stats,
                           on='game_date',
                           left_by='pitcher_id', right_by='pitcher',
                           direction='backward', allow_exact_matches=False)
        if len(df_merged) != original_len: logger.warning("merge_asof changed row count.")

        # Drop the extra 'pitcher' column from the merge
        df_merged = df_merged.drop(columns=['pitcher'], errors='ignore')

        # Handle NaNs from merge (pitchers with no prior history) using global median
        for col in [roll_k_pct_col, roll_whiff_col, roll_fb_col]:
            if col in df_merged.columns:
                 median_val = game_level_rolling_stats[col].median() # Use historical median
                 fill_val = median_val if pd.notna(median_val) else 0
                 nan_count = df_merged[col].isnull().sum()
                 if nan_count > 0:
                      logger.info(f"Imputing {nan_count} missing '{col}' values with median {fill_val:.4f} (or 0).")
                      df_merged[col] = df_merged[col].fillna(fill_val)

        logger.info("Rolling pitcher stats calculation complete.")
        return df_merged

    except Exception as merge_e:
        logger.error(f"Error merging rolling pitcher stats: {merge_e}", exc_info=True)
        # Return df with NaNs for rolling columns
        return df_copy

def calculate_rolling_opponent_stats(df, historical_team_df):
    """Calculates rolling opponent team stats with robust index handling."""
    logger.info("Calculating rolling opponent team stats...")
    df_copy = df.copy().reset_index(drop=True) # Ensure clean index

    opp_k_pct_col = 'roll_opp_k_pct_vs_hand'; opp_bb_pct_col = 'roll_opp_bb_pct_vs_hand'
    df_copy[opp_k_pct_col] = np.nan; df_copy[opp_bb_pct_col] = np.nan

    if historical_team_df is None or historical_team_df.empty:
        logger.warning("Historical team game data empty. Returning NaNs for rolling opponent stats.")
        return df_copy

    required_cols = ['game_date', 'team', 'k_percent', 'bb_percent', 'game_pk', 'opponent', 'home_team']
    if not all(col in historical_team_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in historical_team_df.columns]
        logger.error(f"Missing required columns in historical_team_df: {missing}. Returning NaNs.")
        return df_copy

    try:
        hist_team_df_copy = historical_team_df.copy() # Work on copy
        hist_team_df_copy['game_date'] = pd.to_datetime(hist_team_df_copy['game_date'])
        hist_team_df_copy = optimize_dtypes(hist_team_df_copy)
        # Ensure 'team' column is suitable type for grouping
        if 'team' in hist_team_df_copy.columns:
             hist_team_df_copy['team'] = hist_team_df_copy['team'].astype(str) # Assume team abbr is string
        else:
             logger.error("'team' column missing from historical team data.")
             return df_copy
        hist_team_df_copy = hist_team_df_copy.sort_values(by=['team', 'game_date']) # Sort for rolling
    except Exception as e:
        logger.error(f"Error during historical team data prep: {e}")
        return df_copy # Return df with defaults

    # --- Rolling Calculations (Game-based window) ---
    group_col = 'team'; date_col = 'game_date'
    grouped_team = hist_team_df_copy.groupby(group_col)
    logger.info(f"Applying opponent rolling window ({ROLLING_OPPONENT_WINDOW} games, min={MIN_GAMES_FOR_ROLLING})...")
    # Use shift(1) to exclude current game from calculation
    roll_opp_k_pct = grouped_team['k_percent'].shift(1).rolling(ROLLING_OPPONENT_WINDOW, min_periods=MIN_GAMES_FOR_ROLLING).mean().astype(np.float32)
    roll_opp_bb_pct = grouped_team['bb_percent'].shift(1).rolling(ROLLING_OPPONENT_WINDOW, min_periods=MIN_GAMES_FOR_ROLLING).mean().astype(np.float32)

    # Add results back to the historical df copy
    hist_team_df_copy = hist_team_df_copy.assign(
         roll_opp_k_pct = roll_opp_k_pct,
         roll_opp_bb_pct = roll_opp_bb_pct
    )
    del roll_opp_k_pct, roll_opp_bb_pct, grouped_team; gc.collect()

    # --- Merge onto main df ---
    logger.info("Merging rolling opponent stats onto main dataframe...")
    opponent_rolling_stats = hist_team_df_copy[['team', 'game_date', 'roll_opp_k_pct', 'roll_opp_bb_pct']].copy()
    opponent_rolling_stats = opponent_rolling_stats.rename(columns={'team': 'opponent_team'}) # RENAME to merge key

    try:
        df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
        if 'opponent_team' not in df_copy.columns:
             logger.error("Main dataframe 'df' missing 'opponent_team' column.")
             return df_copy # Return df with defaults

        # Ensure types match for merge
        df_copy['opponent_team'] = df_copy['opponent_team'].astype(str)
        opponent_rolling_stats['opponent_team'] = opponent_rolling_stats['opponent_team'].astype(str)

        df_copy = df_copy.sort_values(by='game_date') # Sort target df by date
        opponent_rolling_stats = opponent_rolling_stats.sort_values(by='game_date') # Sort source df by date
        if not opponent_rolling_stats['game_date'].is_monotonic_increasing:
            logger.error("FATAL: Opponent rolling stats 'game_date' column NOT monotonic!")
            return df_copy
    except Exception as e:
         logger.error(f"Error preparing DFs for opponent merge: {e}")
         return df_copy

    original_len = len(df_copy)
    df_merged = pd.merge_asof(df_copy, opponent_rolling_stats,
                       on='game_date', by='opponent_team',
                       direction='backward', allow_exact_matches=False)
    if len(df_merged) != original_len: logger.warning("Opponent merge_asof changed row count.")

    # --- Vs Hand Calculation (Placeholder) ---
    # Assign overall stats as placeholder - **IMPLEMENT HAND-SPECIFIC LOGIC HERE IF NEEDED**
    df_merged[opp_k_pct_col] = df_merged['roll_opp_k_pct']
    df_merged[opp_bb_pct_col] = df_merged['roll_opp_bb_pct']
    if 'roll_opp_k_pct' in df_merged.columns: # Only log if column exists
        logger.warning("Assigning overall opponent stats to 'vs_hand' columns. Implement specific logic if required.")

    # Handle teams with no prior rolling stats
    for col, default_val in zip([opp_k_pct_col, opp_bb_pct_col], [0.22, 0.08]): # Use default constants
        if col in df_merged.columns:
             median_val = df_merged[col].median() # Use median of calculated values
             fill_val = median_val if pd.notna(median_val) else default_val
             nan_count = df_merged[col].isnull().sum()
             if nan_count > 0:
                  logger.info(f"Imputing {nan_count} missing '{col}' values with median {fill_val:.4f} (or default).")
                  df_merged[col] = df_merged[col].fillna(fill_val)

    # Drop intermediate columns
    df_merged = df_merged.drop(columns=['roll_opp_k_pct', 'roll_opp_bb_pct'], errors='ignore')

    logger.info("Rolling opponent stats calculation complete.")
    return df_merged # Return modified dataframe

def calculate_interaction_features(df):
    """Calculates interaction features between pitcher, opponent, and park."""
    logger.info("Calculating interaction features...")
    df_copy = df.copy() # Work on copy

    # Define component column names
    park_factor_col = 'park_factor_k_dynamic'
    pitcher_k_col = 'roll_pitcher_k_pct'
    pitcher_whiff_col = 'roll_pitcher_whiff_pct'
    pitcher_fb_col = 'roll_pitcher_fastball_pct'
    opp_k_col = 'roll_opp_k_pct_vs_hand'

    # Define default values for safety
    default_park = 1.0; default_pitcher_k = 0.20; default_pitcher_whiff = 0.10
    default_pitcher_fb = 0.50; default_opp_k = 0.22

    # Calculate interactions, using .get() with defaults
    df_copy['interact_park_pitcher_k'] = df_copy.get(park_factor_col, default_park) * df_copy.get(pitcher_k_col, default_pitcher_k)
    df_copy['interact_whiff_opp_k'] = df_copy.get(pitcher_whiff_col, default_pitcher_whiff) * df_copy.get(opp_k_col, default_opp_k)
    df_copy['interact_fb_opp_k'] = df_copy.get(pitcher_fb_col, default_pitcher_fb) * df_copy.get(opp_k_col, default_opp_k)

    logger.info("Interaction features calculation complete.")
    return df_copy # Return modified copy

# --- Main Function ---
def create_advanced_features(args):
    """Loads data, calculates advanced features, and saves to the database."""
    db_path = Path(DBConfig.PATH)
    prediction_mode = args.prediction_date is not None
    max_historical_date_str = (datetime.strptime(args.prediction_date, '%Y-%m-%d').date() - timedelta(days=1)).strftime('%Y-%m-%d') if prediction_mode else '9999-12-31'

    if prediction_mode:
        logger.info(f"Running in PREDICTION mode for date: {args.prediction_date}")
        input_table = "prediction_features" # Baseline features for the prediction date
        output_table = "prediction_features_advanced"
    else:
        logger.info("Running in TRAINING mode (processing train and test sets).")
        input_table_train = "train_features" # Baseline train features
        input_table_test = "test_features"   # Baseline test features
        output_table_train = "train_features_advanced"
        output_table_test = "test_features_advanced"

    # --- Load Data ---
    logger.info("Loading baseline features and historical data...")
    df = pd.DataFrame()
    historical_pitch_df = pd.DataFrame()
    historical_game_df = pd.DataFrame()
    load_success = True
    try:
        with DBConnection(db_path) as conn:
            # Load baseline features
            if prediction_mode:
                df = pd.read_sql_query(f"SELECT * FROM {input_table}", conn)
                logger.info(f"Loaded {len(df)} baseline rows from '{input_table}' for prediction date {args.prediction_date}.")
                if df.empty: logger.error(f"No baseline prediction features found for {args.prediction_date}."); load_success = False
            else:
                train_df = pd.read_sql_query(f"SELECT * FROM {input_table_train}", conn)
                test_df = pd.read_sql_query(f"SELECT * FROM {input_table_test}", conn)
                logger.info(f"Loaded {len(train_df)} baseline train rows from '{input_table_train}'.")
                logger.info(f"Loaded {len(test_df)} baseline test rows from '{input_table_test}'.")
                if train_df.empty and test_df.empty: logger.error("Both train and test baseline features are empty."); load_success = False
                elif train_df.empty: logger.warning("Train baseline features are empty.")
                elif test_df.empty: logger.warning("Test baseline features are empty.")
                df = pd.concat([train_df, test_df], ignore_index=True)
                logger.info(f"Combined baseline train/test features: {len(df)} rows.")
                del train_df, test_df; gc.collect()

            # Load Historical Data (only if baseline loaded successfully)
            if load_success:
                logger.info(f"Loading historical data strictly before {max_historical_date_str}...")
                # Load PITCH data (statcast_pitchers)
                pitch_cols = ['pitcher', 'game_pk', 'game_date', 'pitch_type', 'description', 'events', 'type', 'at_bat_number', 'pitch_number']
                pitch_query = f"SELECT {', '.join(pitch_cols)} FROM statcast_pitchers WHERE DATE(game_date) < '{max_historical_date_str}'"
                historical_pitch_df = pd.read_sql_query(pitch_query, conn)
                logger.info(f"Loaded {len(historical_pitch_df)} historical pitches.")
                historical_pitch_df = optimize_dtypes(historical_pitch_df); gc.collect()

                # Load GAME data (game_level_team_stats)
                game_cols = ['game_pk', 'game_date', 'team', 'opponent', 'home_team', 'k_percent', 'bb_percent'] # Add others if needed
                game_query = f"SELECT {', '.join(game_cols)} FROM game_level_team_stats WHERE DATE(game_date) < '{max_historical_date_str}'"
                historical_game_df = pd.read_sql_query(game_query, conn)
                logger.info(f"Loaded {len(historical_game_df)} historical team games.")
                historical_game_df = optimize_dtypes(historical_game_df); gc.collect()

    except sqlite3.OperationalError as oe:
         logger.error(f"Database Error during data loading: {oe}. Check table names and existence.")
         if "no such table" in str(oe).lower(): logger.error(f"Missing table: {str(oe).split(':')[-1].strip()}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error during initial data loading phase: {e}", exc_info=True); sys.exit(1)

    if not load_success or df.empty:
         logger.error("Baseline feature loading failed or resulted in empty dataframe. Cannot proceed.")
         sys.exit(1)

    # --- Add Ballpark Info ---
    # (Keep the existing ballpark mapping logic here)
    if 'home_team' in df.columns:
         logger.info("Mapping teams to ballparks...")
         df['ballpark'] = df['home_team'].map(TEAM_TO_BALLPARK).fillna(DEFAULT_BALLPARK)
         unmapped = df[df['ballpark'] == DEFAULT_BALLPARK]['home_team'].unique()
         if len(unmapped) > 0: logger.warning(f"Unmapped teams assigned '{DEFAULT_BALLPARK}': {list(unmapped)}")
    else:
         logger.warning("Column 'home_team' not found. Cannot map ballparks.")
         df['ballpark'] = DEFAULT_BALLPARK

    # --- Calculate Advanced Features ---
    logger.info("Starting advanced feature calculation pipeline...")
    try:
        # Apply calculations sequentially, passing the result of one to the next
        df_processed = calculate_dynamic_park_factors(df, historical_game_df); gc.collect()
        # Check essential columns after each step if needed
        if 'pitcher_id' not in df_processed.columns: raise KeyError("Missing 'pitcher_id' after park factors")
        df_processed = calculate_rolling_pitcher_stats(df_processed, historical_pitch_df); gc.collect()
        if 'opponent_team' not in df_processed.columns: raise KeyError("Missing 'opponent_team' after rolling pitcher stats")
        df_processed = calculate_rolling_opponent_stats(df_processed, historical_game_df); gc.collect()
        df_processed = calculate_interaction_features(df_processed); gc.collect()
        df = df_processed # Assign the final processed dataframe back to df
        logger.info("Advanced feature calculation pipeline finished.")
    except Exception as e:
         logger.error(f"Error during advanced feature calculation: {e}", exc_info=True)
         sys.exit(1) # Exit if calculation fails
    finally:
         # Clean up large historical dataframes explicitly
         del historical_pitch_df, historical_game_df; gc.collect()

    # --- Final check for NaNs/Infs introduced by calculations ---
    logger.info("Performing final NaN/Inf check and imputation...")
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns
        # Replace Inf first, using assignment
        df = df.replace([np.inf, -np.inf], np.nan)

        # Handle NaNs using assignment
        nan_mask = df[numeric_cols].isnull()
        if nan_mask.any().any():
            logger.warning(f"NaNs found after calculation. Applying median imputation...")
            cols_with_nans = numeric_cols[nan_mask.any()]
            imputation_values = {} # Store medians
            for col in cols_with_nans:
                 median_val = df[col].median()
                 imputation_values[col] = median_val if pd.notna(median_val) else 0
                 logger.debug(f"Median for '{col}': {imputation_values[col]}")
            # Apply fillna using the calculated medians
            df = df.fillna(value=imputation_values)
            # Verify (optional)
            remaining_nans = df[numeric_cols].isnull().sum().sum()
            if remaining_nans > 0:
                 logger.error(f"{remaining_nans} NaNs remain after imputation!")

    except Exception as e:
         logger.error(f"Error during final NaN/Inf handling: {e}", exc_info=True)


    # --- Save Results ---
    logger.info("Saving advanced features to database...")
    try:
        with DBConnection(db_path) as conn:
            if prediction_mode:
                 # Final check before saving prediction features
                 if df.empty: logger.error("Prediction features dataframe is empty before saving.")
                 else:
                     df.to_sql(output_table, conn, if_exists='replace', index=False)
                     logger.info(f"Saved {len(df)} rows to '{output_table}' for date {args.prediction_date}.")
            else:
                # Split back into train/test and save separately
                if 'season' not in df.columns:
                     logger.error("Cannot split train/test - 'season' column missing.")
                     sys.exit(1)
                train_mask = df['season'].isin(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)
                test_mask = df['season'].isin(StrikeoutModelConfig.DEFAULT_TEST_YEARS)

                train_advanced_df = df[train_mask].copy() # Explicit copy
                test_advanced_df = df[test_mask].copy()   # Explicit copy

                if train_advanced_df.empty: logger.warning(f"Train advanced features dataframe is empty before saving to {output_table_train}.")
                else:
                    train_advanced_df.to_sql(output_table_train, conn, if_exists='replace', index=False)
                    logger.info(f"Saved {len(train_advanced_df)} rows to '{output_table_train}'.")

                if test_advanced_df.empty: logger.warning(f"Test advanced features dataframe is empty before saving to {output_table_test}.")
                else:
                    test_advanced_df.to_sql(output_table_test, conn, if_exists='replace', index=False)
                    logger.info(f"Saved {len(test_advanced_df)} rows to '{output_table_test}'.")

    except Exception as e:
        logger.error(f"Error saving advanced features to database: {e}", exc_info=True); sys.exit(1)

    logger.info("Advanced feature creation finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    # Validate prediction date format if provided
    if args.prediction_date:
        try: datetime.strptime(args.prediction_date, '%Y-%m-%d')
        except ValueError: logger.error(f"Invalid date format: {args.prediction_date}."); sys.exit(1)

    create_advanced_features(args)
    logger.info("--- Create Advanced Features Script Completed ---")

