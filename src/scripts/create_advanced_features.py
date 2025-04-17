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
from datetime import datetime # Ensure datetime is imported

# Suppress specific warnings if needed (e.g., from rolling operations)
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

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
logger = setup_logger('create_advanced_features', LogConfig.LOG_DIR / 'create_advanced_features.log')

# --- Ballpark Mapping ---
TEAM_TO_BALLPARK = {
    'ARI': 'Chase Field', 'ATL': 'Truist Park', 'BAL': 'Oriole Park at Camden Yards',
    'BOS': 'Fenway Park', 'CHC': 'Wrigley Field', 'CWS': 'Guaranteed Rate Field',
    'CIN': 'Great American Ball Park', 'CLE': 'Progressive Field', 'COL': 'Coors Field',
    'DET': 'Comerica Park', 'HOU': 'Minute Maid Park', 'KC': 'Kauffman Stadium',
    'LAA': 'Angel Stadium', 'LAD': 'Dodger Stadium', 'MIA': 'LoanDepot Park',
    'MIL': 'American Family Field', 'MIN': 'Target Field', 'NYM': 'Citi Field',
    'NYY': 'Yankee Stadium', 'OAK': 'Oakland-Alameda County Coliseum', 'PHI': 'Citizens Bank Park',
    'PIT': 'PNC Park', 'SD': 'Petco Park', 'SF': 'Oracle Park',
    'SEA': 'T-Mobile Park', 'STL': 'Busch Stadium', 'TB': 'Tropicana Field',
    'TEX': 'Globe Life Field', 'TOR': 'Rogers Centre', 'WSH': 'Nationals Park'
}
DEFAULT_BALLPARK = 'Unknown Park' # Fallback for unmapped teams

# --- Constants ---
ROLLING_PITCHER_WINDOW = '90d' # Rolling window for pitcher pitch-level stats
ROLLING_OPPONENT_WINDOW = 20 # Rolling window (games) for opponent team stats
MIN_PITCHES_FOR_ROLLING = 50 # Min pitches in window for pitcher stats
MIN_GAMES_FOR_ROLLING = 5 # Min games in window for team stats
ID_COLS_TO_EXCLUDE_FROM_DOWNCAST = ['pitcher', 'batter', 'game_pk', 'pitcher_id'] # Add other potential ID columns if needed

# --- Memory Optimization Helper ---
def optimize_dtypes(df):
    """Attempt to reduce memory usage by downcasting dtypes, excluding specified ID columns."""
    logger.debug("Optimizing dtypes...")
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning later if df is a slice
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
            # Downcast floats to float32
            df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
         except Exception as e:
            logger.warning(f"Could not downcast float column '{col}': {e}")
    logger.debug("Dtype optimization attempt complete.")
    return df_copy

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Create Advanced Features for Training or Prediction.")
    parser.add_argument("--prediction-date", type=str, default=None,
                        help="If specified, generate features only for this date (YYYY-MM-DD) for prediction.")
    # Add other arguments if needed (e.g., --force-recompute, --historical-years)
    return parser.parse_args()


# --- Feature Calculation Functions (Implemented Examples) ---

def calculate_dynamic_park_factors(df, historical_game_df):
    """
    Calculates dynamic park factors based on historical game data.
    Example: K% park factor based on rolling 1-year average from game_level_team_stats.
    """
    logger.info("Calculating dynamic park factors using team K%...")
    if historical_game_df.empty:
        logger.warning("Historical game data (game_level_team_stats) is empty, cannot calculate park factors. Returning default.")
        df['park_factor_k_dynamic'] = 1.0
        return df

    # Ensure correct data types
    try:
        # Work on a copy to avoid modifying the original df passed to other functions
        hist_game_df_copy = historical_game_df.copy()
        hist_game_df_copy['game_date'] = pd.to_datetime(hist_game_df_copy['game_date'])
        hist_game_df_copy['season'] = hist_game_df_copy['game_date'].dt.year
    except Exception as e:
        logger.error(f"Error processing dates in historical_game_df: {e}")
        df['park_factor_k_dynamic'] = 1.0
        return df

    # Use team k_percent as the basis for the park factor
    stat_col = 'k_percent'
    if stat_col not in hist_game_df_copy.columns:
         logger.error(f"Required column '{stat_col}' not found in historical game data (game_level_team_stats) for park factors. Returning default.")
         df['park_factor_k_dynamic'] = 1.0
         return df

    # Map ballpark using home_team
    if 'home_team' in hist_game_df_copy.columns:
        hist_game_df_copy['ballpark'] = hist_game_df_copy['home_team'].map(TEAM_TO_BALLPARK).fillna(DEFAULT_BALLPARK)
    else:
        logger.warning("'home_team' column not found in historical data. Cannot map ballparks accurately.")
        hist_game_df_copy['ballpark'] = DEFAULT_BALLPARK

    # Calculate overall average stat per season
    league_avg_per_season = hist_game_df_copy.groupby('season')[stat_col].mean().reset_index()
    league_avg_per_season.rename(columns={stat_col: 'league_avg_stat'}, inplace=True)

    # Calculate average stat per park per season
    park_avg_per_season = hist_game_df_copy.groupby(['season', 'ballpark'])[stat_col].mean().reset_index()
    park_avg_per_season.rename(columns={stat_col: 'park_avg_stat'}, inplace=True)

    # Merge league and park averages
    park_factors = pd.merge(park_avg_per_season, league_avg_per_season, on='season', how='left')

    # Calculate park factor (handle division by zero)
    park_factors['park_factor_k_dynamic'] = np.where(
        park_factors['league_avg_stat'] > 0,
        park_factors['park_avg_stat'] / park_factors['league_avg_stat'],
        1.0 # Default to 1 if league average is 0 or NaN
    )
    park_factors['park_factor_k_dynamic'] = park_factors['park_factor_k_dynamic'].fillna(1.0) # Fill NaNs resulting from merge/division

    # Select relevant columns for merging
    park_factors = park_factors[['season', 'ballpark', 'park_factor_k_dynamic']]

    # Merge park factors onto the main dataframe
    # Ensure df has 'season' and 'ballpark' columns first
    df_copy = df.copy() # Work on copy
    try:
        df_copy['season'] = pd.to_datetime(df_copy['game_date']).dt.year
    except Exception as e:
         logger.error(f"Error processing game_date in main df: {e}")
         df_copy['season'] = datetime.now().year # Example fallback
    if 'ballpark' not in df_copy.columns: # Add ballpark mapping if not done already
         if 'home_team' in df_copy.columns:
              df_copy['ballpark'] = df_copy['home_team'].map(TEAM_TO_BALLPARK).fillna(DEFAULT_BALLPARK)
         else: df_copy['ballpark'] = DEFAULT_BALLPARK

    original_len = len(df_copy)
    df_copy = pd.merge(df_copy, park_factors, on=['season', 'ballpark'], how='left')
    df_copy['park_factor_k_dynamic'] = df_copy['park_factor_k_dynamic'].fillna(1.0)
    assert len(df_copy) == original_len, "Merge changed row count unexpectedly!"

    logger.info("Dynamic park factors calculation complete.")
    return df_copy # Return the modified copy

def calculate_rolling_pitcher_stats(df, historical_pitch_df):
    """
    Calculates rolling pitcher stats from historical pitch data (statcast_pitchers).
    Example: Rolling K%, Whiff%, Fastball Usage% over past N days/pitches.
    Refactored to reduce memory usage during intermediate steps.
    """
    logger.info("Calculating rolling pitcher stats...")
    if historical_pitch_df.empty:
        logger.warning("Historical pitch data (statcast_pitchers) is empty, cannot calculate rolling pitcher stats.")
        df['roll_pitcher_k_pct'] = 0.20; df['roll_pitcher_whiff_pct'] = 0.10; df['roll_pitcher_fastball_pct'] = 0.50
        return df

    # --- Data Prep ---
    logger.info(f"Processing {len(historical_pitch_df)} historical pitches...")
    required_cols = ['pitcher', 'game_date', 'events', 'description', 'type', 'pitch_type', 'game_pk', 'at_bat_number', 'pitch_number']
    if not all(col in historical_pitch_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in historical_pitch_df.columns]
        logger.error(f"Missing required columns in historical_pitch_df: {missing}. Cannot calculate rolling pitcher stats.")
        df['roll_pitcher_k_pct'] = 0.20; df['roll_pitcher_whiff_pct'] = 0.10; df['roll_pitcher_fastball_pct'] = 0.50
        return df

    try:
        # Work on a copy to avoid modifying original passed df
        hist_pitch_df_copy = historical_pitch_df.copy()
        hist_pitch_df_copy['game_date'] = pd.to_datetime(hist_pitch_df_copy['game_date'])
        hist_pitch_df_copy = optimize_dtypes(hist_pitch_df_copy)
        # Use assignment for sort_values
        hist_pitch_df_copy = hist_pitch_df_copy.sort_values(by=['pitcher', 'game_date', 'at_bat_number', 'pitch_number'])
    except Exception as e:
        logger.error(f"Error during historical pitch data prep: {e}")
        df['roll_pitcher_k_pct'] = 0.20; df['roll_pitcher_whiff_pct'] = 0.10; df['roll_pitcher_fastball_pct'] = 0.50
        return df

    # --- Create Flags/Metrics per Pitch ---
    loc_indexer = hist_pitch_df_copy.index
    hist_pitch_df_copy.loc[loc_indexer, 'is_k'] = hist_pitch_df_copy['events'].isin(['strikeout', 'strikeout_double_play']).astype(np.int8)
    hist_pitch_df_copy.loc[loc_indexer, 'is_swstr'] = ((hist_pitch_df_copy['type'] == 'S') &
                                       (hist_pitch_df_copy['description'].isin(['swinging_strike', 'swinging_strike_blocked']))).astype(np.int8)
    hist_pitch_df_copy.loc[loc_indexer, 'is_swing'] = hist_pitch_df_copy['description'].isin([
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score'
    ]).astype(np.int8)
    hist_pitch_df_copy.loc[loc_indexer, 'is_fastball'] = hist_pitch_df_copy['pitch_type'].isin(['FF', 'SI', 'FT', 'FC']).astype(np.int8)

    # --- Rolling Calculations ---
    logger.info(f"Applying rolling window ({ROLLING_PITCHER_WINDOW}, min_pitches={MIN_PITCHES_FOR_ROLLING})...")
    pitch_df_indexed = hist_pitch_df_copy.set_index('game_date')
    grouped_pitcher_time = pitch_df_indexed.groupby('pitcher') # Group by pitcher ID

    # Calculate rolling sums directly
    rolling_sums_df = pd.DataFrame({
        'roll_k_sum': grouped_pitcher_time['is_k'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum(),
        'roll_swstr_sum': grouped_pitcher_time['is_swstr'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum(),
        'roll_swing_sum': grouped_pitcher_time['is_swing'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum(),
        'roll_fastball_sum': grouped_pitcher_time['is_fastball'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').sum(),
        'roll_pitches_count': grouped_pitcher_time['pitcher'].rolling(ROLLING_PITCHER_WINDOW, min_periods=MIN_PITCHES_FOR_ROLLING, closed='left').count()
    })
    gc.collect()

    # Calculate rates (handle division by zero) directly on the result
    rolling_sums_df['roll_pitcher_k_pct'] = np.where(rolling_sums_df['roll_pitches_count'] > 0, rolling_sums_df['roll_k_sum'] / rolling_sums_df['roll_pitches_count'], np.nan).astype(np.float32)
    rolling_sums_df['roll_pitcher_whiff_pct'] = np.where(rolling_sums_df['roll_swing_sum'] > 0, rolling_sums_df['roll_swstr_sum'] / rolling_sums_df['roll_swing_sum'], np.nan).astype(np.float32)
    rolling_sums_df['roll_pitcher_fastball_pct'] = np.where(rolling_sums_df['roll_pitches_count'] > 0, rolling_sums_df['roll_fastball_sum'] / rolling_sums_df['roll_pitches_count'], np.nan).astype(np.float32)

    # Select only the rate columns and reset index
    rolling_rates = rolling_sums_df[['roll_pitcher_k_pct', 'roll_pitcher_whiff_pct', 'roll_pitcher_fastball_pct']].reset_index()
    del rolling_sums_df; gc.collect() # Free memory

    # --- Aggregate to Game Level ---
    logger.info("Aggregating rolling stats to game level...")
    # Get the context needed (last pitch per game) from the original df copy
    last_pitch_context = hist_pitch_df_copy.sort_values(by=['pitcher', 'game_date', 'at_bat_number', 'pitch_number']) \
                                           .drop_duplicates(subset=['pitcher', 'game_pk'], keep='last') \
                                           [['pitcher', 'game_date', 'game_pk']]
    last_pitch_context['game_date'] = pd.to_datetime(last_pitch_context['game_date'])

    # Merge the calculated rates onto this game-level context
    # Ensure types are compatible for merge
    rolling_rates['game_date'] = pd.to_datetime(rolling_rates['game_date'])
    if 'pitcher' in last_pitch_context.columns and 'pitcher' in rolling_rates.columns:
        # Ensure both are the same type before merge, default to int64 if unsure
        common_pitcher_type = np.int64
        try:
             common_pitcher_type = rolling_rates['pitcher'].dtype
             last_pitch_context['pitcher'] = last_pitch_context['pitcher'].astype(common_pitcher_type)
        except Exception as e:
             logger.warning(f"Could not align pitcher types for merge, defaulting to int64. Error: {e}")
             last_pitch_context['pitcher'] = last_pitch_context['pitcher'].astype(np.int64)
             rolling_rates['pitcher'] = rolling_rates['pitcher'].astype(np.int64)


    game_level_rolling_stats = pd.merge(
        last_pitch_context,
        rolling_rates,
        on=['pitcher', 'game_date'],
        how='left'
    )
    # Cleanup intermediate dfs
    del rolling_rates, last_pitch_context, hist_pitch_df_copy, pitch_df_indexed, grouped_pitcher_time; gc.collect()

    # --- Merge onto main df ---
    df_copy = df.copy() # Work on copy
    try:
        df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
        if 'pitcher_id' not in df_copy.columns:
             logger.error("'pitcher_id' column missing from main dataframe 'df'. Cannot merge rolling pitcher stats.")
             return df_copy # Return df without these features
        # Ensure types match for merge
        if 'pitcher' in game_level_rolling_stats.columns:
             df_copy['pitcher_id'] = df_copy['pitcher_id'].astype(game_level_rolling_stats['pitcher'].dtype)
        else: # Should not happen if merge above worked, but safety check
             logger.warning("'pitcher' column missing from game_level_rolling_stats after merge.")

        # *** Explicitly sort BOTH dataframes primarily by game_date before merge_asof ***
        logger.debug("Sorting left dataframe (df_copy) by game_date...")
        df_copy = df_copy.sort_values(by='game_date') # USE ASSIGNMENT

    except Exception as e:
        logger.error(f"Error processing game_date or pitcher_id in main df: {e}")
        return df_copy

    try:
        game_level_rolling_stats['game_date'] = pd.to_datetime(game_level_rolling_stats['game_date'])
        # *** Explicitly sort RIGHT dataframe by game_date before merge_asof ***
        logger.debug("Sorting right dataframe (game_level_rolling_stats) by game_date...")
        game_level_rolling_stats = game_level_rolling_stats.sort_values(by='game_date') # USE ASSIGNMENT

        # --- Add sort verification ---
        if not game_level_rolling_stats['game_date'].is_monotonic_increasing:
             logger.error("FATAL: Right dataframe 'game_date' column is NOT monotonically increasing after sort! Cannot perform merge_asof.")
             diffs = game_level_rolling_stats['game_date'].diff()
             non_sorted_indices = diffs[diffs < pd.Timedelta(0)].index
             if len(non_sorted_indices) > 0:
                  logger.error(f"First non-sorted index: {non_sorted_indices[0]}. Value: {game_level_rolling_stats.loc[non_sorted_indices[0], 'game_date']}, Previous value: {game_level_rolling_stats.loc[non_sorted_indices[0]-1, 'game_date']}")
             return df_copy # Exit function before merge fails catastrophically
        else:
             logger.debug("Right dataframe 'game_date' column verified as sorted.")

    except Exception as e:
         logger.error(f"Error processing or sorting game_level_rolling_stats: {e}")
         return df_copy


    logger.info("Merging game-level rolling stats onto main dataframe using merge_asof...")
    original_len = len(df_copy)
    # Use merge_asof to get the latest stats strictly BEFORE the current game_date
    df_copy = pd.merge_asof(df_copy, game_level_rolling_stats,
                       on='game_date',
                       left_by='pitcher_id', # Column in the left df (df)
                       right_by='pitcher',   # Column in the right df (game_level_rolling_stats)
                       direction='backward',
                       allow_exact_matches=False) # Only take stats from BEFORE the game
    assert len(df_copy) == original_len, "Merge changed row count unexpectedly!"

    # Handle pitchers with no prior rolling stats (fill with median of calculated stats)
    fill_values = {
        'roll_pitcher_k_pct': df_copy['roll_pitcher_k_pct'].median(),
        'roll_pitcher_whiff_pct': df_copy['roll_pitcher_whiff_pct'].median(),
        'roll_pitcher_fastball_pct': df_copy['roll_pitcher_fastball_pct'].median()
    }
    # Use assignment instead of inplace=True on a slice
    for col, value in fill_values.items():
        if col in df_copy.columns:
             # Use .fillna directly on the column Series
             df_copy[col] = df_copy[col].fillna(value)

    logger.info("Rolling pitcher stats calculation complete.")
    return df_copy # Return the modified copy


def calculate_rolling_opponent_stats(df, historical_team_df):
    """
    Calculates rolling opponent team stats based on historical game data (game_level_team_stats).
    Example: Rolling K%, BB%. Uses 'opponent_team' column from main df.
    """
    logger.info("Calculating rolling opponent team stats...")
    if historical_team_df.empty:
        logger.warning("Historical team game data (game_level_team_stats) is empty, cannot calculate rolling opponent stats.")
        # Add default columns expected later
        df['roll_opp_k_pct_vs_hand'] = 0.22
        df['roll_opp_bb_pct_vs_hand'] = 0.08
        return df

    # --- Data Prep ---
    logger.info(f"Processing {len(historical_team_df)} historical team games...")
    # Use confirmed column names, including 'opponent' from game_level_team_stats
    required_cols = ['game_date', 'team', 'k_percent', 'bb_percent', 'game_pk', 'opponent', 'home_team'] # Use opponent
    if not all(col in historical_team_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in historical_team_df.columns]
        logger.error(f"Missing required columns in historical_team_df: {missing}. Cannot calculate rolling opponent stats.")
        df['roll_opp_k_pct_vs_hand'] = 0.22; df['roll_opp_bb_pct_vs_hand'] = 0.08
        return df

    try:
        # Work on a copy
        hist_team_df_copy = historical_team_df.copy()
        hist_team_df_copy['game_date'] = pd.to_datetime(hist_team_df_copy['game_date'])
        hist_team_df_copy = optimize_dtypes(hist_team_df_copy) # Optimize dtypes
        # Use assignment for sort_values
        hist_team_df_copy = hist_team_df_copy.sort_values(by=['team', 'game_date'])
    except Exception as e:
        logger.error(f"Error during historical team data prep: {e}")
        df['roll_opp_k_pct_vs_hand'] = 0.22; df['roll_opp_bb_pct_vs_hand'] = 0.08
        return df

    # --- Rolling Calculations ---
    grouped_team = hist_team_df_copy.groupby('team')
    logger.info(f"Applying rolling window ({ROLLING_OPPONENT_WINDOW} games, min_games={MIN_GAMES_FOR_ROLLING})...")
    # Use shift(1) to ensure we use data PRIOR to the current game
    hist_team_df_copy['roll_opp_k_pct'] = grouped_team['k_percent'].shift(1).rolling(ROLLING_OPPONENT_WINDOW, min_periods=MIN_GAMES_FOR_ROLLING).mean().astype(np.float32)
    hist_team_df_copy['roll_opp_bb_pct'] = grouped_team['bb_percent'].shift(1).rolling(ROLLING_OPPONENT_WINDOW, min_periods=MIN_GAMES_FOR_ROLLING).mean().astype(np.float32)

    # --- Merge onto main df ---
    logger.info("Merging rolling opponent stats onto main dataframe...")
    # Prepare the lookup table with opponent stats
    opponent_rolling_stats = hist_team_df_copy[['team', 'game_date', 'roll_opp_k_pct', 'roll_opp_bb_pct']].copy()
    # Rename 'team' to 'opponent_team' to match the column in the main df 'df'
    opponent_rolling_stats.rename(columns={'team': 'opponent_team'}, inplace=True) # RENAME to opponent_team for merge target

    # Ensure game_date is datetime in main df and sort both dfs
    df_copy = df.copy() # Work on copy
    try:
        df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
        # Ensure 'opponent_team' column exists in df (as confirmed by user)
        if 'opponent_team' not in df_copy.columns:
             logger.error("Main dataframe 'df' is missing the 'opponent_team' column required for merging opponent stats.")
             # Add default columns before returning
             df_copy['roll_opp_k_pct_vs_hand'] = 0.22
             df_copy['roll_opp_bb_pct_vs_hand'] = 0.08
             return df_copy # Return df without these features
        # Ensure types match for merge
        # Ensure opponent_team column type is compatible (likely object/string)
        if opponent_rolling_stats['opponent_team'].dtype != df_copy['opponent_team'].dtype:
             logger.warning(f"Attempting to align opponent_team column types: {df_copy['opponent_team'].dtype} (df) vs {opponent_rolling_stats['opponent_team'].dtype} (stats)")
             try:
                  # Convert df column to match stats column type (usually object/string)
                  df_copy['opponent_team'] = df_copy['opponent_team'].astype(opponent_rolling_stats['opponent_team'].dtype)
             except Exception as type_e:
                  logger.error(f"Could not align opponent_team column types: {type_e}")
                  # Add default columns before returning
                  df_copy['roll_opp_k_pct_vs_hand'] = 0.22
                  df_copy['roll_opp_bb_pct_vs_hand'] = 0.08
                  return df_copy

        # Use assignment for sort_values - sort primarily by date for merge_asof
        df_copy = df_copy.sort_values(by='game_date') # Sort target df by date
    except Exception as e:
         logger.error(f"Error processing game_date or opponent_team column in main df: {e}")
         # Add default columns before returning
         df_copy['roll_opp_k_pct_vs_hand'] = 0.22
         df_copy['roll_opp_bb_pct_vs_hand'] = 0.08
         return df_copy

    # Use assignment for sort_values - sort primarily by date for merge_asof
    opponent_rolling_stats = opponent_rolling_stats.sort_values(by='game_date') # Sort source df by date
    # --- Add sort verification ---
    if not opponent_rolling_stats['game_date'].is_monotonic_increasing:
        logger.error("FATAL: Opponent rolling stats 'game_date' column is NOT monotonically increasing after sort! Cannot perform merge_asof.")
        # Add default columns before returning
        df_copy['roll_opp_k_pct_vs_hand'] = 0.22
        df_copy['roll_opp_bb_pct_vs_hand'] = 0.08
        return df_copy
    else:
        logger.debug("Opponent rolling stats 'game_date' column verified as sorted.")


    original_len = len(df_copy)
    # Use merge_asof to get the latest opponent stats strictly BEFORE the current game_date
    df_copy = pd.merge_asof(df_copy, opponent_rolling_stats,
                       on='game_date',
                       by='opponent_team', # Use 'opponent_team' column
                       direction='backward',
                       allow_exact_matches=False) # Get stats strictly before the game date
    assert len(df_copy) == original_len, "Merge changed row count unexpectedly!"

    # --- Vs Hand Calculation (Placeholder) ---
    df_copy['roll_opp_k_pct_vs_hand'] = df_copy['roll_opp_k_pct'] # Use overall as placeholder
    df_copy['roll_opp_bb_pct_vs_hand'] = df_copy['roll_opp_bb_pct'] # Use overall as placeholder
    logger.warning("Rolling opponent stats 'vs_hand' are currently using overall stats. Implement hand-specific logic if needed.")

    # Handle teams with no prior rolling stats
    fill_values = {
        'roll_opp_k_pct_vs_hand': df_copy['roll_opp_k_pct_vs_hand'].median(),
        'roll_opp_bb_pct_vs_hand': df_copy['roll_opp_bb_pct_vs_hand'].median(),
    }
    # Fix FutureWarning: Use assignment instead of inplace=True on a slice
    for col, value in fill_values.items():
         if col in df_copy.columns:
              # Use .fillna directly on the column Series
              df_copy[col] = df_copy[col].fillna(value)
    # Drop intermediate columns used for merging if no longer needed
    df_copy.drop(columns=['roll_opp_k_pct', 'roll_opp_bb_pct'], errors='ignore', inplace=True)

    logger.info("Rolling opponent stats calculation complete.")
    return df_copy # Return modified copy

def calculate_interaction_features(df):
    """Calculates interaction features between pitcher, opponent, and park."""
    logger.info("Calculating interaction features...")
    df_copy = df.copy() # Work on copy
    # Ensure component features exist, using .get() with defaults for safety

    # Park x Pitcher K rate (using rolling pitcher K% now)
    df_copy['interact_park_pitcher_k'] = df_copy.get('park_factor_k_dynamic', 1.0) * df_copy.get('roll_pitcher_k_pct', 0.20)

    # Pitcher Whiff Rate x Opponent K Rate
    df_copy['interact_whiff_opp_k'] = df_copy.get('roll_pitcher_whiff_pct', 0.10) * df_copy.get('roll_opp_k_pct_vs_hand', 0.22)

    # Pitcher Fastball Usage x Opponent K Rate
    df_copy['interact_fb_opp_k'] = df_copy.get('roll_pitcher_fastball_pct', 0.50) * df_copy.get('roll_opp_k_pct_vs_hand', 0.22)

    logger.info("Interaction features calculation complete.")
    return df_copy # Return modified copy

# --- Main Function ---
def create_advanced_features(args):
    """Loads data, calculates advanced features, and saves to the database."""
    db_path = Path(DBConfig.PATH)
    prediction_mode = args.prediction_date is not None
    max_historical_date = args.prediction_date if prediction_mode else '9999-12-31'

    if prediction_mode:
        logger.info(f"Running in PREDICTION mode for date: {args.prediction_date}")
        input_table = "prediction_features"
        output_table = "prediction_features_advanced"
        date_filter = f"WHERE DATE(game_date) = '{args.prediction_date}'"
    else:
        logger.info("Running in TRAINING mode (processing train and test sets).")
        input_table_train = "train_features"
        input_table_test = "test_features"
        output_table_train = "train_features_advanced"
        output_table_test = "test_features_advanced"

    # --- Load Data ---
    logger.info("Loading baseline features and historical data...")
    try:
        with DBConnection(db_path) as conn:
            # Load baseline features (train/test or prediction)
            if prediction_mode:
                df = pd.read_sql_query(f"SELECT * FROM {input_table} {date_filter}", conn)
                logger.info(f"Loaded {len(df)} rows from '{input_table}' for prediction date.")
                if df.empty:
                    logger.error(f"No baseline prediction features found for {args.prediction_date}. Run engineer_features first.")
                    sys.exit(1)
            else:
                train_df = pd.read_sql_query(f"SELECT * FROM {input_table_train}", conn)
                test_df = pd.read_sql_query(f"SELECT * FROM {input_table_test}", conn)
                df = pd.concat([train_df, test_df], ignore_index=True)
                logger.info(f"Loaded {len(train_df)} train rows, {len(test_df)} test rows. Total: {len(df)}")
                del train_df, test_df; gc.collect()

            # --- Load Necessary Historical Data ---
            logger.info(f"Loading historical data strictly before {max_historical_date}...")

            # Load PITCH data (statcast_pitchers) - Select only necessary columns
            logger.info("Loading historical pitch data (statcast_pitchers)...")
            # Reduce columns loaded to absolute minimum needed for current calcs
            pitch_cols = ['pitcher', 'game_pk', 'game_date', 'pitch_type',
                          'description', 'events', 'type', 'at_bat_number', 'pitch_number']
            try:
                pitch_query = f"SELECT {', '.join(pitch_cols)} FROM statcast_pitchers WHERE DATE(game_date) < '{max_historical_date}'"
                historical_pitch_df = pd.read_sql_query(pitch_query, conn)
                logger.info(f"Loaded {len(historical_pitch_df)} historical pitches.")
                # Optimize after loading, before processing
                historical_pitch_df = optimize_dtypes(historical_pitch_df)
                gc.collect()
            except Exception as e:
                logger.error(f"Failed to load historical pitch data: {e}", exc_info=True)
                historical_pitch_df = pd.DataFrame() # Ensure it's an empty df if loading fails

            # Load GAME data (game_level_team_stats) - Use confirmed columns
            logger.info("Loading historical game data (game_level_team_stats)...")
            # Select columns needed for opponent stats and park factors - USE 'opponent'
            game_cols = ['game_pk', 'game_date', 'team', 'opponent', 'home_team', # Use opponent
                         'k_percent', 'bb_percent'] # Add others if needed
            try:
                game_query = f"SELECT {', '.join(game_cols)} FROM game_level_team_stats WHERE DATE(game_date) < '{max_historical_date}'"
                historical_game_df = pd.read_sql_query(game_query, conn)
                logger.info(f"Loaded {len(historical_game_df)} historical team games.")
                # Optimize after loading
                historical_game_df = optimize_dtypes(historical_game_df)
                gc.collect()
            except sqlite3.OperationalError as oe: # Catch specific error
                 # Check for both opponent and opponent_team just in case
                 if "no such column: opponent" in str(oe).lower() or "no such column: opponent_team" in str(oe).lower():
                      logger.error(f"Failed to load historical game data: Column 'opponent' or 'opponent_team' not found in 'game_level_team_stats'. Please check table schema. Error: {oe}")
                 else:
                      logger.error(f"Failed to load historical game data (OperationalError): {oe}", exc_info=True)
                 historical_game_df = pd.DataFrame() # Ensure it's an empty df if loading fails
            except Exception as e:
                 logger.error(f"Failed to load historical game data: {e}", exc_info=True)
                 historical_game_df = pd.DataFrame() # Ensure it's an empty df if loading fails

    except Exception as e:
        logger.error(f"Error during initial data loading phase: {e}", exc_info=True); sys.exit(1)

    # --- Add Ballpark Info ---
    if 'home_team' in df.columns:
         logger.info("Mapping teams to ballparks...")
         df['ballpark'] = df['home_team'].map(TEAM_TO_BALLPARK).fillna(DEFAULT_BALLPARK)
         logger.info(f"Ballpark mapping complete. Found {df['ballpark'].nunique()} unique parks.")
         unmapped_teams = df[df['ballpark'] == DEFAULT_BALLPARK]['home_team'].unique()
         if len(unmapped_teams) > 0:
              logger.warning(f"Unmapped teams found, assigned '{DEFAULT_BALLPARK}': {list(unmapped_teams)}")
    else:
         logger.warning("Column 'home_team' not found in baseline features. Cannot map ballparks accurately.")
         df['ballpark'] = DEFAULT_BALLPARK

    # --- Calculate Advanced Features ---
    logger.info("Starting advanced feature calculation pipeline...")
    # Wrap calculations in try-except blocks for better error isolation
    try:
        with tqdm(total=4, desc="Advanced Features") as pbar:
            df = calculate_dynamic_park_factors(df, historical_game_df); pbar.update(1); gc.collect()
            df = calculate_rolling_pitcher_stats(df, historical_pitch_df); pbar.update(1); gc.collect()
            # Clean up large pitch df as soon as it's not needed
            if 'historical_pitch_df' in locals(): del historical_pitch_df; gc.collect()
            df = calculate_rolling_opponent_stats(df, historical_game_df); pbar.update(1); gc.collect()
            # Clean up game df
            if 'historical_game_df' in locals(): del historical_game_df; gc.collect()
            df = calculate_interaction_features(df); pbar.update(1); gc.collect()
        logger.info("Advanced feature calculation pipeline finished.")
    except Exception as e:
         logger.error(f"Error during advanced feature calculation: {e}", exc_info=True)
         sys.exit(1) # Exit if calculation fails

    # Final check for NaNs/Infs introduced by calculations
    logger.info("Performing final NaN/Inf check and imputation...")
    try:
        # Convert potential object columns resulting from merges/calcs to numeric if possible
        for col in df.select_dtypes(include='object').columns:
             # Ensure 'opponent_team' (and 'opponent') is also excluded if it exists
            if col not in ['player_name', 'home_team', 'away_team', 'p_throws', 'team', 'opponent', 'opponent_team', 'ballpark']:
                 try:
                      df[col] = pd.to_numeric(df[col], errors='coerce')
                      # logger.debug(f"Converted object column '{col}' to numeric.")
                 except Exception:
                      logger.warning(f"Could not convert object column '{col}' to numeric.")

        numeric_cols = df.select_dtypes(include=np.number).columns
        nan_mask = df[numeric_cols].isnull()
        if nan_mask.any().any():
            logger.warning(f"NaNs found after feature calculation in numeric columns: {nan_mask.sum().loc[lambda x: x>0].index.tolist()}.")
            logger.info("Applying simple median imputation for remaining NaNs...")
            for col in tqdm(numeric_cols[nan_mask.any()], desc="Imputing NaNs", leave=False):
                median_val = df[col].median()
                if pd.isna(median_val): median_val = 0 # Fallback if median is NaN
                # Fix FutureWarning: Use assignment instead of inplace=True
                df[col] = df[col].fillna(median_val)

        inf_mask = np.isinf(df.select_dtypes(include=np.number))
        if inf_mask.any().any():
            logger.warning(f"Infinite values found after calculation in columns: {inf_mask.sum().loc[lambda x: x>0].index.tolist()}. Replacing with NaN then imputing.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Re-run imputation for columns that now have NaNs
            numeric_cols = df.select_dtypes(include=np.number).columns # Re-select numeric cols
            nan_mask = df[numeric_cols].isnull()
            if nan_mask.any().any():
                 for col in tqdm(numeric_cols[nan_mask.any()], desc="Re-Imputing NaNs after Inf replacement", leave=False):
                    median_val = df[col].median()
                    if pd.isna(median_val): median_val = 0
                    # Fix FutureWarning: Use assignment instead of inplace=True
                    df[col] = df[col].fillna(median_val)
    except Exception as e:
         logger.error(f"Error during final NaN/Inf handling: {e}", exc_info=True)
         # Decide whether to exit or proceed with potentially problematic data
         # sys.exit(1)


    # --- Save Results ---
    logger.info("Saving advanced features to database...")
    try:
        with DBConnection(db_path) as conn:
            target_table = output_table if prediction_mode else None # Use specific table in pred mode

            if prediction_mode:
                 # Ensure prediction_mode df has necessary columns if schema exists
                 # Or just replace entirely
                 df.to_sql(target_table, conn, if_exists='replace', index=False)
                 logger.info(f"Saved {len(df)} rows to '{target_table}' for date {args.prediction_date}.")
            else:
                # Split back into train/test and save separately for training mode
                if 'season' not in df.columns:
                     logger.error("Cannot split train/test - 'season' column missing after feature generation.")
                     sys.exit(1)
                train_mask = df['season'].isin(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)
                test_mask = df['season'].isin(StrikeoutModelConfig.DEFAULT_TEST_YEARS)

                train_advanced_df = df[train_mask]
                test_advanced_df = df[test_mask]

                train_advanced_df.to_sql(output_table_train, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(train_advanced_df)} rows to '{output_table_train}'.")
                test_advanced_df.to_sql(output_table_test, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(test_advanced_df)} rows to '{output_table_test}'.")

    except Exception as e:
        logger.error(f"Error saving advanced features to database: {e}", exc_info=True); sys.exit(1)

    logger.info("Advanced feature creation finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    create_advanced_features(args)
    logger.info("--- Create Advanced Features Script Completed ---")

