import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path
import sys
import time
import gc
import argparse

# --- Setup ---
try:
    # Assuming script is run via python -m src.features.create_advanced_features
    from src.config import DBConfig
    from src.data.utils import setup_logger, DBConnection
except ImportError:
    # Fallback if running directly or src path issue
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config import DBConfig
    from src.data.utils import setup_logger, DBConnection

logger = setup_logger('create_advanced_features', level=logging.INFO)
db_path = Path(DBConfig.PATH)
MIN_HISTORY_FOR_FACTOR = 50 # Min games needed at park for dynamic factor calc
ROLLING_PITCH_WINDOW = 200 # Window size for rolling pitch-level stats
ROLLING_GAME_WINDOW = 20 # Window size for rolling game-level stats
MIN_PITCH_PERIODS = 50 # Min pitches needed for rolling pitch stats
MIN_GAME_PERIODS = 5 # Min games needed for rolling game stats

# --- Ballpark Mapping ---
TEAM_BALLPARK_MAP = {
    'ARI': 'Chase Field','ATL': 'Truist Park','BAL': 'Oriole Park at Camden Yards',
    'BOS': 'Fenway Park','CHC': 'Wrigley Field','CWS': 'Guaranteed Rate Field',
    'CIN': 'Great American Ball Park','CLE': 'Progressive Field','COL': 'Coors Field',
    'DET': 'Comerica Park','HOU': 'Minute Maid Park','KC':  'Kauffman Stadium',
    'LAA': 'Angel Stadium','LAD': 'Dodger Stadium','MIA': 'loanDepot park',
    'MIL': 'American Family Field','MIN': 'Target Field','NYM': 'Citi Field',
    'NYY': 'Yankee Stadium','ATH': 'Oakland Coliseum', 'OAK': 'Oakland Coliseum',
    'PHI': 'Citizens Bank Park','PIT': 'PNC Park','SD':  'Petco Park',
    'SF':  'Oracle Park','SEA': 'T-Mobile Park','STL': 'Busch Stadium',
    'TB':  'Tropicana Field','TEX': 'Globe Life Field','TOR': 'Rogers Centre',
    'WSH': 'Nationals Park', 'WAS': 'Nationals Park'
}

# --- Functions ---

def update_team_mapping_with_ballparks(db_path):
    """Reads team_mapping, adds ballpark, saves back."""
    # (Identical to previous version)
    logger.info("Updating team_mapping table with ballpark information...")
    table_name = 'team_mapping'; abbr_col = 'team_abbr' # Default expected column
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            cursor = conn.cursor(); cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone(): logger.error(f"Table '{table_name}' not found."); return False
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            if df.empty: logger.error(f"'{table_name}' table is empty."); return False
            if 'team_abbr' not in df.columns: # Try to find alternative
                 abbr_col = next((c for c in ['Abbreviation', 'Abbr', 'TeamID'] if c in df.columns), None)
                 if not abbr_col: logger.error(f"Could not find team abbreviation column in {table_name}"); return False
                 logger.warning(f"Using '{abbr_col}' as team abbreviation column.")

            df['ballpark'] = df[abbr_col].map(TEAM_BALLPARK_MAP)
            missing = df[df['ballpark'].isnull()][abbr_col].unique()
            if len(missing) > 0: logger.warning(f"Could not map ballparks for teams: {missing}")
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Successfully updated '{table_name}' with ballpark column.")
            return True
    except Exception as e: logger.error(f"Error updating {table_name}: {e}", exc_info=True); return False

def add_ballpark_column(df, db_path):
    """Adds ballpark column to a DataFrame by merging with team_mapping."""
    # (Identical to previous version)
    logger.info("Adding ballpark column...")
    if 'home_team' not in df.columns: logger.error("Input DataFrame missing 'home_team'."); return df
    try:
        with DBConnection(db_path) as conn: team_map_df = pd.read_sql_query("SELECT team_abbr, ballpark FROM team_mapping", conn)
        if team_map_df.empty: raise ValueError("team_mapping table is empty.")
        if 'team_abbr' not in team_map_df.columns: # Find alternative abbr col if needed
             abbr_col = next((c for c in ['Abbreviation', 'Abbr', 'TeamID'] if c in team_map_df.columns), None)
             if not abbr_col: raise ValueError("Abbreviation column not found in team_mapping")
             team_map_df = team_map_df.rename(columns={abbr_col:'team_abbr'})

        df_with_ballpark = pd.merge(df, team_map_df[['team_abbr', 'ballpark']], left_on='home_team', right_on='team_abbr', how='left')
        if 'team_abbr' in df_with_ballpark.columns: df_with_ballpark = df_with_ballpark.drop(columns=['team_abbr'])
        missing = df_with_ballpark['ballpark'].isnull().sum()
        if missing > 0: logger.warning(f"{missing} rows could not be mapped to a ballpark.")
        logger.info("Ballpark column added.")
        return df_with_ballpark
    except Exception as e: logger.error(f"Error adding ballpark column: {e}", exc_info=True); return df

def calculate_dynamic_park_factors(db_path, target_col='strikeouts'):
    """Calculates historical average target value per ballpark (expanding window)."""
    # (Identical to previous version)
    logger.info(f"Calculating dynamic park factors based on historical '{target_col}'...")
    query = f"SELECT game_pk, game_date, home_team, {target_col} FROM game_level_pitchers" # Assumes target is here
    try:
        with DBConnection(db_path) as conn: game_data = pd.read_sql_query(query, conn)
        if game_data.empty: raise ValueError("game_level_pitchers is empty.")
        if target_col not in game_data.columns: raise ValueError(f"Target column '{target_col}' not found.")
        game_data['game_date'] = pd.to_datetime(game_data['game_date'])
        game_data = game_data.sort_values(by='game_date')
        game_data = add_ballpark_column(game_data, db_path) # Add ballpark info
        if 'ballpark' not in game_data.columns or game_data['ballpark'].isnull().all(): raise ValueError("Ballpark column missing/null.")
        global_mean_target = game_data[target_col].mean(); logger.info(f"Global mean for '{target_col}': {global_mean_target:.3f}")
        game_data[f'{target_col}_lag1'] = game_data.groupby('ballpark')[target_col].shift(1)
        expanding_mean = game_data.groupby('ballpark')[f'{target_col}_lag1'].expanding(min_periods=MIN_HISTORY_FOR_FACTOR).mean()
        expanding_mean = expanding_mean.reset_index(level=0, drop=True) # Align index back
        game_data['park_factor_dynamic'] = expanding_mean
        nans = game_data['park_factor_dynamic'].isnull().sum()
        if nans > 0: logger.info(f"Filling {nans} missing dynamic park factors with global mean {global_mean_target:.3f}"); game_data['park_factor_dynamic'].fillna(global_mean_target, inplace=True)
        park_factors_df = game_data[['game_pk', 'park_factor_dynamic']].copy(); logger.info("Dynamic park factors calculated.")
        return park_factors_df
    except Exception as e: logger.error(f"Error calculating dynamic park factors: {e}", exc_info=True); return pd.DataFrame()

# --- Expanded Granular Pitcher Stats ---
def calculate_granular_pitcher_stats(db_path):
    """
    Calculates rolling pitcher stats (e.g., whiff rates, usage) from Statcast.
    WARNING: Loads full statcast_pitchers table!
    """
    logger.info("Calculating granular pitcher stats (Expanding examples)...")
    logger.warning("This function loads the full 'statcast_pitchers' table - may require significant memory!")
    table_name = 'statcast_pitchers'
    # Define required columns for calculations
    required_cols = ['pitcher', 'game_pk', 'game_date', 'pitch_type', 'description', 'balls', 'strikes']
    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor(); cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone(): raise FileNotFoundError(f"Table '{table_name}' not found.")
            sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
            if not all(col in sample.columns for col in required_cols): raise ValueError(f"Missing required columns in '{table_name}'. Need: {required_cols}")
            # Load necessary columns only
            df = pd.read_sql_query(f"SELECT {', '.join(required_cols)} FROM {table_name}", conn)
            if df.empty: raise ValueError(f"'{table_name}' table is empty.")

        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['pitcher', 'game_date', 'game_pk']) # Sort for rolling calc

        # --- Feature 1: Rolling Whiff Rate on Breaking Balls ---
        breaking_types = ['SL', 'CU', 'KC', 'CS', 'SC', 'ST'] # Added ST (Sweeper)
        df['is_breaking'] = df['pitch_type'].isin(breaking_types)
        df['is_whiff'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
        df['is_breaking_whiff'] = df['is_breaking'] & df['is_whiff']
        grouped = df.groupby('pitcher')
        rolling_breaking_whiffs = grouped['is_breaking_whiff'].shift(1).rolling(window=ROLLING_PITCH_WINDOW, min_periods=MIN_PITCH_PERIODS).sum()
        rolling_breaking_pitches = grouped['is_breaking'].shift(1).rolling(window=ROLLING_PITCH_WINDOW, min_periods=MIN_PITCH_PERIODS).sum()
        df['roll_breaking_whiff_pct'] = rolling_breaking_whiffs / rolling_breaking_pitches

        # --- Feature 2: Rolling Fastball Usage % ---
        # Define fastball types (adjust as needed)
        fastball_types = ['FF', 'SI', 'FT', 'FC'] # 4-Seam, Sinker, 2-Seam, Cutter
        df['is_fastball'] = df['pitch_type'].isin(fastball_types)
        rolling_fastballs = grouped['is_fastball'].shift(1).rolling(window=ROLLING_PITCH_WINDOW, min_periods=MIN_PITCH_PERIODS).sum()
        rolling_pitches = grouped['pitcher'].shift(1).rolling(window=ROLLING_PITCH_WINDOW, min_periods=MIN_PITCH_PERIODS).count() # Count pitches
        df['roll_fastball_usage_pct'] = rolling_fastballs / rolling_pitches

        # --- Feature 3: Rolling K% when Ahead in Count ---
        df['is_ahead_count'] = df['strikes'] > df['balls']
        df['is_k_outcome'] = df['description'].str.contains('strikeout', na=False) # Check description for K
        # Calculate K when ahead, and pitches thrown when ahead
        df['is_k_when_ahead'] = df['is_k_outcome'] & df['is_ahead_count']
        df['is_pitch_when_ahead'] = df['is_ahead_count']
        rolling_k_ahead = grouped['is_k_when_ahead'].shift(1).rolling(window=ROLLING_PITCH_WINDOW, min_periods=MIN_PITCH_PERIODS).sum()
        rolling_pitches_ahead = grouped['is_pitch_when_ahead'].shift(1).rolling(window=ROLLING_PITCH_WINDOW, min_periods=MIN_PITCH_PERIODS).sum()
        df['roll_k_pct_ahead_count'] = rolling_k_ahead / rolling_pitches_ahead

        # --- Aggregate to Game Level ---
        # Take the value from the last pitch of each game for each pitcher
        final_game_stats = df.loc[df.groupby(['pitcher', 'game_pk']).tail(1).index].copy()
        final_game_stats = final_game_stats[['pitcher', 'game_pk', 'roll_breaking_whiff_pct', 'roll_fastball_usage_pct', 'roll_k_pct_ahead_count']]

        # Fill NaNs with median or other sensible default for each new feature
        for col in ['roll_breaking_whiff_pct', 'roll_fastball_usage_pct', 'roll_k_pct_ahead_count']:
            median_val = final_game_stats[col].median()
            median_val = median_val if pd.notna(median_val) else 0 # Ensure not NaN
            nan_count = final_game_stats[col].isnull().sum()
            if nan_count > 0:
                logger.warning(f"Filling {nan_count} NaNs in '{col}' with median {median_val:.3f}")
                final_game_stats[col].fillna(median_val, inplace=True)

        logger.info(f"Calculated granular stats examples for {len(final_game_stats)} pitcher-games.")
        # Rename pitcher to pitcher_id for merging
        final_game_stats = final_game_stats.rename(columns={'pitcher':'pitcher_id'})
        return final_game_stats

    except Exception as e:
        logger.error(f"Error calculating granular pitcher stats: {e}", exc_info=True)
        return pd.DataFrame()

# --- Expanded Opponent Details ---
def calculate_opponent_details(db_path):
    """
    Calculates rolling opponent team stats vs pitcher handedness.
    WARNING: Loads full statcast_batters table!
    """
    logger.info("Calculating opponent details (Expanding examples)...")
    logger.warning("This function loads the full 'statcast_batters' table - may require significant memory!")
    table_name = 'statcast_batters'
    required_cols = ['game_pk', 'game_date', 'events', 'p_throws', 'inning_topbot', 'home_team', 'away_team', 'description', 'type']
    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor(); cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone(): raise FileNotFoundError(f"Table '{table_name}' not found.")
            sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
            if not all(col in sample.columns for col in required_cols): raise ValueError(f"Missing required columns in '{table_name}'. Need: {required_cols}")
            df = pd.read_sql_query(f"SELECT {', '.join(required_cols)} FROM {table_name}", conn)
            if df.empty: raise ValueError(f"'{table_name}' table is empty.")

        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['game_date'])

        # Determine batting team
        df['batting_team'] = df.apply(lambda row: row['away_team'] if row['inning_topbot'] == 'Top' else row['home_team'], axis=1)

        # Identify outcomes (PA, K, Swing, Contact)
        df['is_k'] = df['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)
        pa_events = ['strikeout', 'walk', 'hit_by_pitch', 'field_out', 'single', 'double', 'triple', 'home_run', 'force_out', 'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out', 'sac_fly', 'sac_bunt', 'double_play', 'triple_play', 'strikeout_double_play']
        df['is_pa'] = df['events'].isin(pa_events).astype(int)
        df['is_swing'] = df['description'].isin(['hit_into_play', 'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked', 'hit_into_play_no_out', 'hit_into_play_score']).astype(int)
        # Contact = Swing - Whiff (swinging strike)
        df['is_contact_swing'] = (df['is_swing'] == 1) & (~df['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip']))

        # Aggregate outcomes per game, batting team, pitcher hand faced
        game_agg = df.groupby(['game_pk', 'game_date', 'batting_team', 'p_throws'])[['is_k', 'is_pa', 'is_swing', 'is_contact_swing']].sum().reset_index()

        # Calculate rolling stats per team vs hand
        game_agg = game_agg.sort_values(['batting_team', 'p_throws', 'game_date'])
        grouped = game_agg.groupby(['batting_team', 'p_throws'])

        # Shift by 1 to use past games only
        lagged_k = grouped['is_k'].shift(1)
        lagged_pa = grouped['is_pa'].shift(1)
        lagged_swing = grouped['is_swing'].shift(1)
        lagged_contact = grouped['is_contact_swing'].shift(1)

        # Calculate rolling sums
        rolling_k = lagged_k.rolling(window=ROLLING_GAME_WINDOW, min_periods=MIN_GAME_PERIODS).sum()
        rolling_pa = lagged_pa.rolling(window=ROLLING_GAME_WINDOW, min_periods=MIN_GAME_PERIODS).sum()
        rolling_swing = lagged_swing.rolling(window=ROLLING_GAME_WINDOW, min_periods=MIN_GAME_PERIODS).sum()
        rolling_contact = lagged_contact.rolling(window=ROLLING_GAME_WINDOW, min_periods=MIN_GAME_PERIODS).sum()

        # Calculate rolling rates
        game_agg[f'roll_{ROLLING_GAME_WINDOW}g_k_pct_vs_hand'] = rolling_k / rolling_pa
        game_agg[f'roll_{ROLLING_GAME_WINDOW}g_swing_pct_vs_hand'] = rolling_swing / rolling_pa # Swing rate per PA
        game_agg[f'roll_{ROLLING_GAME_WINDOW}g_contact_pct_vs_hand'] = rolling_contact / rolling_swing # Contact rate per Swing

        # Prepare for merging
        opp_stats = game_agg[['game_pk', 'batting_team', 'p_throws',
                              f'roll_{ROLLING_GAME_WINDOW}g_k_pct_vs_hand',
                              f'roll_{ROLLING_GAME_WINDOW}g_swing_pct_vs_hand',
                              f'roll_{ROLLING_GAME_WINDOW}g_contact_pct_vs_hand']].copy()
        opp_stats = opp_stats.rename(columns={'batting_team': 'opponent_team', 'p_throws': 'pitcher_hand_faced'})

        logger.info(f"Calculated rolling opponent details vs hand for {len(opp_stats)} game-team-hand rows.")
        return opp_stats

    except Exception as e:
        logger.error(f"Error calculating opponent details: {e}", exc_info=True)
        return pd.DataFrame()

# --- Expanded Interaction Features ---
def create_interaction_features(df):
    """Creates interaction features."""
    logger.info("Creating interaction features (Expanding examples)...")
    prefix = "interact_"

    # Example 1: Dynamic Park Factor * Pitcher's Rolling K/9
    f1, f2 = 'park_factor_dynamic', 'ewma_10g_k_per_9'
    if f1 in df.columns and f2 in df.columns:
         df[f'{prefix}park_pitcher_k9'] = df[f1] * df[f2]; logger.info(f"Created '{prefix}park_pitcher_k9'.")

    # Example 2: Opponent Rolling K% vs Pitcher Hand * Pitcher Rolling Breaking Whiff %
    f1, f2 = f'roll_{ROLLING_GAME_WINDOW}g_k_pct_vs_hand', 'roll_100p_breaking_whiff_pct'
    if f1 in df.columns and f2 in df.columns:
        # Fill NaNs first (using means/medians calculated during merge step is better)
        df[f1].fillna(df[f1].median(), inplace=True)
        df[f2].fillna(df[f2].median(), inplace=True)
        df[f'{prefix}opp_k_pitcher_break_whiff'] = df[f1] * df[f2]; logger.info(f"Created '{prefix}opp_k_pitcher_break_whiff'.")

    # Example 3: Pitcher Velocity vs Opponent Contact Rate
    f1, f2 = 'ewma_10g_avg_velocity', f'roll_{ROLLING_GAME_WINDOW}g_contact_pct_vs_hand'
    if f1 in df.columns and f2 in df.columns:
        df[f1].fillna(df[f1].median(), inplace=True)
        df[f2].fillna(df[f2].median(), inplace=True)
        df[f'{prefix}velo_opp_contact'] = df[f1] * (1 - df[f2]) # Interact with non-contact rate
        logger.info(f"Created '{prefix}velo_opp_contact'.")

    # Example 4: Days Rest * Rolling Pitch Count (EWMA)
    f1, f2 = 'days_since_last_game', 'ewma_10g_total_pitches'
    if f1 in df.columns and f2 in df.columns:
         # Fill NaNs for days_since_last_game (e.g., for debuts) - assume avg rest?
         df[f1].fillna(5, inplace=True)
         df[f2].fillna(df[f2].median(), inplace=True)
         df[f'{prefix}rest_workload'] = df[f1] * df[f2]; logger.info(f"Created '{prefix}rest_workload'.")

    return df

# --- Main Orchestration ---
def main(args):
    """Main function to run the advanced feature engineering pipeline."""
    logger.info(f"--- Starting Advanced Feature Engineering (Mode: {args.mode}) ---")
    start_time = time.time()

    # Step 1: Update team mapping (Run once or periodically)
    if args.update_mapping:
        update_team_mapping_with_ballparks(db_path)

    # Step 2: Determine Input/Output Table Names
    # (Same as previous version)
    if args.mode == 'train': input_table, output_table = 'train_features', 'train_features_advanced'
    elif args.mode == 'test': input_table, output_table = 'test_features', 'test_features_advanced'
    elif args.mode == 'predict': input_table, output_table = 'prediction_features', 'prediction_features_advanced'
    else: logger.error(f"Invalid mode: {args.mode}"); return
    if args.mode == 'predict' and not args.date: logger.error("--date required for --mode=predict"); return
    if args.mode == 'predict': logger.info(f"Processing prediction features for date: {args.date}")

    # Step 3: Load base features
    logger.info(f"Loading base features from '{input_table}'...")
    try:
        with DBConnection(db_path) as conn:
            if args.mode == 'predict': query = f"SELECT * FROM {input_table} WHERE date(game_date) = ?"; params = (args.date,)
            else: query = f"SELECT * FROM {input_table}"; params = None
            base_df = pd.read_sql_query(query, conn, params=params)
        if base_df.empty: raise ValueError(f"{input_table} table is empty or no data for date.")
        base_df['game_date'] = pd.to_datetime(base_df['game_date']).dt.date # Ensure date type
        base_df = base_df.sort_values(['pitcher_id', 'game_date']).reset_index(drop=True)
        logger.info(f"Loaded {len(base_df)} rows from {input_table}.")
    except Exception as e: logger.error(f"Failed to load base data: {e}", exc_info=True); return

    # --- Run Feature Creation Steps ---
    # Step 4: Add Ballpark Column
    final_df = add_ballpark_column(base_df, db_path)
    del base_df; gc.collect()
    if 'ballpark' not in final_df.columns: logger.error("Failed to add ballpark column."); return

    # Step 5: Calculate and Merge Dynamic Park Factor
    logger.info("Calculating/Merging Dynamic Park Factor...")
    park_factors_df = calculate_dynamic_park_factors(db_path, target_col='strikeouts')
    if not park_factors_df.empty:
        if 'game_pk' in final_df.columns and 'game_pk' in park_factors_df.columns:
            final_df = pd.merge(final_df, park_factors_df[['game_pk', 'park_factor_dynamic']], on='game_pk', how='left')
            missing = final_df['park_factor_dynamic'].isnull().sum()
            if missing > 0:
                 fill_val = park_factors_df['park_factor_dynamic'].mean() # Use mean from calculated factors
                 fill_val = fill_val if pd.notna(fill_val) else 5.5 # Fallback global mean
                 logger.warning(f"Filling {missing} missing dynamic park factors after merge with {fill_val:.3f}.")
                 final_df['park_factor_dynamic'].fillna(fill_val, inplace=True)
        else: logger.warning("Missing 'game_pk', cannot merge dynamic park factors.")
    else: logger.warning("Could not calculate dynamic park factors. Skipping.")
    gc.collect()

    # Step 6: Calculate & Merge Granular Pitcher Stats (Example)
    logger.info("Calculating/Merging Granular Pitcher Stats Example...")
    granular_pitcher_stats = calculate_granular_pitcher_stats(db_path)
    if not granular_pitcher_stats.empty:
        if 'game_pk' in final_df.columns and 'pitcher_id' in final_df.columns:
            final_df = pd.merge(final_df, granular_pitcher_stats, on=['game_pk', 'pitcher_id'], how='left')
            # Handle NaNs created by the merge or rolling calc for new columns
            for col in ['roll_breaking_whiff_pct', 'roll_fastball_usage_pct', 'roll_k_pct_ahead_count']:
                 if col in final_df.columns and final_df[col].isnull().any():
                      fill_val = final_df[col].median()
                      fill_val = fill_val if pd.notna(fill_val) else 0
                      logger.warning(f"Filling {final_df[col].isnull().sum()} NaNs in '{col}' with median {fill_val:.3f}")
                      final_df[col].fillna(fill_val, inplace=True)
        else: logger.warning("Missing 'game_pk' or 'pitcher_id', cannot merge granular pitcher stats.")
    else: logger.warning("Could not calculate granular pitcher stats. Skipping.")
    gc.collect()

    # Step 7: Calculate & Merge Opponent Details (Example)
    logger.info("Calculating/Merging Opponent Details Example...")
    opponent_details = calculate_opponent_details(db_path)
    if not opponent_details.empty:
        if 'game_pk' in final_df.columns and 'opponent_team' in final_df.columns and 'p_throws' in final_df.columns:
             final_df = pd.merge(final_df, opponent_details,
                                 left_on=['game_pk', 'opponent_team', 'p_throws'],
                                 right_on=['game_pk', 'opponent_team', 'pitcher_hand_faced'], # Corrected right_on
                                 how='left')
             final_df = final_df.drop(columns=['pitcher_hand_faced'], errors='ignore')
             # Handle NaNs for new opponent columns
             for col in [f'roll_{ROLLING_GAME_WINDOW}g_k_pct_vs_hand', f'roll_{ROLLING_GAME_WINDOW}g_swing_pct_vs_hand', f'roll_{ROLLING_GAME_WINDOW}g_contact_pct_vs_hand']:
                  if col in final_df.columns and final_df[col].isnull().any():
                       fill_val = final_df[col].median()
                       fill_val = fill_val if pd.notna(fill_val) else 0.22 # Example fallback
                       logger.warning(f"Filling {final_df[col].isnull().sum()} NaNs in '{col}' with median {fill_val:.3f}")
                       final_df[col].fillna(fill_val, inplace=True)
        else: logger.warning("Missing keys, cannot merge opponent details.")
    else: logger.warning("Could not calculate opponent details. Skipping.")
    gc.collect()

    # Step 8: Create Interaction Features
    final_df = create_interaction_features(final_df)

    # Step 9: Final NaN check (optional)
    # ... (can add more checks if needed) ...

    # Step 10: Save the enriched dataset
    logger.info(f"Saving final dataset with advanced features to '{output_table}'...")
    try:
        with DBConnection(db_path) as conn:
            # Ensure game_date is TEXT for SQLite compatibility if needed
            if 'game_date' in final_df.columns and not pd.api.types.is_string_dtype(final_df['game_date']):
                 final_df['game_date'] = final_df['game_date'].astype(str)
            final_df.to_sql(output_table, conn, if_exists='replace', index=False, chunksize=5000)
        logger.info(f"Successfully saved {len(final_df)} rows to '{output_table}'.")
    except Exception as e:
        logger.error(f"Failed to save final dataset to {output_table}: {e}", exc_info=True)

    total_time = time.time() - start_time
    logger.info(f"--- Advanced Feature Engineering Finished in {total_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Advanced Features (Granular, Park, Interactions).")
    parser.add_argument("--mode", required=True, choices=['train', 'test', 'predict'], help="Which dataset to process.")
    parser.add_argument("--date", type=str, default=None, help="Required date (YYYY-MM-DD) if mode='predict'.")
    parser.add_argument("--update-mapping", action="store_true", help="Run team-to-ballpark mapping update first.")
    args = parser.parse_args()
    if args.mode == 'predict' and not args.date: logger.error("--date required for --mode=predict"); sys.exit(1)
    if args.mode == 'predict' and args.date:
         try: datetime.strptime(args.date, "%Y-%m-%d")
         except ValueError: logger.error(f"Invalid format for --date: {args.date}. Use YYYY-MM-DD."); sys.exit(1)
    main(args)