# src/scripts/engineer_features.py (Removed Import Fallbacks)
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta, date
import traceback
import pickle

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Direct Imports (Script will fail here if any are missing) ---
from src.data.utils import setup_logger, DBConnection
from src.data.aggregate_pitchers import aggregate_pitchers_to_game_level
from src.data.aggregate_batters import aggregate_batters_to_game_level
from src.data.aggregate_teams import aggregate_teams_to_game_level
from src.features.pitch_features import create_pitcher_features
from src.features.batter_features import create_batter_features
from src.features.team_features import create_team_features
from src.features.advanced_pitch_features import create_advanced_pitcher_features
from src.features.advanced_opp_features import create_advanced_opponent_features
from src.config import StrikeoutModelConfig, DBConfig
# Attempt to import store_data_to_sql - script will fail if data_fetcher is missing/broken
from src.scripts.data_fetcher import store_data_to_sql

# --- Logger Setup ---
# Setup logger directly - assumes setup_logger was imported successfully
logger = setup_logger('engineer_features')

# --- Helper function combine_all_features (Keep as is) ---
def combine_all_features(pitcher_features_df, base_team_features_df, adv_opp_features_df):
    # (This function remains the same as the previous version)
    logger.info(f"Combining all historical features...")
    db_path = project_root / DBConfig.PATH
    try:
        if pitcher_features_df is None or pitcher_features_df.empty: logger.error("Input pitcher_features_df empty."); return pd.DataFrame()
        combined_df = pitcher_features_df.copy()
        id_cols_needed = ['game_date', 'season', 'pitcher_id', 'game_pk', 'home_team', 'away_team', 'is_home']
        if not all(col in combined_df.columns for col in id_cols_needed): missing_cols = [col for col in id_cols_needed if col not in combined_df.columns]; logger.error(f"Pitcher features missing essential columns: {missing_cols}"); return pd.DataFrame()
        combined_df['game_date'] = pd.to_datetime(combined_df['game_date']).dt.date
        logger.info("Identifying opponent team...")
        combined_df['is_home'] = pd.to_numeric(combined_df['is_home'], errors='coerce'); combined_df.dropna(subset=['is_home'], inplace=True); combined_df['is_home'] = combined_df['is_home'].astype(int)
        def get_opponent(row): 
            try: 
                return row['away_team'] if row['is_home'] == 1 else row['home_team'] 
            except KeyError: return None
        combined_df['opponent_team'] = combined_df.apply(get_opponent, axis=1); combined_df.dropna(subset=['opponent_team'], inplace=True); combined_df['opponent_team'] = combined_df['opponent_team'].astype(str)
        if base_team_features_df is not None and not base_team_features_df.empty:
             logger.info("Merging base opponent features (lagged)...")
             if 'team' in base_team_features_df.columns and 'season' in base_team_features_df.columns:
                 base_team_features_df['team'] = base_team_features_df['team'].astype(str); base_team_features_df['season'] = pd.to_numeric(base_team_features_df['season'], errors='coerce').astype('Int64')
                 base_lagged = base_team_features_df.add_prefix('opp_base_'); combined_df['prior_season'] = pd.to_numeric(combined_df['season'], errors='coerce').astype('Int64') - 1
                 combined_df = pd.merge(combined_df, base_lagged, left_on=['opponent_team', 'prior_season'], right_on=['opp_base_team', 'opp_base_season'], how='left', suffixes=('', '_dup'))
                 combined_df.drop(columns=[c for c in combined_df if c.endswith('_dup') or c in ['opp_base_team', 'opp_base_season', 'prior_season']], errors='ignore', inplace=True)
             else: logger.error("Base team features missing keys.");
        else: logger.warning("No base team features. Skipping merge.")
        if adv_opp_features_df is not None and not adv_opp_features_df.empty:
            logger.info("Merging advanced opponent features...")
            if 'game_pk' in adv_opp_features_df.columns and 'team' in adv_opp_features_df.columns:
                 adv_opp_features_df['team'] = adv_opp_features_df['team'].astype(str); adv_opp = adv_opp_features_df.add_prefix('opp_adv_')
                 combined_df = pd.merge(combined_df, adv_opp, left_on=['game_pk', 'opponent_team'], right_on=['opp_adv_game_pk', 'opp_adv_team'], how='left', suffixes=('', '_dup'))
                 adv_drop_cols = ['opp_adv_game_pk', 'opp_adv_game_date', 'opp_adv_team', 'opp_adv_opponent', 'opp_adv_is_home', 'opp_adv_season']
                 combined_df.drop(columns=[c for c in combined_df if c.endswith('_dup') or c in adv_drop_cols], errors='ignore', inplace=True)
            else: logger.error("Adv opp features missing keys.");
        else: logger.warning("No advanced opponent features. Skipping merge.")
        logger.info("Merging umpire data...")
        umpire_df = pd.DataFrame()
        try:
            with DBConnection(db_path) as conn:
                 if conn is None: raise ConnectionError("DB connection failed for umpire data.")
                 umpire_query = "SELECT game_date, home_team, away_team, umpire FROM umpire_data"
                 umpire_df = pd.read_sql_query(umpire_query, conn)
            if not umpire_df.empty:
                umpire_df['game_date'] = pd.to_datetime(umpire_df['game_date']).dt.date; umpire_df['home_team'] = umpire_df['home_team'].astype(str); umpire_df['away_team'] = umpire_df['away_team'].astype(str)
                combined_df['home_team'] = combined_df['home_team'].astype(str); combined_df['away_team'] = combined_df['away_team'].astype(str)
                combined_df = pd.merge(combined_df, umpire_df[['game_date', 'home_team', 'away_team', 'umpire']], on=['game_date', 'home_team', 'away_team'], how='left')
            else: logger.warning("Umpire data empty."); combined_df['umpire'] = None
        except Exception as e: logger.error(f"Error merging umpire data: {e}."); logger.error(traceback.format_exc()); combined_df['umpire'] = None
        if 'umpire' in combined_df.columns:
            rows_before_drop = len(combined_df); combined_df.dropna(subset=['umpire'], inplace=True); rows_dropped = rows_before_drop - len(combined_df)
            if rows_dropped > 0: logger.info(f"Dropped {rows_dropped} rows missing umpire info.")
            if not combined_df.empty: combined_df['umpire'] = combined_df['umpire'].astype(str)
        else: logger.warning("Cannot drop rows: 'umpire' column not found.")
        opp_cols = [c for c in combined_df.columns if 'opp_' in c]
        if opp_cols: logger.info(f"Filling NaNs in opponent features...");
        for col in opp_cols:
            if combined_df[col].isnull().any() and pd.api.types.is_numeric_dtype(combined_df[col]): combined_df[col].fillna(0, inplace=True)
        combined_df.drop(columns=['opponent_team'], errors='ignore', inplace=True)
        logger.info(f"Final combined shape before split: {combined_df.shape}")
        return combined_df
    except Exception as e: logger.error(f"Error in combine_all_features: {e}", exc_info=True); logger.error(traceback.format_exc()); return pd.DataFrame()


# --- run_historical_feature_pipeline function (MODIFIED with Checkpoint Logic) ---
def run_historical_feature_pipeline(args):
    logger.info("--- Running Historical Feature Generation Mode ---")
    start_time = time.time()
    db_path = project_root / DBConfig.PATH
    TARGET_VARIABLE = 'strikeouts'
    checkpoint_dir = project_root / 'data' / 'checkpoints'; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'combined_hist_checkpoint.pkl'
    use_checkpoint = False

    # Check for Checkpoint Load
    if not args.ignore_checkpoint and checkpoint_path.exists():
        try:
            logger.info(f"Attempting load from checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f: combined_hist_df = pickle.load(f)
            logger.info(f"Loaded {len(combined_hist_df)} rows from checkpoint.")
            use_checkpoint = True
            if not isinstance(combined_hist_df, pd.DataFrame) or combined_hist_df.empty: logger.warning("Checkpoint invalid/empty."); use_checkpoint = False
            elif 'season' not in combined_hist_df.columns or TARGET_VARIABLE not in combined_hist_df.columns or 'umpire' not in combined_hist_df.columns: logger.warning("Checkpoint missing columns."); use_checkpoint = False
        except Exception as e: logger.error(f"Failed load/validate checkpoint: {e}."); use_checkpoint = False

    if not use_checkpoint:
        logger.info("Running full pipeline (no checkpoint used)...")
        # 1. Aggregations
        logger.info("Running aggregations...")
        try:
            if not aggregate_pitchers_to_game_level(): return False
            if not aggregate_batters_to_game_level(): return False
            if not aggregate_teams_to_game_level(): return False
            logger.info("Aggregations complete.")
        except Exception as e: logger.error(f"Aggregation error: {e}.", exc_info=True); logger.error(traceback.format_exc()); return False

        # 2. Feature Creation
        try:
            logger.info("Creating individual features...")
            pitcher_features_df = pd.DataFrame(); adv_opp_features_df = pd.DataFrame(); adv_pitch_df = pd.DataFrame(); team_features_base = pd.DataFrame()
            with DBConnection(db_path) as conn:
                if conn is None: raise ConnectionError("DB Connection failed.")
                try: pitcher_agg_df = pd.read_sql_query("SELECT * FROM game_level_pitchers", conn)
                except Exception as e: logger.error(f"Failed load game_level_pitchers: {e}"); return False
                if args.advanced_opponent:
                    try: team_agg_df = pd.read_sql_query("SELECT * FROM game_level_team_stats", conn)
                    except Exception as e: logger.warning(f"Failed load game_level_team_stats: {e}."); team_agg_df = pd.DataFrame()
                else: team_agg_df = pd.DataFrame()

            if not pitcher_agg_df.empty: pitcher_features_df = create_pitcher_features(pitcher_agg_df.copy(), "all")
            else: logger.error("Pitcher aggregation empty."); return False

            if args.advanced_pitcher:
                 logger.info("Creating/Merging advanced pitcher features..."); adv_pitch_df = create_advanced_pitcher_features()
                 if not adv_pitch_df.empty and not pitcher_features_df.empty:
                     merge_keys_adv_p = ['pitcher_id', 'game_pk', 'game_date']
                     if all(k in pitcher_features_df.columns for k in merge_keys_adv_p) and all(k in adv_pitch_df.columns for k in merge_keys_adv_p):
                           pitcher_features_df['game_date'] = pd.to_datetime(pitcher_features_df['game_date']).dt.date
                           adv_pitch_df['game_date'] = pd.to_datetime(adv_pitch_df['game_date']).dt.date
                           pitcher_features_df = pd.merge(pitcher_features_df, adv_pitch_df, on=merge_keys_adv_p, how='left', suffixes=('', '_adv'))
                           pitcher_features_df.drop(columns=[c for c in pitcher_features_df if c.endswith('_adv')], inplace=True)
                     else: logger.warning("Cannot merge adv pitcher features - missing keys.")
                 elif adv_pitch_df.empty: logger.warning("Adv pitcher features empty.")
                 else: logger.warning("Base pitcher features empty.")

            if args.advanced_opponent:
                 logger.info("Creating advanced opponent features..."); adv_opp_df = create_advanced_opponent_features(window_games=10)
                 if adv_opp_df.empty: logger.warning("Adv opponent features empty.")
                 else: adv_opp_features_df = adv_opp_df

            team_features_base = create_team_features();
            if team_features_base.empty: logger.warning("Base team features empty.")
            if pitcher_features_df.empty: logger.error("Pitcher features DF empty."); return False

        except Exception as e: logger.error(f"Feature creation error: {e}", exc_info=True); logger.error(traceback.format_exc()); return False

        # 3. Combine All Features
        combined_hist_df = combine_all_features(pitcher_features_df, team_features_base, adv_opp_features_df)
        if combined_hist_df.empty: logger.error("Combining features failed."); return False

        # Save Checkpoint
        try:
            logger.info(f"Saving checkpoint: {checkpoint_path}");
            with open(checkpoint_path, 'wb') as f: pickle.dump(combined_hist_df, f)
            logger.info("Checkpoint saved.")
        except Exception as e: logger.error(f"Failed save checkpoint: {e}") # Continue


    # Steps from Checkpoint or after creation
    if combined_hist_df.empty: logger.error("Combined DF empty."); return False
    # ... (Validation checks for season, target, umpire - same as before) ...
    if 'season' not in combined_hist_df.columns: logger.error("Missing 'season'."); return False
    if TARGET_VARIABLE not in combined_hist_df.columns: logger.error(f"Target '{TARGET_VARIABLE}' not found."); return False
    if 'umpire' not in combined_hist_df.columns or combined_hist_df['umpire'].isnull().all(): logger.error("'umpire' missing or all null."); return False

    # 4. Split into Train/Test
    logger.info("Splitting data...")
    # ... (Splitting logic - same as before) ...
    train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS; test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    combined_hist_df['season'] = pd.to_numeric(combined_hist_df['season'], errors='coerce').astype('Int64')
    train_df = combined_hist_df[combined_hist_df['season'].isin(train_seasons)].copy()
    test_df = combined_hist_df[combined_hist_df['season'].isin(test_seasons)].copy()
    if train_df.empty: logger.error("Train DF empty."); return False
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 5. Calculate and Apply Target Encoding
    logger.info(f"Calculating Target Encoding for 'umpire'...")
    # ... (Target encoding logic + saving map - same as before) ...
    try:
        global_mean = train_df[TARGET_VARIABLE].mean(); logger.info(f"Global mean {TARGET_VARIABLE}: {global_mean:.4f}")
        umpire_encoding_map = train_df.groupby('umpire')[TARGET_VARIABLE].mean()
        train_df['umpire_target_encoded'] = train_df['umpire'].map(umpire_encoding_map)
        test_df['umpire_target_encoded'] = test_df['umpire'].map(umpire_encoding_map)
        train_nan_count = train_df['umpire_target_encoded'].isnull().sum(); test_nan_count = test_df['umpire_target_encoded'].isnull().sum()
        train_df['umpire_target_encoded'].fillna(global_mean, inplace=True); test_df['umpire_target_encoded'].fillna(global_mean, inplace=True)
        logger.info(f"Applied umpire target encoding. Filled {train_nan_count} NaNs train, {test_nan_count} test.")
        output_dir = project_root / 'models'; output_dir.mkdir(parents=True, exist_ok=True)
        map_filename = output_dir / f'umpire_target_encoding_map_{datetime.now().strftime("%Y%m%d")}.pkl'
        with open(map_filename, 'wb') as f: pickle.dump({'map': umpire_encoding_map, 'fallback': global_mean}, f)
        logger.info(f"Saved umpire encoding map: {map_filename}")
    except Exception as e: logger.error(f"Target encoding error: {e}"); logger.error(traceback.format_exc()); return False

    # 6. Save Final Train/Test Feature Sets
    logger.info("Saving final feature sets...")
    train_save_success = False; test_save_success = False
    # --- Convert object columns and game_date explicitly before saving ---
    for df_to_save, df_name in [(train_df, "Train"), (test_df, "Test")]:
        if not df_to_save.empty:
            object_cols = df_to_save.select_dtypes(include=['object']).columns
            other_object_cols = [col for col in object_cols if col != 'game_date']
            if len(other_object_cols) > 0:
                logger.info(f"Converting {len(other_object_cols)} obj cols to string for {df_name}: {other_object_cols}")
                for col in other_object_cols: df_to_save[col] = df_to_save[col].astype(str)
            if 'game_date' in df_to_save.columns:
                    logger.info(f"Converting 'game_date' to YYYY-MM-DD string for {df_name}.")
                    try: df_to_save['game_date'] = df_to_save['game_date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, date) and pd.notnull(x) else str(x))
                    except Exception as date_e: logger.error(f"Date conversion failed for {df_name}: {date_e}."); df_to_save['game_date'] = df_to_save['game_date'].astype(str)

    # --- Remove check for DATA_FETCHER_IMPORT_OK as store_data_to_sql is imported directly ---
    logger.info("--- Train DataFrame Info Before Save ---"); train_df.info(verbose=False); logger.info("--- End Train Info ---")
    train_save_success = store_data_to_sql(train_df, 'train_combined_features', db_path, if_exists='replace')
    if not test_df.empty:
            logger.info("--- Test DataFrame Info Before Save ---"); test_df.info(verbose=False); logger.info("--- End Test Info ---")
            test_save_success = store_data_to_sql(test_df, 'test_combined_features', db_path, if_exists='replace')
    else: logger.info("Test DataFrame empty, skipping save."); test_save_success = True

    if not train_save_success or not test_save_success: logger.error("Failed to save one or both final feature sets."); return False

    logger.info(f"Historical pipeline completed in {time.time() - start_time:.2f}s.")
    return True


# --- generate_prediction_features function (Keep as is for now) ---
def generate_prediction_features(prediction_date_str):
    # (Keep the existing function, which needs updating later to use the saved map)
    logger.warning("generate_prediction_features NOT YET UPDATED to apply saved umpire target encoding.")
    # ... (existing logic) ...
    pass


# --- Main Execution Block (Add --ignore-checkpoint argument) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLB feature engineering pipeline.")
    parser.add_argument("--advanced-pitcher", action="store_true", help="Include advanced PITCHER features.")
    parser.add_argument("--advanced-opponent", action="store_true", help="Include advanced OPPONENT features.")
    parser.add_argument("--real-world", action="store_true", help="Generate features for prediction for a specific date.")
    parser.add_argument("--prediction-date", type=str, help="Date (YYYY-MM-DD) for prediction features (use with --real-world).")
    # --- Updated ARGUMENT ---
    parser.add_argument("--ignore-checkpoint", action="store_true", help="Ignore existing combined feature checkpoint and force full rebuild of features.")

    args = parser.parse_args()
    success = False
    try:
        # --- Direct imports ensure failure if modules missing ---
        if args.real_world:
            # ... (real-world logic) ...
            pass
        else:
            success = run_historical_feature_pipeline(args) # Pass args
    except NameError as ne:
        # Catch specific NameError if a function (like store_data_to_sql) wasn't imported
        logger.error(f"Import failed or function not defined: {ne}")
        logger.error("Ensure all required modules and functions (like store_data_to_sql from data_fetcher) are correctly imported.")
        success = False
    except Exception as main_e:
        logger.error(f"Unhandled exception in main: {main_e}", exc_info=True)
        logger.error(traceback.format_exc())
        success = False

    if success: logger.info("--- Feature Engineering Finished Successfully ---"); sys.exit(0)
    else: logger.error("--- Feature Engineering Finished With Errors ---"); sys.exit(1)