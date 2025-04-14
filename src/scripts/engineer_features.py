# src/scripts/engineer_features.py
import os
import sys
import logging
from pathlib import Path # Make sure Path is imported
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Attempt imports, handle potential errors
try:
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
    # Import store_data_to_sql for use in generate_prediction_features
    from src.scripts.data_fetcher import store_data_to_sql
    MODULE_IMPORTS_OK = True
except ImportError as e:
     print(f"ERROR: Failed to import required modules: {e}")
     MODULE_IMPORTS_OK = False
     # Dummy definitions... (same as before)
     def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
     class DBConnection:
          def __init__(self, p): self.p=p
          def __enter__(self): print("WARN: DBConnection disabled"); return None
          def __exit__(self,t,v,tb): pass
     class StrikeoutModelConfig: DEFAULT_TRAIN_YEARS=(); DEFAULT_TEST_YEARS=()
     class DBConfig: PATH="data/pitcher_stats.db"
     def store_data_to_sql(df, tn, dp, if_exists): print(f"Dummy store {tn}"); return True


logger = setup_logger('engineer_features') if MODULE_IMPORTS_OK else logging.getLogger('engineer_features_fallback')


# --- run_historical_feature_pipeline function (Identical) ---
def run_historical_feature_pipeline(args):
    logger.info("Running standard historical feature engineering pipeline...")
    start_time = time.time()
    db_path = project_root / DBConfig.PATH
    # Aggregations
    logger.info("Running/Verifying base game-level aggregations...")
    try:
        force_rebuild_agg = getattr(args, 'force_rebuild_agg', False)
        aggregate_pitchers_to_game_level(force_rebuild=force_rebuild_agg)
        aggregate_batters_to_game_level()
        aggregate_teams_to_game_level()
        logger.info("Aggregations complete/verified.")
    except Exception as e: logger.error(f"Aggregation error: {e}.", exc_info=True); return False
    # Feature Creation
    try:
        logger.info("Creating historical features...")
        with DBConnection(db_path) as conn:
             pitcher_hist_df = pd.read_sql_query("SELECT * FROM game_level_pitchers", conn)
             batter_hist_df = pd.read_sql_query("SELECT * FROM game_level_batters", conn)
        if pitcher_hist_df.empty or batter_hist_df.empty: logger.error("Hist pitcher/batter empty."); return False
        pitcher_features_df = create_pitcher_features(pitcher_hist_df.copy(), "all")
        batter_features_df = create_batter_features(batter_hist_df.copy(), "all")
        adv_opp_features_df = pd.DataFrame()
        if args.advanced_pitcher:
             logger.info("Creating & Merging advanced pitcher features..."); adv_pitch_df = create_advanced_pitcher_features()
             if not adv_pitch_df.empty: pitcher_features_df = pd.merge(pitcher_features_df, adv_pitch_df, on=['pitcher_id', 'game_pk', 'game_date'], how='left')
             else: logger.warning("Adv pitcher features empty.")
        if args.advanced_opponent:
             logger.info("Creating advanced opponent features..."); adv_opp_df = create_advanced_opponent_features(window_games=10)
             if not adv_opp_df.empty: adv_opp_features_df = adv_opp_df
             else: logger.warning("Adv opponent features empty.")
        team_features_base = create_team_features();
        if team_features_base.empty: logger.warning("Base team features empty.")
        # Splitting and Combining
        logger.info("Splitting data & combining train/test features...")
        train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS; test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
        train_pitcher_features = pitcher_features_df[pitcher_features_df['season'].isin(train_seasons)].copy()
        test_pitcher_features = pitcher_features_df[pitcher_features_df['season'].isin(test_seasons)].copy()
        create_combined_features_final(pitcher_features_df=train_pitcher_features, base_team_features_df=team_features_base, adv_opp_features_df=adv_opp_features_df, dataset_type="train")
        create_combined_features_final(pitcher_features_df=test_pitcher_features, base_team_features_df=team_features_base, adv_opp_features_df=adv_opp_features_df, dataset_type="test")
        logger.info(f"Historical pipeline completed in {time.time() - start_time:.2f}s.")
        return True
    except Exception as e: logger.error(f"Error in historical pipeline: {e}", exc_info=True); return False


# --- generate_prediction_features function (MODIFIED final column selection) ---
def generate_prediction_features(prediction_date_str):
    logger.info(f"Generating prediction features for date: {prediction_date_str}")
    start_time = time.time()
    db_path = project_root / DBConfig.PATH

    # Load Expected Feature Columns
    expected_feature_cols = None
    try:
        model_dir = project_root / 'models'; feature_files = sorted(model_dir.glob('*_feature_columns_*.pkl'), reverse=True)
        if not feature_files: logger.error("No feature columns file found."); return False
        latest_feature_file = feature_files[0]
        import pickle
        with open(latest_feature_file, 'rb') as f: expected_feature_cols = pickle.load(f)
        logger.info(f"Loaded {len(expected_feature_cols)} expected feature cols from {latest_feature_file}")
        if not expected_feature_cols: logger.error("Loaded feature list empty!"); return False
    except Exception as e: logger.error(f"Error loading feature columns: {e}", exc_info=True); return False

    # Load Data
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed")
            query_today = f"SELECT * FROM mlb_api WHERE game_date = '{prediction_date_str}'"
            try: today_games_df = pd.read_sql_query(query_today, conn)
            except Exception as e:
                 if "no such table: mlb_api" in str(e): logger.error(f"'mlb_api' table not found.")
                 else: logger.error(f"Error loading mlb_api table: {e}")
                 return False
            if today_games_df.empty: logger.warning(f"No games in 'mlb_api' for {prediction_date_str}."); return True

            logger.info(f"Loaded {len(today_games_df)} games for {prediction_date_str}.")
            logger.info("Loading historical pitcher feature tables (train & test)...")
            query_ptrain = "SELECT * FROM train_predictive_pitch_features"; query_ptest = "SELECT * FROM test_predictive_pitch_features"
            try: hist_p_train = pd.read_sql_query(query_ptrain, conn); logger.info(f"Loaded {len(hist_p_train)} train pitcher features.")
            except: logger.error(f"Failed load train_predictive_pitch_features."); hist_p_train = pd.DataFrame()
            try: hist_p_test = pd.read_sql_query(query_ptest, conn); logger.info(f"Loaded {len(hist_p_test)} test pitcher features.")
            except: logger.error(f"Failed load test_predictive_pitch_features."); hist_p_test = pd.DataFrame()
            if hist_p_train.empty and hist_p_test.empty: logger.error("Pitcher feature tables empty/missing."); return False
            hist_pitcher_features = pd.concat([hist_p_train, hist_p_test], ignore_index=True); hist_pitcher_features['game_date'] = pd.to_datetime(hist_pitcher_features['game_date']).dt.date
            logger.info(f"Combined historical pitcher features: {len(hist_pitcher_features)} rows.")

            query_opp = "SELECT * FROM advanced_opponent_game_features"
            try: hist_opp_features = pd.read_sql_query(query_opp, conn); hist_opp_features['game_date'] = pd.to_datetime(hist_opp_features['game_date']).dt.date; logger.info(f"Loaded {len(hist_opp_features)} adv opp features.")
            except Exception as e: logger.warning(f"Could not load adv opp features: {e}."); hist_opp_features = pd.DataFrame()

        prediction_date = datetime.strptime(prediction_date_str, "%Y-%m-%d").date()
    except Exception as e: logger.error(f"Error loading data: {e}", exc_info=True); return False

    # Generate Features
    prediction_feature_list = []
    logger.info(f"Processing {len(today_games_df)} games for {prediction_date_str}...")
    # --- (Loop through games, find features, combine, select - NO CHANGE IN THIS LOGIC) ---
    for _, game in today_games_df.iterrows():
        game_pk = game['gamePk']; home_pitcher_id = game['home_probable_pitcher_id']; away_pitcher_id = game['away_probable_pitcher_id']
        home_team_id = game['home_team_id']; away_team_id = game['away_team_id']
        # Process Home Pitcher
        if pd.notna(home_pitcher_id):
             pitcher_id = int(home_pitcher_id); opponent_team_id = away_team_id
             p_hist = hist_pitcher_features[(hist_pitcher_features['pitcher_id'] == pitcher_id) & (hist_pitcher_features['game_date'] < prediction_date)].sort_values(by='game_date', ascending=False)
             if not p_hist.empty:
                  latest_p_feats = p_hist.iloc[0:1].copy()
                  opp_hist = hist_opp_features[(hist_opp_features['team'] == opponent_team_id) & (hist_opp_features['game_date'] < prediction_date)].sort_values(by='game_date', ascending=False) if not hist_opp_features.empty else pd.DataFrame()
                  comb_feats = latest_p_feats.reset_index(drop=True)
                  if not opp_hist.empty: latest_opp = opp_hist.iloc[0:1].reset_index(drop=True).add_prefix('opp_adv_'); comb_feats = pd.concat([comb_feats, latest_opp], axis=1)
                  else: logger.debug(f"No prior adv opp features for team {opponent_team_id}")
                  final_f = pd.DataFrame(columns=expected_feature_cols); final_f = pd.concat([final_f, comb_feats[comb_feats.columns.intersection(expected_feature_cols)]], ignore_index=True)
                  final_f['gamePk']=game_pk; final_f['game_date']=prediction_date_str; final_f['pitcher_id']=pitcher_id; final_f['team_id']=home_team_id; final_f['opponent_team_id']=opponent_team_id; final_f['is_home']=1
                  final_f.fillna(0, inplace=True); prediction_feature_list.append(final_f)
             else: logger.warning(f"No history found for home pitcher {pitcher_id}")
        # Process Away Pitcher
        if pd.notna(away_pitcher_id):
             pitcher_id = int(away_pitcher_id); opponent_team_id = home_team_id
             p_hist = hist_pitcher_features[(hist_pitcher_features['pitcher_id'] == pitcher_id) & (hist_pitcher_features['game_date'] < prediction_date)].sort_values(by='game_date', ascending=False)
             if not p_hist.empty:
                  latest_p_feats = p_hist.iloc[0:1].copy()
                  opp_hist = hist_opp_features[(hist_opp_features['team'] == opponent_team_id) & (hist_opp_features['game_date'] < prediction_date)].sort_values(by='game_date', ascending=False) if not hist_opp_features.empty else pd.DataFrame()
                  comb_feats = latest_p_feats.reset_index(drop=True)
                  if not opp_hist.empty: latest_opp = opp_hist.iloc[0:1].reset_index(drop=True).add_prefix('opp_adv_'); comb_feats = pd.concat([comb_feats, latest_opp], axis=1)
                  else: logger.debug(f"No prior adv opp features for team {opponent_team_id}")
                  final_f = pd.DataFrame(columns=expected_feature_cols); final_f = pd.concat([final_f, comb_feats[comb_feats.columns.intersection(expected_feature_cols)]], ignore_index=True)
                  final_f['gamePk']=game_pk; final_f['game_date']=prediction_date_str; final_f['pitcher_id']=pitcher_id; final_f['team_id']=away_team_id; final_f['opponent_team_id']=opponent_team_id; final_f['is_home']=0
                  final_f.fillna(0, inplace=True); prediction_feature_list.append(final_f)
             else: logger.warning(f"No history found for away pitcher {pitcher_id}")

    if not prediction_feature_list: logger.error(f"No prediction features generated for {prediction_date_str}."); return False
    final_prediction_df = pd.concat(prediction_feature_list, ignore_index=True)

    # --- MODIFIED: Ensure final column order correctly ---
    id_cols = ['gamePk', 'game_date', 'pitcher_id', 'team_id', 'opponent_team_id', 'is_home']
    # Ensure all expected feature cols exist, adding any missing ones and filling with 0
    for col in expected_feature_cols:
        if col not in final_prediction_df.columns:
            logger.warning(f"Expected feature column '{col}' was missing, adding with value 0.")
            final_prediction_df[col] = 0

    # Create the final list of columns, starting with IDs then unique features
    final_columns_ordered = list(id_cols)
    for col in expected_feature_cols:
        if col not in final_columns_ordered:
            final_columns_ordered.append(col)
        # No need for elif check here, as we already ensured expected_cols are added if missing above

    # Select only the necessary columns in the desired order
    try:
        final_prediction_df = final_prediction_df[final_columns_ordered]
        # Check for duplicates *after* selection (shouldn't happen now)
        if final_prediction_df.columns.duplicated().any():
             dups = final_prediction_df.columns[final_prediction_df.columns.duplicated()].tolist()
             logger.error(f"Duplicate columns STILL exist after selection! Dups: {dups}")
             return False
    except KeyError as e:
         logger.error(f"Missing expected columns during final selection: {e}")
         logger.error(f"Available columns: {list(final_prediction_df.columns)}")
         return False
    # --- END MODIFICATION ---


    # --- Save prediction features ---
    output_table_name = "prediction_features"
    logger.info(f"Saving {len(final_prediction_df)} prediction feature rows to '{output_table_name}' (replacing)...")
    # Reuse storage function from data_fetcher - make sure it's imported
    # from src.scripts.data_fetcher import store_data_to_sql
    if 'store_data_to_sql' not in globals(): # Check if import failed earlier
         logger.error("store_data_to_sql function not available. Cannot save.")
         return False
    success = store_data_to_sql(final_prediction_df, output_table_name, db_path, if_exists='replace')

    total_time = time.time() - start_time
    if success: logger.info(f"Prediction feature generation complete in {total_time:.2f}s."); return True
    else: logger.error("Failed to save prediction features."); return False


# --- Helper function create_combined_features_final (Identical) ---
def create_combined_features_final(pitcher_features_df, base_team_features_df, adv_opp_features_df, dataset_type="all"):
    logger.info(f"Creating FINAL combined features for {dataset_type} dataset...")
    db_path = project_root / DBConfig.PATH
    try:
        if pitcher_features_df.empty: logger.error("Pitcher features empty."); return pd.DataFrame()
        combined_df = pitcher_features_df.copy()
        combined_df['game_date'] = pd.to_datetime(combined_df['game_date']).dt.date
        if not all(col in combined_df.columns for col in ['home_team_id', 'away_team_id', 'is_home']):
            logger.warning("Missing team ID/is_home in pitcher features."); combined_df['opponent_team_id'] = pd.NA
        else:
            combined_df['opponent_team_id'] = combined_df.apply(lambda r: int(r['away_team_id']) if r['is_home']==1 else int(r['home_team_id']), axis=1); combined_df['opponent_team_id'] = pd.to_numeric(combined_df['opponent_team_id'], errors='coerce').astype('Int64')
        if base_team_features_df is not None and not base_team_features_df.empty:
             logger.info("Merging base opponent features (lagged)...")
             combined_df['prior_season'] = combined_df['season'] - 1; base_lagged = base_team_features_df.add_prefix('opp_base_')
             base_lagged['opp_base_team_id'] = pd.to_numeric(base_lagged.get('opp_base_team_id', pd.Series(dtype='Int64')), errors='coerce').astype('Int64'); base_lagged['opp_base_season'] = pd.to_numeric(base_lagged.get('opp_base_season', pd.Series(dtype='Int64')), errors='coerce').astype('Int64')
             combined_df['opponent_team_id'] = combined_df['opponent_team_id'].astype('Int64'); combined_df['prior_season'] = combined_df['prior_season'].astype('Int64')
             combined_df = pd.merge(combined_df, base_lagged, left_on=['opponent_team_id', 'prior_season'], right_on=['opp_base_team_id', 'opp_base_season'], how='left', suffixes=('', '_dup'))
             combined_df.drop(columns=[c for c in combined_df if c.endswith('_dup') or c in ['opp_base_team_id','opp_base_season','prior_season']], errors='ignore', inplace=True); logger.info(f"Shape after base opp merge: {combined_df.shape}")
        if adv_opp_features_df is not None and not adv_opp_features_df.empty:
            logger.info("Merging advanced opponent features...")
            adv_opp = adv_opp_features_df.add_prefix('opp_adv_'); adv_opp['opp_adv_team'] = pd.to_numeric(adv_opp.get('opp_adv_team', pd.Series(dtype='Int64')), errors='coerce').astype('Int64'); adv_opp['opp_adv_game_date'] = pd.to_datetime(adv_opp.get('opp_adv_game_date', pd.Series(dtype='datetime64[ns]'))).dt.date
            combined_df['opponent_team_id'] = combined_df['opponent_team_id'].astype('Int64');
            combined_df = pd.merge(combined_df, adv_opp, left_on=['opponent_team_id', 'game_date'], right_on=['opp_adv_team', 'opp_adv_game_date'], how='left', suffixes=('', '_dup'))
            combined_df.drop(columns=[c for c in combined_df if c.endswith('_dup') or c in ['opp_adv_game_pk','opp_adv_team','opp_adv_opponent','opp_adv_game_date']], errors='ignore', inplace=True); logger.info(f"Shape after adv opp merge: {combined_df.shape}")
        opp_cols = [c for c in combined_df.columns if 'opp_' in c]; logger.info(f"Filling NaNs in {len(opp_cols)} opponent columns...")
        for col in opp_cols:
             if combined_df[col].isnull().any(): combined_df[col].fillna(0, inplace=True)
        combined_df.drop(columns=['opponent_team_id'], errors='ignore', inplace=True)
        table_name = f"{dataset_type}_combined_features"; logger.info(f"Saving {len(combined_df)} combined features for {dataset_type} to {table_name}...")
        # from src.scripts.data_fetcher import store_data_to_sql # Already imported at top level
        store_data_to_sql(combined_df, table_name, db_path, if_exists='replace')
        return combined_df
    except Exception as e: logger.error(f"Error combining features {dataset_type}: {e}", exc_info=True); return pd.DataFrame()


# --- Main Execution Block (Identical) ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    parser = argparse.ArgumentParser(description="Run MLB feature engineering pipeline.")
    parser.add_argument("--advanced-pitcher", action="store_true", help="Include advanced PITCHER features.")
    parser.add_argument("--advanced-opponent", action="store_true", help="Include advanced OPPONENT features.")
    parser.add_argument("--force-rebuild-agg", action="store_true", help="Force rebuild of base aggregation tables.")
    parser.add_argument("--real-world", action="store_true", help="Generate features for prediction for a specific date.")
    parser.add_argument("--prediction-date", type=str, help="Date (YYYY-MM-DD) for prediction features (use with --real-world).")
    args = parser.parse_args()
    if args.real_world:
        if not args.prediction_date: logger.error("--prediction-date required with --real-world."); sys.exit(1)
        try: datetime.strptime(args.prediction_date, "%Y-%m-%d")
        except ValueError: logger.error(f"Invalid date format: {args.prediction_date}."); sys.exit(1)
        logger.info(f"--- Running REAL-WORLD Mode for Date: {args.prediction_date} ---")
        success = generate_prediction_features(args.prediction_date)
    else:
        logger.info("--- Running Historical Feature Generation Mode ---")
        success = run_historical_feature_pipeline(args)
    if success: logger.info("--- Feature Engineering Finished Successfully ---"); sys.exit(0)
    else: logger.error("--- Feature Engineering Finished With Errors ---"); sys.exit(1)