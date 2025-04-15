# src/data/aggregate_teams.py (Rewritten)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import time
import traceback # Added for better error logging

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2] # Use Path for consistency
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.data.utils import setup_logger, DBConnection
    from src.config import DataConfig # For chunk size maybe
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed import in aggregate_teams: {e}")
    MODULE_IMPORTS_OK = False
    # Dummy definitions
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, p=None): self.p=p or "dummy.db"
        def __enter__(self): print("WARN: Dummy DB"); return None
        def __exit__(self,t,v,tb): pass
    class DataConfig: pass

logger = setup_logger('aggregate_teams') if MODULE_IMPORTS_OK else logging.getLogger('aggregate_teams_fallback')

def aggregate_teams_to_game_level():
    """
    Aggregates Statcast data to create game-level team statistics,
    including batting vs LHP/RHP and pitching vs LHB/RHB.
    Handles potential missing 'batter' column in older pitcher data.
    """
    start_time = time.time()
    if not MODULE_IMPORTS_OK: logger.error("Exiting: Module imports failed."); return False
    logger.info("Loading Statcast data for game-level TEAM aggregation...")

    all_game_teams_data = []

    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            # Determine seasons available in statcast data (use batter data as primary source)
            try:
                seasons = pd.read_sql_query("SELECT DISTINCT season FROM statcast_batters ORDER BY season", conn)['season'].tolist()
                if not seasons:
                     logger.error("No seasons found in statcast_batters table.")
                     return False
                logger.info(f"Found seasons in statcast_batters: {seasons}")
            except Exception as e:
                 logger.error(f"Failed to query seasons from statcast_batters: {e}")
                 return False

            # Get available columns from both tables once
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(statcast_batters)")
            batter_cols_avail = {info[1] for info in cursor.fetchall()}
            cursor.execute("PRAGMA table_info(statcast_pitchers)")
            pitcher_cols_avail = {info[1] for info in cursor.fetchall()}

            batter_cols_needed = ['game_pk', 'game_date', 'home_team', 'away_team', 'batter', 'pitcher', 'events', 'description',
                                  'balls', 'strikes', 'p_throws', 'stand', 'at_bat_number', 'pitch_number', 'woba_value', 'season']
            pitcher_cols_needed = ['game_pk', 'game_date', 'home_team', 'away_team', 'pitcher', 'batter', 'events', 'description',
                                  'balls', 'strikes', 'p_throws', 'stand', 'at_bat_number', 'pitch_number', 'season']

            # Select only available columns
            batter_cols_select = [col for col in batter_cols_needed if col in batter_cols_avail]
            pitcher_cols_select = [col for col in pitcher_cols_needed if col in pitcher_cols_avail]

            # Critical check for needed columns
            if not all(c in batter_cols_select for c in ['game_pk', 'at_bat_number', 'batter', 'pitcher', 'events', 'p_throws', 'stand']):
                logger.error(f"Essential columns missing from statcast_batters. Needed: game_pk, at_bat_number, batter, pitcher, events, p_throws, stand. Found: {batter_cols_select}")
                return False
            if not all(c in pitcher_cols_select for c in ['game_pk', 'at_bat_number', 'pitcher']):
                 logger.error(f"Essential columns missing from statcast_pitchers. Needed: game_pk, at_bat_number, pitcher. Found: {pitcher_cols_select}")
                 return False

            logger.info(f"Columns selected from statcast_batters: {batter_cols_select}")
            logger.info(f"Columns selected from statcast_pitchers: {pitcher_cols_select}")

            for season in seasons:
                logger.info(f"Processing team aggregation for season: {season}")
                query_batt = f"SELECT {', '.join(batter_cols_select)} FROM statcast_batters WHERE season = {season}"
                query_pitch = f"SELECT {', '.join(pitcher_cols_select)} FROM statcast_pitchers WHERE season = {season}"

                try:
                    # Use pandas chunking for large tables if memory is a concern
                    df_batt_chunks = pd.read_sql_query(query_batt, conn, chunksize=500000)
                    df_pitch_chunks = pd.read_sql_query(query_pitch, conn, chunksize=500000)
                    df_batt = pd.concat(df_batt_chunks, ignore_index=True)
                    df_pitch = pd.concat(df_pitch_chunks, ignore_index=True)
                    logger.info(f"Season {season}: Loaded {len(df_batt)} batter rows, {len(df_pitch)} pitcher rows.")
                except Exception as e:
                    logger.error(f"Failed to load data for season {season}: {e}")
                    continue

                if df_batt.empty: # Pitcher data might be less critical if only calculating batting splits
                     logger.warning(f"Skipping season {season} due to missing batter data.")
                     continue

                # --- Data Preparation ---
                df_batt['game_date'] = pd.to_datetime(df_batt['game_date'], errors='coerce')
                df_batt.dropna(subset=['game_pk', 'game_date', 'home_team', 'away_team', 'at_bat_number', 'pitcher', 'batter', 'events'], inplace=True)

                # Identify PA ends (using batter data)
                df_batt['is_pa_end'] = df_batt['pitch_number'] == df_batt.groupby(['game_pk', 'at_bat_number'])['pitch_number'].transform('max')
                pa_ends_df = df_batt[df_batt['is_pa_end']].copy()

                # Calculate outcomes needed (K, BB, wOBA contribution) - use PA ends data
                pa_ends_df['is_strikeout'] = pa_ends_df['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)
                pa_ends_df['is_walk'] = pa_ends_df['events'].isin(['walk', 'hit_by_pitch']).astype(int)
                pa_ends_df['is_ab'] = pa_ends_df['events'].notna() & ~pa_ends_df['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf', 'intent_walk'])
                pa_ends_df['is_sac_fly'] = (pa_ends_df['events'] == 'sac_fly').astype(int)

                # wOBA Contribution
                woba_weights = {'walk': 0.69, 'hbp': 0.72, 'single': 0.89, 'double': 1.27, 'triple': 1.62, 'home_run': 2.10}
                if 'woba_value' in pa_ends_df.columns and pa_ends_df['woba_value'].notna().any():
                    pa_ends_df['woba_contribution'] = pa_ends_df['woba_value'].fillna(0)
                else:
                    pa_ends_df['woba_contribution'] = 0.0
                    for event, weight in woba_weights.items():
                         pa_ends_df.loc[pa_ends_df['events'].str.contains(event, case=False, na=False), 'woba_contribution'] = weight
                    pa_ends_df.loc[pa_ends_df['events'] == 'hit_by_pitch', 'woba_contribution'] = woba_weights['hbp'] # Explicitly add HBP

                # --- Aggregate PA outcomes by game, distinguishing pitcher hand and batter stance ---
                pa_agg = pa_ends_df.groupby(['game_pk', 'game_date', 'home_team', 'away_team', 'p_throws', 'stand']).agg(
                    pa=('at_bat_number', 'size'), # Count PA ends
                    k=('is_strikeout', 'sum'),
                    bb=('is_walk', 'sum'),
                    ab=('is_ab', 'sum'),
                    sf=('is_sac_fly', 'sum'),
                    woba_num=('woba_contribution', 'sum'),
                ).reset_index()

                # --- Calculate Team-Level Game Stats (Batting and Pitching perspectives) ---
                # For this, we need to identify which PAs belong to the home team batting vs. away team batting.
                # Simplification: Aggregate ALL PAs first, then pivot by batter stance (for pitching stats) and pitcher hand (for batting stats).
                # Assigning home/away batting roles requires roster info or complex inference.
                # Let's create stats *for the game as a whole* first.

                # Batting stats vs RHP/LHP
                batting_vs_hand = pa_agg.groupby(['game_pk', 'game_date', 'home_team', 'away_team', 'p_throws']).agg(
                    pa_vs_hand=('pa', 'sum'),
                    k_vs_hand=('k', 'sum'),
                    woba_num_vs_hand=('woba_num', 'sum'),
                    ab_vs_hand=('ab', 'sum'),
                    bb_vs_hand=('bb', 'sum'),
                    sf_vs_hand=('sf', 'sum'),
                ).reset_index()
                woba_denom_hand = batting_vs_hand['ab_vs_hand'] + batting_vs_hand['bb_vs_hand'] + batting_vs_hand['sf_vs_hand']
                batting_vs_hand['k_pct_vs_hand'] = (batting_vs_hand['k_vs_hand'] / batting_vs_hand['pa_vs_hand'].replace(0, np.nan)).fillna(0)
                batting_vs_hand['woba_vs_hand'] = (batting_vs_hand['woba_num_vs_hand'] / woba_denom_hand.replace(0, np.nan)).fillna(0)
                batting_pivot = batting_vs_hand.pivot_table(
                    index=['game_pk', 'game_date', 'home_team', 'away_team'], columns='p_throws',
                    values=['k_pct_vs_hand', 'woba_vs_hand', 'pa_vs_hand']
                )
                batting_pivot.columns = [f'batting_{metric}_{hand}' for metric, hand in batting_pivot.columns]
                batting_pivot.reset_index(inplace=True)

                # Pitching stats vs RHB/LHB
                pitching_vs_batter = pa_agg.groupby(['game_pk', 'game_date', 'home_team', 'away_team', 'stand']).agg(
                     pa_vs_batter=('pa', 'sum'),
                     k_vs_batter=('k', 'sum'),
                     woba_num_vs_batter=('woba_num', 'sum'),
                     ab_vs_batter=('ab', 'sum'),
                     bb_vs_batter=('bb', 'sum'),
                     sf_vs_batter=('sf', 'sum'),
                ).reset_index()
                woba_denom_batter = pitching_vs_batter['ab_vs_batter'] + pitching_vs_batter['bb_vs_batter'] + pitching_vs_batter['sf_vs_batter']
                pitching_vs_batter['k_pct_vs_batter'] = (pitching_vs_batter['k_vs_batter'] / pitching_vs_batter['pa_vs_batter'].replace(0, np.nan)).fillna(0)
                pitching_vs_batter['woba_vs_batter'] = (pitching_vs_batter['woba_num_vs_batter'] / woba_denom_batter.replace(0, np.nan)).fillna(0)

                pitching_pivot = pitching_vs_batter.pivot_table(
                    index=['game_pk', 'game_date', 'home_team', 'away_team'], columns='stand',
                    values=['k_pct_vs_batter', 'woba_vs_batter', 'pa_vs_batter']
                )
                pitching_pivot.columns = [f'pitching_{metric}_{stand}' for metric, stand in pitching_pivot.columns]
                pitching_pivot.reset_index(inplace=True)

                # Overall game totals
                game_totals = pa_agg.groupby(['game_pk', 'game_date', 'home_team', 'away_team']).agg(
                    total_pa = ('pa', 'sum'),
                    total_k = ('k', 'sum'),
                    total_bb = ('bb', 'sum'),
                    total_ab = ('ab', 'sum'),
                    total_sf = ('sf', 'sum'),
                    total_woba_num = ('woba_num', 'sum'),
                ).reset_index()
                game_woba_denom = game_totals['total_ab'] + game_totals['total_bb'] + game_totals['total_sf']
                game_totals['overall_k_pct'] = (game_totals['total_k'] / game_totals['total_pa'].replace(0,np.nan)).fillna(0)
                game_totals['overall_woba'] = (game_totals['total_woba_num'] / game_woba_denom.replace(0,np.nan)).fillna(0)


                # --- Merge All Game Stats ---
                game_summary = pd.merge(game_totals, batting_pivot, on=['game_pk', 'game_date', 'home_team', 'away_team'], how='left')
                game_summary = pd.merge(game_summary, pitching_pivot, on=['game_pk', 'game_date', 'home_team', 'away_team'], how='left')
                game_summary['season'] = season # Add season back

                # Fill NaNs resulting from merges/pivots
                for col in game_summary.columns:
                     if game_summary[col].isnull().any() and pd.api.types.is_numeric_dtype(game_summary[col]):
                          game_summary[col].fillna(0, inplace=True)

                all_game_teams_data.append(game_summary)
                logger.info(f"Finished team aggregation for season {season}. Games aggregated: {len(game_summary)}")


        if not all_game_teams_data:
             logger.error("No game-level team data could be aggregated across all seasons.")
             return False

        # Combine data across all seasons
        final_team_game_data = pd.concat(all_game_teams_data, ignore_index=True)
        logger.info(f"Total aggregated team-game records: {len(final_team_game_data)}")

        # Store to database
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed post-aggregation.")
            # Ensure table is dropped if it exists before replacing
            try:
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS game_level_team_stats")
                conn.commit()
                logger.info("Dropped existing game_level_team_stats table (if any).")
            except Exception as drop_e:
                 logger.warning(f"Could not drop game_level_team_stats table: {drop_e}")

            final_team_game_data.to_sql('game_level_team_stats', conn, if_exists='replace', index=False, chunksize=5000) # Use chunksize for writing
            logger.info("Stored aggregated data to 'game_level_team_stats'.")

        total_time = time.time() - start_time
        logger.info(f"Team aggregation completed in {total_time:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error during team aggregation: {e}", exc_info=True)
        logger.error(traceback.format_exc()) # Print full traceback
        return False

if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Module imports failed.")
    logger.info("Starting team game-level aggregation process...")
    success = aggregate_teams_to_game_level()
    if success:
        logger.info("Team game-level aggregation process finished successfully.")
    else:
        logger.error("Team game-level aggregation process failed.")