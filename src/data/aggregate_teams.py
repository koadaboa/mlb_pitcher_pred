# src/data/aggregate_teams.py
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import time

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.config import DataConfig # For chunk size maybe

logger = setup_logger('aggregate_teams')

def aggregate_teams_to_game_level():
    """
    Aggregates Statcast data to create game-level team statistics,
    including batting vs LHP/RHP and pitching vs LHB/RHB.
    """
    start_time = time.time()
    logger.info("Loading Statcast data for game-level TEAM aggregation...")

    # Load necessary data: statcast_batters and statcast_pitchers
    # Ideally load both to get batting and pitching stats per team per game.
    # This can be memory intensive. Load necessary columns in chunks.

    all_game_teams_data = []

    try:
        with DBConnection() as conn:
            # Determine seasons available in statcast data
            seasons = pd.read_sql_query("SELECT DISTINCT season FROM statcast_batters ORDER BY season", conn)['season'].tolist()
            logger.info(f"Found seasons in statcast_batters: {seasons}")

            for season in seasons:
                logger.info(f"Processing team aggregation for season: {season}")
                # Load batter data for the season (adjust chunksize as needed)
                query_batt = f"""
                    SELECT game_pk, game_date, home_team, away_team, batter, pitcher, events, description,
                           balls, strikes, p_throws, stand, at_bat_number, pitch_number, woba_value
                    FROM statcast_batters WHERE season = {season}
                """
                # Load pitcher data for the season (fewer columns might be needed)
                query_pitch = f"""
                     SELECT game_pk, game_date, home_team, away_team, pitcher, events, description,
                           balls, strikes, p_throws, stand, at_bat_number, pitch_number
                    FROM statcast_pitchers WHERE season = {season}
                """
                try:
                    df_batt = pd.read_sql_query(query_batt, conn) # Consider chunking if memory is an issue
                    df_pitch = pd.read_sql_query(query_pitch, conn) # Consider chunking
                    logger.info(f"Season {season}: Loaded {len(df_batt)} batter rows, {len(df_pitch)} pitcher rows.")
                except Exception as e:
                    logger.error(f"Failed to load data for season {season}: {e}")
                    continue

                if df_batt.empty or df_pitch.empty:
                     logger.warning(f"Skipping season {season} due to missing batter or pitcher data.")
                     continue

                # Combine batter and pitcher data for a full game view
                # Minimal combination for identifying teams and PAs
                df_game = pd.concat([
                     df_batt[['game_pk', 'game_date', 'home_team', 'away_team', 'at_bat_number', 'pitch_number', 'batter', 'pitcher', 'events', 'description', 'p_throws', 'stand', 'woba_value']],
                     df_pitch[['game_pk', 'game_date', 'home_team', 'away_team', 'at_bat_number', 'pitch_number', 'batter', 'pitcher', 'events', 'description', 'p_throws', 'stand']]
                     ], ignore_index=True).drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number']) # Basic deduplication

                df_game['game_date'] = pd.to_datetime(df_game['game_date'], errors='coerce')
                df_game.dropna(subset=['game_pk', 'game_date', 'home_team', 'away_team', 'at_bat_number', 'pitcher', 'batter'], inplace=True)


                # Determine batting team based on inning (top/bottom) - Approximation needed if inning not available
                # Simple approach: Assume batter is on away team if pitcher is on home team, etc.
                # This requires mapping pitcher/batter IDs to teams for that game, complex.
                # ALTERNATIVE: Use home_team/away_team directly. Aggregate stats *for* the home team and *for* the away team separately.

                # Identify PA ends
                df_game['is_pa_end'] = df_game['pitch_number'] == df_game.groupby(['game_pk', 'at_bat_number'])['pitch_number'].transform('max')

                # Calculate outcomes needed (K, BB, wOBA contribution)
                df_game['is_strikeout'] = df_game['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)
                df_game['is_walk'] = df_game['events'].isin(['walk', 'hit_by_pitch']).astype(int)
                # Add other events if needed for wOBA
                df_game['is_ab'] = df_game['events'].notna() & ~df_game['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf', 'intent_walk'])
                df_game['is_sac_fly'] = (df_game['events'] == 'sac_fly').astype(int)

                # wOBA Contribution
                woba_weights = {'walk': 0.69, 'hbp': 0.72, 'single': 0.89, 'double': 1.27, 'triple': 1.62, 'home_run': 2.10}
                if 'woba_value' in df_game.columns and df_game['woba_value'].notna().any():
                    df_game['woba_contribution'] = df_game['woba_value'].fillna(0)
                else:
                    # Calculate based on events (simplified)
                     df_game['woba_contribution'] = 0.0
                     # ... (logic to assign weights based on events, same as aggregate_batters) ...


                # --- Aggregate Batting Stats per Team per Game ---
                logger.info(f"Aggregating batting stats per team for season {season}...")
                # Group PAs by game and the batting team
                # We need to know which team the batter belonged to for each PA.
                # This mapping isn't directly in Statcast. A roster lookup or approximation is needed.
                # Approximation: If home_team == pitching team, batter is on away_team.
                # This requires knowing the pitching team for each PA.
                # Let's assume we can approximate team_batting and team_pitching roles later

                # Aggregate PA outcomes by game, separating Home/Away perspectives
                pa_outcomes_game = df_game[df_game['is_pa_end']].groupby(['game_pk', 'game_date', 'home_team', 'away_team', 'p_throws', 'stand']).agg(
                    pa=('at_bat_number', 'nunique'), # Count unique PAs
                    k=('is_strikeout', 'sum'),
                    bb=('is_walk', 'sum'),
                    ab=('is_ab', 'sum'),
                    sf=('is_sac_fly', 'sum'),
                    woba_num=('woba_contribution', 'sum'),
                ).reset_index()

                # Add team perspective - This is complex. Simplifying:
                # Calculate overall game stats first, then try to assign home/away later if needed.
                game_stats = pa_outcomes_game.groupby(['game_pk', 'game_date', 'home_team', 'away_team']).agg(
                    total_pa = ('pa', 'sum'),
                    total_k = ('k', 'sum'),
                    total_bb = ('bb', 'sum'),
                    total_ab = ('ab', 'sum'),
                    total_sf = ('sf', 'sum'),
                    total_woba_num = ('woba_num', 'sum'),
                ).reset_index()

                # --- Batting Splits vs LHP/RHP ---
                batting_vs_hand = pa_outcomes_game.groupby(['game_pk', 'game_date', 'home_team', 'away_team', 'p_throws']).agg(
                    pa_vs_hand=('pa', 'sum'),
                    k_vs_hand=('k', 'sum'),
                    woba_num_vs_hand=('woba_num', 'sum'),
                    ab_vs_hand=('ab', 'sum'),
                    bb_vs_hand=('bb', 'sum'),
                    sf_vs_hand=('sf', 'sum'),
                ).reset_index()
                # Calculate rates
                woba_denom_hand = batting_vs_hand['ab_vs_hand'] + batting_vs_hand['bb_vs_hand'] + batting_vs_hand['sf_vs_hand']
                batting_vs_hand['k_pct_vs_hand'] = (batting_vs_hand['k_vs_hand'] / batting_vs_hand['pa_vs_hand'].replace(0, np.nan)).fillna(0)
                batting_vs_hand['woba_vs_hand'] = (batting_vs_hand['woba_num_vs_hand'] / woba_denom_hand.replace(0, np.nan)).fillna(0)
                # Pivot
                batting_pivot = batting_vs_hand.pivot_table(
                    index=['game_pk', 'game_date', 'home_team', 'away_team'],
                    columns='p_throws',
                    values=['k_pct_vs_hand', 'woba_vs_hand', 'pa_vs_hand']
                )
                batting_pivot.columns = [f'batting_{metric}_{hand}' for metric, hand in batting_pivot.columns]
                batting_pivot.reset_index(inplace=True)

                # --- Pitching Splits vs LHB/RHB (Similar logic using 'stand') ---
                pitching_vs_batter = pa_outcomes_game.groupby(['game_pk', 'game_date', 'home_team', 'away_team', 'stand']).agg(
                     pa_vs_batter=('pa', 'sum'),
                     k_vs_batter=('k', 'sum'),
                     # ... other pitching outcomes ...
                ).reset_index()
                pitching_vs_batter['k_pct_vs_batter'] = (pitching_vs_batter['k_vs_batter'] / pitching_vs_batter['pa_vs_batter'].replace(0, np.nan)).fillna(0)
                # Pivot
                pitching_pivot = pitching_vs_batter.pivot_table(
                    index=['game_pk', 'game_date', 'home_team', 'away_team'],
                    columns='stand',
                    values=['k_pct_vs_batter', 'pa_vs_batter']
                )
                pitching_pivot.columns = [f'pitching_{metric}_{stand}' for metric, stand in pitching_pivot.columns]
                pitching_pivot.reset_index(inplace=True)


                # --- Merge All Game Stats ---
                game_summary = pd.merge(game_stats, batting_pivot, on=['game_pk', 'game_date', 'home_team', 'away_team'], how='left')
                game_summary = pd.merge(game_summary, pitching_pivot, on=['game_pk', 'game_date', 'home_team', 'away_team'], how='left')
                game_summary['season'] = season # Add season back

                # Fill NaNs resulting from merges/pivots
                for col in game_summary.columns:
                     if game_summary[col].isnull().any() and pd.api.types.is_numeric_dtype(game_summary[col]):
                          game_summary[col].fillna(0, inplace=True)

                all_game_teams_data.append(game_summary)
                logger.info(f"Finished team aggregation for season {season}. Games found: {len(game_summary)}")


        if not all_game_teams_data:
             logger.error("No game-level team data could be aggregated.")
             return False

        # Combine data across all seasons
        final_team_game_data = pd.concat(all_game_teams_data, ignore_index=True)
        logger.info(f"Total aggregated team-game records: {len(final_team_game_data)}")

        # Store to database
        with DBConnection() as conn:
            final_team_game_data.to_sql('game_level_team_stats', conn, if_exists='replace', index=False)
            logger.info("Stored aggregated data to 'game_level_team_stats'.")

        total_time = time.time() - start_time
        logger.info(f"Team aggregation completed in {total_time:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error during team aggregation: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting team game-level aggregation process...")
    aggregate_teams_to_game_level()
    logger.info("Team game-level aggregation process finished.")