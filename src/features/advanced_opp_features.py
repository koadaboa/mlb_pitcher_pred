# src/features/advanced_opp_features.py (Updated for Game-Level Stats)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

# ... (imports and setup logger same as before) ...
from src.data.utils import setup_logger, DBConnection
from src.config import StrikeoutModelConfig

logger = setup_logger('advanced_opp_features')

def create_advanced_opponent_features(window_games=10):
    """
    Calculates advanced opponent features using GAME-LEVEL team stats.
    Expects 'game_level_team_stats' table to exist.

    Args:
        window_games (int): Number of recent games for rolling calculations.

    Returns:
        pandas.DataFrame: DataFrame with game_pk, team (home/away), and advanced opponent features.
    """
    logger.info("Starting advanced opponent feature creation using GAME-LEVEL stats...")
    try:
        with DBConnection() as conn:
            logger.info("Loading game_level_team_stats data...")
            query = "SELECT * FROM game_level_team_stats ORDER BY game_date" # Order essential for rolling
            try:
                df_game_teams = pd.read_sql_query(query, conn)
                logger.info(f"Loaded {len(df_game_teams)} rows from game_level_team_stats")
            except Exception as e:
                 logger.error(f"Failed to load game_level_team_stats: {e}. Run aggregate_teams.py first.")
                 return pd.DataFrame()

        if df_game_teams.empty:
            logger.error("game_level_team_stats table is empty.")
            return pd.DataFrame()

        df_game_teams['game_date'] = pd.to_datetime(df_game_teams['game_date'])
        # Data needs to be processed per team (home and away perspectives)
        # Create separate entries for each team's perspective in a game

        home_df = df_game_teams.copy()
        home_df['team'] = home_df['home_team']
        home_df['opponent'] = home_df['away_team']
        home_df['is_home'] = 1
        # Rename columns to reflect 'team' perspective (e.g., team_batting_k_pct_vs_hand_R)
        home_df.rename(columns={col: f"team_{col}" for col in home_df.columns if col.startswith(('batting_', 'pitching_'))}, inplace=True)


        away_df = df_game_teams.copy()
        away_df['team'] = away_df['away_team']
        away_df['opponent'] = away_df['home_team']
        away_df['is_home'] = 0
        # Rename columns similarly
        away_df.rename(columns={col: f"team_{col}" for col in away_df.columns if col.startswith(('batting_', 'pitching_'))}, inplace=True)

        # Combine into one long dataframe indexed by team and game
        df_long = pd.concat([home_df, away_df], ignore_index=True)
        df_long = df_long.sort_values(['team', 'game_date']).reset_index(drop=True)

        # --- Feature Calculation (Rolling Game Averages) ---
        logger.info(f"Calculating rolling {window_games}-game features...")
        advanced_features = {}
        # Key columns to keep
        key_cols = ['game_pk', 'game_date', 'team', 'opponent', 'is_home', 'season']
        for col in key_cols: advanced_features[col] = df_long[col]


        # Define metrics from game_level_team_stats to roll
        # Example: focus on batting K% vs R/L and pitching K% vs R/L batters
        metrics_to_roll = [
            'team_batting_k_pct_vs_hand_R', 'team_batting_k_pct_vs_hand_L',
            'team_batting_woba_vs_hand_R', 'team_batting_woba_vs_hand_L',
            'team_pitching_k_pct_vs_batter_R', 'team_pitching_k_pct_vs_batter_L',
            # Add overall metrics if needed: 'total_k', 'total_pa', etc.
        ]

        # Verify columns exist
        cols_to_process = [m for m in metrics_to_roll if m in df_long.columns]
        missing_cols = [m for m in metrics_to_roll if m not in df_long.columns]
        if missing_cols: logger.warning(f"Missing columns for rolling opponent features: {missing_cols}.")
        if not cols_to_process:
             logger.error("No valid columns found for rolling opponent features.")
             return pd.DataFrame(advanced_features) # Return keys even if no features calculated


        grouped_team = df_long.groupby('team')

        for metric in cols_to_process:
            # Shift metric within each team group, sorted by date
            shifted_metric = grouped_team[metric].shift(1)

            # Calculate rolling mean on shifted data
            roll_mean_col = f'rolling_{window_games}g_{metric}'
            rolling_mean = shifted_metric.rolling(window_games, min_periods=max(1, window_games // 2)).mean()
            advanced_features[roll_mean_col] = rolling_mean.reset_index(level=0, drop=True)

            # Calculate rolling std dev on shifted data
            roll_std_col = f'rolling_{window_games}g_{metric}_std'
            rolling_std = shifted_metric.rolling(window_games, min_periods=max(2, window_games // 2)).std()
            advanced_features[roll_std_col] = rolling_std.reset_index(level=0, drop=True)


        # Convert dictionary to DataFrame
        advanced_features_df = pd.DataFrame(advanced_features)

        # --- Handle NaNs introduced by shifting/rolling ---
        newly_created_cols = [col for col in advanced_features_df.columns if col.startswith('rolling_')]
        logger.info(f"Filling NaNs in {len(newly_created_cols)} rolling opponent feature columns...")
        for col in newly_created_cols:
            if advanced_features_df[col].isnull().any():
                median_fill = advanced_features_df[col].median() # Use overall median
                advanced_features_df[col].fillna(median_fill, inplace=True)
                # logger.debug(f"Filled NaNs in {col} with median {median_fill:.4f}")

        # --- Save to Database ---
        table_name = 'advanced_opponent_game_features' # New table name
        with DBConnection() as conn:
             advanced_features_df.to_sql(table_name, conn, if_exists='replace', index=False)
             logger.info(f"Stored {len(advanced_features_df)} advanced opponent game features to '{table_name}'")

        logger.info("Successfully created advanced opponent features using game-level stats.")
        return advanced_features_df

    except Exception as e:
        logger.error(f"Error creating advanced opponent features: {str(e)}", exc_info=True)
        return pd.DataFrame()

if __name__ == "__main__":
    logger.info("Running advanced opponent feature creation (game-level) directly for testing...")
    test_features = create_advanced_opponent_features()
    # ... (add testing print statements if desired) ...