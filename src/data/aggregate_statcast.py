# src/data/aggregate_statcast.py (Fully Refactored with SQL Aggregation - Added is_home flag)
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import time
import pickle
import gc
from sklearn.impute import KNNImputer

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.utils import setup_logger, DBConnection

# Setup logger
logger = setup_logger('aggregate_statcast')

# Create checkpoint directory
checkpoint_dir = project_root / 'data' / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def aggregate_statcast_pitchers_sql(use_checkpoint=True, force_reprocess=False):
    """
    Aggregate raw Statcast pitch-by-pitch data to PITCHER game level using SQL.
    Focused on starting pitchers, excluding spring training, with smart imputation.
    Calculates 'is_home' flag based on inning_topbot.

    Args:
        use_checkpoint (bool): Whether to try loading the FINAL aggregated result
                               from a checkpoint first.
        force_reprocess (bool): Force reprocessing even if checkpoint exists.

    Returns:
        pd.DataFrame: DataFrame with game-level pitcher metrics including 'is_home'.
    """
    start_time = time.time()
    logger.info("Starting PITCHER Statcast aggregation using SQL...")

    # Define checkpoint path for the FINAL aggregated data
    pitcher_checkpoint = checkpoint_dir / 'pitcher_game_level_sql.pkl'

    # --- Checkpoint Loading ---
    if use_checkpoint and not force_reprocess and pitcher_checkpoint.exists():
        try:
            logger.info(f"Loading from pitcher checkpoint: {pitcher_checkpoint}")
            with open(pitcher_checkpoint, 'rb') as f:
                game_level = pickle.load(f)

            if isinstance(game_level, pd.DataFrame) and not game_level.empty:
                # Check for essential columns including the new is_home flag
                if 'pitcher_id' in game_level.columns and 'strikeouts' in game_level.columns and 'is_home' in game_level.columns:
                    logger.info(f"Successfully loaded {len(game_level)} pitcher records from checkpoint")
                    return game_level
                else:
                    logger.warning("Pitcher checkpoint data missing required columns (possibly 'is_home'). Re-aggregating.")
            else:
                logger.warning("Invalid or empty pitcher checkpoint data")
        except Exception as e:
            logger.warning(f"Failed to load pitcher checkpoint: {e}")

    logger.info("Processing pitcher data using SQL aggregation (no valid checkpoint found)...")

    # --- SQL Aggregation Query for Pitchers (SQLite Compatible - Added is_home) ---
    # Assumes 'inning_topbot' column exists in 'statcast_pitchers' table
    sql_query_pitchers = """
    WITH PitchLevelFlags AS (
        SELECT
            pitcher_id,
            player_name,
            game_pk,
            game_date,
            game_type,
            home_team,
            away_team,
            p_throws,
            inning,
            inning_topbot, -- Include inning_topbot
            at_bat_number,
            pitch_number,
            events,
            description,
            pitch_type,
            release_speed,
            release_spin_rate,
            pfx_x,
            pfx_z,
            zone,
            MAX(CASE WHEN inning = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY pitcher_id, game_pk) as is_starter_flag,
            ROW_NUMBER() OVER (PARTITION BY pitcher_id, game_pk, at_bat_number ORDER BY pitch_number DESC) as pitch_rank_in_ab,
            ROW_NUMBER() OVER (PARTITION BY pitcher_id, game_pk ORDER BY pitch_number ASC) as game_pitch_seq, -- Sequence number for FIRST_VALUE
            CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END as is_strikeout_event,
            CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END as is_swinging_strike,
            CASE WHEN description = 'called_strike' THEN 1 ELSE 0 END as is_called_strike,
            CASE WHEN pitch_type IN ('FF', 'FT', 'FC', 'SI', 'FS') THEN 1 ELSE 0 END as is_fastball,
            CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'KN') THEN 1 ELSE 0 END as is_breaking,
            CASE WHEN pitch_type IN ('CH', 'SC', 'FO') THEN 1 ELSE 0 END as is_offspeed,
            CASE WHEN zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END as is_in_zone
        FROM
            statcast_pitchers -- Ensure this table has inning_topbot
        WHERE
            game_type = 'R'
    ),
    GameLevelAgg AS (
        SELECT
            pitcher_id,
            game_pk,
            game_date,
            MAX(player_name) as player_name,
            MAX(home_team) as home_team,
            MAX(away_team) as away_team,
            MAX(p_throws) as p_throws,
            -- *** FIX: Determine is_home flag using FIRST_VALUE and inning_topbot ***
            FIRST_VALUE(CASE WHEN inning_topbot = 'Top' THEN 1 WHEN inning_topbot = 'Bot' THEN 0 ELSE NULL END) OVER (PARTITION BY pitcher_id, game_pk ORDER BY game_pitch_seq ASC) as is_home,
            SUM(CASE WHEN pitch_rank_in_ab = 1 THEN is_strikeout_event ELSE 0 END) as strikeouts,
            COUNT(DISTINCT CASE WHEN pitch_rank_in_ab = 1 THEN at_bat_number ELSE NULL END) as batters_faced,
            COUNT(*) as total_pitches,
            SUM(is_swinging_strike) as total_swinging_strikes,
            SUM(is_called_strike) as total_called_strikes,
            SUM(is_fastball) as total_fastballs,
            SUM(is_breaking) as total_breaking,
            SUM(is_offspeed) as total_offspeed,
            SUM(is_in_zone) as total_in_zone,
            AVG(release_speed) as avg_velocity,
            MAX(release_speed) as max_velocity,
            AVG(release_spin_rate) as avg_spin_rate,
            AVG(pfx_x) as avg_horizontal_break,
            AVG(pfx_z) as avg_vertical_break
        FROM
            PitchLevelFlags
        WHERE
            is_starter_flag = 1
        GROUP BY
            pitcher_id,
            game_pk,
            game_date
    )
    SELECT DISTINCT -- Use DISTINCT because FIRST_VALUE might duplicate rows before final aggregation in some DBs
        g.pitcher_id,
        g.game_pk,
        g.game_date,
        g.player_name,
        g.home_team,
        g.away_team,
        g.p_throws,
        g.is_home, -- Include the new flag
        g.strikeouts,
        g.batters_faced,
        g.total_pitches,
        g.total_swinging_strikes,
        g.total_called_strikes,
        g.total_fastballs,
        g.total_breaking,
        g.total_offspeed,
        g.total_in_zone,
        g.avg_velocity,
        g.max_velocity,
        g.avg_spin_rate,
        g.avg_horizontal_break,
        g.avg_vertical_break,
        CAST(STRFTIME('%Y', g.game_date) AS INTEGER) as season,
        CAST(g.batters_faced AS REAL) / 3.0 as innings_pitched,
        CAST(g.strikeouts AS REAL) / NULLIF(g.batters_faced, 0) as k_percent,
        CAST(g.strikeouts AS REAL) * 9.0 / NULLIF(CAST(g.batters_faced AS REAL) / 3.0, 0) as k_per_9,
        CAST(g.total_swinging_strikes AS REAL) / NULLIF(g.total_pitches, 0) as swinging_strike_percent,
        CAST(g.total_called_strikes AS REAL) / NULLIF(g.total_pitches, 0) as called_strike_percent,
        CAST(g.total_fastballs AS REAL) / NULLIF(g.total_pitches, 0) as fastball_percent,
        CAST(g.total_breaking AS REAL) / NULLIF(g.total_pitches, 0) as breaking_percent,
        CAST(g.total_offspeed AS REAL) / NULLIF(g.total_pitches, 0) as offspeed_percent,
        CAST(g.total_in_zone AS REAL) / NULLIF(g.total_pitches, 0) as zone_percent
    FROM
        GameLevelAgg g
    WHERE g.is_home IS NOT NULL -- Ensure we could determine home/away status
    ORDER BY
        g.game_date, g.game_pk, g.pitcher_id;
    """

    # --- Execute SQL Query ---
    game_level = pd.DataFrame()
    try:
        logger.info("Executing SQL query for PITCHER game-level aggregation (incl. is_home)...")
        with DBConnection() as conn:
            if conn is None:
                raise ConnectionError("DB Connection failed.")
            game_level = pd.read_sql_query(sql_query_pitchers, conn)
            logger.info(f"SQL aggregation returned {len(game_level)} PITCHER game-level records for starters")

            # Verify is_home column exists and has values
            if 'is_home' not in game_level.columns:
                 logger.error("'is_home' column missing from SQL result. Check query and source table.")
                 # Handle error - maybe return empty or raise exception
                 return pd.DataFrame()
            if game_level['is_home'].isnull().any():
                 logger.warning(f"Found {game_level['is_home'].isnull().sum()} records where 'is_home' could not be determined (likely missing inning_topbot data). These records were excluded.")
                 # Note: The WHERE g.is_home IS NOT NULL clause in SQL handles this exclusion.

            # Convert game_date back to datetime if it's not already
            if 'game_date' in game_level.columns and not pd.api.types.is_datetime64_any_dtype(game_level['game_date']):
                 game_level['game_date'] = pd.to_datetime(game_level['game_date'])

    except Exception as e:
        logger.error(f"Error executing PITCHER SQL aggregation query: {e}", exc_info=True)
        return pd.DataFrame() # Return empty dataframe on error

    if game_level.empty:
        logger.error("PITCHER SQL aggregation resulted in an empty DataFrame.")
        return pd.DataFrame()

    # --- Optional: Filter Pitcher Data by Umpire Data ---
    # This step remains the same, but now operates on data that includes 'is_home'
    logger.info("Filtering PITCHER results based on available umpire data...")
    try:
        with DBConnection() as conn:
            if conn is not None:
                # *** Use correct table name 'umpire_data' ***
                umpire_query = "SELECT DISTINCT game_date, home_team, away_team FROM umpire_data"
                umpire_df = pd.read_sql_query(umpire_query, conn)

                if not umpire_df.empty:
                    umpire_df['game_date'] = pd.to_datetime(umpire_df['game_date']).dt.normalize()
                    game_level['game_date'] = pd.to_datetime(game_level['game_date']).dt.normalize()
                    initial_count = len(game_level)
                    game_level = pd.merge(
                        game_level,
                        umpire_df[['game_date', 'home_team', 'away_team']],
                        on=['game_date', 'home_team', 'away_team'],
                        how='inner'
                    )
                    logger.info(f"Filtered PITCHER records to {len(game_level)} with umpire data (from {initial_count})")
                else:
                    logger.warning("No umpire data found for PITCHER filtering.")
            else:
                 logger.warning("DB connection failed, skipping PITCHER umpire filtering.")
    except Exception as e:
        # Catch specific OperationalError for "no such table"
        if "no such table: umpire_data" in str(e):
             logger.error(f"Umpire filtering failed: Table 'umpire_data' not found in the database.", exc_info=False)
             logger.warning("Proceeding without umpire filtering for pitchers.")
        else:
             logger.warning(f"Failed to filter PITCHER data using umpire data: {e}", exc_info=True)
        # Continue execution without filtering if the table is missing or another error occurs

    # Check if empty *after* attempting filter, even if filter failed
    if game_level.empty:
        logger.error("PITCHER DataFrame is empty (potentially after umpire filtering attempt).")
        return pd.DataFrame()

    # --- Pitcher Imputation ---
    # (Imputation code remains the same as previous version)
    logger.info("Performing smart imputation for PITCHER missing values...")
    numeric_features = [
        'avg_velocity', 'max_velocity', 'avg_spin_rate',
        'avg_horizontal_break', 'avg_vertical_break',
        'k_percent', 'k_per_9', 'swinging_strike_percent', 'zone_percent',
        'fastball_percent', 'breaking_percent', 'offspeed_percent', 'innings_pitched'
    ]
    available_features = [col for col in numeric_features if col in game_level.columns]

    # 1. Median Imputation (Pitcher-specific + Global Fallback)
    for col in available_features:
        if game_level[col].isnull().sum() > 0:
            logger.info(f"Imputing PITCHER NaNs in '{col}' using pitcher/global median...")
            pitcher_medians = game_level.groupby('pitcher_id')[col].transform('median')
            game_level[f'{col}_imputed_median'] = game_level[col].isnull()
            game_level[col] = game_level[col].fillna(pitcher_medians)
            if game_level[col].isnull().sum() > 0:
                global_median = game_level[col].median()
                if pd.notna(global_median):
                    game_level[col] = game_level[col].fillna(global_median)
                    logger.info(f"  Used global median fallback ({global_median:.4f}) for PITCHER '{col}'")
                else:
                    game_level[col] = game_level[col].fillna(0)
                    logger.warning(f"  Global median for PITCHER '{col}' is NaN. Used 0 as fallback.")

    # 2. KNN Imputation
    logger.info("Performing KNN imputation for remaining complex PITCHER features...")
    complex_features = ['avg_velocity', 'avg_spin_rate', 'avg_horizontal_break', 'avg_vertical_break']
    available_complex = [col for col in complex_features if col in game_level.columns and game_level[col].isnull().any()]

    if available_complex and len(game_level) > 20:
        helper_cols = ['k_per_9', 'swinging_strike_percent', 'fastball_percent', 'innings_pitched']
        helper_cols = [col for col in helper_cols if col in game_level.columns and not game_level[col].isnull().all()]

        if helper_cols:
            imputer_cols = available_complex + helper_cols
            logger.info(f"Applying PITCHER KNN Imputation using features: {imputer_cols}")
            rows_to_impute_mask = game_level[available_complex].isnull().any(axis=1)
            imputer_data = game_level.loc[rows_to_impute_mask, imputer_cols].dropna(subset=helper_cols)

            if not imputer_data.empty and len(imputer_data) >= 5:
                try:
                    n_neighbors = min(5, len(imputer_data) -1)
                    if n_neighbors < 1: n_neighbors = 1
                    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                    imputed_values = imputer.fit_transform(imputer_data)
                    imputed_df = pd.DataFrame(imputed_values, columns=imputer_cols, index=imputer_data.index)
                    for col in available_complex:
                        target_mask = rows_to_impute_mask & game_level[col].isnull()
                        aligned_imputed, _ = imputed_df[col].align(target_mask, join='right', axis=0)
                        game_level.loc[target_mask, col] = aligned_imputed[target_mask]
                        game_level[f'{col}_imputed_knn'] = target_mask & game_level[col].notnull()
                    logger.info(f"Successfully applied PITCHER KNN imputation to relevant rows for columns: {', '.join(available_complex)}")
                except Exception as e:
                    logger.warning(f"PITCHER KNN imputation failed: {e}. Falling back to median/0.", exc_info=True)
                    for col in available_complex:
                         if game_level[col].isnull().sum() > 0:
                              median_val = game_level[col].median()
                              game_level[col] = game_level[col].fillna(median_val if pd.notna(median_val) else 0)
            elif imputer_data.empty:
                 logger.info("No PITCHER rows suitable for KNN imputation after filtering helpers.")
            else:
                 logger.info(f"Not enough samples ({len(imputer_data)}) for PITCHER KNN imputation.")
        else:
             logger.warning("PITCHER KNN imputation skipped: Not enough valid helper columns. Falling back to median/0.")
             for col in available_complex:
                 if game_level[col].isnull().sum() > 0:
                     median_val = game_level[col].median()
                     game_level[col] = game_level[col].fillna(median_val if pd.notna(median_val) else 0)

    # --- Final Check for Pitcher NaNs ---
    # (Code remains the same)
    final_nan_check = game_level[available_features].isnull().sum()
    logger.info(f"PITCHER NaN counts after imputation:\n{final_nan_check[final_nan_check > 0]}")
    if final_nan_check.sum() > 0:
         logger.warning("Some PITCHER NaNs remain after imputation. Filling with 0.")
         for col in available_features:
              if game_level[col].isnull().any():
                   game_level[col] = game_level[col].fillna(0)


    # --- Save Pitcher Checkpoint and Database ---
    # (Code remains the same)
    try:
        logger.info(f"Saving final aggregated PITCHER data checkpoint: {pitcher_checkpoint}")
        with open(pitcher_checkpoint, 'wb') as f:
            pickle.dump(game_level, f)
        logger.info("PITCHER checkpoint saved")
    except Exception as e:
        logger.error(f"Failed to save PITCHER checkpoint: {e}", exc_info=True)

    try:
        logger.info("Saving final aggregated PITCHER data to database table 'game_level_pitchers'...")
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            game_level.to_sql('game_level_pitchers', conn, if_exists='replace', index=False, chunksize=10000)
            logger.info(f"Saved {len(game_level)} PITCHER records to game_level_pitchers table")
    except Exception as e:
        logger.error(f"Failed to save aggregated PITCHER data to database: {e}", exc_info=True)


    total_time = time.time() - start_time
    logger.info(f"PITCHER SQL aggregation completed in {total_time:.2f}s")

    return game_level


def aggregate_statcast_batters_sql(use_checkpoint=True, force_reprocess=False):
    """
    Aggregate raw Statcast pitch-by-pitch data to TEAM/BATTER game level using SQL.
    Excludes spring training, calculates team batting stats per game.

    Args:
        use_checkpoint (bool): Whether to try loading the FINAL aggregated result
                               from a checkpoint first.
        force_reprocess (bool): Force reprocessing even if checkpoint exists.

    Returns:
        pd.DataFrame: DataFrame with game-level team batting metrics.
    """
    start_time = time.time()
    logger.info("Starting BATTER/TEAM Statcast aggregation using SQL...")

    # Define checkpoint path for the FINAL aggregated data
    batter_checkpoint = checkpoint_dir / 'team_game_level_sql.pkl'

    # --- Checkpoint Loading ---
    # (Code remains the same)
    if use_checkpoint and not force_reprocess and batter_checkpoint.exists():
        try:
            logger.info(f"Loading from batter/team checkpoint: {batter_checkpoint}")
            with open(batter_checkpoint, 'rb') as f:
                team_game_level = pickle.load(f)

            if isinstance(team_game_level, pd.DataFrame) and not team_game_level.empty:
                if 'team' in team_game_level.columns and 'k_percent' in team_game_level.columns:
                    logger.info(f"Successfully loaded {len(team_game_level)} team records from checkpoint")
                    return team_game_level
                else:
                    logger.warning("Team checkpoint data missing required columns")
            else:
                logger.warning("Invalid or empty team checkpoint data")
        except Exception as e:
            logger.warning(f"Failed to load team checkpoint: {e}")


    logger.info("Processing batter/team data using SQL aggregation (no valid checkpoint found)...")

    # --- SQL Aggregation Query for Batters/Teams (SQLite Compatible - No Comments) ---
    # **IMPORTANT**: Adjust the source table name 'statcast_pitchers' if needed!
    sql_query_batters = """
    WITH PitchLevelFlags AS (
        SELECT
            game_pk,
            game_date,
            game_type,
            home_team,
            away_team,
            inning_topbot,
            at_bat_number,
            pitch_number,
            events,
            description,
            zone,
            CASE
                WHEN inning_topbot = 'Top' THEN away_team
                ELSE home_team
            END as batting_team,
            CASE
                WHEN inning_topbot = 'Top' THEN home_team
                ELSE away_team
            END as fielding_team,
            ROW_NUMBER() OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number DESC) as pitch_rank_in_ab,
            CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END as is_strikeout_event,
            CASE WHEN events IN ('walk', 'hit_by_pitch') THEN 1 ELSE 0 END as is_walk_event,
            CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END as is_hit_event,
            CASE WHEN description LIKE '%swing%' OR description LIKE '%foul%' OR description LIKE '%hit_into_play%' THEN 1 ELSE 0 END as is_swing,
            CASE WHEN description LIKE '%foul%' OR description LIKE '%hit_into_play%' THEN 1 ELSE 0 END as is_contact,
            CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END as is_swinging_strike,
            CASE WHEN zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END as is_in_zone
        FROM
            statcast_pitchers
        WHERE
            game_type = 'R'
    ),
    TeamGameAgg AS (
        SELECT
            game_pk,
            game_date,
            batting_team as team,
            fielding_team as opponent,
            MAX(home_team) as home_team,
            MAX(away_team) as away_team,
            MAX(CASE WHEN batting_team = home_team THEN 1 ELSE 0 END) as is_home,
            COUNT(DISTINCT at_bat_number) as pa,
            SUM(CASE WHEN pitch_rank_in_ab = 1 THEN is_strikeout_event ELSE 0 END) as strikeouts,
            SUM(CASE WHEN pitch_rank_in_ab = 1 THEN is_walk_event ELSE 0 END) as walks,
            SUM(CASE WHEN pitch_rank_in_ab = 1 THEN is_hit_event ELSE 0 END) as hits,
            COUNT(*) as pitches_faced,
            SUM(is_swing) as swings,
            SUM(is_contact) as contact,
            SUM(is_swinging_strike) as swinging_strikes,
            SUM(is_in_zone) as zone_pitches,
            SUM(CASE WHEN is_in_zone = 0 AND is_swing = 1 THEN 1 ELSE 0 END) as chases,
            SUM(CASE WHEN is_in_zone = 1 AND is_swing = 1 THEN 1 ELSE 0 END) as zone_swings,
            SUM(CASE WHEN is_in_zone = 1 AND is_contact = 1 THEN 1 ELSE 0 END) as zone_contact
        FROM
            PitchLevelFlags
        GROUP BY
            game_pk,
            game_date,
            batting_team,
            fielding_team
    )
    SELECT
        t.*,
        CAST(STRFTIME('%Y', t.game_date) AS INTEGER) as season,
        CAST(t.strikeouts AS REAL) / NULLIF(t.pa, 0) as k_percent,
        CAST(t.walks AS REAL) / NULLIF(t.pa, 0) as bb_percent,
        CAST(t.swings AS REAL) / NULLIF(t.pitches_faced, 0) as swing_percent,
        CAST(t.contact AS REAL) / NULLIF(t.swings, 0) as contact_percent,
        CAST(t.swinging_strikes AS REAL) / NULLIF(t.pitches_faced, 0) as swinging_strike_percent,
        CAST(t.chases AS REAL) / NULLIF(t.pitches_faced - t.zone_pitches, 0) as chase_percent,
        CAST(t.zone_contact AS REAL) / NULLIF(t.zone_swings, 0) as zone_contact_percent
    FROM
        TeamGameAgg t
    ORDER BY
        t.game_date, t.game_pk, t.team;
    """

    # --- Execute SQL Query ---
    # (Code remains the same)
    team_game_level = pd.DataFrame()
    try:
        logger.info("Executing SQL query for BATTER/TEAM game-level aggregation...")
        with DBConnection() as conn:
            if conn is None:
                raise ConnectionError("DB Connection failed.")
            team_game_level = pd.read_sql_query(sql_query_batters, conn)
            logger.info(f"SQL aggregation returned {len(team_game_level)} TEAM game-level records")

            if 'game_date' in team_game_level.columns and not pd.api.types.is_datetime64_any_dtype(team_game_level['game_date']):
                 team_game_level['game_date'] = pd.to_datetime(team_game_level['game_date'])

    except Exception as e:
        logger.error(f"Error executing BATTER/TEAM SQL aggregation query: {e}", exc_info=True)
        return pd.DataFrame()

    if team_game_level.empty:
        logger.error("BATTER/TEAM SQL aggregation resulted in an empty DataFrame.")
        return pd.DataFrame()


    # --- Optional: Filter Team Data by Umpire Data ---
    # (Code remains the same, including the fix for table name)
    logger.info("Filtering TEAM results based on available umpire data...")
    try:
        with DBConnection() as conn:
            if conn is not None:
                umpire_query = "SELECT DISTINCT game_date, home_team, away_team FROM umpire_data" # Corrected table name
                umpire_df = pd.read_sql_query(umpire_query, conn)

                if not umpire_df.empty and 'home_team' in team_game_level.columns and 'away_team' in team_game_level.columns:
                    umpire_df['game_date'] = pd.to_datetime(umpire_df['game_date']).dt.normalize()
                    team_game_level['game_date'] = pd.to_datetime(team_game_level['game_date']).dt.normalize()
                    initial_count = len(team_game_level)
                    team_game_level = pd.merge(
                        team_game_level,
                        umpire_df[['game_date', 'home_team', 'away_team']],
                        on=['game_date', 'home_team', 'away_team'],
                        how='inner'
                    )
                    logger.info(f"Filtered TEAM records to {len(team_game_level)} with umpire data (from {initial_count})")
                elif 'home_team' not in team_game_level.columns or 'away_team' not in team_game_level.columns:
                    logger.warning("Cannot filter TEAM data by umpire: home_team/away_team columns missing from SQL result.")
                else:
                    logger.warning("No umpire data found for TEAM filtering.")
            else:
                 logger.warning("DB connection failed, skipping TEAM umpire filtering.")
    except Exception as e:
        if "no such table: umpire_data" in str(e):
             logger.error(f"Umpire filtering failed: Table 'umpire_data' not found in the database.", exc_info=False)
             logger.warning("Proceeding without umpire filtering for teams.")
        else:
             logger.warning(f"Failed to filter TEAM data using umpire data: {e}", exc_info=True)


    if team_game_level.empty:
        logger.warning("TEAM DataFrame is empty after optional umpire filtering. Proceeding without filter if possible.")


    # --- Team Imputation (Using Median) ---
    # (Code remains the same)
    logger.info("Performing median imputation for TEAM missing values...")
    team_numeric_features = [
        'k_percent', 'bb_percent', 'swing_percent', 'contact_percent',
        'swinging_strike_percent', 'chase_percent', 'zone_contact_percent'
    ]
    team_available_features = [col for col in team_numeric_features if col in team_game_level.columns]

    for col in team_available_features:
        if team_game_level[col].isnull().sum() > 0:
            logger.info(f"Imputing TEAM NaNs in '{col}' using team/global median...")
            team_medians = team_game_level.groupby('team')[col].transform('median')
            team_game_level[f'{col}_imputed_median'] = team_game_level[col].isnull()
            team_game_level[col] = team_game_level[col].fillna(team_medians)
            if team_game_level[col].isnull().sum() > 0:
                global_median = team_game_level[col].median()
                if pd.notna(global_median):
                    team_game_level[col] = team_game_level[col].fillna(global_median)
                    logger.info(f"  Used global median fallback ({global_median:.4f}) for TEAM '{col}'")
                else:
                    team_game_level[col] = team_game_level[col].fillna(0)
                    logger.warning(f"  Global median for TEAM '{col}' is NaN. Used 0 as fallback.")

    # --- Final Check for Team NaNs ---
    # (Code remains the same)
    final_nan_check_team = team_game_level[team_available_features].isnull().sum()
    logger.info(f"TEAM NaN counts after imputation:\n{final_nan_check_team[final_nan_check_team > 0]}")
    if final_nan_check_team.sum() > 0:
         logger.warning("Some TEAM NaNs remain after imputation. Filling with 0.")
         for col in team_available_features:
              if team_game_level[col].isnull().any():
                   team_game_level[col] = team_game_level[col].fillna(0)


    # --- Save Team Checkpoint and Database ---
    # (Code remains the same)
    try:
        logger.info(f"Saving final aggregated TEAM data checkpoint: {batter_checkpoint}")
        with open(batter_checkpoint, 'wb') as f:
            pickle.dump(team_game_level, f)
        logger.info("TEAM checkpoint saved")
    except Exception as e:
        logger.error(f"Failed to save TEAM checkpoint: {e}", exc_info=True)

    try:
        logger.info("Saving final aggregated TEAM data to database table 'game_level_team_stats'...")
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            team_game_level.to_sql('game_level_team_stats', conn, if_exists='replace', index=False, chunksize=10000)
            logger.info(f"Saved {len(team_game_level)} records to game_level_team_stats table")
    except Exception as e:
        logger.error(f"Failed to save aggregated TEAM data to database: {e}", exc_info=True)


    total_time = time.time() - start_time
    logger.info(f"BATTER/TEAM SQL aggregation completed in {total_time:.2f}s")

    return team_game_level


# Example usage (called from engineer_features.py)
if __name__ == "__main__":
    # This part is just for testing the functions directly if needed
    logger.info("Running aggregate_statcast functions directly for testing...")

    # Test pitcher aggregation
    logger.info("--- Testing Pitcher Aggregation ---")
    pitcher_data = aggregate_statcast_pitchers_sql(force_reprocess=False) # Set True to force run
    if not pitcher_data.empty:
        logger.info("Pitcher test run completed successfully.")
        # print(pitcher_data.head())
        # print(pitcher_data.info())
    else:
        logger.error("Pitcher test run failed or produced no data.")

    # Test batter/team aggregation
    logger.info("--- Testing Batter/Team Aggregation ---")
    team_data = aggregate_statcast_batters_sql(force_reprocess=False) # Set True to force run
    if not team_data.empty:
        logger.info("Batter/Team test run completed successfully.")
        # print(team_data.head())
        # print(team_data.info())
    else:
        logger.error("Batter/Team test run failed or produced no data.")

