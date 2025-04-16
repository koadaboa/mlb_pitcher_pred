# src/features/pitcher_features.py (Fixed ValueError + Merge on game_pk)
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import gc # Import gc

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.utils import setup_logger, DBConnection

# Setup logger
logger = setup_logger('pitcher_features')

# (create_recency_weighted_features, create_trend_features, create_rest_features,
#  create_pitch_arsenal_features functions remain the same as the previous version)
# ...

def create_recency_weighted_features(df):
    """
    Create features with higher weights for more recent games.

    Args:
        df: DataFrame with pitcher game data sorted by pitcher_id and game_date

    Returns:
        DataFrame with added recency-weighted features
    """
    logger.info("Creating recency-weighted pitcher features...")

    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure data is sorted correctly
    result_df = result_df.sort_values(['pitcher_id', 'game_date'])

    # Group by pitcher
    grouped = result_df.groupby('pitcher_id')

    # --- Exponentially Weighted Features ---
    # These give more weight to recent games

    # CRITICAL: First shift all metrics to prevent data leakage
    metrics_to_weight = [
        'strikeouts', 'batters_faced', 'innings_pitched',
        'total_pitches', 'swinging_strikes', 'called_strikes', # Corrected: use aggregated counts
        'avg_velocity', 'max_velocity', 'zone_percent',
        'swinging_strike_percent', 'fastball_percent',
        'breaking_percent', 'offspeed_percent',
        'k_percent', 'k_per_9' # Add key ratios
    ]

    # Check which metrics are actually in the dataframe
    avail_metrics = [col for col in metrics_to_weight if col in result_df.columns]
    logger.info(f"Available metrics for weighting: {avail_metrics}")


    # Create shifted versions of each metric
    for metric in avail_metrics:
        result_df[f'{metric}_lag1'] = grouped[metric].shift(1)

    # Apply exponential weighting with different spans
    spans = [3, 5, 10]  # Roughly correspond to half-lives in games

    for span in spans:
        # Use higher alpha values (0.5-0.7) to weight recent games more heavily
        # alpha = 2 / (span + 1)  # Convert span to alpha (not directly used by .ewm span)

        for metric in avail_metrics:
            lag_col = f'{metric}_lag1'
            # Apply EWM to lagged values
            # Ensure the column exists before transform
            if lag_col in result_df.columns:
                # Handle potential all-NaN slices within a group gracefully
                try:
                    result_df[f'ewma_{span}g_{metric}'] = grouped[lag_col].transform(
                        lambda x: x.ewm(span=span, adjust=True, min_periods=1).mean()
                    )
                except Exception as e:
                     logger.warning(f"Could not calculate EWMA for {metric} (span {span}): {e}")
                     result_df[f'ewma_{span}g_{metric}'] = np.nan # Assign NaN if calculation fails
            else:
                 logger.warning(f"Lagged column {lag_col} not found for EWMA calculation.")
                 result_df[f'ewma_{span}g_{metric}'] = np.nan


    # --- Calculate Derived Metrics from EWMA components (if components exist) ---
    # K/9 from weighted components
    for span in spans:
        ewma_k_col = f'ewma_{span}g_strikeouts'
        ewma_ip_col = f'ewma_{span}g_innings_pitched'
        if ewma_k_col in result_df.columns and ewma_ip_col in result_df.columns:
            result_df[f'ewma_{span}g_k_per_9'] = (
                result_df[ewma_k_col] * 9 /
                result_df[ewma_ip_col].replace(0, np.nan)
            ).fillna(0)
        else:
             result_df[f'ewma_{span}g_k_per_9'] = np.nan # Indicate metric couldn't be calculated

        # K% from weighted components
        ewma_bf_col = f'ewma_{span}g_batters_faced'
        if ewma_k_col in result_df.columns and ewma_bf_col in result_df.columns:
            result_df[f'ewma_{span}g_k_percent'] = (
                result_df[ewma_k_col] /
                result_df[ewma_bf_col].replace(0, np.nan)
            ).fillna(0)
        else:
             result_df[f'ewma_{span}g_k_percent'] = np.nan

    # --- Explicit Last N Games Features ---
    # Direct features specifically for most recent games

    # Last game features
    for metric in avail_metrics:
        lag1_col = f'{metric}_lag1' # Use already created lag column
        if lag1_col in result_df.columns:
             result_df[f'last_game_{metric}'] = result_df[lag1_col]
        else:
             result_df[f'last_game_{metric}'] = np.nan


    # Second-to-last game features
    for metric in avail_metrics:
        result_df[f'{metric}_lag2'] = grouped[metric].shift(2) # Calculate lag 2
        lag2_col = f'{metric}_lag2'
        if lag2_col in result_df.columns:
             result_df[f'second_last_game_{metric}'] = result_df[lag2_col]
        else:
             result_df[f'second_last_game_{metric}'] = np.nan

    # Calculate weighted combinations with higher weight to most recent game
    last_k_col = 'last_game_strikeouts'
    last_ip_col = 'last_game_innings_pitched'
    sec_last_k_col = 'second_last_game_strikeouts'
    sec_last_ip_col = 'second_last_game_innings_pitched'

    if last_k_col in result_df.columns and last_ip_col in result_df.columns:
        # Last game K/9
        result_df['last_game_k_per_9'] = (
            result_df[last_k_col] * 9 /
            result_df[last_ip_col].replace(0, np.nan)
        ).fillna(0)

        # Second last game K/9 (check if columns exist)
        if sec_last_k_col in result_df.columns and sec_last_ip_col in result_df.columns:
            result_df['second_last_game_k_per_9'] = (
                result_df[sec_last_k_col] * 9 /
                result_df[sec_last_ip_col].replace(0, np.nan)
            ).fillna(0)

            # Weighted average of last two games (70% last, 30% second last)
            result_df['weighted_recent_k_per_9'] = (
                0.7 * result_df['last_game_k_per_9'] +
                0.3 * result_df['second_last_game_k_per_9']
            )
        else:
             result_df['second_last_game_k_per_9'] = np.nan
             result_df['weighted_recent_k_per_9'] = result_df['last_game_k_per_9'] # Fallback to just last game


    # K% combinations if available
    last_bf_col = 'last_game_batters_faced'
    sec_last_bf_col = 'second_last_game_batters_faced'
    if last_k_col in result_df.columns and last_bf_col in result_df.columns:
        # Last game K%
        result_df['last_game_k_percent'] = (
            result_df[last_k_col] /
            result_df[last_bf_col].replace(0, np.nan)
        ).fillna(0)

        # Second last game K%
        if sec_last_k_col in result_df.columns and sec_last_bf_col in result_df.columns:
            result_df['second_last_game_k_percent'] = (
                result_df[sec_last_k_col] /
                result_df[sec_last_bf_col].replace(0, np.nan)
            ).fillna(0)

            # Weighted average (70/30 split)
            result_df['weighted_recent_k_percent'] = (
                0.7 * result_df['last_game_k_percent'] +
                0.3 * result_df['second_last_game_k_percent']
            )
        else:
            result_df['second_last_game_k_percent'] = np.nan
            result_df['weighted_recent_k_percent'] = result_df['last_game_k_percent'] # Fallback

    # Drop intermediate lag columns if desired
    # result_df.drop(columns=[f'{m}_lag1' for m in avail_metrics if f'{m}_lag1' in result_df], inplace=True)
    # result_df.drop(columns=[f'{m}_lag2' for m in avail_metrics if f'{m}_lag2' in result_df], inplace=True)

    logger.info("Completed recency-weighted feature creation.")
    return result_df

def create_trend_features(df):
    """
    Create features that capture performance trends and changes in form.

    Args:
        df: DataFrame with pitcher game data sorted by pitcher_id and game_date

    Returns:
        DataFrame with added trend and form features
    """
    logger.info("Creating trend and form change features...")

    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure data is sorted
    result_df = result_df.sort_values(['pitcher_id', 'game_date'])

    # Group by pitcher
    grouped = result_df.groupby('pitcher_id')

    # --- Performance Change Features ---

    change_metrics = ['strikeouts', 'innings_pitched', 'batters_faced',
                     'swinging_strike_percent', 'avg_velocity', 'k_percent', 'k_per_9']

    avail_metrics = [col for col in change_metrics if col in result_df.columns]
    logger.info(f"Available metrics for trend features: {avail_metrics}")

    # --- Streak/Trend Features ---

    # Is the pitcher improving or declining?
    k_col = 'strikeouts' # Or use k_per_9 or k_percent if preferred
    if k_col in result_df.columns:
        k_lag1 = grouped[k_col].shift(1)
        k_lag2 = grouped[k_col].shift(2)
        k_lag3 = grouped[k_col].shift(3)
        # Check if lag1 > lag2 AND lag2 > lag3
        result_df['k_trend_up_lagged'] = ((k_lag1 > k_lag2) & (k_lag2 > k_lag3)).fillna(False).astype(int)
        # Check if lag1 < lag2 AND lag2 < lag3
        result_df['k_trend_down_lagged'] = ((k_lag1 < k_lag2) & (k_lag2 < k_lag3)).fillna(False).astype(int)

    # --- Form Volatility/Consistency Features ---

    # Standard deviation to detect volatility (use shifted values!)
    vol_windows = [3, 5]
    for metric in avail_metrics:
        shifted_metric = grouped[metric].shift(1)

        for window in vol_windows:
             # Need at least 2 periods for std dev
             min_periods_std = max(2, window // 2)
             try:
                  result_df[f'{metric}_volatility_{window}g'] = shifted_metric.transform(
                       lambda x: x.rolling(window, min_periods=min_periods_std).std()
                  )
             except Exception as e:
                  logger.warning(f"Could not calculate volatility for {metric} (window {window}): {e}")
                  result_df[f'{metric}_volatility_{window}g'] = np.nan


    # --- Form Change Detection ---

    # Compare recent performance to baseline
    for metric in avail_metrics:
        # Compare last 2 game average to 10-game baseline (use lagged values)
        lag1 = grouped[metric].shift(1)
        lag2 = grouped[metric].shift(2)
        # Ensure we have enough data for rolling mean
        min_periods_roll = 5
        try:
            baseline_10g = lag1.transform(
                lambda x: x.rolling(10, min_periods=min_periods_roll).mean()
            )
            # Calculate only if lag1 and lag2 are not NaN
            result_df[f'{metric}_last2g_vs_baseline'] = (
                (lag1 + lag2) / 2 - baseline_10g
            )
        except Exception as e:
             logger.warning(f"Could not calculate last2g_vs_baseline for {metric}: {e}")
             result_df[f'{metric}_last2g_vs_baseline'] = np.nan


    # Velocity change is particularly important
    velo_col = 'avg_velocity'
    if velo_col in result_df.columns:
        lag1_velo = grouped[velo_col].shift(1)
        min_periods_roll = 5
        try:
            # Compare recent velocity to baseline
            baseline_10g_velo = lag1_velo.transform(
                lambda x: x.rolling(10, min_periods=min_periods_roll).mean()
            )
            result_df['velocity_vs_baseline'] = lag1_velo - baseline_10g_velo

            # Flag significant velocity drops (potential injury/fatigue)
            result_df['significant_velo_drop'] = (
                result_df['velocity_vs_baseline'] < -1.0 # 1 mph drop
            ).fillna(False).astype(int) # Handle NaNs
        except Exception as e:
             logger.warning(f"Could not calculate velocity trend features: {e}")
             result_df['velocity_vs_baseline'] = np.nan
             result_df['significant_velo_drop'] = 0


    logger.info("Completed trend and form change feature creation.")
    return result_df

def create_rest_features(df):
    """
    Create features related to pitcher rest and schedule.

    Args:
        df: DataFrame with pitcher game data sorted by pitcher_id and game_date

    Returns:
        DataFrame with added rest-related features
    """
    logger.info("Creating pitcher rest features...")

    # Make a copy
    result_df = df.copy()

    # Ensure data is sorted
    result_df = result_df.sort_values(['pitcher_id', 'game_date'])

    # Calculate days since last start
    result_df['days_since_last_game'] = result_df.groupby('pitcher_id')['game_date'].diff().dt.days

    # Create rest day categories
    result_df['rest_days_4_less'] = (
        (result_df['days_since_last_game'] > 0) &
        (result_df['days_since_last_game'] <= 4)
    ).fillna(False).astype(int) # Handle NaNs for first game

    result_df['rest_days_5'] = (
        result_df['days_since_last_game'] == 5
    ).fillna(False).astype(int)

    result_df['rest_days_6_more'] = (
        result_df['days_since_last_game'] >= 6
    ).fillna(False).astype(int)

    # Extra long rest (potential IL stint or All-Star break)
    result_df['extended_rest'] = (
        result_df['days_since_last_game'] >= 10
    ).fillna(False).astype(int)

    # Calculate rolling workload metrics
    ip_col = 'innings_pitched'
    pitch_col = 'total_pitches'
    if ip_col in result_df.columns and pitch_col in result_df.columns:

        # Reset index before groupby.apply to prevent ValueError
        # Ensure a unique index for alignment after apply
        original_index = result_df.index # Keep original index if needed later
        result_df = result_df.reset_index(drop=True)

        # Define apply function for rolling sum on date index
        # This function now returns a series with the group's original index
        def rolling_sum_past_days(group, metric_col, days):
            group_original_index = group.index # Capture index of the group passed to apply
            group = group.set_index('game_date')
            rolled = group[metric_col].shift(1).rolling(f'{days}D').sum()
            # Reindex back to the original group index before returning
            return rolled.reindex(group.index).set_axis(group_original_index)

        logger.info("Calculating rolling workload (IP)...")
        try:
            ip_last_15d = result_df.groupby('pitcher_id', group_keys=False).apply(
                 rolling_sum_past_days, metric_col=ip_col, days=15
            )
            # Assign the result - should align with reset index
            result_df['ip_last_15d'] = ip_last_15d
        except ValueError as ve:
             logger.error(f"ValueError during IP workload calculation: {ve}", exc_info=True)
             logger.error(f"Index type: {type(result_df.index)}, is unique: {result_df.index.is_unique}")
             if 'ip_last_15d' in locals(): logger.error(f"Result index type: {type(ip_last_15d.index)}, is unique: {ip_last_15d.index.is_unique}, length: {len(ip_last_15d)}")
             raise ve # Re-raise after logging
        except Exception as e:
             logger.error(f"Error calculating IP workload: {e}", exc_info=True)
             result_df['ip_last_15d'] = np.nan # Assign NaN on error


        logger.info("Calculating rolling workload (Pitches)...")
        try:
            pitches_last_15d = result_df.groupby('pitcher_id', group_keys=False).apply(
                 rolling_sum_past_days, metric_col=pitch_col, days=15
            )
            result_df['pitches_last_15d'] = pitches_last_15d
        except ValueError as ve:
             logger.error(f"ValueError during Pitch workload calculation: {ve}", exc_info=True)
             logger.error(f"Index type: {type(result_df.index)}, is unique: {result_df.index.is_unique}")
             if 'pitches_last_15d' in locals(): logger.error(f"Result index type: {type(pitches_last_15d.index)}, is unique: {pitches_last_15d.index.is_unique}, length: {len(pitches_last_15d)}")
             raise ve # Re-raise after logging
        except Exception as e:
             logger.error(f"Error calculating Pitch workload: {e}", exc_info=True)
             result_df['pitches_last_15d'] = np.nan # Assign NaN on error


        # Restore original index if it was kept and needed downstream
        # result_df.index = original_index # Only if drop=False was used and index needed

    logger.info("Completed rest feature creation.")
    return result_df


def create_pitch_arsenal_features(df):
    """
    Create features related to a pitcher's arsenal and pitch mix from game-level data.

    Args:
        df: DataFrame with pitcher game data including aggregated pitch type info
            (e.g., fastball_percent, breaking_percent, offspeed_percent)

    Returns:
        DataFrame with added pitch arsenal features
    """
    logger.info("Creating pitch arsenal features (from game-level data)...")

    # Make a copy
    result_df = df.copy()

    # Pitch mix trend features based on game-level percentages
    arsenal_metrics = ['fastball_percent', 'breaking_percent', 'offspeed_percent']
    avail_metrics = [col for col in arsenal_metrics if col in result_df.columns]
    logger.info(f"Available arsenal metrics for trends: {avail_metrics}")


    if not avail_metrics:
         logger.warning("No game-level pitch percentage columns found. Skipping arsenal trends.")
         return result_df

    # Ensure data is sorted
    result_df = result_df.sort_values(['pitcher_id', 'game_date'])

    # Calculate trends in pitch usage
    for metric in avail_metrics:
        # Group by pitcher
        grouped = result_df.groupby('pitcher_id')

        # Change in usage from previous game (shift first)
        lag1 = grouped[metric].shift(1)

        # Compare to 5-game baseline (properly shifted)
        min_periods_roll = 3
        try:
            baseline_5g = lag1.transform(
                lambda x: x.rolling(5, min_periods=min_periods_roll).mean()
            )
            result_df[f'{metric}_vs_baseline'] = lag1 - baseline_5g
        except Exception as e:
             logger.warning(f"Could not calculate baseline comparison for {metric}: {e}")
             result_df[f'{metric}_vs_baseline'] = np.nan


    # Pitch effectiveness features (using game-level proxies if available)
    # Example: Use swinging_strike_percent as a proxy for effectiveness
    effectiveness_proxy = 'swinging_strike_percent'
    if effectiveness_proxy in result_df.columns:
         logger.info(f"Using '{effectiveness_proxy}' as effectiveness proxy.")
         grouped = result_df.groupby('pitcher_id')
         shifted_metric = grouped[effectiveness_proxy].shift(1)

         # EWMA with different spans
         for span in [3, 5]:
              try:
                   result_df[f'ewma_{span}g_{effectiveness_proxy}'] = shifted_metric.transform(
                        lambda x: x.ewm(span=span, adjust=True, min_periods=1).mean()
                   )
              except Exception as e:
                   logger.warning(f"Could not calculate EWMA for {effectiveness_proxy} (span {span}): {e}")
                   result_df[f'ewma_{span}g_{effectiveness_proxy}'] = np.nan

    else:
         logger.warning("Effectiveness proxy 'swinging_strike_percent' not found. Skipping related features.")


    logger.info("Completed pitch arsenal feature creation.")
    return result_df

def create_opponent_team_features(df, team_game_stats_df=None):
    """
    Create features related to the opposing team using aggregated game-level team stats.
    Merges based on game_pk to correctly handle doubleheaders.

    Args:
        df: DataFrame with pitcher game data including opponent team info and game_pk
        team_game_stats_df: DataFrame with aggregated team batting stats per game
                            (e.g., from game_level_team_stats table), must include game_pk

    Returns:
        DataFrame with added opponent features
    """
    logger.info("Creating opponent team features using game-level stats (merging on game_pk)...")

    # Make a copy
    result_df = df.copy()

    # --- Input Validation ---
    if team_game_stats_df is None or team_game_stats_df.empty:
        logger.warning("No team game stats data provided. Skipping opponent features.")
        return result_df

    required_pitcher_cols = ['game_pk', 'game_date', 'home_team', 'away_team', 'is_home']
    if not all(col in result_df.columns for col in required_pitcher_cols):
        logger.error(f"Pitcher data missing required columns for opponent features ({required_pitcher_cols}). Skipping.")
        return result_df

    required_team_cols = ['game_pk', 'game_date', 'team']
    if not all(col in team_game_stats_df.columns for col in required_team_cols):
        logger.error(f"Team stats data missing required columns for opponent features ({required_team_cols}). Skipping.")
        return result_df

    # Identify opponent team using is_home flag
    result_df['opponent_team'] = np.where(result_df['is_home'] == 1,
                                          result_df['away_team'],
                                          result_df['home_team'])

    # --- Prepare Team Stats ---
    team_metrics_to_use = [
        'k_percent', 'bb_percent', 'swing_percent', 'contact_percent',
        'swinging_strike_percent', 'chase_percent', 'zone_contact_percent'
    ]
    team_cols = required_team_cols + [col for col in team_metrics_to_use if col in team_game_stats_df.columns]

    if len(team_cols) <= len(required_team_cols): # Only identifiers present
         logger.warning("No usable metrics found in team_game_stats_df. Skipping opponent features.")
         return result_df

    team_stats_subset = team_game_stats_df[team_cols].copy()

    # Rename columns to avoid clashes and indicate opponent stats
    rename_dict = {col: f"opp_{col}" for col in team_metrics_to_use if col in team_stats_subset.columns}
    team_stats_subset.rename(columns=rename_dict, inplace=True)
    logger.info(f"Using opponent metrics: {list(rename_dict.values())}")


    # --- Calculate Rolling Opponent Averages ---
    # Calculate rolling averages for opponent teams based on their past games
    team_stats_subset = team_stats_subset.sort_values(['team', 'game_date'])

    opp_metrics_renamed = list(rename_dict.values())
    rolling_windows = [5, 10, 20] # Games

    for metric in opp_metrics_renamed:
        grouped_team = team_stats_subset.groupby('team')
        # Shift to prevent leakage from the game being predicted
        shifted_metric = grouped_team[metric].shift(1)
        for window in rolling_windows:
            min_periods_roll = max(2, window // 2)
            try:
                # Use rolling by game count (needs default index)
                # Or rolling by date if preferred (needs date index)
                # Sticking to game count for simplicity here
                team_stats_subset[f'opp_roll_{window}g_{metric}'] = shifted_metric.transform(
                    lambda x: x.rolling(window, min_periods=min_periods_roll).mean()
                )
            except Exception as e:
                 logger.warning(f"Could not calculate rolling avg for {metric} (window {window}): {e}")
                 team_stats_subset[f'opp_roll_{window}g_{metric}'] = np.nan


    # --- Merge Opponent Stats with Pitcher Data ---
    # *** FIX: Merge using game_pk as well to handle doubleheaders ***

    # Select only the rolling average columns and merge keys
    opp_feature_cols = ['game_pk', 'team'] + [col for col in team_stats_subset.columns if col.startswith('opp_roll_')]
    opp_features_to_merge = team_stats_subset[opp_feature_cols].copy()

    # Ensure merge key types are consistent if necessary (though game_pk should be numeric)
    # result_df['game_pk'] = pd.to_numeric(result_df['game_pk'], errors='coerce')
    # opp_features_to_merge['game_pk'] = pd.to_numeric(opp_features_to_merge['game_pk'], errors='coerce')
    # result_df.dropna(subset=['game_pk'], inplace=True) # Drop if game_pk is missing
    # opp_features_to_merge.dropna(subset=['game_pk'], inplace=True)

    logger.info(f"Merging opponent features using opponent_team and game_pk. Left shape: {result_df.shape}, Right shape: {opp_features_to_merge.shape}")

    # Merge opponent features based on the opponent team and game_pk
    result_df_merged = pd.merge(
        result_df,
        opp_features_to_merge,
        left_on=['opponent_team', 'game_pk'], # Use game_pk
        right_on=['team', 'game_pk'],       # Use game_pk
        how='left',
        suffixes=('', '_opponent_merge')
    )

    # Check shape after merge
    logger.info(f"Shape after opponent feature merge: {result_df_merged.shape}")
    if len(result_df_merged) != len(result_df):
         logger.warning(f"Row count changed during opponent merge ({len(result_df)} -> {len(result_df_merged)}). Check for duplicate game_pk in opponent stats.")
         # Optional: Investigate duplicates
         # dups = opp_features_to_merge[opp_features_to_merge.duplicated(subset=['team', 'game_pk'], keep=False)]
         # logger.warning(f"Duplicate keys found in opponent features:\n{dups}")


    # Drop redundant columns from merge
    result_df_merged.drop(columns=['team'], inplace=True, errors='ignore')

    # --- Handle Missing Opponent Features ---
    opp_feature_cols_final = [col for col in result_df_merged.columns if col.startswith('opp_roll_')]
    for col in opp_feature_cols_final:
        if result_df_merged[col].isnull().any():
            mean_val = result_df_merged[col].mean()
            fill_val = mean_val if pd.notna(mean_val) else 0
            logger.info(f"Filling {result_df_merged[col].isnull().sum()} missing values in {col} with mean ({fill_val:.4f})")
            result_df_merged[col].fillna(fill_val, inplace=True)

    logger.info("Completed opponent team feature creation.")
    return result_df_merged # Return the merged dataframe


def create_umpire_features(df, umpire_df=None):
    """
    Create features related to the umpire assigned to the game.

    Args:
        df: DataFrame with pitcher game data
        umpire_df: DataFrame with umpire assignments by game (must contain 'umpire' column)

    Returns:
        DataFrame with added umpire features
    """
    logger.info("Creating umpire features...")

    # Make a copy
    result_df = df.copy()

    # If no umpire data provided or empty, return original
    if umpire_df is None or umpire_df.empty:
        logger.warning("No umpire data provided or empty. Skipping umpire features.")
        return result_df

    # Ensure 'umpire' column exists in umpire_df
    if 'umpire' not in umpire_df.columns:
         logger.warning("Column 'umpire' not found in umpire_df. Skipping umpire features.")
         return result_df

    # Merge umpire assignments
    if 'game_date' in result_df.columns and 'home_team' in result_df.columns and 'away_team' in result_df.columns:
        # Ensure date formats match (use normalized date)
        result_df['game_date_norm'] = pd.to_datetime(result_df['game_date']).dt.normalize()
        umpire_df['game_date_norm'] = pd.to_datetime(umpire_df['game_date']).dt.normalize()

        # Select only relevant columns from umpire_df
        umpire_merge_cols = ['game_date_norm', 'home_team', 'away_team', 'umpire']
        if not all(col in umpire_df.columns for col in umpire_merge_cols):
             logger.warning(f"Umpire data missing one of required columns: {umpire_merge_cols}. Skipping umpire merge.")
             result_df.drop(columns=['game_date_norm'], inplace=True, errors='ignore')
             return result_df

        # Merge umpire data
        # Avoid duplicates in umpire data before merge if necessary
        umpire_data_to_merge = umpire_df[umpire_merge_cols].drop_duplicates(subset=['game_date_norm', 'home_team', 'away_team'])
        result_df = pd.merge(
            result_df,
            umpire_data_to_merge,
            on=['game_date_norm', 'home_team', 'away_team'],
            how='left'
        )

        # Drop temporary column
        result_df.drop(columns=['game_date_norm'], inplace=True, errors='ignore')
        logger.info(f"Merged umpire assignments. Found umpire for {result_df['umpire'].notnull().sum()} / {len(result_df)} records.")

    else:
        logger.warning("Missing required columns (game_date, home_team, away_team) for umpire merge. Skipping.")
        return result_df

    # Calculate historical umpire tendencies based on game-level data
    # Use K/9 as the example tendency metric
    ump_tendency_metric = 'k_per_9'
    if 'umpire' in result_df.columns and ump_tendency_metric in result_df.columns:
        logger.info(f"Calculating historical umpire tendencies based on '{ump_tendency_metric}'...")

        # Sort chronologically by umpire and date for expanding mean
        result_df = result_df.sort_values(['umpire', 'game_date'])

        # Calculate historical average K/9 for each umpire
        # CRITICAL: Exclude current game (use shift) before calculating expanding mean
        grouped_ump = result_df.groupby('umpire')
        shifted_metric = grouped_ump[ump_tendency_metric].shift(1)

        # Calculate expanding mean on shifted data
        try:
            result_df['umpire_historical_k_per_9'] = shifted_metric.transform(
                lambda x: x.expanding(min_periods=5).mean() # Require min 5 games for stable avg
            )
        except Exception as e:
             logger.warning(f"Could not calculate expanding mean for umpire tendency: {e}")
             result_df['umpire_historical_k_per_9'] = np.nan


        # Fill missing values (e.g., umpire's first few games) with global average K/9
        # Calculate global average excluding NaNs
        global_k_per_9_mean = result_df[ump_tendency_metric].mean()
        if pd.isna(global_k_per_9_mean): global_k_per_9_mean = 7.5 # Estimate if mean is NaN
        logger.info(f"Filling missing umpire historical K/9 with global average: {global_k_per_9_mean:.4f}")
        result_df['umpire_historical_k_per_9'].fillna(global_k_per_9_mean, inplace=True)

        # Create pitcher-umpire interaction feature (Optional)
        # Example: How does this umpire's tendency compare to league average, scaled by pitcher's recent form?
        pitcher_recent_form_metric = 'ewma_5g_k_per_9' # Use a recent EWMA K/9
        if pitcher_recent_form_metric in result_df.columns:
            result_df['pitcher_umpire_k_boost'] = (
                (result_df['umpire_historical_k_per_9'] - global_k_per_9_mean) *
                 result_df[pitcher_recent_form_metric].fillna(global_k_per_9_mean) # Fill pitcher NaNs too
            )
        else:
             logger.warning(f"Pitcher recent form metric '{pitcher_recent_form_metric}' not found for interaction.")

    else:
         logger.warning(f"Cannot calculate umpire tendencies. Missing 'umpire' or '{ump_tendency_metric}' column.")


    logger.info("Completed umpire feature creation.")
    return result_df


def create_platoon_features(df, pitch_data=None):
    """
    Create features related to pitcher performance against different handed batters.
    Placeholder: Requires pitch-level data which is not passed in SQL workflow.

    Args:
        df: DataFrame with pitcher game data
        pitch_data: DataFrame with pitch-level data including batter handedness (Currently None)

    Returns:
        DataFrame (potentially unmodified if pitch_data is None)
    """
    logger.info("Attempting to create platoon split features...")

    # Make a copy
    result_df = df.copy()

    # If no pitch data provided (as expected in SQL workflow), skip this
    if pitch_data is None or pitch_data.empty:
        logger.warning("No pitch-level data provided (pitch_data=None). Skipping platoon features.")
        # Add placeholder columns if needed downstream, otherwise just return
        result_df['k_percent_vs_rhb'] = np.nan
        result_df['k_percent_vs_lhb'] = np.nan
        result_df['platoon_split_k_pct'] = np.nan
        return result_df

    # --- The following code would run if pitch_data WAS provided ---
    logger.info("Processing pitch-level data for platoon features...")

    # Check if needed columns exist
    required_cols = ['pitcher_id', 'game_pk', 'game_date', 'at_bat_number', 'stand', 'events']
    if not all(col in pitch_data.columns for col in required_cols):
        logger.warning(f"Pitch data missing required columns ({required_cols}) for platoon features. Skipping.")
        return result_df

    # Add handedness indicators
    pitch_data['batter_stands_right'] = (pitch_data['stand'] == 'R').astype(int)

    # Get at-bat outcomes with handedness info (last pitch of each at-bat)
    # Ensure correct sorting
    pitch_data = pitch_data.sort_values(['pitcher_id', 'game_pk', 'game_date', 'at_bat_number', 'pitch_number'])
    ab_last_pitch = pitch_data.groupby(['pitcher_id', 'game_pk', 'game_date', 'at_bat_number']).last().reset_index()

    # Mark strikeouts
    ab_last_pitch['is_strikeout'] = ab_last_pitch['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)

    # Calculate vs RHB
    vs_rhb = ab_last_pitch[ab_last_pitch['batter_stands_right'] == 1].groupby(
        ['pitcher_id', 'game_pk', 'game_date']).agg(
            k_vs_rhb=('is_strikeout', 'sum'),
            pa_vs_rhb=('at_bat_number', 'count') # Count ABs vs RHB
        ).reset_index()
    vs_rhb['k_percent_vs_rhb'] = (vs_rhb['k_vs_rhb'] / vs_rhb['pa_vs_rhb'].replace(0, np.nan)).fillna(0)

    # Calculate vs LHB
    vs_lhb = ab_last_pitch[ab_last_pitch['batter_stands_right'] == 0].groupby(
        ['pitcher_id', 'game_pk', 'game_date']).agg(
            k_vs_lhb=('is_strikeout', 'sum'),
            pa_vs_lhb=('at_bat_number', 'count') # Count ABs vs LHB
        ).reset_index()
    vs_lhb['k_percent_vs_lhb'] = (vs_lhb['k_vs_lhb'] / vs_lhb['pa_vs_lhb'].replace(0, np.nan)).fillna(0)

    # Merge with game_level
    result_df = pd.merge(
        result_df,
        vs_rhb[['pitcher_id', 'game_pk', 'game_date', 'pa_vs_rhb', 'k_vs_rhb', 'k_percent_vs_rhb']],
        on=['pitcher_id', 'game_pk', 'game_date'],
        how='left'
    )

    result_df = pd.merge(
        result_df,
        vs_lhb[['pitcher_id', 'game_pk', 'game_date', 'pa_vs_lhb', 'k_vs_lhb', 'k_percent_vs_lhb']],
        on=['pitcher_id', 'game_pk', 'game_date'],
        how='left'
    )

    # Fill missing values (games with 0 PA vs one handedness)
    for col in ['pa_vs_rhb', 'k_vs_rhb', 'k_percent_vs_rhb', 'pa_vs_lhb', 'k_vs_lhb', 'k_percent_vs_lhb']:
        if col in result_df.columns:
            result_df[col].fillna(0, inplace=True)

    # Calculate platoon split strength
    if 'k_percent_vs_rhb' in result_df.columns and 'k_percent_vs_lhb' in result_df.columns:
        result_df['platoon_split_k_pct'] = result_df['k_percent_vs_lhb'] - result_df['k_percent_vs_rhb']

    # Create recent platoon performance metrics (EWMA on game-level platoon splits)
    split_cols = ['k_percent_vs_rhb', 'k_percent_vs_lhb']
    result_df = result_df.sort_values(['pitcher_id', 'game_date']) # Ensure sort order

    for col in split_cols:
        if col in result_df.columns:
            grouped = result_df.groupby('pitcher_id')
            shifted_metric = grouped[col].shift(1)
            try:
                result_df[f'ewma_5g_{col}'] = shifted_metric.transform(
                    lambda x: x.ewm(span=5, adjust=True, min_periods=1).mean()
                )
            except Exception as e:
                 logger.warning(f"Could not calculate EWMA for platoon split {col}: {e}")
                 result_df[f'ewma_5g_{col}'] = np.nan


    logger.info("Completed platoon features creation.")
    return result_df


def create_pitcher_features(game_level_df, pitch_level_data=None, team_batting_data=None, umpire_data=None):
    """
    Main function to create comprehensive pitcher features.
    Accepts game-level data as primary input. Pitch-level data is optional.

    Args:
        game_level_df: DataFrame with aggregated game-level pitcher data (Primary Input)
        pitch_level_data: DataFrame with raw pitch-level data (Optional, maybe None)
        team_batting_data: DataFrame with team batting statistics (game-level or seasonal)
        umpire_data: DataFrame with umpire assignments

    Returns:
        DataFrame with complete set of engineered features
    """
    logger.info("Starting pitcher feature engineering pipeline...")

    if game_level_df is None or game_level_df.empty:
        logger.error("Game level data (game_level_df) is missing or empty. Cannot create features.")
        return pd.DataFrame()

    # Ensure data is sorted chronologically
    if 'game_date' not in game_level_df.columns:
        logger.error("Missing 'game_date' column in game level data.")
        return pd.DataFrame()

    # Make sure game_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(game_level_df['game_date']):
        try:
            game_level_df['game_date'] = pd.to_datetime(game_level_df['game_date'])
        except Exception as e:
             logger.error(f"Could not convert game_date to datetime: {e}")
             return pd.DataFrame()

    # Sort data is crucial for rolling/lagging features
    # Reset index here BEFORE passing to sub-functions to ensure clean state
    game_level_df = game_level_df.sort_values(['pitcher_id', 'game_date']).reset_index(drop=True)
    logger.info(f"Input data shape to feature engineering: {game_level_df.shape}")


    # Apply feature creation functions in sequence
    # Pass the main dataframe (game_level_df) to each function
    df = game_level_df.copy()
    intermediate_dfs = [] # To help with memory management

    # 1. Create recency-weighted features
    df = create_recency_weighted_features(df)
    logger.info(f"Shape after recency features: {df.shape}")
    intermediate_dfs.append(df) # Store for potential later use if needed, or just track shape

    # 2. Create trend and form change features
    df = create_trend_features(df)
    logger.info(f"Shape after trend features: {df.shape}")
    intermediate_dfs.append(df)

    # 3. Create rest-related features
    df = create_rest_features(df)
    logger.info(f"Shape after rest features: {df.shape}")
    intermediate_dfs.append(df)


    # 4. Create pitch arsenal features (uses game-level %s)
    df = create_pitch_arsenal_features(df)
    logger.info(f"Shape after arsenal features: {df.shape}")
    intermediate_dfs.append(df)

    # 5. Create opponent team features (uses team_batting_data)
    df = create_opponent_team_features(df, team_batting_data)
    logger.info(f"Shape after opponent features: {df.shape}")
    intermediate_dfs.append(df)

    # 6. Create umpire features (uses umpire_data)
    df = create_umpire_features(df, umpire_data)
    logger.info(f"Shape after umpire features: {df.shape}")
    intermediate_dfs.append(df)

    # 7. Create platoon features (uses optional pitch_level_data)
    # This will likely just add NaN columns if pitch_level_data is None
    df = create_platoon_features(df, pitch_level_data)
    logger.info(f"Shape after platoon features: {df.shape}")
    intermediate_dfs.append(df)

    # Clean up intermediate dataframes if needed (optional)
    del intermediate_dfs
    gc.collect()


    # 8. Handle any remaining missing values introduced by feature engineering
    # Use a more robust filling strategy - fill with 0 for counts/sums, median for ratios/averages
    logger.info("Performing final NaN cleanup...")
    for col in df.columns:
         if df[col].isnull().any():
              if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                   fill_value = 0 # Default fill
                   # Use median for ratios/percentages/averages if possible
                   if 'percent' in col or 'rate' in col or 'avg' in col or 'per_9' in col or 'split' in col or 'volatility' in col or 'boost' in col or 'vs_baseline' in col:
                        median_val = df[col].median()
                        # Ensure median is not NaN before using it
                        if pd.notna(median_val):
                             fill_value = median_val
                   logger.info(f"Filling {df[col].isnull().sum()} NaNs in numeric column '{col}' with {fill_value:.4f}")
                   df[col].fillna(fill_value, inplace=True)
              # else: # Handle non-numeric if necessary
              #      logger.info(f"Filling {df[col].isnull().sum()} NaNs in non-numeric column '{col}' with 'Unknown'")
              #      df[col].fillna('Unknown', inplace=True)


    logger.info(f"Completed pitcher feature engineering. Final shape: {df.shape}")
    # Log final columns for verification
    logger.debug(f"Final columns: {df.columns.tolist()}")
    return df
