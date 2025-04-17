# src/features/pitcher_features.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging
from datetime import datetime, timedelta

# Use standard logging setup assumed to be configured elsewhere
logger = logging.getLogger(__name__)

# --- Helper Functions ---
# (Assuming these exist or are defined elsewhere if needed)
# def calculate_ewma(series, span):
#     return series.ewm(span=span, adjust=False).mean()
#
# def calculate_volatility(series, window):
#     return series.rolling(window=window, min_periods=max(1, window // 2)).std()

# --- Feature Creation Functions ---

def create_recency_weighted_features(df, metrics, spans=[3, 5, 10], group_col='pitcher_id', date_col='game_date'):
    """Creates Exponentially Weighted Moving Average features."""
    logger.info("Creating recency-weighted pitcher features...")
    df_copy = df.copy() # Work on a copy
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    # Use assignment for sort_values
    df_copy = df_copy.sort_values(by=[group_col, date_col])

    logger.debug(f"Available metrics for weighting: {metrics}")
    present_metrics = [m for m in metrics if m in df_copy.columns]
    missing_metrics = [m for m in metrics if m not in df_copy.columns]
    if missing_metrics:
        logger.warning(f"Metrics not found for EWMA calculation: {missing_metrics}")

    for metric in tqdm(present_metrics, desc="Recency Features", leave=False):
        # Calculate EWMA only if the metric exists
        grouped = df_copy.groupby(group_col)[metric]
        for span in spans:
            # Shift(1) to prevent using current game's data
            # Use assignment to add the new column
            df_copy[f'ewma_{span}g_{metric}'] = grouped.shift(1).ewm(span=span, adjust=False).mean()

    logger.info("Completed recency-weighted feature creation.")
    return df_copy # Return the modified copy

def create_trend_features(df, metrics, windows=[3, 5], group_col='pitcher_id', date_col='game_date'):
    """Creates features based on recent trends and volatility."""
    logger.info("Creating trend and form change features...")
    df_copy = df.copy() # Work on a copy
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    # Use assignment for sort_values
    df_copy = df_copy.sort_values(by=[group_col, date_col])

    logger.debug(f"Available metrics for trend features: {metrics}")
    present_metrics = [m for m in metrics if m in df_copy.columns]
    missing_metrics = [m for m in metrics if m not in df_copy.columns]
    if missing_metrics:
        logger.warning(f"Metrics not found for trend calculation: {missing_metrics}")

    for metric in tqdm(present_metrics, desc="Trend Features", leave=False):
        grouped = df_copy.groupby(group_col)[metric]
        # Lagged features (ensure shift(1) to use previous game)
        # Use assignment to add new columns
        lag1_col = f'{metric}_lag1'
        lag2_col = f'{metric}_lag2'
        df_copy[lag1_col] = grouped.shift(1)
        df_copy[lag2_col] = grouped.shift(2)

        # Change vs previous game
        # Ensure lag1 column exists before calculating change
        if lag1_col in df_copy.columns:
             df_copy[f'{metric}_change'] = df_copy[metric].sub(df_copy[lag1_col])
        else:
             logger.warning(f"Lag1 column '{lag1_col}' not found, cannot calculate '{metric}_change'.")
             df_copy[f'{metric}_change'] = np.nan


        # Volatility (standard deviation over windows) - use lagged data
        for window in windows:
            # Calculate volatility on the shifted (lagged) series
            # Use assignment to add new columns
            df_copy[f'{metric}_volatility_{window}g'] = grouped.shift(1).rolling(window=window, min_periods=max(1,window//2)).std()

        # Performance vs Baseline (e.g., last 2 games vs EWMA_10)
        ewma_col = f'ewma_10g_{metric}'
        if ewma_col in df_copy.columns and lag1_col in df_copy.columns:
             # Calculate rolling mean of last 2 lagged values
             roll_2g = grouped.shift(1).rolling(window=2, min_periods=1).mean()
             # Use assignment to add new columns
             df_copy[f'{metric}_last2g_vs_baseline'] = roll_2g - df_copy[ewma_col]
        elif ewma_col not in df_copy.columns:
             logger.warning(f"Baseline column '{ewma_col}' not found for vs_baseline calc for {metric}.")
             df_copy[f'{metric}_last2g_vs_baseline'] = np.nan


    # Example specific trend: K% trend based on lags
    k_lag1 = 'k_percent_lag1'
    k_lag2 = 'k_percent_lag2'
    if k_lag1 in df_copy.columns and k_lag2 in df_copy.columns:
        # Use assignment to add new columns
        df_copy['k_trend_up_lagged'] = (df_copy[k_lag1] > df_copy[k_lag2]).astype(int)
        df_copy['k_trend_down_lagged'] = (df_copy[k_lag1] < df_copy[k_lag2]).astype(int)
    else:
        logger.warning(f"Lag columns ('{k_lag1}', '{k_lag2}') not found, skipping k_trend calculation.")
        df_copy['k_trend_up_lagged'] = 0
        df_copy['k_trend_down_lagged'] = 0


    logger.info("Completed trend and form change feature creation.")
    return df_copy # Return modified copy

def create_rest_features(df, group_col='pitcher_id', date_col='game_date'):
    """Creates features related to pitcher rest and recent workload."""
    logger.info("Creating pitcher rest features...")
    df_copy = df.copy() # Work on a copy

    # --- START: Pre-processing before rolling ---
    logger.debug("Preprocessing data before rolling calculations...")
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        # 1. Ensure uniqueness based on pitcher and date FIRST
        duplicates_mask = df_copy.duplicated(subset=[group_col, date_col], keep=False)
        if duplicates_mask.any():
            num_duplicates = duplicates_mask.sum()
            logger.warning(f"Found {num_duplicates} rows involved in duplicate {group_col}/{date_col} entries. Keeping the first entry.")
            original_row_count = len(df_copy)
            df_copy = df_copy.drop_duplicates(subset=[group_col, date_col], keep='first')
            logger.info(f"Resolved duplicates. Row count changed from {original_row_count} to {len(df_copy)}.")

        # 2. Explicitly sort *again* right before setting index
        df_copy = df_copy.sort_values(by=[group_col, date_col])
        logger.debug("Ensured data is sorted by group and date before setting index.")

        # 3. Set index and check overall monotonicity (for debugging)
        df_indexed = df_copy.set_index(date_col)
        if not df_indexed.index.is_monotonic_increasing:
            logger.warning("Index is not strictly monotonic overall after setting. Group-level check will occur during rolling.")

        # 4. Group by pitcher ID
        grouped = df_indexed.groupby(group_col)
        logger.debug("Data preprocessing complete.")

    except Exception as preproc_e:
        logger.error(f"Error during preprocessing in create_rest_features: {preproc_e}", exc_info=True)
        # Add default columns and return if preprocessing fails
        df_copy['days_since_last_game'] = np.nan
        df_copy['rest_days_4_less'] = 0
        df_copy['rest_days_5'] = 0
        df_copy['rest_days_6_more'] = 0
        df_copy['extended_rest'] = 0
        df_copy['ip_last_15d'] = np.nan
        df_copy['pitches_last_15d'] = np.nan
        return df_copy
    # --- END: Pre-processing before rolling ---


    # Days since last game (calculated on the potentially modified df_copy)
    # Need to calculate diff on the original structure before setting index
    df_copy['days_since_last_game'] = df_copy.groupby(group_col)[date_col].diff().dt.days

    # Rest day categories (example)
    df_copy['rest_days_4_less'] = (df_copy['days_since_last_game'] <= 4).astype(int)
    df_copy['rest_days_5'] = (df_copy['days_since_last_game'] == 5).astype(int)
    df_copy['rest_days_6_more'] = (df_copy['days_since_last_game'] >= 6).astype(int)
    df_copy['extended_rest'] = (df_copy['days_since_last_game'] > 7).astype(int)

    # Recent workload calculations using the pre-processed df_indexed and grouped
    logger.info("Calculating rolling workload (IP)...")
    ip_col = 'innings_pitched'
    if ip_col in df_copy.columns: # Check df_copy for original column presence
        try:
            rolling_ip = grouped[ip_col].shift(1).rolling('15D', closed='left').sum()
            rolling_ip.name = 'ip_last_15d'
            df_copy = pd.merge(df_copy, rolling_ip.reset_index(), on=[group_col, date_col], how='left')
        except ValueError as ve:
            if "monotonic" in str(ve).lower():
                logger.error(f"ValueError during rolling IP calculation (non-monotonic index within a group): {ve}")
                # (Keep enhanced debugging from previous step here if needed)
                df_copy['ip_last_15d'] = np.nan
            else:
                logger.error(f"Unexpected ValueError during rolling IP calculation: {ve}")
                df_copy['ip_last_15d'] = np.nan
        except Exception as e:
             logger.error(f"Error calculating rolling IP: {e}", exc_info=True)
             df_copy['ip_last_15d'] = np.nan
    else:
        logger.warning(f"Column '{ip_col}' not found for workload calculation.")
        df_copy['ip_last_15d'] = np.nan

    logger.info("Calculating rolling workload (Pitches)...")
    pitches_col = 'total_pitches'
    if pitches_col in df_copy.columns:
        try:
            rolling_pitches = grouped[pitches_col].shift(1).rolling('15D', closed='left').sum()
            rolling_pitches.name = 'pitches_last_15d'
            df_copy = pd.merge(df_copy, rolling_pitches.reset_index(), on=[group_col, date_col], how='left')
        except ValueError as ve:
             if "monotonic" in str(ve).lower():
                  logger.error(f"ValueError during rolling Pitches calculation (non-monotonic index within a group): {ve}")
                  # (Keep enhanced debugging from previous step here if needed)
                  df_copy['pitches_last_15d'] = np.nan
             else:
                  logger.error(f"Unexpected ValueError during rolling Pitches calculation: {ve}")
                  df_copy['pitches_last_15d'] = np.nan
        except Exception as e:
             logger.error(f"Error calculating rolling Pitches: {e}", exc_info=True)
             df_copy['pitches_last_15d'] = np.nan
    else:
        logger.warning(f"Column '{pitches_col}' not found for workload calculation.")
        df_copy['pitches_last_15d'] = np.nan

    logger.info("Completed rest feature creation.")
    return df_copy

def create_arsenal_features(df, arsenal_metrics, effectiveness_metric='swinging_strike_percent', group_col='pitcher_id', date_col='game_date'):
    """Creates features related to pitch arsenal usage and effectiveness trends."""
    logger.info("Creating pitch arsenal features (from game-level data)...")
    df_copy = df.copy() # Work on copy
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    # Use assignment for sort_values
    df_copy = df_copy.sort_values(by=[group_col, date_col])

    logger.debug(f"Available arsenal metrics for trends: {arsenal_metrics}")
    logger.debug(f"Using '{effectiveness_metric}' as effectiveness proxy.")

    present_metrics = [m for m in arsenal_metrics if m in df_copy.columns]
    missing_metrics = [m for m in arsenal_metrics if m not in df_copy.columns]
    if missing_metrics:
        logger.warning(f"Arsenal metrics not found for calculation: {missing_metrics}")


    for metric in tqdm(present_metrics, desc="Arsenal Features", leave=False):
        lag1_col = f'{metric}_lag1'
        ewma_col = f'ewma_10g_{metric}'
        # Check if BOTH lag and ewma columns exist before calculating difference
        if lag1_col in df_copy.columns and ewma_col in df_copy.columns:
            # Compare recent usage (lag1) to baseline (ewma10)
            # Use assignment to add new column
            df_copy[f'{metric}_vs_baseline'] = df_copy[lag1_col] - df_copy[ewma_col]
        else:
             # Log which specific column was missing
             if lag1_col not in df_copy.columns:
                 logger.warning(f"Lag column '{lag1_col}' not found for arsenal vs baseline calc.")
             if ewma_col not in df_copy.columns:
                 logger.warning(f"EWMA column '{ewma_col}' not found for arsenal vs baseline calc.")
             # Assign NaN if components are missing
             df_copy[f'{metric}_vs_baseline'] = np.nan


    # Example: Velocity trend vs baseline
    velo_lag1 = 'avg_velocity_lag1'
    velo_ewma = 'ewma_10g_avg_velocity'
    if velo_lag1 in df_copy.columns and velo_ewma in df_copy.columns:
        # Use assignment to add new columns
        df_copy['velocity_vs_baseline'] = df_copy[velo_lag1] - df_copy[velo_ewma]
        df_copy['significant_velo_drop'] = (df_copy['velocity_vs_baseline'] < -1.0).astype(int) # Example threshold: 1 mph drop
    else:
        logger.warning(f"Velocity columns ('{velo_lag1}', '{velo_ewma}') not found, skipping velocity trend.")
        df_copy['velocity_vs_baseline'] = np.nan
        df_copy['significant_velo_drop'] = 0


    logger.info("Completed pitch arsenal feature creation.")
    return df_copy # Return modified copy

def create_opponent_features(pitcher_df, team_stats_df, opponent_metrics, group_col='pitcher_id', date_col='game_date'):
    """Merges pre-calculated rolling opponent team stats onto pitcher data."""
    logger.info("Creating opponent team features using game-level stats (merging)...")
    pitcher_df_copy = pitcher_df.copy() # Work on copy
    team_stats_copy = team_stats_df.copy() # Work on copy

    # Prepare team stats: select relevant metrics, rename team -> opponent_team
    required_team_cols = ['game_pk', 'team'] + opponent_metrics
    missing_team_cols = [col for col in required_team_cols if col not in team_stats_copy.columns]
    if missing_team_cols:
         logger.error(f"Missing required columns in team_stats_df: {missing_team_cols}. Cannot create opponent features.")
         # Add empty columns to pitcher_df to avoid errors later
         for metric in opponent_metrics:
              pitcher_df_copy[f'opp_{metric}'] = np.nan
         return pitcher_df_copy # Return pitcher_df_copy

    opp_stats_to_merge = team_stats_copy[required_team_cols].copy()
    # Rename 'team' to match the opponent column in pitcher_df (assume 'opponent_team')
    opponent_col_name = 'opponent_team'
    if opponent_col_name not in pitcher_df_copy.columns:
        logger.error(f"Pitcher data is missing the '{opponent_col_name}' column needed for merging opponent stats.")
        # Add empty columns to pitcher_df to avoid errors later
        for metric in opponent_metrics:
              pitcher_df_copy[f'opp_{metric}'] = np.nan
        return pitcher_df_copy # Return pitcher_df_copy

    # Use assignment for rename
    opp_stats_to_merge = opp_stats_to_merge.rename(columns={'team': opponent_col_name})

    # Rename metric columns to indicate they are opponent stats
    opp_rename_dict = {metric: f'opp_{metric}' for metric in opponent_metrics}
    # Use assignment for rename
    opp_stats_to_merge = opp_stats_to_merge.rename(columns=opp_rename_dict)

    # Merge opponent stats onto pitcher data using game_pk
    logger.info(f"Merging opponent features using {opponent_col_name} and game_pk. Left shape: {pitcher_df_copy.shape}, Right shape: {opp_stats_to_merge.shape}")
    original_len = len(pitcher_df_copy)
    # Need to handle potential duplicate game_pk if team_stats has both teams per game_pk
    # Ensure we only merge the stats for the correct opponent
    # Use assignment for merge
    result_df_merged = pd.merge(
        pitcher_df_copy,
        opp_stats_to_merge,
        on=['game_pk', opponent_col_name], # Merge on game and the specific opponent
        how='left'
    )
    if len(result_df_merged) != original_len:
         logger.warning(f"Merge changed row count from {original_len} to {len(result_df_merged)}. Check for duplicates in team_stats_df or merge keys.")
         # Fallback or error handling might be needed here

    # Impute missing opponent stats (e.g., if opponent data was missing for that game_pk)
    for col in opp_rename_dict.values():
        if col in result_df_merged.columns and result_df_merged[col].isnull().any():
            fill_val = result_df_merged[col].mean() # Impute with mean of available opponent stats
            logger.info(f"Filling {result_df_merged[col].isnull().sum()} missing values in {col} with mean ({fill_val:.4f})")
            # Use assignment instead of inplace
            result_df_merged[col] = result_df_merged[col].fillna(fill_val)

    logger.info("Completed opponent team feature creation.")
    return result_df_merged # Return merged df


def create_umpire_features(pitcher_df, umpire_df, historical_metric='k_per_9'):
    """Merges umpire assignments and calculates historical umpire tendencies."""
    logger.info("Creating umpire features...")
    result_df = pitcher_df.copy() # Work on copy
    umpire_data_copy = umpire_df.copy() # Work on copy

    # Ensure types are correct for merging
    try:
        result_df['game_date'] = pd.to_datetime(result_df['game_date']).dt.strftime('%Y-%m-%d')
        umpire_data_copy['game_date'] = pd.to_datetime(umpire_data_copy['game_date']).dt.strftime('%Y-%m-%d')
        # Strip whitespace from team names in both dataframes before merge
        for col in ['home_team', 'away_team']:
             if col in result_df.columns and result_df[col].dtype == 'object':
                  result_df[col] = result_df[col].str.strip()
             if col in umpire_data_copy.columns and umpire_data_copy[col].dtype == 'object':
                  umpire_data_copy[col] = umpire_data_copy[col].str.strip()
        ump_col = 'umpire' # Use the correct column name from umpire_data
        if ump_col in umpire_data_copy.columns and umpire_data_copy[ump_col].dtype == 'object':
             umpire_data_copy[ump_col] = umpire_data_copy[ump_col].str.strip()

    except Exception as e:
         logger.error(f"Error preparing columns for umpire merge: {e}")
         # Add default columns and return
         result_df['umpire'] = 'Unknown'
         result_df['umpire_historical_k_per_9'] = np.nan
         result_df['pitcher_umpire_k_boost'] = np.nan
         return result_df # Return result_df

    # Merge umpire name based on date, home_team, away_team
    merge_cols = ['game_date', 'home_team', 'away_team']
    ump_col = 'umpire' # Define umpire column name
    cols_to_merge_from_ump = merge_cols + [ump_col]

    if not all(col in result_df.columns for col in merge_cols):
         logger.error(f"Missing columns required for umpire merge in main df: {[c for c in merge_cols if c not in result_df.columns]}")
         result_df[ump_col] = 'Unknown'
    elif not all(col in umpire_data_copy.columns for col in cols_to_merge_from_ump):
         logger.error(f"Missing columns required for umpire merge in umpire_df: {[c for c in cols_to_merge_from_ump if c not in umpire_data_copy.columns]}")
         result_df[ump_col] = 'Unknown'
    else:
        logger.info(f"Merging umpire assignments using {merge_cols}...")
        original_len = len(result_df)
        # Select only needed columns from umpire data before merge
        ump_subset_to_merge = umpire_data_copy[cols_to_merge_from_ump].drop_duplicates(subset=merge_cols, keep='first')
        # Use assignment for merge
        result_df = pd.merge(
            result_df,
            ump_subset_to_merge,
            on=merge_cols,
            how='left'
        )
        if len(result_df) != original_len:
             # This can happen if pitcher_df had duplicate game_date/home/away combos
             logger.warning(f"Umpire merge changed row count from {original_len} to {len(result_df)}. Check for duplicates in input pitcher_df.")
        # Fill missing umpires if any
        missing_umps = result_df[ump_col].isnull().sum()
        if missing_umps > 0:
             logger.warning(f"Could not find umpire match for {missing_umps} records. Filling with 'Unknown'.")
             # Use assignment for fillna
             result_df[ump_col] = result_df[ump_col].fillna('Unknown')
        logger.info(f"Merged umpire assignments. Found umpire for {original_len - missing_umps} / {original_len} records.")

    # Calculate historical umpire tendency (e.g., average K/9 in games they HP'd)
    logger.info(f"Calculating historical umpire tendencies based on '{historical_metric}'...")
    hist_ump_col = f'umpire_historical_{historical_metric}'
    boost_col = f'pitcher_umpire_k_boost' # Use consistent naming

    if historical_metric in result_df.columns and ump_col in result_df.columns:
        # Calculate average of the metric per umpire historically (use data BEFORE current game)
        result_df['game_date_dt'] = pd.to_datetime(result_df['game_date'])
        # Use assignment for sort_values
        result_df = result_df.sort_values(by=[ump_col, 'game_date_dt'])
        # Group by umpire, shift the metric, calculate expanding mean
        # Use assignment to add new column
        result_df[hist_ump_col] = result_df.groupby(ump_col)[historical_metric].shift(1).expanding().mean()
        # Use assignment for drop
        result_df = result_df.drop(columns=['game_date_dt'])

        # Impute missing historical values (e.g., first game for umpire) with global mean
        global_metric_mean = result_df[historical_metric].mean()
        missing_hist = result_df[hist_ump_col].isnull().sum()
        if missing_hist > 0:
             logger.info(f"Filling {missing_hist} missing umpire historical {historical_metric} with global average: {global_metric_mean:.4f}")
             # Use assignment instead of inplace
             result_df[hist_ump_col] = result_df[hist_ump_col].fillna(global_metric_mean)

        # Create interaction term (e.g., pitcher's recent K/9 vs umpire's historical K/9)
        ewma_k_metric = f'ewma_10g_{historical_metric}' # Example EWMA column
        if ewma_k_metric in result_df.columns:
             # Use assignment to add new column
             result_df[boost_col] = result_df[ewma_k_metric] - result_df[hist_ump_col]
        else:
             logger.warning(f"Could not create pitcher-umpire interaction: '{ewma_k_metric}' missing.")
             result_df[boost_col] = np.nan

    else:
        logger.warning(f"Required columns ('{historical_metric}', '{ump_col}') not found for historical umpire calculation.")
        result_df[hist_ump_col] = np.nan
        result_df[boost_col] = np.nan


    logger.info("Completed umpire feature creation.")
    return result_df # Return modified df

def create_platoon_features(df, pitch_data, group_col='pitcher_id', date_col='game_date'):
    """Creates features based on pitcher performance vs LHB/RHB."""
    logger.info("Attempting to create platoon split features...")
    df_copy = df.copy() # Work on copy
    # Add default columns first, will be overwritten if successful
    lhb_col = 'ewma_5g_k_percent_vs_lhb'
    rhb_col = 'ewma_5g_k_percent_vs_rhb'
    df_copy[lhb_col] = np.nan
    df_copy[rhb_col] = np.nan

    if pitch_data is None or pitch_data.empty:
        logger.warning("Pitch-level data is None or empty, cannot create platoon features.")
        return df_copy # Return with NaN columns

    required_cols = ['pitcher', 'game_pk', 'game_date', 'stand', 'events']
    if not all(col in pitch_data.columns for col in required_cols):
         missing = [col for col in required_cols if col not in pitch_data.columns]
         logger.warning(f"Pitch data missing required columns for platoon splits: {missing}. Skipping.")
         return df_copy # Return with NaN columns

    # --- Calculations (same as before) ---
    logger.info("Processing pitch-level data for platoon features...")
    pitch_data_copy = pitch_data.copy() # Work on copy of pitch_data
    pitch_data_copy['game_date'] = pd.to_datetime(pitch_data_copy['game_date'])
    pitch_data_copy['is_k'] = pitch_data_copy['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)
    pa_end_events = ['strikeout', 'walk', 'hit_by_pitch', 'field_out', 'force_out', 'sac_fly', 'sac_bunt',
                     'double_play', 'triple_play', 'fielders_choice', 'field_error', 'grounded_into_double_play',
                     'single', 'double', 'triple', 'home_run']
    pitch_data_copy['is_pa_end'] = pitch_data_copy['events'].isin(pa_end_events).astype(int)
    platoon_game = pitch_data_copy.groupby(['pitcher', 'game_pk', 'game_date', 'stand']).agg(
        k_vs_stand=('is_k', 'sum'),
        pa_vs_stand=('is_pa_end', 'sum')
    ).reset_index()
    platoon_game['k_percent_vs_stand'] = np.where(
        platoon_game['pa_vs_stand'] > 0,
        platoon_game['k_vs_stand'] / platoon_game['pa_vs_stand'],
        np.nan
    )
    platoon_pivot = platoon_game.pivot_table(
        index=['pitcher', 'game_date'], columns='stand', values='k_percent_vs_stand'
    ).reset_index()
    platoon_pivot = platoon_pivot.rename(columns={'L': 'k_percent_vs_lhb', 'R': 'k_percent_vs_rhb'})
    platoon_pivot = platoon_pivot[['pitcher', 'game_date', 'k_percent_vs_lhb', 'k_percent_vs_rhb']]
    platoon_pivot = platoon_pivot.sort_values(by=['pitcher', 'game_date'])
    grouped = platoon_pivot.groupby('pitcher')
    platoon_pivot[lhb_col] = grouped['k_percent_vs_lhb'].shift(1).ewm(span=5, adjust=False).mean()
    platoon_pivot[rhb_col] = grouped['k_percent_vs_rhb'].shift(1).ewm(span=5, adjust=False).mean()
    # --- End Calculations ---

    # Merge onto main dataframe
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    # Ensure pitcher_id type matches pitcher type
    if group_col in df_copy.columns and 'pitcher' in platoon_pivot.columns:
         try:
            df_copy[group_col] = df_copy[group_col].astype(platoon_pivot['pitcher'].dtype)
         except Exception as e:
            logger.warning(f"Could not align pitcher types for platoon merge: {e}")
            try: # Fallback
                df_copy[group_col] = pd.to_numeric(df_copy[group_col], errors='coerce').astype('Int64')
                platoon_pivot['pitcher'] = pd.to_numeric(platoon_pivot['pitcher'], errors='coerce').astype('Int64')
            except Exception as e2:
                logger.error(f"Failed fallback type alignment for platoon merge: {e2}")
                return df_copy # Return df without platoon features if merge fails

    original_len = len(df_copy)
    cols_to_merge = ['pitcher', date_col, lhb_col, rhb_col]
    platoon_pivot_subset = platoon_pivot[cols_to_merge].copy() # Work with a copy
    # Ensure date column type matches for merge
    platoon_pivot_subset[date_col] = pd.to_datetime(platoon_pivot_subset[date_col])

    logger.debug(f"Merging platoon features. Left shape: {df_copy.shape}, Right shape: {platoon_pivot_subset.shape}")
    logger.debug(f"Left merge keys ({group_col}, {date_col}): Types {df_copy[group_col].dtype}, {df_copy[date_col].dtype}")
    logger.debug(f"Right merge keys ('pitcher', {date_col}): Types {platoon_pivot_subset['pitcher'].dtype}, {platoon_pivot_subset[date_col].dtype}")

    # Use assignment for merge
    df_merged = pd.merge(df_copy, platoon_pivot_subset,
                       left_on=[group_col, date_col], right_on=['pitcher', date_col], how='left')

    if len(df_merged) != original_len:
        logger.warning(f"Merge for platoon features changed row count from {original_len} to {len(df_merged)}!")

    # --- START: Added Check and Robust FillNA ---
    # Check if the columns were actually added by the merge
    if lhb_col not in df_merged.columns:
        logger.error(f"Merge failed to add '{lhb_col}' column. Filling with NaN.")
        df_merged[lhb_col] = np.nan # Ensure column exists even if merge failed
    if rhb_col not in df_merged.columns:
        logger.error(f"Merge failed to add '{rhb_col}' column. Filling with NaN.")
        df_merged[rhb_col] = np.nan # Ensure column exists

    # Fill NaNs (e.g., first game, pitchers who only faced one type of batter, or merge failures)
    # Calculate medians on the potentially updated df_merged
    median_lhb = df_merged[lhb_col].median()
    median_rhb = df_merged[rhb_col].median()
    fallback_lhb = 0.20 # Define fallback values
    fallback_rhb = 0.22
    logger.info(f"Calculated medians for platoon fill: LHB={median_lhb:.4f}, RHB={median_rhb:.4f}")

    # Use assignment for fillna, applying fallbacks if median itself is NaN
    df_merged[lhb_col] = df_merged[lhb_col].fillna(median_lhb if pd.notna(median_lhb) else fallback_lhb)
    df_merged[rhb_col] = df_merged[rhb_col].fillna(median_rhb if pd.notna(median_rhb) else fallback_rhb)
    # --- END: Added Check and Robust FillNA ---

    # Drop intermediate columns if needed
    # Use assignment for drop
    df_merged = df_merged.drop(columns=['pitcher'], errors='ignore')

    logger.info("Completed platoon features creation.")
    return df_merged # Return the merged and filled df

def final_cleanup_and_imputation(df):
    """Performs final checks, NaN filling, and type conversions."""
    logger.info("Performing final NaN cleanup...")
    df_copy = df.copy() # Work on copy
    numeric_cols = df_copy.select_dtypes(include=np.number).columns

    for col in tqdm(numeric_cols, desc="Final Imputation", leave=False):
        if df_copy[col].isnull().any():
            # Impute with median
            median_val = df_copy[col].median()
            fill_value = median_val if pd.notna(median_val) else 0 # Fallback to 0 if median is NaN
            nan_count = df_copy[col].isnull().sum()
            if nan_count > 0:
                 logger.info(f"Filling {nan_count} NaNs in numeric column '{col}' with median ({fill_value:.4f})")
                 # Use assignment instead of inplace
                 df_copy[col] = df_copy[col].fillna(fill_value)

    # Final check for infinities
    inf_mask = np.isinf(df_copy.select_dtypes(include=np.number))
    if inf_mask.any().any():
        inf_cols = df_copy.columns[inf_mask.any()].tolist()
        logger.warning(f"Infinite values found after imputation in columns: {inf_cols}. Replacing with 0.")
        # Use assignment for replace
        numeric_subset = df_copy.select_dtypes(include=np.number)
        numeric_subset = numeric_subset.replace([np.inf, -np.inf], 0)
        # Update the original dataframe
        df_copy.update(numeric_subset)


    return df_copy # Return modified df

# --- Main Feature Pipeline Function ---
def create_pitcher_features(pitcher_data, team_stats_data, umpire_data, pitch_data):
    """
    Orchestrates the creation of all pitcher-related features.

    Args:
        pitcher_data (pd.DataFrame): Game-level pitcher stats (e.g., from game_level_pitchers).
                                      MUST contain 'pitcher_id', 'game_pk', 'game_date',
                                      'home_team', 'away_team', 'opponent_team'.
        team_stats_data (pd.DataFrame): Game-level team stats (e.g., from game_level_team_stats).
        umpire_data (pd.DataFrame): Umpire assignments (game_date, home_team, away_team, umpire).
        pitch_data (pd.DataFrame): Minimal pitch-level data (for platoon splits).

    Returns:
        pd.DataFrame: DataFrame with all features included. Returns empty DataFrame on critical error.
    """
    logger.info("Starting pitcher feature engineering pipeline...")
    if pitcher_data is None or pitcher_data.empty:
        logger.error("Input pitcher_data is None or empty. Cannot create features.")
        return pd.DataFrame()

    # Ensure pitcher_data is a copy to avoid modifying the original DataFrame outside this function
    features_df = pitcher_data.copy()

    logger.info(f"Input data shape to feature engineering: {features_df.shape}")

    # Check for essential columns early
    required_cols = ['pitcher_id', 'game_pk', 'game_date', 'home_team', 'away_team', 'opponent_team']
    missing = [c for c in required_cols if c not in features_df.columns]
    if missing:
         logger.error(f"Input pitcher_data missing required columns: {missing}. Cannot proceed.")
         return pd.DataFrame()


    # Define metric groups - check if columns exist in input df
    recency_metrics_all = ['strikeouts', 'batters_faced', 'innings_pitched', 'total_pitches',
                           'avg_velocity', 'max_velocity', 'zone_percent', 'swinging_strike_percent',
                           'fastball_percent', 'breaking_percent', 'offspeed_percent',
                           'k_percent', 'k_per_9']
    trend_metrics_all = ['strikeouts', 'innings_pitched', 'batters_faced', 'swinging_strike_percent',
                         'avg_velocity', 'k_percent', 'k_per_9']
    arsenal_metrics_all = ['fastball_percent', 'breaking_percent', 'offspeed_percent']
    opponent_metrics_all = ['k_percent', 'bb_percent', 'swing_percent', 'contact_percent',
                            'swinging_strike_percent', 'chase_percent', 'zone_contact_percent'] # From game_level_team_stats

    # Filter metrics based on columns actually present in the input dataframes
    recency_metrics = [m for m in recency_metrics_all if m in features_df.columns]
    trend_metrics = [m for m in trend_metrics_all if m in features_df.columns]
    arsenal_metrics = [m for m in arsenal_metrics_all if m in features_df.columns]
    opponent_metrics = [m for m in opponent_metrics_all if team_stats_data is not None and m in team_stats_data.columns]

    # Pipeline Steps (wrap each major step in try-except)
    try:
        features_df = create_recency_weighted_features(features_df, recency_metrics)
        logger.info(f"Shape after recency features: {features_df.shape}")

        features_df = create_trend_features(features_df, trend_metrics)
        logger.info(f"Shape after trend features: {features_df.shape}")
        # DEBUG LOGGING ADDED HERE
        logger.debug(f"Columns BEFORE create_rest_features: {features_df.columns.tolist()}") # <-- ADDED

        features_df = create_rest_features(features_df)
        logger.info(f"Shape after rest features: {features_df.shape}")
        # DEBUG LOGGING ADDED HERE
        logger.debug(f"Columns AFTER create_rest_features: {features_df.columns.tolist()}") # <-- ADDED
        if 'fastball_percent_lag1' not in features_df.columns and 'fastball_percent' in trend_metrics : # Check if it should exist
            logger.error("CRITICAL DEBUG: 'fastball_percent_lag1' is MISSING after create_rest_features!")
            # Optionally: raise Exception("Lag column lost after rest features") to stop execution

        features_df = create_arsenal_features(features_df, arsenal_metrics)
        logger.info(f"Shape after arsenal features: {features_df.shape}")

        features_df = create_opponent_features(features_df, team_stats_data, opponent_metrics)
        logger.info(f"Shape after opponent features: {features_df.shape}")

        features_df = create_umpire_features(features_df, umpire_data)
        logger.info(f"Shape after umpire features: {features_df.shape}")

        features_df = create_platoon_features(features_df, pitch_data)
        logger.info(f"Shape after platoon features: {features_df.shape}")

        # Final Cleanup
        features_df = final_cleanup_and_imputation(features_df)

    except Exception as e:
        logger.error(f"Error occurred during feature creation steps: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

    logger.info(f"Completed pitcher feature engineering. Final shape: {features_df.shape}")
    return features_df