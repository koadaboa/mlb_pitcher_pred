# src/features/pitcher_features.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging
from datetime import datetime, timedelta

# Use standard logging setup assumed to be configured elsewhere
logger = logging.getLogger(__name__)

# --- Helper Functions ---
# (Keep helper functions like calculate_ewma, calculate_volatility if defined)
# Example:
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

def create_trend_features(df, metrics, windows=[3, 5, 10], group_col='pitcher_id', date_col='game_date'):
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

        # Change vs previous game (lag1 - lag2 to avoid using current data)
        if lag1_col in df_copy.columns and lag2_col in df_copy.columns:
            # Use assignment to add new columns
            df_copy[f'{metric}_change'] = df_copy[lag1_col] - df_copy[lag2_col]
        else:
            logger.warning(f"Cannot calculate '{metric}_change' (requires '{lag1_col}' and '{lag2_col}').")
            # Use assignment to add new columns
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
             # Use assignment to add new columns
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
        # Use assignment to add new columns
        df_copy['k_trend_up_lagged'] = 0
        df_copy['k_trend_down_lagged'] = 0


    logger.info("Completed trend and form change feature creation.")
    return df_copy # Return modified copy

def create_rest_features(df, group_col='pitcher_id', date_col='game_date'):
    """Creates features related to pitcher rest and recent workload."""
    logger.info("Creating pitcher rest features...")
    # Start with a fresh index guaranteed to be unique
    df_copy = df.copy().reset_index(drop=True)

    # --- Pre-processing ---
    logger.debug("Preprocessing data before rolling calculations...")
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        # Ensure uniqueness based on pitcher and date FIRST
        duplicates_mask = df_copy.duplicated(subset=[group_col, date_col], keep=False)
        if duplicates_mask.any():
            num_duplicates = duplicates_mask.sum()
            logger.warning(f"Found {num_duplicates} duplicate {group_col}/{date_col} entries. Keeping first.")
            df_copy = df_copy.drop_duplicates(subset=[group_col, date_col], keep='first').reset_index(drop=True) # Reset index after drop

        # Explicitly sort *again* right before setting index
        df_copy = df_copy.sort_values(by=[group_col, date_col]).reset_index(drop=True) # Assign sorted df and reset index
        logger.debug("Ensured data is sorted by group and date.")

    except Exception as preproc_e:
        logger.error(f"Error during preprocessing in create_rest_features: {preproc_e}", exc_info=True)
        # Add default columns and return if preprocessing fails
        # Use assignment to add new columns
        df_copy['days_since_last_game'] = np.nan; df_copy['rest_days_4_less'] = 0
        df_copy['rest_days_5'] = 0; df_copy['rest_days_6_more'] = 0
        df_copy['extended_rest'] = 0; df_copy['ip_last_15d'] = np.nan
        df_copy['pitches_last_15d'] = np.nan
        return df_copy
    # --- END Pre-processing ---

    # Days since last game (calculated on the sorted df_copy)
    # Ensure group_col exists
    if group_col not in df_copy.columns:
         logger.error(f"Grouping column '{group_col}' not found for rest days calculation.")
         # Use assignment to add new columns
         df_copy['days_since_last_game'] = np.nan
    else:
         # Use assignment to add new columns
         df_copy['days_since_last_game'] = df_copy.groupby(group_col)[date_col].diff().dt.days

    # Rest day categories
    # Use assignment to add new columns
    df_copy['rest_days_4_less'] = (df_copy['days_since_last_game'] <= 4).astype(int)
    df_copy['rest_days_5'] = (df_copy['days_since_last_game'] == 5).astype(int)
    df_copy['rest_days_6_more'] = (df_copy['days_since_last_game'] >= 6).astype(int)
    df_copy['extended_rest'] = (df_copy['days_since_last_game'] > 7).astype(int)

    # --- Rolling Workload Calculation ---
    # Ensure datetime index and sorted order for rolling calculations
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(by=[group_col, date_col]).reset_index(drop=True)
    # Set datetime index for rolling
    df_indexed = df_copy.set_index(date_col)

    ip_col = 'innings_pitched'; pitches_col = 'total_pitches'
    ip_workload_col = 'ip_last_15d'; pitches_workload_col = 'pitches_last_15d'
    grouped_pitcher_time = df_indexed.groupby(group_col)

    try:
        # Calculate IP workload
        if ip_col in df_indexed.columns:
            # Use assignment to add new columns
            df_indexed[ip_workload_col] = grouped_pitcher_time[ip_col].shift(1).rolling('15D', closed='left').sum()
        else:
            logger.warning(f"Column '{ip_col}' not found for IP workload calculation.")
            # Use assignment to add new columns
            df_indexed[ip_workload_col] = np.nan # Ensure column exists

        # Calculate Pitches workload
        if pitches_col in df_indexed.columns:
            # Use assignment to add new columns
            df_indexed[pitches_workload_col] = grouped_pitcher_time[pitches_col].shift(1).rolling('15D', closed='left').sum()
        else:
            logger.warning(f"Column '{pitches_col}' not found for Pitches workload calculation.")
            # Use assignment to add new columns
            df_indexed[pitches_workload_col] = np.nan # Ensure column exists

    except ValueError as ve:
        logger.error(f"ValueError during rolling workload calculation: {ve}. Assigning NaN.")
        # Use assignment to add new columns
        df_indexed[ip_workload_col] = np.nan
        df_indexed[pitches_workload_col] = np.nan
    except Exception as e:
        logger.error(f"Error calculating rolling workloads: {e}", exc_info=True)
        # Use assignment to add new columns
        df_indexed[ip_workload_col] = np.nan
        df_indexed[pitches_workload_col] = np.nan


    # Reset index to bring date column back
    df_copy = df_indexed.reset_index()

    logger.info("Completed rest feature creation.")
    # Return the dataframe with the original index structure (potentially modified by drop_duplicates)
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
        lag1_col = f'{metric}_lag1' # Assumes trend_features was run first
        ewma_col = f'ewma_10g_{metric}' # Assumes recency_features was run first
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
             # Use assignment to add new column
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
        # Use assignment to add new columns
        df_copy['velocity_vs_baseline'] = np.nan
        df_copy['significant_velo_drop'] = 0


    logger.info("Completed pitch arsenal feature creation.")
    return df_copy # Return modified copy


# --- *** MODIFIED create_opponent_features Function *** ---
def create_opponent_features(pitcher_df, opponent_stats_input, prediction_mode, rolling_window=30):
    """
    Handles opponent team features.
    - Prediction Mode: Merges pre-calculated rolling stats based on opponent_team.
    - Training Mode: Merges game-level opponent stats based on game_pk and opponent_team.
    """
    pitcher_df_copy = pitcher_df.copy() # Work on copy
    opponent_col_name = 'opponent_team'

    if opponent_col_name not in pitcher_df_copy.columns:
        logger.error(f"Pitcher data missing '{opponent_col_name}' column needed for opponent features.")
        return pitcher_df_copy # Return unchanged df

    if opponent_stats_input is None or opponent_stats_input.empty:
        logger.warning("Opponent stats input is empty. Opponent features will be NaN.")
        # Define expected columns based on mode, add NaN columns if needed
        if prediction_mode:
            expected_opp_metrics = ['k_percent', 'bb_percent', 'swing_percent', 'contact_percent', 'swinging_strike_percent', 'chase_percent', 'zone_contact_percent']
            for metric in expected_opp_metrics:
                 # Use assignment to add new columns
                 pitcher_df_copy[f'opp_{metric}_roll{rolling_window}'] = np.nan
        else:
             expected_opp_metrics = ['k_percent', 'bb_percent', 'swing_percent', 'contact_percent', 'swinging_strike_percent', 'chase_percent', 'zone_contact_percent']
             for metric in expected_opp_metrics:
                 # Use assignment to add new columns
                 pitcher_df_copy[f'opp_{metric}'] = np.nan
        return pitcher_df_copy


    if prediction_mode:
        # --- PREDICTION MODE ---
        logger.info("Creating opponent features [Prediction Mode]: Merging pre-calculated rolling stats...")
        latest_rolling_stats_df = opponent_stats_input.copy() # Input is pre-calculated rolling stats

        # Rename columns in latest_rolling_stats_df to have 'opp_' prefix
        # Assumes columns are named like 'metric_rollWINDOW'
        opp_rename_dict = {col: f'opp_{col}' for col in latest_rolling_stats_df.columns if col != 'team'} # Exclude 'team' if it's the index name
        # Use assignment for rename
        rolling_stats_to_merge = latest_rolling_stats_df.rename(columns=opp_rename_dict)

        # Merge using opponent_team and the index ('team') of rolling_stats_to_merge
        logger.info(f"Merging opponent features. Left shape: {pitcher_df_copy.shape}, Right (rolling stats) shape: {rolling_stats_to_merge.shape}")
        original_len = len(pitcher_df_copy)

        # Use assignment for merge
        result_df_merged = pd.merge(
            pitcher_df_copy,
            rolling_stats_to_merge,
            left_on=opponent_col_name, # Join based on the opponent team in pitcher data
            right_index=True,          # Join based on the 'team' index of the rolling stats data
            how='left'
        )

        if len(result_df_merged) != original_len:
            logger.warning(f"Opponent merge changed row count from {original_len} to {len(result_df_merged)}. Review merge keys.")

        # Impute missing values (e.g., opponent team not found in rolling stats -> maybe new team or insufficient history)
        for col in opp_rename_dict.values(): # Iterate through the expected opponent columns
            if col not in result_df_merged.columns: # Check if column was added
                 logger.warning(f"Opponent column '{col}' was not added by merge. Filling with 0.")
                 # Use assignment to add new columns
                 result_df_merged[col] = 0 # Add column with default value
            elif result_df_merged[col].isnull().any():
                # Impute with 0 or a global average if preferred/available
                fill_val = 0
                nan_count = result_df_merged[col].isnull().sum()
                logger.info(f"Filling {nan_count} missing values in predicted opponent feature '{col}' with {fill_val}")
                # Use assignment for fillna
                result_df_merged[col] = result_df_merged[col].fillna(fill_val)

    else:
        # --- TRAINING MODE ---
        logger.info("Creating opponent features [Training Mode]: Merging game-level stats...")
        team_stats_df = opponent_stats_input.copy() # Input is raw historical game-level stats

        # Define opponent metrics available in the raw data
        opponent_metrics = [
            'k_percent', 'bb_percent', 'swing_percent', 'contact_percent',
            'swinging_strike_percent', 'chase_percent', 'zone_contact_percent'
            # Add other metrics if needed
        ]
        required_team_cols = ['game_pk', 'team'] + [m for m in opponent_metrics if m in team_stats_df.columns]
        present_opponent_metrics = [m for m in opponent_metrics if m in required_team_cols]

        missing_team_cols = [col for col in ['game_pk', 'team'] if col not in team_stats_df.columns]
        if missing_team_cols:
             logger.error(f"Missing required key columns in team_stats_df for training merge: {missing_team_cols}.")
             # Add empty columns as fallback
             for metric in present_opponent_metrics:
                 # Use assignment to add new columns
                 pitcher_df_copy[f'opp_{metric}'] = np.nan
             return pitcher_df_copy

        # Prepare team stats for merging (game_pk, team, metrics)
        opp_stats_to_merge = team_stats_df[required_team_cols].copy()
        # Use assignment for rename
        opp_stats_to_merge = opp_stats_to_merge.rename(columns={'team': opponent_col_name}) # Rename team -> opponent_team for merge

        # Rename metric columns to indicate opponent stats
        opp_rename_dict = {metric: f'opp_{metric}' for metric in present_opponent_metrics}
        # Use assignment for rename
        opp_stats_to_merge = opp_stats_to_merge.rename(columns=opp_rename_dict)

        # Merge opponent stats onto pitcher data using game_pk and opponent_team
        logger.info(f"Merging opponent features using game_pk and {opponent_col_name}. Left shape: {pitcher_df_copy.shape}, Right shape: {opp_stats_to_merge.shape}")
        original_len = len(pitcher_df_copy)

        # Use assignment for merge
        result_df_merged = pd.merge(
            pitcher_df_copy,
            opp_stats_to_merge,
            on=['game_pk', opponent_col_name], # Merge on game and the specific opponent
            how='left'
        )
        if len(result_df_merged) != original_len:
             logger.warning(f"Merge changed row count from {original_len} to {len(result_df_merged)}. Check for duplicates in team_stats_df or merge keys.")

        # Impute missing opponent stats (e.g., if opponent data was missing for that game_pk)
        for col in opp_rename_dict.values():
            if col not in result_df_merged.columns:
                 logger.warning(f"Opponent column '{col}' was not added by training merge. Filling with 0.")
                 # Use assignment to add new columns
                 result_df_merged[col] = 0 # Add column with default value
            elif result_df_merged[col].isnull().any():
                fill_val = result_df_merged[col].mean() # Impute with mean of available opponent stats for that column
                fill_val = fill_val if pd.notna(fill_val) else 0 # Fallback to 0 if mean is NaN
                nan_count = result_df_merged[col].isnull().sum()
                logger.info(f"Filling {nan_count} missing values in training opponent feature '{col}' with mean ({fill_val:.4f}) or 0 fallback.")
                # Use assignment for fillna
                result_df_merged[col] = result_df_merged[col].fillna(fill_val)

    logger.info("Completed opponent team feature creation.")
    return result_df_merged # Return merged df
# --- *** END MODIFIED create_opponent_features Function *** ---


def create_umpire_features(pitcher_df, umpire_df, historical_metric='k_per_9'):
    """
    Merges umpire assignments and calculates historical umpire tendencies.
    Uses 'home_plate_umpire' as the expected column name in umpire_df.
    """
    logger.info("Creating umpire features using 'home_plate_umpire'...")
    result_df = pitcher_df.copy() # Work on copy
    umpire_data_copy = umpire_df.copy() if umpire_df is not None else pd.DataFrame() # Work on copy, handle None

    ump_col = 'home_plate_umpire' # Define the standard umpire column name
    hist_ump_col = f'umpire_historical_{historical_metric}'
    boost_col = 'pitcher_umpire_k_boost'

    # --- Prepare merge keys ---
    try:
        # Create string version of date for reliable merging
        result_df['game_date_str'] = pd.to_datetime(result_df['game_date']).dt.strftime('%Y-%m-%d')
        if not umpire_data_copy.empty:
             umpire_data_copy['game_date_str'] = pd.to_datetime(umpire_data_copy['game_date']).dt.strftime('%Y-%m-%d')
        # Strip whitespace from team names
        for col in ['home_team', 'away_team']:
             if col in result_df.columns and result_df[col].dtype == 'object':
                  # Use assignment for astype/str.strip
                  result_df[col] = result_df[col].astype(str).str.strip()
             if col in umpire_data_copy.columns and umpire_data_copy[col].dtype == 'object':
                  # Use assignment for astype/str.strip
                  umpire_data_copy[col] = umpire_data_copy[col].astype(str).str.strip()

        # Check if umpire_df is valid for merging
        merge_cols = ['game_date_str', 'home_team', 'away_team']
        if umpire_data_copy is None or umpire_data_copy.empty:
             logger.warning("Umpire data (umpire_df) is empty. Cannot merge.")
             # Use assignment to add new columns
             result_df[ump_col] = 'Unknown' # Add placeholder column
        elif not all(col in umpire_data_copy.columns for col in merge_cols + [ump_col]):
            logger.error(f"Umpire data (umpire_df) is missing required columns for merge: {merge_cols + [ump_col]}. Cannot merge.")
            # Use assignment to add new columns
            result_df[ump_col] = 'Unknown' # Add placeholder column
        else:
            # Ensure umpire name is string before merge
            # Use assignment for astype/str.strip
            umpire_data_copy[ump_col] = umpire_data_copy[ump_col].astype(str).str.strip()
            # --- Merge umpire name ---
            logger.info(f"Merging umpire assignments using {merge_cols}...")
            original_len = len(result_df)
            # Select only needed columns and drop duplicates from umpire data before merge
            ump_subset_to_merge = umpire_data_copy[merge_cols + [ump_col]].drop_duplicates(subset=merge_cols, keep='first')
            # Use assignment for merge
            result_df = pd.merge(
                result_df,
                ump_subset_to_merge,
                on=merge_cols,
                how='left'
            )
            if len(result_df) != original_len:
                 logger.warning(f"Umpire merge changed row count from {original_len} to {len(result_df)}. Check input data.")

        # --- Ensure umpire column exists and fill NaNs ---
        if ump_col not in result_df.columns:
             logger.warning(f"Umpire column '{ump_col}' was not added by the merge. Filling with 'Unknown'.")
             # Use assignment to add new columns
             result_df[ump_col] = 'Unknown'
        else:
             missing_umps = result_df[ump_col].isnull().sum()
             if missing_umps > 0:
                  logger.info(f"Filling {missing_umps} missing umpire matches with 'Unknown'.")
                  # Use .loc to avoid SettingWithCopyWarning if result_df is a slice
                  result_df.loc[result_df[ump_col].isnull(), ump_col] = 'Unknown'
             # Convert to string just in case
             result_df.loc[:, ump_col] = result_df[ump_col].astype(str)

    except Exception as e:
         logger.error(f"Error preparing columns or merging umpire data: {e}", exc_info=True)
         # Use assignment to add new columns
         result_df[ump_col] = 'Unknown' # Ensure column exists even on error

    # --- Calculate historical umpire tendency ---
    logger.info(f"Calculating historical umpire tendencies based on '{historical_metric}'...")
    # Initialize columns even if calculation fails
    # Use assignment to add new columns
    result_df[hist_ump_col] = np.nan
    result_df[boost_col] = np.nan

    if historical_metric not in result_df.columns:
        logger.error(f"Required metric '{historical_metric}' not found for historical umpire calc.")
    elif ump_col not in result_df.columns:
         logger.error(f"Umpire column '{ump_col}' missing unexpectedly before tendency calc.")
    else:
        try:
            # Use assignment to add new columns
            result_df['game_date_dt'] = pd.to_datetime(result_df['game_date_str']) # Use dt for sorting
            result_df_sorted = result_df.sort_values(by=[ump_col, 'game_date_dt']) # Work on sorted copy

            # Calculate expanding mean *only* on historical, known umpires
            mask_known_ump = result_df_sorted[ump_col] != 'Unknown'
            known_ump_metric_series = result_df_sorted.loc[mask_known_ump, historical_metric]

            if not known_ump_metric_series.empty:
                 # Calculate shifted expanding mean within each known umpire group
                 hist_tendency = result_df_sorted.loc[mask_known_ump].groupby(ump_col)[historical_metric].shift(1).expanding(min_periods=5).mean() # Add min_periods
                 # Assign calculated values back using the index from hist_tendency
                 result_df_sorted.loc[hist_tendency.index, hist_ump_col] = hist_tendency
            else:
                 logger.warning("No data with known umpires to calculate historical tendency.")

            # Impute missing historical values after calculation
            global_metric_mean = result_df_sorted.loc[mask_known_ump, historical_metric].mean() # Recalculate mean on valid data
            fill_value_hist = global_metric_mean if pd.notna(global_metric_mean) else 0
            missing_hist_mask = result_df_sorted[hist_ump_col].isnull()
            missing_hist_count = missing_hist_mask.sum()
            if missing_hist_count > 0:
                 logger.info(f"Filling {missing_hist_count} missing umpire historical {historical_metric} with average ({fill_value_hist:.4f}) or 0 fallback.")
                 result_df_sorted.loc[missing_hist_mask, hist_ump_col] = fill_value_hist

            # Create interaction term
            ewma_k_metric = f'ewma_10g_{historical_metric}' # Assumes this exists from pitcher features
            if ewma_k_metric in result_df_sorted.columns:
                 # Use assignment to add new columns
                 result_df_sorted[boost_col] = result_df_sorted[ewma_k_metric] - result_df_sorted[hist_ump_col]
                 boost_nan_mask = result_df_sorted[boost_col].isnull()
                 boost_nan_count = boost_nan_mask.sum()
                 if boost_nan_count > 0:
                      logger.info(f"Filling {boost_nan_count} NaNs in '{boost_col}' with 0.")
                      result_df_sorted.loc[boost_nan_mask, boost_col] = 0
            else:
                 logger.warning(f"Could not create pitcher-umpire interaction: '{ewma_k_metric}' missing.")
                 # boost_col already initialized to NaN

            # Assign results back to the original DataFrame index structure
            # Use assignment to add new columns
            result_df[hist_ump_col] = result_df_sorted[hist_ump_col]
            result_df[boost_col] = result_df_sorted[boost_col]

            # Use assignment for drop
            result_df = result_df.drop(columns=['game_date_dt'], errors='ignore')

        except Exception as e:
            logger.error(f"Error during historical umpire tendency calculation: {e}", exc_info=True)
            # Ensure columns exist with NaN if calculation fails
            # Use assignment to add new columns
            result_df[hist_ump_col] = np.nan
            result_df[boost_col] = np.nan

    # --- Final Cleanup ---
    # Drop helper date string column
    # Use assignment for drop
    result_df = result_df.drop(columns=['game_date_str'], errors='ignore')

    logger.info("Completed umpire feature creation.")
    return result_df


def create_platoon_features(df, pitch_data, group_col='pitcher_id', date_col='game_date'):
    """Creates features based on pitcher performance vs LHB/RHB."""
    logger.info("Attempting to create platoon split features...")
    df_copy = df.copy() # Work on copy
    # Add default columns first, will be overwritten if successful
    lhb_col = 'ewma_5g_k_percent_vs_lhb'
    rhb_col = 'ewma_5g_k_percent_vs_rhb'
    # Use assignment to add new columns
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

    # --- Calculations ---
    logger.info("Processing pitch-level data for platoon features...")
    try:
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
            np.nan # Use NaN for division by zero
        )
        # Handle potential NaNs before pivot if necessary
        # platoon_game = platoon_game.dropna(subset=['k_percent_vs_stand']) # Optional: drop rows with NaN results if desired

        platoon_pivot = platoon_game.pivot_table(
            index=['pitcher', 'game_date'], columns='stand', values='k_percent_vs_stand'
        ) # Removed reset_index() initially

        # Check if 'L' and 'R' columns exist after pivot, add if missing
        if 'L' not in platoon_pivot.columns: platoon_pivot['L'] = np.nan
        if 'R' not in platoon_pivot.columns: platoon_pivot['R'] = np.nan

        platoon_pivot = platoon_pivot.reset_index() # Reset index after ensuring L/R exist
        # Use assignment for rename
        platoon_pivot = platoon_pivot.rename(columns={'L': 'k_percent_vs_lhb', 'R': 'k_percent_vs_rhb'})

        # Keep only necessary columns after pivot/rename
        platoon_pivot = platoon_pivot[['pitcher', 'game_date', 'k_percent_vs_lhb', 'k_percent_vs_rhb']]
        platoon_pivot = platoon_pivot.sort_values(by=['pitcher', 'game_date'])

        # Calculate EWMA on the K% splits
        grouped = platoon_pivot.groupby('pitcher')
        # Use assignment to add new columns
        platoon_pivot[lhb_col] = grouped['k_percent_vs_lhb'].shift(1).ewm(span=5, adjust=False).mean()
        platoon_pivot[rhb_col] = grouped['k_percent_vs_rhb'].shift(1).ewm(span=5, adjust=False).mean()

    except Exception as e:
        logger.error(f"Error processing pitch data for platoon features: {e}", exc_info=True)
        return df_copy # Return original df with NaN platoon columns

    # --- Merge onto main dataframe ---
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    # Ensure pitcher_id type matches pitcher type
    if group_col in df_copy.columns and 'pitcher' in platoon_pivot.columns:
         try:
             # Use assignment for astype
             df_copy[group_col] = df_copy[group_col].astype(platoon_pivot['pitcher'].dtype)
         except Exception as e:
            logger.warning(f"Could not align pitcher types for platoon merge: {e}")
            try: # Fallback
                 # Use assignment for astype
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
    # logger.debug(f"Left merge keys ({group_col}, {date_col}): Types {df_copy[group_col].dtype}, {df_copy[date_col].dtype}")
    # logger.debug(f"Right merge keys ('pitcher', {date_col}): Types {platoon_pivot_subset['pitcher'].dtype}, {platoon_pivot_subset[date_col].dtype}")

    # Use assignment for merge
    # Drop potentially duplicated platoon columns before merge if they exist from placeholder assignment
    df_copy = df_copy.drop(columns=[lhb_col, rhb_col], errors='ignore')
    df_merged = pd.merge(df_copy, platoon_pivot_subset,
                       left_on=[group_col, date_col], right_on=['pitcher', date_col], how='left')

    if len(df_merged) != original_len:
        logger.warning(f"Merge for platoon features changed row count from {original_len} to {len(df_merged)}!")

    # --- Robust FillNA ---
    # Check if the columns were actually added by the merge
    if lhb_col not in df_merged.columns:
        logger.error(f"Merge failed to add '{lhb_col}' column. Filling with NaN.")
        # Use assignment to add new columns
        df_merged[lhb_col] = np.nan # Ensure column exists even if merge failed
    if rhb_col not in df_merged.columns:
        logger.error(f"Merge failed to add '{rhb_col}' column. Filling with NaN.")
        # Use assignment to add new columns
        df_merged[rhb_col] = np.nan # Ensure column exists

    # Fill NaNs (e.g., first game, pitchers who only faced one type of batter, or merge failures)
    median_lhb = df_merged[lhb_col].median()
    median_rhb = df_merged[rhb_col].median()
    fallback_lhb = 0.20 # Define fallback values
    fallback_rhb = 0.22
    # logger.info(f"Calculated medians for platoon fill: LHB={median_lhb:.4f}, RHB={median_rhb:.4f}")

    fill_lhb_val = median_lhb if pd.notna(median_lhb) else fallback_lhb
    fill_rhb_val = median_rhb if pd.notna(median_rhb) else fallback_rhb

    nan_lhb_count = df_merged[lhb_col].isnull().sum()
    nan_rhb_count = df_merged[rhb_col].isnull().sum()

    if nan_lhb_count > 0:
        logger.info(f"Filling {nan_lhb_count} NaNs in {lhb_col} with {fill_lhb_val:.4f}")
        # Use assignment for fillna
        df_merged[lhb_col] = df_merged[lhb_col].fillna(fill_lhb_val)
    if nan_rhb_count > 0:
        logger.info(f"Filling {nan_rhb_count} NaNs in {rhb_col} with {fill_rhb_val:.4f}")
        # Use assignment for fillna
        df_merged[rhb_col] = df_merged[rhb_col].fillna(fill_rhb_val)

    # Drop intermediate columns if needed
    # Use assignment for drop
    df_merged = df_merged.drop(columns=['pitcher'], errors='ignore')

    logger.info("Completed platoon features creation.")
    return df_merged # Return the merged and filled df


def final_cleanup_and_imputation(df):
    """Performs final checks, NaN filling, and type conversions."""
    logger.info("Performing final NaN cleanup and imputation...")
    df_copy = df.copy() # Work on copy
    numeric_cols = df_copy.select_dtypes(include=np.number).columns

    for col in tqdm(numeric_cols, desc="Final Imputation", leave=False):
        if df_copy[col].isnull().any():
            # Impute with median
            median_val = df_copy[col].median()
            fill_value = median_val if pd.notna(median_val) else 0 # Fallback to 0 if median is NaN
            nan_count = df_copy[col].isnull().sum()
            if nan_count > 0:
                 # logger.debug(f"Filling {nan_count} NaNs in numeric column '{col}' with median ({fill_value:.4f})") # DEBUG level
                 # Use assignment for fillna
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

    logger.info("Completed final cleanup and imputation.")
    return df_copy # Return modified df


# --- *** MODIFIED Main Feature Pipeline Function *** ---
def create_pitcher_features(
    pitcher_data,                       # Baseline data (prediction rows in pred mode, all historical in train mode)
    historical_pitcher_stats=None,      # Pre-calculated PITCHER features (pred mode only)
    team_stats_data=None,               # Raw historical TEAM stats (train mode only)
    latest_rolling_opponent_stats_data=None, # Pre-calculated ROLLING OPPONENT stats (pred mode only)
    umpire_data=None,                   # Historical (train) or Predicted (pred) umpires
    pitch_data=None,                    # Historical minimal pitch data (for platoon)
    prediction_mode=False,              # Flag to indicate mode
    rolling_window=30                   # Pass rolling window for col naming consistency
    ):
    """
    Orchestrates the creation of all pitcher-related features.
    Handles prediction mode by merging pre-calculated historical context.
    Handles training mode by calculating features directly or merging game-level context.
    """
    logger.info(f"Starting pitcher feature engineering pipeline (Prediction Mode: {prediction_mode})...")

    # Input validation
    if pitcher_data is None or pitcher_data.empty:
        logger.error("Input pitcher_data (baseline features) is None or empty. Cannot create features.")
        return pd.DataFrame()
    if prediction_mode and (historical_pitcher_stats is None or historical_pitcher_stats.empty):
        # Allow running without historical pitcher stats if necessary, but log warning
        logger.warning("Prediction mode: historical_pitcher_stats is missing or empty. Pitcher historical features will be limited.")
    if prediction_mode and (latest_rolling_opponent_stats_data is None or latest_rolling_opponent_stats_data.empty):
        logger.warning("Prediction mode: latest_rolling_opponent_stats_data is missing or empty. Opponent features will be limited/NaN.")
    if not prediction_mode and (team_stats_data is None or team_stats_data.empty):
        logger.warning("Training mode: team_stats_data is missing or empty. Opponent features will be limited/NaN.")


    # Ensure pitcher_data is a copy
    features_df = pitcher_data.copy()
    logger.info(f"Input baseline data shape: {features_df.shape}")

    # --- Define Metric Groups ---
    recency_metrics_all = ['strikeouts', 'batters_faced', 'innings_pitched', 'total_pitches', 'avg_velocity', 'max_velocity', 'zone_percent', 'swinging_strike_percent', 'fastball_percent', 'breaking_percent', 'offspeed_percent', 'k_percent', 'k_per_9']
    trend_metrics_all = ['strikeouts', 'innings_pitched', 'batters_faced', 'swinging_strike_percent', 'avg_velocity', 'k_percent', 'k_per_9']
    arsenal_metrics_all = ['fastball_percent', 'breaking_percent', 'offspeed_percent']
    # Note: Opponent metrics are handled within create_opponent_features based on mode

    # --- Historical PITCHER Feature Calculation/Merging ---
    if prediction_mode:
        logger.info("Prediction Mode: Merging pre-calculated historical PITCHER features...")
        if historical_pitcher_stats is not None and not historical_pitcher_stats.empty:
            # historical_pitcher_stats should contain pre-calculated lags/EWMAs up to the day before prediction_date
            required_pitcher_ids = features_df['pitcher_id'].unique()

            # Filter historical stats to only include pitchers needed for prediction
            hist_filtered = historical_pitcher_stats[historical_pitcher_stats['pitcher_id'].isin(required_pitcher_ids)].copy()

            # Sort by pitcher and date, then keep the last (most recent) entry for each pitcher
            hist_latest = hist_filtered.sort_values(by=['pitcher_id', 'game_date']).drop_duplicates(subset=['pitcher_id'], keep='last')

            # Identify columns generated by historical calculations (lags, EWMAs, trends, rest, arsenal)
            # Make this list comprehensive based on functions above
            hist_feature_cols = [
                col for col in hist_latest.columns if (
                    col.startswith('ewma_') or col.startswith('k_trend_') or col.startswith('significant_velo')
                    or 'days_since' in col or col.startswith('rest_days_') or col == 'extended_rest'
                    or col.startswith('ip_last') or col.startswith('pitches_last')
                    or '_lag' in col or '_volatility' in col or '_last2g' in col
                    or '_vs_baseline' in col or '_change' in col
                    # Add specific metrics if they are directly used as features after calculation (like k_per_9)
                    # or col in ['k_per_9', 'other_calculated_metric']
                ) and col != 'pitcher_id' and col != 'game_date' # Exclude keys
            ]
            cols_to_merge = ['pitcher_id'] + list(set(hist_feature_cols)) # Ensure unique columns

            logger.info(f"Merging {len(cols_to_merge)-1} latest historical PITCHER features onto prediction baseline...")
            # Use assignment for merge
            features_df = pd.merge(features_df, hist_latest[cols_to_merge], on='pitcher_id', how='left')

            # Impute missing historical features (e.g., for pitchers with no history)
            for col in cols_to_merge:
                 if col == 'pitcher_id': continue # Skip merge key
                 if col in features_df.columns and features_df[col].isnull().any():
                      median_val = hist_latest[col].median() # Use median from latest historical
                      fill_val = median_val if pd.notna(median_val) else 0 # Fallback to 0
                      nan_count = features_df[col].isnull().sum()
                      # logger.debug(f"Imputing {nan_count} missing values in '{col}' (likely new pitcher) with median {fill_val:.4f} (or 0).")
                      # Use assignment for fillna
                      features_df[col] = features_df[col].fillna(fill_val)
        else:
            logger.warning("Prediction mode: No historical pitcher stats provided. Skipping merge.")

    else: # Training Mode: Calculate historical PITCHER features directly on the combined train/test baseline data
        logger.info("Training Mode: Calculating historical PITCHER features...")
        # Filter metrics based on columns actually present in the baseline data
        recency_metrics = [m for m in recency_metrics_all if m in features_df.columns]
        trend_metrics = [m for m in trend_metrics_all if m in features_df.columns]
        arsenal_metrics = [m for m in arsenal_metrics_all if m in features_df.columns]

        # Apply historical calculations directly to features_df
        if recency_metrics:
            features_df = create_recency_weighted_features(features_df, recency_metrics)
        if trend_metrics:
            features_df = create_trend_features(features_df, trend_metrics)

        # Rest features depend on date diffs, safe to run directly
        features_df = create_rest_features(features_df)

        if arsenal_metrics:
             # Arsenal relies on lag/ewma columns from above, ensure they were created
             features_df = create_arsenal_features(features_df, arsenal_metrics)
        logger.info(f"Shape after historical PITCHER feature calculation: {features_df.shape}")


    # --- Contextual Features (Applied to the result in both modes) ---
    try:
        # --- Opponent Features ---
        # Pass the correct opponent data based on mode
        opponent_input_data = latest_rolling_opponent_stats_data if prediction_mode else team_stats_data
        features_df = create_opponent_features(features_df, opponent_input_data, prediction_mode, rolling_window)
        logger.info(f"Shape after opponent features: {features_df.shape}")

        # --- Umpire Features ---
        # Use the umpire_data provided (predicted or actual)
        if umpire_data is not None:
            features_df = create_umpire_features(features_df, umpire_data, historical_metric='k_per_9') # Use k_per_9 as example metric
            logger.info(f"Shape after umpire features: {features_df.shape}")
        else:
            logger.warning("No umpire data provided, skipping umpire features.")

        # --- Platoon Features ---
        # Needs historical pitch_data, same for both modes if available
        if pitch_data is not None:
            features_df = create_platoon_features(features_df, pitch_data)
            logger.info(f"Shape after platoon features: {features_df.shape}")
        else:
            logger.warning("No pitch-level data provided, skipping platoon features.")

        # --- Final Cleanup ---
        features_df = final_cleanup_and_imputation(features_df)
        logger.info(f"Shape after final cleanup: {features_df.shape}")

    except Exception as e:
        logger.error(f"Error occurred during contextual feature creation steps: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

    logger.info(f"Completed pitcher feature engineering. Final shape: {features_df.shape}")

    # Final check for required columns before returning
    required_output_cols = ['pitcher_id', 'game_date', 'game_pk'] # Add more as needed
    missing_output_cols = [col for col in required_output_cols if col not in features_df.columns]
    if missing_output_cols:
        logger.error(f"Final feature dataframe missing required columns: {missing_output_cols}")
        logger.error(f"Available columns: {features_df.columns.tolist()}")
        # Decide whether to return empty df or df with missing columns
        # return pd.DataFrame()

    return features_df
# --- *** END MODIFIED Main Feature Pipeline Function *** ---