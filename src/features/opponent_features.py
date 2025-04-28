# src/features/opponent_features.py
import pandas as pd
import numpy as np
import logging
from typing import Callable, List, Dict, Tuple

logger = logging.getLogger(__name__) # Use standard logging practice

def calculate_opponent_rolling_features(
    team_hist_df: pd.DataFrame,
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Calculates rolling features specifically for opponents (teams).

    Args:
        team_hist_df: DataFrame containing team game-level data.
        group_col: The column to group by (e.g., 'team').
        date_col: The date column for sorting and rolling.
        metrics: List of metric columns to calculate rolling stats for.
        windows: List of window sizes.
        min_periods: Minimum number of observations in window required.
        calculate_multi_window_rolling: The function to perform the rolling calculation.

    Returns:
        A tuple containing:
            - DataFrame with opponent rolling features, including 'team' and 'game_date' keys.
              Columns are renamed with an 'opp_' prefix (e.g., 'opp_roll5g_k_percent').
            - Dictionary mapping original rolling column names to renamed opponent column names.
              Returns empty DataFrame and empty dict if input is invalid or empty.
    """
    if team_hist_df is None or team_hist_df.empty:
        logger.warning("Input DataFrame for opponent rolling features is empty.")
        return pd.DataFrame(), {}

    logger.info(f"Calculating opponent/team rolling features (Windows: {windows})...")
    available_metrics = [m for m in metrics if m in team_hist_df.columns]
    if not available_metrics:
        logger.warning("No specified opponent metrics found in the DataFrame.")
        return pd.DataFrame(index=team_hist_df.index), {}

    opponent_rolling_calc = calculate_multi_window_rolling(
        df=team_hist_df,
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods
    )

    # Define and apply rename map
    rename_map = {
        f"{m}_roll{w}g": f"opp_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in opponent_rolling_calc.columns
    }
    opponent_rolling_df = opponent_rolling_calc.rename(columns=rename_map)

    # Add keys back for merging - ensure keys exist in original df
    key_cols = [group_col, date_col]
    if all(col in team_hist_df.columns for col in key_cols):
        opponent_rolling_df[key_cols] = team_hist_df[key_cols]
    else:
        logger.error(f"Missing key columns {key_cols} in team_hist_df for opponent rolling features.")
        # Return empty df if keys are missing, as merge won't work
        return pd.DataFrame(), {}


    logger.info(f"Finished calculating opponent rolling features. Found {len(opponent_rolling_df.columns) - len(key_cols)} features.")
    return opponent_rolling_df, rename_map


def merge_opponent_features_historical(
    final_features_df: pd.DataFrame,
    opponent_rolling_df: pd.DataFrame,
    opp_rename_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Merges historical opponent rolling features using merge_asof.

    Args:
        final_features_df: The main features DataFrame being built.
        opponent_rolling_df: DataFrame containing opponent rolling features (output from calculate_opponent_rolling_features).
        opp_rename_map: The rename map used for opponent features.

    Returns:
        The final_features_df with opponent features merged.
    """
    if opponent_rolling_df is None or opponent_rolling_df.empty:
        logger.warning("Opponent rolling features DataFrame is empty, skipping merge.")
        return final_features_df

    logger.debug("Merging historical opponent rolling features...")

    # Use keys from opp_rename_map to get the actual columns to merge
    opp_roll_cols_to_merge = list(opp_rename_map.values())
    # Check if columns actually exist in the dataframe
    opp_roll_cols_to_merge = [col for col in opp_roll_cols_to_merge if col in opponent_rolling_df.columns]

    if not opp_roll_cols_to_merge:
        logger.warning("No opponent rolling columns identified to merge.")
        return final_features_df

    # Ensure required columns for merge exist in both dataframes
    if 'opponent_team' not in final_features_df.columns:
        logger.error("Missing 'opponent_team' column in final_features_df for opponent merge.")
        return final_features_df
    if 'team' not in opponent_rolling_df.columns or 'game_date' not in opponent_rolling_df.columns:
         logger.error("Missing 'team' or 'game_date' column in opponent_rolling_df for merge.")
         return final_features_df

    # Prepare for merge_asof
    final_features_df_sorted = final_features_df.sort_values('game_date')
    # Ensure correct keys are present in right df for merge_asof
    right_merge_cols = ['team', 'game_date'] + opp_roll_cols_to_merge
    opponent_rolling_df_sorted = opponent_rolling_df[right_merge_cols].sort_values('game_date')

    # Add merge keys safely
    final_features_df_sorted = final_features_df_sorted.copy()
    opponent_rolling_df_sorted = opponent_rolling_df_sorted.copy()
    final_features_df_sorted['merge_key_opponent'] = final_features_df_sorted['opponent_team'].astype(str)
    opponent_rolling_df_sorted['merge_key_team'] = opponent_rolling_df_sorted['team'].astype(str)

    try:
        merged_df = pd.merge_asof(
            final_features_df_sorted,
            opponent_rolling_df_sorted,
            on='game_date',
            left_by='merge_key_opponent',
            right_by='merge_key_team',
            direction='backward', # Use stats from before the game date
            allow_exact_matches=False # Do not use stats from the exact same day
        )
        # Drop merge keys and original 'team' column from right df if it got merged
        merged_df = merged_df.drop(columns=['merge_key_opponent', 'merge_key_team', 'team'], errors='ignore')
        logger.debug("Successfully merged historical opponent features.")
        return merged_df
    except Exception as e:
        logger.error(f"Error during historical opponent feature merge_asof: {e}", exc_info=True)
        # Return original df if merge fails
        return final_features_df


def merge_opponent_features_prediction(
    final_features_df: pd.DataFrame,
    latest_opponent_rolling: pd.DataFrame,
    opp_rename_map: Dict[str, str],
    rolling_windows: List[int]
) -> pd.DataFrame:
    """
    Merges the latest opponent rolling features onto the prediction baseline
    and selects the correct platoon feature based on pitcher handedness.

    Args:
        final_features_df: The prediction baseline DataFrame.
        latest_opponent_rolling: DataFrame with the latest rolling stats per team.
        opp_rename_map: The rename map used for opponent features.
        rolling_windows: List of window sizes used (e.g., [3, 5, 10]).

    Returns:
        The final_features_df with latest opponent features merged and platoons selected.
    """
    if latest_opponent_rolling is None or latest_opponent_rolling.empty:
        logger.warning("Latest opponent rolling DataFrame is empty, skipping merge.")
        return final_features_df
    if 'opponent_team' not in final_features_df.columns:
        logger.error("Missing 'opponent_team' in prediction baseline for merge.")
        return final_features_df
    if 'team' not in latest_opponent_rolling.columns:
        logger.error("Missing 'team' key in latest opponent rolling data.")
        return final_features_df
    if 'p_throws' not in final_features_df.columns:
         logger.error("Missing 'p_throws' in prediction baseline, cannot select opponent platoon.")
         return final_features_df

    logger.debug("Merging latest opponent features for prediction...")

    # Use keys from opp_rename_map to get the actual columns to merge
    opp_merge_cols = ['team'] + list(opp_rename_map.values())
    # Check columns exist
    opp_merge_cols = [col for col in opp_merge_cols if col in latest_opponent_rolling.columns]

    if len(opp_merge_cols) <= 1 : # Only 'team' key present
        logger.warning("No latest opponent rolling columns identified to merge.")
        return final_features_df

    try:
        # Merge latest stats onto baseline based on opponent_team
        merged_df = pd.merge(
            final_features_df,
            latest_opponent_rolling[opp_merge_cols],
            left_on='opponent_team',
            right_on='team',
            how='left'
        )

        # Select Correct Opponent Platoon Feature for each window
        base_metrics_for_platoon = ['k_percent', 'swinging_strike_percent']
        cols_to_drop = []
        for w in rolling_windows:
            for metric_base in base_metrics_for_platoon:
                opp_met_vs_p = f'opp_roll{w}g_{metric_base}_vs_pitcher'
                opp_met_vs_L = f'opp_roll{w}g_{metric_base}_vs_LHP'
                opp_met_vs_R = f'opp_roll{w}g_{metric_base}_vs_RHP'

                # Check if platoon columns exist before attempting selection
                if opp_met_vs_L in merged_df.columns or opp_met_vs_R in merged_df.columns:
                    merged_df[opp_met_vs_p] = np.where(
                        merged_df['p_throws'] == 'L',
                        merged_df.get(opp_met_vs_L, np.nan),
                        merged_df.get(opp_met_vs_R, np.nan)
                    )
                    # Mark original L/R specific columns for dropping
                    cols_to_drop.extend([opp_met_vs_L, opp_met_vs_R])
                else:
                    logger.warning(f"Platoon columns {opp_met_vs_L}/{opp_met_vs_R} not found for selection.")
                    # Ensure the target column exists even if selection fails
                    if opp_met_vs_p not in merged_df.columns:
                        merged_df[opp_met_vs_p] = np.nan


        # Drop original L/R specific columns and the merge key 'team' safely
        cols_to_drop.append('team')
        merged_df = merged_df.drop(columns=list(set(cols_to_drop)), errors='ignore') # Use set for unique cols
        logger.debug("Successfully merged latest opponent features and selected platoons.")
        return merged_df

    except Exception as e:
        logger.error(f"Error during prediction opponent feature merge: {e}", exc_info=True)
        return final_features_df # Return original df on error