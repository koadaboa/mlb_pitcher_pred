# src/features/selection.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__) # Use standard logging

# Define the columns to exclude BEFORE any importance calculation
# (Using the list you provided previously, plus umpire features)
BASE_EXCLUDE_COLS = [
    # Identifiers / Non-Features
    'index', '', 'pitcher_id', 'player_name', 'game_pk', 'home_team', 'away_team', 'opponent',
    'opponent_team_name', 'game_date', 'season', 'game_month', 'year',
    'p_throws', 'stand', 'team', 'Team', # Original categoricals (encoded versions are kept if desired)
    'opp_base_team', 'opp_adv_team', 'opp_adv_opponent', 'ballpark', # Original categoricals
    'home_plate_umpire', # Original umpire string

    # Target Variable
    'strikeouts',

    # --- DIRECT LEAKAGE COLUMNS (Derived from CURRENT game outcome/process) ---
    'batters_faced', 'total_pitches', 'innings_pitched',
    'avg_velocity', 'max_velocity', 'avg_spin_rate', 'avg_horizontal_break', 'avg_vertical_break',
    'k_per_9', 'k_percent', 'swinging_strike_percent', 'called_strike_percent',
    'zone_percent', 'fastball_percent', 'breaking_percent', 'offspeed_percent',
    'total_swinging_strikes', 'total_called_strikes', 'total_fastballs',
    'total_breaking', 'total_offspeed', 'total_in_zone',
    'pa_vs_rhb', 'k_vs_rhb', 'k_percent_vs_rhb', # Current game platoon splits
    'pa_vs_lhb', 'k_vs_lhb', 'k_percent_vs_lhb', # Current game platoon splits
    'platoon_split_k_pct', # Current game platoon split derived metric

    # --- ADDED EXCLUSIONS FOR LEAKAGE (Changes & _pct_change) ---
    'strikeouts_change',    # Directly uses current game target
    'k_percent_change',     # Directly uses current game k_percent (leaky)
    'k_per_9_change',       # Directly uses current game k_per_9 (leaky)
    'batters_faced_change', # Directly uses current game batters_faced (leaky)
    'innings_pitched_change',# Directly uses current game innings_pitched (leaky)
    'swinging_strike_percent_change', # <<< ADDED: Uses current game swstr%
    'called_strike_percent_change', # <<< ADDED: Uses current game cstr%
    'zone_percent_change',           # <<< ADDED: Uses current game zone%
    'fastball_percent_change',       # <<< ADDED: Uses current game fb%
    'breaking_percent_change',     # <<< ADDED: Uses current game brk%
    'offspeed_percent_change',     # <<< ADDED: Uses current game off%
    # --------------------------------------------------------

    # --- EXCLUDE LAGGED/EWMA TARGET-RELATED FEATURES (Highly Suspect) ---
    'strikeouts_lag1', 'strikeouts_lag2',
    'k_percent_lag1', 'k_percent_lag2',
    'k_per_9_lag1', 'k_per_9_lag2',
    'ewma_3g_strikeouts', 'ewma_5g_strikeouts', 'ewma_10g_strikeouts',
    'ewma_3g_k_percent', 'ewma_5g_k_percent', 'ewma_10g_k_percent',
    'ewma_3g_k_per_9', 'ewma_5g_k_per_9', 'ewma_10g_k_per_9',
    'strikeouts_last2g_vs_baseline', 'k_percent_last2g_vs_baseline', 'k_per_9_last2g_vs_baseline',
    'k_trend_up_lagged', 'k_trend_down_lagged',
    # -------------------------------------------------------------------

    # --- UMPIRE FEATURES (Exclude raw/intermediate, keep encoded/historical) ---
    'umpire_historical_k_per_9', # Keep historical umpire stats
    'pitcher_umpire_k_boost', # Keep interaction feature
    # Encoded umpire ('home_plate_umpire_encoded') is KEPT (not listed here)
    # -----------------------------------------------------------------------

    # Other potential post-game info or less relevant features
    'inning', 'score_differential', 'is_close_game', 'is_playoff',

    # Low importance / redundant features
    #'is_home', # Keep this simple numerical feature for now
    'rest_days_6_more', 'rest_days_4_less', 'rest_days_5', # days_since_last_game is likely better

    # Imputation flags (usually not predictive)
    'avg_velocity_imputed_median', 'max_velocity_imputed_median', 'avg_spin_rate_imputed_median',
    'avg_horizontal_break_imputed_median', 'avg_vertical_break_imputed_median',
    'avg_velocity_imputed_knn', 'avg_spin_rate_imputed_knn',
    'avg_horizontal_break_imputed_knn', 'avg_vertical_break_imputed_knn',

    # Keep Lags/Changes/EWMAs of PREDICTORS (e.g., velocity, pitch usage, opponent stats)
    # Example: 'innings_pitched_lag1', 'avg_velocity_change', 'ewma_10g_total_pitches' are generally OK to keep
    # NOTE: We are excluding _change features derived from leaky base metrics above.
]


def select_features(df, target_variable, exclude_cols=None):
    """
    Selects numeric features suitable for training, excluding a predefined list.

    Args:
        df (pd.DataFrame): The input dataframe containing all features.
        target_variable (str): The name of the target variable column.
        exclude_cols (list, optional): A list of columns to explicitly exclude.
                                       Defaults to BASE_EXCLUDE_COLS defined above.

    Returns:
        tuple: (list_of_selected_feature_names, pd.DataFrame_subset_with_features_and_target)
               Returns ([], pd.DataFrame()) if input df is invalid or no features are selected.
    """
    if df is None or df.empty:
        logger.error("Input DataFrame is None or empty.")
        return [], pd.DataFrame()

    if exclude_cols is None:
        exclude_cols = BASE_EXCLUDE_COLS

    # Create a set for efficient lookup, handling potential None values in df.columns
    df_cols_set = set(col for col in df.columns if col is not None)
    exclude_set = set(exclude_cols)

    # Ensure target variable is in the exclusion list (if it exists in df)
    if target_variable in df_cols_set and target_variable not in exclude_set:
        exclude_set.add(target_variable)

    # Select numeric columns ONLY
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Find features that are numeric AND not in the exclusion list
    feature_cols = [
        col for col in numeric_cols
        if col not in exclude_set
        and col in df_cols_set # Ensure the column actually exists in the df
    ]

    # Check for infinite values AFTER identifying potential feature columns
    has_inf = False
    if feature_cols: # Check if list is not empty
        df_subset = df[feature_cols]
        if not df_subset.empty:
            try:
                inf_mask = np.isinf(df_subset).any()
                if inf_mask.any():
                    inf_cols = df_subset.columns[inf_mask].tolist()
                    logger.warning(f"Infinite values found in potential feature columns: {inf_cols}. Consider handling them.")
                    has_inf = True
            except TypeError as e:
                 logger.error(f"TypeError checking for infinite values (potentially mixed types): {e}")
                 # Handle or log columns causing issues if possible
                 pass # Continue without inf check if types are mixed causing error


    if not feature_cols:
        logger.error("No numeric features selected after applying exclusions.")
        return [], pd.DataFrame()

    logger.info(f"Selected {len(feature_cols)} numeric features after initial exclusion (including umpire features).")

    # Return the list of feature names and a DataFrame subset containing
    # only these features and the target variable (if present)
    columns_to_return = feature_cols + ([target_variable] if target_variable in df_cols_set else [])
    # Ensure all columns to return actually exist before slicing
    columns_to_return = [col for col in columns_to_return if col in df_cols_set]
    selected_df_subset = df[columns_to_return].copy()

    return feature_cols, selected_df_subset

