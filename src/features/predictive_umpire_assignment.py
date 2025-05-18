# src/features/predictive_umpire_assignment.py
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Define umpire positions for clarity and consistency
HP_UMPIRE_COL = 'hp_umpire'
FIRST_BASE_UMPIRE_COL = '1b_umpire'
SECOND_BASE_UMPIRE_COL = '2b_umpire'
THIRD_BASE_UMPIRE_COL = '3b_umpire'
UMPIRE_POSITION_COLS = [HP_UMPIRE_COL, FIRST_BASE_UMPIRE_COL, SECOND_BASE_UMPIRE_COL, THIRD_BASE_UMPIRE_COL]

# --- Helper Functions ---

def _identify_series_games(
    target_game_date: pd.Timestamp,
    target_home_team: str,
    target_away_team: str,
    historical_assignments_df: pd.DataFrame,
    series_date_window_days: int = 4 # Games within this window around target could be part of the same series
) -> pd.DataFrame:
    """
    Identifies historical games that likely belong to the same series as the target game.
    A series is defined by the same home team, same away team, and close game dates.

    Args:
        target_game_date: The date of the game for which we want to find series context.
        target_home_team: The home team of the target game.
        target_away_team: The away team of the target game.
        historical_assignments_df: DataFrame with historical game data, including
                                   'game_date', 'home_team', 'away_team', and umpire columns.
        series_date_window_days: Number of days before and after the target_game_date
                                 to consider for finding games in the same series.

    Returns:
        A DataFrame of games belonging to the identified series, sorted by date.
        Returns an empty DataFrame if no relevant series games are found or if input is invalid.
    """
    required_hist_cols = ['game_date', 'home_team', 'away_team']
    if not all(col in historical_assignments_df.columns for col in required_hist_cols):
        logger.error(f"Historical assignments DataFrame missing one or more required columns: {required_hist_cols}.")
        return pd.DataFrame()

    # Ensure 'game_date' is in datetime format
    df_copy = historical_assignments_df.copy() # Work on a copy
    if not pd.api.types.is_datetime64_any_dtype(df_copy['game_date']):
        try:
            df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
        except Exception as e:
            logger.error(f"Could not convert 'game_date' to datetime in historical_assignments_df: {e}")
            return pd.DataFrame()

    # Filter for games between the same two teams, with the same home team (defines the series venue)
    series_df = df_copy[
        (df_copy['home_team'] == target_home_team) &
        (df_copy['away_team'] == target_away_team)
    ].copy()

    if series_df.empty:
        logger.debug(f"No historical games found with {target_home_team} as home and {target_away_team} as away.")
        return pd.DataFrame()

    # Define a date window to find games in the same series.
    # A series is typically 2-4 games over consecutive days.
    # We look a few days before the target game to find the start of its series.
    min_series_date = target_game_date - timedelta(days=series_date_window_days)
    # We also look slightly after, in case the target_game_date is the first known game in the series
    # and we want to see if subsequent games confirm a pattern (though for prediction, we only use prior data).
    max_series_date = target_game_date + timedelta(days=1) # Only look up to target date for prediction context

    series_df = series_df[
        (series_df['game_date'] >= min_series_date) &
        (series_df['game_date'] <= max_series_date) # Include target date if it's in historical
    ]
    
    series_df = series_df.sort_values(by='game_date').drop_duplicates(subset=['game_pk'], keep='last')

    logger.debug(f"Identified {len(series_df)} potential games in the series for {target_home_team} vs {target_away_team} leading up to or on {target_game_date.date()}.")
    return series_df


def _get_crew_and_assignments_from_game_row(game_row: pd.Series) -> Tuple[Optional[List[str]], Optional[Dict[str, str]]]:
    """
    Extracts the umpire crew (as a list of unique names) and their specific position assignments
    from a single game row. Verifies that all four positions have non-null umpires.

    Args:
        game_row: A pandas Series representing a single game, containing umpire columns.

    Returns:
        A tuple: (crew_list, assignments_dict).
        - crew_list: List of unique umpire names in the crew.
        - assignments_dict: Dictionary mapping position (e.g., 'hp_umpire') to umpire name.
        Returns (None, None) if assignments are incomplete or invalid.
    """
    if game_row.empty or not all(pos_col in game_row for pos_col in UMPIRE_POSITION_COLS):
        logger.debug(f"Game row (game_pk: {game_row.get('game_pk', 'N/A')}) is empty or missing standard umpire columns.")
        return None, None

    assignments = {}
    crew_set = set() # Use a set to store unique umpire names for the crew

    for pos_col in UMPIRE_POSITION_COLS:
        umpire_name = game_row.get(pos_col)
        if pd.isna(umpire_name) or str(umpire_name).strip() == "":
            logger.debug(f"Incomplete umpire assignment: {pos_col} is missing for game_pk {game_row.get('game_pk', 'N/A')}.")
            return None, None # Incomplete crew information
        assignments[pos_col] = str(umpire_name).strip()
        crew_set.add(str(umpire_name).strip())
    
    # Typically expect a 4-man crew for standard MLB games
    if len(crew_set) != 4:
        logger.warning(f"Game_pk {game_row.get('game_pk', 'N/A')} has an unusual crew size: {len(crew_set)} members ({crew_set}). Expected 4. Assignments: {assignments}")
        # Depending on strictness, you might still proceed or return None.
        # For now, let's be strict for rotation prediction.
        return None, None
        
    return sorted(list(crew_set)), assignments


def _get_previous_game_assignments_in_series(
    series_games_df: pd.DataFrame,
    target_game_date: pd.Timestamp
) -> Tuple[Optional[List[str]], Optional[Dict[str, str]], Optional[pd.Timestamp]]:
    """
    Gets the full umpire crew and their assignments from the game *immediately preceding*
    the target_game_date within the provided series_games_df.

    Args:
        series_games_df: DataFrame of games identified as belonging to the same series, sorted by date.
        target_game_date: The date of the game for which we are predicting.

    Returns:
        A tuple: (crew_list, assignments_dict, previous_game_date).
        Returns (None, None, None) if no valid preceding game with full assignments is found.
    """
    if series_games_df.empty:
        return None, None, None

    # Filter for games strictly before the target game date within the identified series
    # Ensure target_game_date is timezone-naive if series_games_df['game_date'] is, or vice-versa
    # Assuming both are pd.Timestamp and can be compared.
    previous_games_in_series = series_games_df[series_games_df['game_date'] < target_game_date].copy()
    
    if previous_games_in_series.empty:
        logger.debug(f"No previous games found in the identified series before {target_game_date.date()}.")
        return None, None, None

    # The last row of this sorted DataFrame is the game immediately prior
    last_game_row = previous_games_in_series.iloc[-1]
    
    crew, assignments = _get_crew_and_assignments_from_game_row(last_game_row)
    
    if crew and assignments:
        prev_game_date = last_game_row['game_date']
        logger.debug(f"Found previous game assignments for series on {prev_game_date.date()}: HP was {assignments.get(HP_UMPIRE_COL)}")
        return crew, assignments, prev_game_date
    else:
        logger.warning(f"Previous game in series (Date: {last_game_row['game_date'].date()}, PK: {last_game_row.get('game_pk', 'N/A')}) had incomplete umpire data.")
        return None, None, None

# --- Main Prediction Function ---

def predict_home_plate_umpire(
    target_game_info: Dict,
    historical_assignments_df: pd.DataFrame,
    rotation_pattern_logic: str = "1B_TO_HP" # Defines how next HP is chosen
) -> Optional[str]:
    """
    Predicts the Home Plate umpire for an upcoming game.
    Relies on finding the previous game in the same series and applying a rotation pattern.

    Args:
        target_game_info: Dictionary with game details for the prediction target.
                          Expected keys: 'game_date' (str or pd.Timestamp), 
                                         'home_team' (str), 'away_team' (str),
                                         'game_pk' (int or str, for logging).
        historical_assignments_df: DataFrame with historical game data including columns:
                                   'game_pk', 'game_date', 'home_team', 'away_team',
                                   'hp_umpire', '1b_umpire', '2b_umpire', '3b_umpire'.
                                   All umpire columns should contain umpire names/IDs.
        rotation_pattern_logic: Specifies the rule for predicting the next HP umpire.
                                Currently supports:
                                - "1B_TO_HP": Assumes the umpire at 1st Base in the
                                              previous game moves to Home Plate.

    Returns:
        The predicted name/ID of the Home Plate umpire, or None if a prediction cannot be made.
    """
    required_target_keys = ['game_date', 'home_team', 'away_team']
    if not target_game_info or not all(k in target_game_info for k in required_target_keys):
        logger.error(f"Target game info is incomplete. Missing one of {required_target_keys}.")
        return None
    if historical_assignments_df.empty:
        logger.warning("Historical assignments DataFrame is empty. Cannot predict umpire.")
        return None

    try:
        target_game_date = pd.to_datetime(target_game_info['game_date'])
    except Exception as e:
        logger.error(f"Invalid target_game_info['game_date']: {target_game_info['game_date']}. Error: {e}")
        return None
        
    target_home_team = target_game_info['home_team']
    target_away_team = target_game_info['away_team']
    target_game_pk_log = target_game_info.get('game_pk', 'N/A_PK')

    logger.info(f"Attempting to predict HP umpire for game PK '{target_game_pk_log}' on {target_game_date.date()} ({target_away_team} @ {target_home_team}).")

    # 1. Identify games in the current series (leading up to and including target_game_date)
    # This helps find the immediately preceding game.
    series_games_df = _identify_series_games(
        target_game_date, target_home_team, target_away_team, historical_assignments_df
    )

    if series_games_df.empty:
        logger.warning(f"Could not identify any relevant series games for PK '{target_game_pk_log}'. Cannot determine crew or rotation.")
        return None

    # 2. Get assignments from the previous game in this series
    prev_crew, prev_assignments, prev_game_date = _get_previous_game_assignments_in_series(
        series_games_df, target_game_date # Pass the target game's date
    )

    if not prev_crew or not prev_assignments:
        logger.warning(f"No valid previous game assignments found in the series for PK '{target_game_pk_log}' before {target_game_date.date()}. Prediction difficult.")
        # This could be the first game of the series, or prior games had incomplete data.
        # A more advanced version might try to find the crew's last assignment in a *different* series.
        return None

    # 3. Apply Rotation Logic
    predicted_hp_umpire = None
    if rotation_pattern_logic == "1B_TO_HP":
        # Assumes the umpire at 1st Base in the previous game moves to Home Plate.
        predicted_hp_umpire = prev_assignments.get(FIRST_BASE_UMPIRE_COL)
        if predicted_hp_umpire:
            logger.info(f"Predicted HP Umpire for PK '{target_game_pk_log}' is: '{predicted_hp_umpire}' (was 1B on {prev_game_date.date()}).")
        else:
            # This should ideally not happen if _get_previous_game_assignments_in_series returned valid assignments
            logger.error(f"Previous game assignments found, but '{FIRST_BASE_UMPIRE_COL}' was missing. Assignments: {prev_assignments}")
    else:
        logger.error(f"Unknown or unsupported rotation_pattern_logic: '{rotation_pattern_logic}'")
        return None

    return predicted_hp_umpire


if __name__ == '__main__':
    # Configure logger for direct script execution testing
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing predictive_umpire_assignment.py ---")

    # Create more comprehensive dummy historical data
    dummy_data_list = [
        # Series 1: LAA vs. HOU (3 games)
        {'game_pk': 101, 'game_date': '2025-05-10', 'home_team': 'LAA', 'away_team': 'HOU', HP_UMPIRE_COL: 'UmpA', FIRST_BASE_UMPIRE_COL: 'UmpB', SECOND_BASE_UMPIRE_COL: 'UmpC', THIRD_BASE_UMPIRE_COL: 'UmpD'},
        {'game_pk': 102, 'game_date': '2025-05-11', 'home_team': 'LAA', 'away_team': 'HOU', HP_UMPIRE_COL: 'UmpB', FIRST_BASE_UMPIRE_COL: 'UmpC', SECOND_BASE_UMPIRE_COL: 'UmpD', THIRD_BASE_UMPIRE_COL: 'UmpA'}, # UmpB (prev 1B) is HP
        {'game_pk': 103, 'game_date': '2025-05-12', 'home_team': 'LAA', 'away_team': 'HOU', HP_UMPIRE_COL: 'UmpC', FIRST_BASE_UMPIRE_COL: 'UmpD', SECOND_BASE_UMPIRE_COL: 'UmpA', THIRD_BASE_UMPIRE_COL: 'UmpB'}, # UmpC (prev 1B) is HP
        
        # Series 2: NYY vs. BOS (2 games)
        {'game_pk': 201, 'game_date': '2025-05-10', 'home_team': 'NYY', 'away_team': 'BOS', HP_UMPIRE_COL: 'UmpE', FIRST_BASE_UMPIRE_COL: 'UmpF', SECOND_BASE_UMPIRE_COL: 'UmpG', THIRD_BASE_UMPIRE_COL: 'UmpH'},
        {'game_pk': 202, 'game_date': '2025-05-11', 'home_team': 'NYY', 'away_team': 'BOS', HP_UMPIRE_COL: 'UmpF', FIRST_BASE_UMPIRE_COL: 'UmpG', SECOND_BASE_UMPIRE_COL: 'UmpH', THIRD_BASE_UMPIRE_COL: 'UmpE'},
        
        # Series 3: LAA vs. TEX (Starts later, for testing "first game of series" scenario)
        # No prior LAA vs TEX games in this dummy set before 05-15 for LAA as home.
        
        # Incomplete game data example
        {'game_pk': 301, 'game_date': '2025-05-09', 'home_team': 'LAA', 'away_team': 'HOU', HP_UMPIRE_COL: 'UmpX', FIRST_BASE_UMPIRE_COL: 'UmpY', SECOND_BASE_UMPIRE_COL: None, THIRD_BASE_UMPIRE_COL: 'UmpZ'},

        # Game for which target game itself has data (for testing that path)
        {'game_pk': 104, 'game_date': '2025-05-13', 'home_team': 'LAA', 'away_team': 'HOU', HP_UMPIRE_COL: 'UmpD_Actual', FIRST_BASE_UMPIRE_COL: 'UmpA_Actual', SECOND_BASE_UMPIRE_COL: 'UmpB_Actual', THIRD_BASE_UMPIRE_COL: 'UmpC_Actual'},

    ]
    historical_df = pd.DataFrame(dummy_data_list)
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date'])

    # --- Test Cases ---
    logger.info("\n--- Test Case 1: Predict for Game 3 of LAA vs HOU series (PK 103, Date 2025-05-12) ---")
    # Previous game is PK 102 (2025-05-11), 1B umpire was UmpC. Expected HP for PK 103: UmpC.
    target_game_1_info = {'game_pk': 103, 'game_date': '2025-05-12', 'home_team': 'LAA', 'away_team': 'HOU'}
    predicted_ump1 = predict_home_plate_umpire(target_game_1_info, historical_df)
    logger.info(f"Prediction for LAA@HOU on 2025-05-12 (PK 103): {predicted_ump1} (Expected: UmpC)")

    logger.info("\n--- Test Case 2: Predict for Game 2 of LAA vs HOU series (PK 102, Date 2025-05-11) ---")
    # Previous game is PK 101 (2025-05-10), 1B umpire was UmpB. Expected HP for PK 102: UmpB.
    target_game_2_info = {'game_pk': 102, 'game_date': '2025-05-11', 'home_team': 'LAA', 'away_team': 'HOU'}
    predicted_ump2 = predict_home_plate_umpire(target_game_2_info, historical_df)
    logger.info(f"Prediction for LAA@HOU on 2025-05-11 (PK 102): {predicted_ump2} (Expected: UmpB)")

    logger.info("\n--- Test Case 3: Predict for First Game of LAA vs HOU series (PK 101, Date 2025-05-10) ---")
    # No prior game in this series in the dummy data. Expected: None
    target_game_3_info = {'game_pk': 101, 'game_date': '2025-05-10', 'home_team': 'LAA', 'away_team': 'HOU'}
    predicted_ump3 = predict_home_plate_umpire(target_game_3_info, historical_df)
    logger.info(f"Prediction for LAA@HOU on 2025-05-10 (PK 101): {predicted_ump3} (Expected: None)")
    
    logger.info("\n--- Test Case 4: Predict for a new series (LAA vs TEX, Date 2025-05-15) ---")
    # No prior LAA vs TEX games where LAA is home. Expected: None
    target_game_4_info = {'game_pk': 401, 'game_date': '2025-05-15', 'home_team': 'LAA', 'away_team': 'TEX'}
    predicted_ump4 = predict_home_plate_umpire(target_game_4_info, historical_df)
    logger.info(f"Prediction for LAA@TEX on 2025-05-15 (PK 401): {predicted_ump4} (Expected: None)")

    logger.info("\n--- Test Case 5: Previous game had incomplete umpire data (using game before PK 301) ---")
    # Add a valid game before the incomplete one to test fallback
    valid_game_before_incomplete = {'game_pk': 300, 'game_date': '2025-05-08', 'home_team': 'LAA', 'away_team': 'HOU', HP_UMPIRE_COL: 'UmpS', FIRST_BASE_UMPIRE_COL: 'UmpT', SECOND_BASE_UMPIRE_COL: 'UmpU', THIRD_BASE_UMPIRE_COL: 'UmpV'}
    historical_df_extended = pd.concat([historical_df, pd.DataFrame([valid_game_before_incomplete])], ignore_index=True)
    historical_df_extended['game_date'] = pd.to_datetime(historical_df_extended['game_date'])
    
    # Target game is PK 301 (2025-05-09), its own data is incomplete.
    # Previous game (PK 300) is complete. 1B was UmpT.
    target_game_5_info = {'game_pk': 'TestPK_after_incomplete', 'game_date': '2025-05-09', 'home_team': 'LAA', 'away_team': 'HOU'}
    predicted_ump5 = predict_home_plate_umpire(target_game_5_info, historical_df_extended) # PK 301 is the one with incomplete data
    logger.info(f"Prediction for LAA@HOU on 2025-05-09 (after incomplete game): {predicted_ump5} (Expected: UmpT, from PK 300)")

    logger.info("\n--- Test Case 6: Target game itself (PK 104) exists in historical_df with umpire data ---")
    # The function should ideally not predict but could return actual if logic allows (current doesn't explicitly do this for target)
    # The current logic finds previous game. If target_game_date is the *latest* in series_games_df,
    # _get_previous_game_assignments_in_series will look for games *before* it.
    # If target_game_info's date is '2025-05-13', previous game is 103 (2025-05-12). 1B was UmpD. Expected HP: UmpD
    target_game_6_info = {'game_pk': 104, 'game_date': '2025-05-13', 'home_team': 'LAA', 'away_team': 'HOU'}
    predicted_ump6 = predict_home_plate_umpire(target_game_6_info, historical_df)
    logger.info(f"Prediction for LAA@HOU on 2025-05-13 (PK 104): {predicted_ump6} (Expected: UmpD)")

