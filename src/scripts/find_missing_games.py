# src/scripts/find_missing_games.py

import pandas as pd
import re
import warnings
from datetime import datetime

# Project imports using the structure from provided files
from src.data.utils import setup_logger, DBConnection
# Import necessary configurations AND constants from config.py
from src.config import (
    DBConfig,
    FileConfig,
    MLB_BOXSCORES_TABLE,
    MASTER_SCHEDULE_TABLE,
    TEAM_MAPPING_TABLE,
    TEAM_NAME_MAP_FULL_NAME_COL,
    TEAM_NAME_MAP_ABBR_COL
)

# Setup logger using the utility function - giving explicit name
logger = setup_logger('find_missing_games') # Changed from __name__

# Define a start date specific to this analysis if not in config
ANALYSIS_START_DATE = '2016-01-01' # Or adjust as needed


def get_day_night(time_str):
    """
    Parses a time string (like '7:10 PM ET' or '13:05') and determines
    if it's a Day ('D') or Night ('N') game based on the hour.
    Threshold: Before 5 PM (17:00) is Day. Logs warnings for parsing issues.
    """
    if pd.isna(time_str):
        return None # Handle missing time data

    match = re.search(r'(\d{1,2}:\d{2})\s*(AM|PM)?', str(time_str).strip(), re.IGNORECASE)
    if not match:
        logger.warning(f"Could not parse time format: {time_str}")
        return None

    time_part = match.group(1)
    am_pm = match.group(2)
    try:
        hour, minute = map(int, time_part.split(':'))
        if am_pm:
            am_pm = am_pm.upper()
            if am_pm == 'PM' and hour != 12: hour += 12
            elif am_pm == 'AM' and hour == 12: hour = 0
        return 'D' if 0 <= hour < 17 else 'N'
    except ValueError:
        logger.warning(f"Could not convert extracted time to int: {time_part}")
        return None


def find_missing_games():
    """
    Compares master_schedule and mlb_boxscores tables to find games
    present in the schedule but missing boxscore data.
    Uses configuration constants for tables and columns from src.config.
    Filters dates within pandas for robustness.
    """
    logger.info("Starting script to find missing MLB boxscores...")

    # --- Configuration ---
    start_date = ANALYSIS_START_DATE
    end_date = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"Target date range for analysis: {start_date} to {end_date}")

    # Use imported constants from config.py
    boxscores_table = MLB_BOXSCORES_TABLE
    schedule_table = MASTER_SCHEDULE_TABLE
    mapping_table = TEAM_MAPPING_TABLE
    map_full_name_col = TEAM_NAME_MAP_FULL_NAME_COL
    map_abbr_col = TEAM_NAME_MAP_ABBR_COL

    try:
        # Use DBConnection with path from DBConfig
        with DBConnection() as conn:
            # 1. Load Team Mapping
            logger.info(f"Loading team abbreviation-to-ID mapping from table '{mapping_table}'...")
            try:
                # Use the correct column name from your CSV for all abbreviations
                all_abbr_col = 'team_abbr'
                id_col = 'team_id'

                # Load necessary columns
                # Ensure team_id is loaded correctly, handling potential '.0' if loaded as float
                df_mapping = pd.read_sql_query(f"SELECT \"{all_abbr_col}\", \"{id_col}\" FROM {mapping_table}", conn)
                logger.info(f"Loaded {len(df_mapping)} rows from team mapping table.")

                # Create the map: {AnyAbbr -> team_id}
                if not df_mapping.empty:
                    # Ensure no NaN values interfere
                    df_mapping = df_mapping.dropna(subset=[all_abbr_col, id_col])
                    # Convert team_id to integer after handling potential float load
                    df_mapping[id_col] = df_mapping[id_col].astype(float).astype(int)
                    abbr_to_id_map = pd.Series(df_mapping[id_col].values, index=df_mapping[all_abbr_col]).to_dict()
                    # Create set of valid team_ids for potential later use (optional here)
                    valid_team_ids = set(df_mapping[id_col].unique())
                    logger.info(f"Created map for {len(abbr_to_id_map)} abbreviations to {len(valid_team_ids)} unique team IDs.")
                    if not valid_team_ids:
                        raise ValueError("No valid team IDs identified.")
                else:
                    raise ValueError("Team mapping table is empty or missing required columns.")

            except Exception as e:
                logger.error(f"Error loading or processing team mapping for standardization: {e}", exc_info=True)
                raise

            # --- Load DataFrames (as before) ---
            # Load mlb_boxscores data (within date range)
            logger.info(f"Loading boxscore data from table '{boxscores_table}'...")
            query_box = f"SELECT game_date, home_team, away_team, first_pitch FROM {boxscores_table} WHERE game_date BETWEEN ? AND ?"
            df_boxscores = pd.read_sql_query(query_box, conn, params=(start_date, end_date))
            logger.info(f"Loaded {len(df_boxscores)} boxscore records.")

            # Load master_schedule data (parsing dates)
            logger.info(f"Loading schedule data from table '{schedule_table}' and parsing dates...")
            query_sched = f"SELECT game_date, home_team, away_team, day_night FROM {schedule_table}"
            try: # Using parse_dates approach
                df_schedule_all = pd.read_sql_query(query_sched, conn, parse_dates=['game_date'])
            except Exception as e: # Fallback...
                logger.error(f"Failed parse_dates: {e}. Falling back...", exc_info=True)
                df_schedule_all = pd.read_sql_query(query_sched, conn)
                df_schedule_all['game_date'] = pd.to_datetime(df_schedule_all['game_date'], errors='coerce')
            logger.info(f"Loaded {len(df_schedule_all)} total schedule records.")

        # --- Start Processing (connection closed) ---

        # Filter schedule by date (as before)
        logger.info(f"Filtering schedule data for dates between {start_date} and {end_date}...")
        conversion_nas = df_schedule_all['game_date'].isna().sum()
        if conversion_nas > 0: logger.warning(f"{conversion_nas} schedule records resulted in NaT dates.")
        df_schedule = df_schedule_all.dropna(subset=['game_date'])
        df_schedule = df_schedule[
            (df_schedule['game_date'] >= pd.to_datetime(start_date)) &
            (df_schedule['game_date'] <= pd.to_datetime(end_date))
        ].copy()
        logger.info(f"{len(df_schedule)} schedule records remain after pandas date filtering.")


        # --- Standardize Team Abbreviations to Team IDs ---
        logger.info("Standardizing teams to team_id in both dataframes...")

        # Standardize schedule teams
        df_schedule['home_team_id'] = df_schedule['home_team'].map(abbr_to_id_map)
        df_schedule['away_team_id'] = df_schedule['away_team'].map(abbr_to_id_map)
        sched_unmapped_home = df_schedule[df_schedule['home_team_id'].isna()]['home_team'].unique()
        sched_unmapped_away = df_schedule[df_schedule['away_team_id'].isna()]['away_team'].unique()
        if len(sched_unmapped_home) > 0: logger.warning(f"Unmapped home teams found IN SCHEDULE when converting to ID: {list(sched_unmapped_home)}")
        if len(sched_unmapped_away) > 0: logger.warning(f"Unmapped away teams found IN SCHEDULE when converting to ID: {list(sched_unmapped_away)}")

        # Standardize boxscore teams
        df_boxscores['home_team_id'] = df_boxscores['home_team'].map(abbr_to_id_map)
        df_boxscores['away_team_id'] = df_boxscores['away_team'].map(abbr_to_id_map)
        box_unmapped_home = df_boxscores[df_boxscores['home_team_id'].isna()]['home_team'].unique()
        box_unmapped_away = df_boxscores[df_boxscores['away_team_id'].isna()]['away_team'].unique()
        if len(box_unmapped_home) > 0: logger.warning(f"Unmapped home teams found IN BOXSCORES when converting to ID: {list(box_unmapped_home)}")
        if len(box_unmapped_away) > 0: logger.warning(f"Unmapped away teams found IN BOXSCORES when converting to ID: {list(box_unmapped_away)}")

        # --- Filter by SUCCESSFUL Team ID Mapping ---
        # Keep only rows where both teams were successfully mapped to an ID
        logger.info("Filtering both dataframes for games with valid team IDs...")

        initial_schedule_count = len(df_schedule)
        df_schedule = df_schedule.dropna(subset=['home_team_id', 'away_team_id']).copy()
        # Convert IDs to int just in case they loaded as float from map
        df_schedule['home_team_id'] = df_schedule['home_team_id'].astype(int)
        df_schedule['away_team_id'] = df_schedule['away_team_id'].astype(int)
        logger.info(f"{len(df_schedule)} schedule records remain after ensuring valid team IDs (removed {initial_schedule_count - len(df_schedule)}).")

        initial_boxscore_count = len(df_boxscores)
        df_boxscores = df_boxscores.dropna(subset=['home_team_id', 'away_team_id']).copy()
        # Convert IDs to int just in case they loaded as float from map
        df_boxscores['home_team_id'] = df_boxscores['home_team_id'].astype(int)
        df_boxscores['away_team_id'] = df_boxscores['away_team_id'].astype(int)
        logger.info(f"{len(df_boxscores)} boxscore records remain after ensuring valid team IDs (removed {initial_boxscore_count - len(df_boxscores)}).")

        # Ensure DataFrames are not empty before proceeding
        if df_boxscores.empty or df_schedule.empty:
            logger.warning("No data found for one or both tables after filtering IDs/dates. Cannot compare.")
            return

        # Ensure day_night is uppercase AFTER filtering
        df_schedule['day_night'] = df_schedule['day_night'].str.upper()

        # 4. Preprocess df_boxscores
        logger.info("Preprocessing boxscore data...")
        # Convert first_pitch to day_night
        df_boxscores['calc_day_night'] = df_boxscores['first_pitch'].apply(get_day_night)

        # Create composite key using original team columns (now filtered)
        # Handle potential NaNs from get_day_night if needed
        df_boxscores_processed = df_boxscores.dropna(
            subset=['game_date', 'home_team_id', 'away_team_id', 'calc_day_night'] # Use _id cols
        ).copy()
        df_boxscores_processed['game_date_str'] = pd.to_datetime(df_boxscores_processed['game_date']).dt.strftime('%Y-%m-%d')
        # Order-independent key using TEAM IDs (sorted numerically)
        # Ensure IDs are numeric before min/max
        id1_b = df_boxscores_processed[['home_team_id', 'away_team_id']].min(axis=1)
        id2_b = df_boxscores_processed[['home_team_id', 'away_team_id']].max(axis=1)
        df_boxscores_processed['composite_key'] = \
            df_boxscores_processed['game_date_str'] + '_' + \
            id1_b.astype(str) + '_' + id2_b.astype(str) + '_' + \
            df_boxscores_processed['calc_day_night']

        # 5. Preprocess df_schedule (already filtered by date)
        logger.info("Preprocessing schedule data...")
        df_schedule_processed = df_schedule.dropna(
            subset=['game_date', 'home_team_id', 'away_team_id', 'day_night'] # Use _id cols
        ).copy()
        df_schedule_processed['game_date_str'] = df_schedule_processed['game_date'].dt.strftime('%Y-%m-%d')
        # Order-independent key using TEAM IDs (sorted numerically)
        # Ensure IDs are numeric before min/max
        id1_s = df_schedule_processed[['home_team_id', 'away_team_id']].min(axis=1)
        id2_s = df_schedule_processed[['home_team_id', 'away_team_id']].max(axis=1)
        df_schedule_processed['composite_key'] = \
            df_schedule_processed['game_date_str'] + '_' + \
            id1_s.astype(str) + '_' + id2_s.astype(str) + '_' + \
            df_schedule_processed['day_night']

        # 6. Compare Keys
        logger.info("Comparing schedule keys against boxscore keys...")
        schedule_keys_set = set(df_schedule_processed['composite_key'])
        boxscores_keys_set = set(df_boxscores_processed['composite_key'])
        missing_keys = schedule_keys_set - boxscores_keys_set

        # --- ADD Debugging for Key Mismatch ---
        intersection_keys = schedule_keys_set.intersection(boxscores_keys_set)
        boxscore_only_keys = boxscores_keys_set - schedule_keys_set
        # Use INFO level to ensure visibility
        logger.info(f"Intersection size (keys in both): {len(intersection_keys)}")
        logger.info(f"Keys unique to boxscores (should ideally be 0): {len(boxscore_only_keys)}")
        if len(boxscore_only_keys) > 0:
            # Log first 5 sample keys unique to boxscores
            logger.info(f"Sample keys unique to boxscores: {list(boxscore_only_keys)[:5]}")
        # Verify missing calculation:
        logger.info(f"Keys unique to schedule (re-calc check): {len(schedule_keys_set - boxscores_keys_set)}")
        # --- END Debugging ---

        # 7. Report Missing Games (as before)
        logger.info("--- Comparison Results ---")
        logger.info(f"Total unique games in schedule (filtered): {len(schedule_keys_set)}")
        logger.info(f"Total unique games in boxscores (filtered & processed): {len(boxscores_keys_set)}")
        logger.info(f"Number of games missing from boxscores: {len(missing_keys)}")

        if len(missing_keys) > 0:
            logger.info("Sample of missing games (from schedule perspective):")
            missing_games_details = df_schedule_processed[df_schedule_processed['composite_key'].isin(missing_keys)]
            missing_games_details_sorted = missing_games_details.sort_values(by=['game_date', 'home_team'])
            # Display using game_date_str now
            logger.info(f"\n{missing_games_details_sorted[['game_date_str', 'away_team', 'home_team', 'day_night']].head(20).to_string()}")

            logger.info("--- Next Steps Suggestion ---")
            logger.info("To fetch missing boxscores, you likely need 'gamePk'. Query an MLB API schedule endpoint using date/teams to find 'gamePk'.")

            try:
                output_path = FileConfig.DATA_DIR / 'missing_games_report.csv'
                # Save relevant columns
                missing_games_details_sorted[['game_date_str', 'away_team', 'home_team', 'day_night']].to_csv(output_path, index=False, header=['game_date','away_team','home_team','day_night'])
                logger.info(f"Full list of missing games saved to '{output_path}'")
            except Exception as e:
                logger.error(f"Failed to save missing games report: {e}", exc_info=True)
        else:
            logger.info("No missing games found based on the comparison criteria!")

        logger.info("Successfully finished finding missing games.")

    except Exception as e:
        logger.error(f"An error occurred during the find_missing_games execution: {e}", exc_info=True)


if __name__ == "__main__":
    find_missing_games()