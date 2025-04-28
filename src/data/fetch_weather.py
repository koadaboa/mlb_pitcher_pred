# src/data/fetch_weather.py

# --- Keep all imports and other functions the same ---
import argparse
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
try:
    import mlbstatsapi
    from src.config import DBConfig, DataConfig, BALLPARK_COORDS, FileConfig
    from src.data.utils import setup_logger, DBConnection, ensure_dir
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    # Fallbacks omitted for brevity
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, db_path): self.db_path = db_path
        def __enter__(self): import sqlite3; self.conn = sqlite3.connect(self.db_path); return self.conn
        def __exit__(self,et,ev,tb): self.conn.close()
    class DBConfig: PATH = "data/pitcher_stats.db"
    class DataConfig: SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]; RATE_LIMIT_PAUSE = 5
    BALLPARK_COORDS = {}
    class FileConfig: DATA_DIR = project_root / 'data'

logger = setup_logger(
    'fetch_weather',
    log_file=project_root / 'logs' / 'fetch_weather.log'
) if MODULE_IMPORTS_OK else logging.getLogger('fetch_weather_fallback')

WEATHER_TABLE_NAME = "weather_data"
OPENMETEO_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARIABLES = [
    "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum",
    "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max",
    "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration"
]
WEATHER_VARIABLES_STR = ",".join(WEATHER_VARIABLES)
SEASON_START_MONTH_DAY = "03-15"
SEASON_END_MONTH_DAY = "11-10"
MAX_RETRIES = 2 # Max number of retries on 429 error
RETRY_WAIT_SECONDS = 60 # Wait time in seconds before retrying

# --- Database Functions (init_db, check_date_in_db, save_data_to_db) - NO CHANGES ---
# (Code omitted for brevity)
def init_db(db_path):
    logger.info(f"Initializing database table '{WEATHER_TABLE_NAME}'...")
    sql_friendly_weather_vars = [v.replace('-', '_') for v in WEATHER_VARIABLES]
    columns_sql = ", ".join([f"{col} REAL" for col in sql_friendly_weather_vars])
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
        game_date TEXT NOT NULL,
        ballpark_name TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        {columns_sql},
        fetch_timestamp TEXT,
        PRIMARY KEY (game_date, ballpark_name)
    );
    """
    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
        logger.info(f"Table '{WEATHER_TABLE_NAME}' initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database table: {e}", exc_info=True)
        raise

def check_date_in_db(db_path, date_str):
    logger.debug(f"Checking database for existing data on date: {date_str}")
    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT 1 FROM {WEATHER_TABLE_NAME} WHERE game_date = ? LIMIT 1"
            cursor.execute(query, (date_str,))
            exists = cursor.fetchone() is not None
            logger.debug(f"Data for {date_str} {'found' if exists else 'not found'} in DB.")
            return exists
    except sqlite3.Error as e:
        logger.error(f"Error checking database for date {date_str}: {e}", exc_info=True)
        return False

def save_data_to_db(db_path, weather_df, mode='replace'):
    if weather_df.empty:
        logger.warning("Weather DataFrame is empty. Nothing to save.")
        return
    logger.info(f"Saving {len(weather_df)} weather records to database (mode: {mode})...")
    required_cols = ['game_date', 'ballpark_name']
    if not all(col in weather_df.columns for col in required_cols):
        logger.error(f"DataFrame missing required columns for saving: {required_cols}")
        return
    try:
        with DBConnection(db_path) as conn:
            weather_df.to_sql(WEATHER_TABLE_NAME, conn, if_exists=mode, index=False, chunksize=DBConfig.BATCH_SIZE)
        logger.info(f"Successfully saved data to '{WEATHER_TABLE_NAME}' table.")
    except sqlite3.IntegrityError as e:
        logger.error(f"Integrity error saving data (likely duplicate date/ballpark): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving data to database: {e}", exc_info=True)
        raise

# --- MLB Schedule Fetching (load_team_ballpark_map_from_db, get_mlb_schedule) - NO CHANGES ---
# (Code omitted for brevity)
def load_team_ballpark_map_from_db(db_path):
    logger.info(f"Loading team->ballpark mapping from database table 'team_mapping'...")
    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_mapping'")
            if not cursor.fetchone(): logger.error("Database table 'team_mapping' not found."); return {}
            query = "SELECT team_id, ballpark FROM team_mapping"
            mapping_df = pd.read_sql_query(query, conn)
        mapping_df['team_id'] = pd.to_numeric(mapping_df['team_id'], errors='coerce').astype('Int64')
        mapping_df = mapping_df.dropna(subset=['team_id', 'ballpark'])
        team_to_ballpark = dict(zip(mapping_df['team_id'], mapping_df['ballpark']))
        logger.info(f"Loaded mapping for {len(team_to_ballpark)} teams from database.")
        return team_to_ballpark
    except sqlite3.Error as e: logger.error(f"SQLite error loading team mapping from database: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error loading team mapping from database: {e}", exc_info=True)
    return {}

def get_mlb_schedule(start_date_str, end_date_str, team_ballpark_map):
    if not team_ballpark_map: logger.error("Team to ballpark map is empty. Cannot map games."); return []
    logger.info(f"Fetching MLB schedule from {start_date_str} to {end_date_str}...")
    game_ballparks = set(); schedule_obj = None
    api_params = {'start_date': start_date_str, 'end_date': end_date_str, 'sportId': 1, 'gameTypes': ['R']}
    logger.debug(f"API call parameters: {api_params}")
    try:
        mlb = mlbstatsapi.Mlb(); schedule_obj = mlb.get_schedule(**api_params)
        if not schedule_obj or not hasattr(schedule_obj, 'dates') or not schedule_obj.dates: logger.warning(f"No schedule data returned or schedule object has no 'dates' for range {start_date_str} - {end_date_str}."); return []
        logger.info(f"Received schedule object with {len(schedule_obj.dates)} date entries for range {start_date_str} - {end_date_str}.")
        game_count, mapped_count = 0, 0; unmapped_teams = set(); first_game_date, last_game_date = None, None
        for daily_schedule in schedule_obj.dates:
            game_date = getattr(daily_schedule, 'date', None);
            if not game_date: continue
            if first_game_date is None: first_game_date = game_date
            last_game_date = game_date
            games_on_date = getattr(daily_schedule, 'games', []);
            if not games_on_date: continue
            for game in games_on_date:
                 home_team_id, home_team_name = None, 'N/A'
                 try: home_team_data = game.teams.home.team; home_team_id = getattr(home_team_data, 'id', None); home_team_name = getattr(home_team_data, 'name', 'N/A')
                 except AttributeError:
                      logger.debug(f"Attribute access failed for team data on {game_date}. Trying dict access...")
                      try: home_team_data = game.teams.get('home', {}).get('team', {}); home_team_id = home_team_data.get('id'); home_team_name = home_team_data.get('name', 'N/A')
                      except Exception as inner_e: logger.error(f"Failed to extract home team ID for game on {game_date} using dict method: {inner_e}")
                 game_count += 1
                 if home_team_id:
                    ballpark_name = team_ballpark_map.get(home_team_id)
                    if ballpark_name:
                        if ballpark_name in BALLPARK_COORDS: game_ballparks.add((game_date, ballpark_name)); mapped_count += 1
                        else:
                             if ballpark_name not in unmapped_teams: logger.warning(f"Mapped team ID {home_team_id} ({home_team_name}) to ballpark '{ballpark_name}', but coordinates not found in config.py."); unmapped_teams.add(ballpark_name)
                    else:
                        if home_team_id not in unmapped_teams: logger.warning(f"Could not map home team ID {home_team_id} ({home_team_name}) to a known ballpark using the database mapping."); unmapped_teams.add(home_team_id)
        logger.info(f"Processed {game_count} games, mapped {mapped_count} for date range {start_date_str} - {end_date_str}.")
        if first_game_date and last_game_date: logger.info(f"Date range covered in processed data: {first_game_date} to {last_game_date}")
        elif game_count > 0: logger.warning("Processed games but could not determine date range from data.")
        return list(game_ballparks)
    except AttributeError as ae: logger.error(f"AttributeError accessing schedule object for {start_date_str}-{end_date_str}: {ae}.", exc_info=True); return []
    except Exception as e: logger.error(f"Error fetching/processing schedule for {start_date_str}-{end_date_str}: {e}", exc_info=True); return []


# --- Weather Fetching Function --- [REVISED FUNCTION v4 - Added Retry Logic]
def fetch_weather_for_games(game_ballpark_list, ballpark_coords_map):
    """
    Fetches weather data from Open-Meteo for the required ballparks and dates.
    Uses the strategy of fetching full season ranges per park/year, then filtering.
    Correctly handles the end_date for the current year for the historical API.
    Includes retry logic for 429 Rate Limit errors.
    """
    if not game_ballpark_list:
        logger.warning("No game/ballpark combinations provided to fetch weather for.")
        return pd.DataFrame()

    parks_years = set()
    required_game_dates = set(game_ballpark_list)

    for game_date_str, ballpark_name in game_ballpark_list:
        year = int(game_date_str[:4])
        parks_years.add((ballpark_name, year))

    logger.info(f"Need weather data for {len(required_game_dates)} game dates across {len(parks_years)} park/year combinations.")

    all_weather_dfs = []
    api_call_count = 0
    current_year = datetime.now().year
    yesterday_date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    # Use base rate limit pause value from config, ensure it's float
    base_rate_limit_pause = float(DataConfig.RATE_LIMIT_PAUSE)
    logger.info(f"Using base rate limit pause of {base_rate_limit_pause} seconds between weather API calls.")

    # Loop through each unique park/year combination needed
    for ballpark_name, year in sorted(list(parks_years)):
        if ballpark_name not in ballpark_coords_map:
            logger.warning(f"Coordinates not found for '{ballpark_name}' in config. Skipping weather fetch for this park in {year}.")
            continue

        lat, lon = ballpark_coords_map[ballpark_name]

        # Determine correct start/end dates for the weather API call
        fetch_start_date = f"{year}-{SEASON_START_MONTH_DAY}"
        if year == current_year:
            fetch_end_date = yesterday_date_str
            if fetch_start_date > fetch_end_date:
                logger.warning(f"Start date {fetch_start_date} is after end date {fetch_end_date} for current year {year} at {ballpark_name}. Skipping weather fetch.")
                continue
        elif year > current_year:
             logger.warning(f"Year {year} is in the future. Skipping weather fetch for {ballpark_name}.")
             continue
        else:
            fetch_end_date = f"{year}-{SEASON_END_MONTH_DAY}"

        logger.info(f"Fetching weather for {ballpark_name} ({lat:.4f}, {lon:.4f}) for year {year} ({fetch_start_date} to {fetch_end_date})")

        params = {
            "latitude": lat, "longitude": lon, "start_date": fetch_start_date,
            "end_date": fetch_end_date, "daily": WEATHER_VARIABLES_STR, "timezone": "auto"
        }

        # --- Retry Loop ---
        retries = 0
        success = False
        while retries <= MAX_RETRIES and not success:
            # Apply the base delay *before* each attempt (even the first)
            # Avoids hammering the API right after the previous call or a retry wait.
            # Use a slightly shorter delay for non-retry calls if desired, but base delay is safer.
            actual_pause = base_rate_limit_pause * (2 ** retries) if retries > 0 else base_rate_limit_pause # Exponential backoff example
            # actual_pause = RETRY_WAIT_SECONDS * (retries + 1) if retries > 0 else base_rate_limit_pause # Linear backoff
            actual_pause = RETRY_WAIT_SECONDS if retries > 0 else base_rate_limit_pause # Fixed long retry wait

            # Add a standard pause between all requests regardless of retry status
            # Note: This means the first attempt waits base_rate_limit_pause seconds.
            # If base_rate_limit_pause is large (like 5s), this slows down the whole process.
            # Consider moving this sleep *after* a successful call or *only* in the retry logic.
            # Let's try putting the base pause *after* a successful call or failed attempt.

            if retries > 0:
                wait_time = RETRY_WAIT_SECONDS
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {retries}/{MAX_RETRIES} for {ballpark_name} {year}...")
                time.sleep(wait_time)
            else:
                 # Apply standard delay between non-retry requests
                 # Don't sleep before the very first request in the outer loop
                 if api_call_count > 0:
                      logger.debug(f"Pausing {base_rate_limit_pause} seconds before next request...")
                      time.sleep(base_rate_limit_pause)


            try:
                response = requests.get(OPENMETEO_API_URL, params=params, timeout=45)
                api_call_count += 1

                if response.status_code == 429:
                    # Explicitly handle 429 within the try block
                    logger.warning(f"Received 429 Too Many Requests (Retry {retries}/{MAX_RETRIES}) for {ballpark_name} {year}.")
                    retries += 1
                    # Let the loop handle the wait based on the incremented retries
                    continue # Go to next retry iteration

                response.raise_for_status() # Check for other errors (4xx, 5xx)
                data = response.json()

                if 'daily' in data and 'time' in data['daily']:
                    df = pd.DataFrame(data['daily'])
                    df['ballpark_name'] = ballpark_name
                    df['latitude'] = lat
                    df['longitude'] = lon
                    df = df.rename(columns={col: col.replace('-', '_') for col in df.columns})
                    if 'time' in df.columns: df = df.rename(columns={'time': 'game_date'})
                    all_weather_dfs.append(df)
                    logger.debug(f"  Successfully fetched {len(df)} days for {ballpark_name} in {year}.")
                    success = True # Mark as success to exit retry loop
                else:
                    logger.warning(f"  No 'daily' data in successful response for {ballpark_name} {year}.")
                    logger.debug(f"  Response: {data}")
                    success = True # Treat as success (valid response, just no data) to exit retry loop

            except requests.exceptions.HTTPError as http_err:
                 # Handle non-429 HTTP errors
                 logger.error(f"  HTTP error occurred fetching weather for {ballpark_name} {year}: {http_err}")
                 logger.error(f"  Failing URL: {response.url}")
                 break # Break inner retry loop, move to next park/year
            except requests.exceptions.Timeout:
                 logger.error(f"  Timeout fetching weather for {ballpark_name} {year}. (Retry {retries}/{MAX_RETRIES})")
                 retries += 1 # Count timeout as a retry attempt
            except requests.exceptions.RequestException as e:
                logger.error(f"  Request error occurred fetching weather for {ballpark_name} {year}: {e}")
                break # Break inner retry loop
            except Exception as e:
                 logger.error(f"  An unexpected error occurred processing weather for {ballpark_name} {year}: {e}", exc_info=True)
                 break # Break inner retry loop

        if not success:
            logger.error(f"Failed to fetch weather for {ballpark_name} {year} after {MAX_RETRIES} retries. Stopping fetch.")
            # Decide whether to stop entirely or just skip this park/year
            break # Stop the entire weather fetching process

    # --- End Retry Loop ---

    logger.info(f"Made {api_call_count} API calls to Open-Meteo.")
    if not all_weather_dfs:
        logger.warning("No weather data was successfully fetched (possibly due to errors or rate limits).")
        return pd.DataFrame()

    # --- Filtering and Timestamping remain the same ---
    combined_df = pd.concat(all_weather_dfs, ignore_index=True)
    logger.info(f"Fetched a total of {len(combined_df)} weather days before filtering.")
    if 'game_date' not in combined_df.columns:
        logger.error("Column 'game_date' not found after combining weather data. Cannot filter.")
        return pd.DataFrame()
    combined_df['filter_key'] = list(zip(combined_df['game_date'], combined_df['ballpark_name']))
    final_weather_df = combined_df[combined_df['filter_key'].isin(required_game_dates)].copy()
    final_weather_df = final_weather_df.drop(columns=['filter_key'])
    final_weather_df['fetch_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Filtered down to {len(final_weather_df)} weather records matching actual game dates.")
    return final_weather_df


# --- Main Execution ---
# (main function remains the same - code omitted for brevity)
def main():
    if not MODULE_IMPORTS_OK:
        sys.exit("Exiting due to missing module imports.")

    parser = argparse.ArgumentParser(description="Fetch MLB game weather data.")
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to fetch data for (YYYY-MM-DD). If omitted, fetches full history.",
    )
    args = parser.parse_args()

    db_path = project_root / DBConfig.PATH
    team_ballpark_map = load_team_ballpark_map_from_db(db_path)
    if not team_ballpark_map:
        logger.error("Failed to load team-ballpark mapping from database. Exiting.")
        sys.exit(1)
    ballpark_coords = BALLPARK_COORDS

    if args.date:
        # Single Date Mode
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y-%m-%d')
            logger.info(f"Running in single date mode for: {target_date}")
        except ValueError:
            logger.error(f"Invalid date format: '{args.date}'. Please use YYYY-MM-DD.")
            sys.exit(1)

        if check_date_in_db(db_path, target_date):
            logger.info(f"Weather data for {target_date} already exists in the database. Nothing to do.")
        else:
            logger.info(f"Data for {target_date} not found in DB. Fetching...")
            game_ballparks_today = get_mlb_schedule(target_date, target_date, team_ballpark_map)
            if game_ballparks_today:
                weather_df_today = fetch_weather_for_games(game_ballparks_today, ballpark_coords)
                init_db(db_path)
                save_data_to_db(db_path, weather_df_today, mode='append')
            else:
                logger.info(f"No regular season games found scheduled for {target_date}.")
    else:
        # Full History Mode
        logger.info("Running in full historical mode.")
        years_to_fetch = [year for year in DataConfig.SEASONS if year != 2020]
        if not years_to_fetch:
            logger.error("No valid years found in DataConfig.SEASONS to fetch.")
            sys.exit(1)

        all_game_ballparks = []
        today = datetime.now()
        yesterday = today - timedelta(days=1)

        for year in sorted(years_to_fetch):
            logger.info(f"--- Fetching schedule for year: {year} ---")
            start_date_str = f"{year}-01-01"
            if year == yesterday.year:
                end_date_str = yesterday.strftime('%Y-%m-%d')
                if start_date_str > end_date_str:
                     logger.warning(f"Start date {start_date_str} is after end date {end_date_str} for current year {year}. Skipping schedule fetch for this year.")
                     continue
            elif year > yesterday.year:
                 logger.warning(f"Year {year} is in the future. Skipping schedule fetch.")
                 continue
            else:
                end_date_str = f"{year}-12-31"

            logger.info(f"Requesting schedule from {start_date_str} to {end_date_str}")
            yearly_game_ballparks = get_mlb_schedule(start_date_str, end_date_str, team_ballpark_map)
            all_game_ballparks.extend(yearly_game_ballparks)
            logger.info(f"Fetched {len(yearly_game_ballparks)} game dates for {year}. Total unique found so far: {len(set(all_game_ballparks))}")
            time.sleep(0.5) # Delay between yearly schedule fetches

        unique_game_ballparks = sorted(list(set(all_game_ballparks)))
        logger.info(f"Total unique game/ballpark combinations found across all years: {len(unique_game_ballparks)}")

        if unique_game_ballparks:
            full_weather_df = fetch_weather_for_games(unique_game_ballparks, ballpark_coords)
            init_db(db_path)
            if not full_weather_df.empty:
                 logger.info("Proceeding to save fetched weather data.")
                 save_data_to_db(db_path, full_weather_df, mode='replace')
            else:
                 logger.warning("Weather data fetching did not complete successfully or returned no data. Database will not be updated/overwritten.")
        else:
            logger.warning("No regular season games found for the entire historical period after checking all years.")

    logger.info("Fetch weather script finished.")


if __name__ == "__main__":
    main()