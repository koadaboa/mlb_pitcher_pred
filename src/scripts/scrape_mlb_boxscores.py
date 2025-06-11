# src/scripts/scrape_mlb_boxscores.py
# Fetches box score data and stores results in the ``mlb_boxscores`` table.

import httpx
# Removed BeautifulSoup import as it's no longer needed for get_game_pks
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import re
import argparse
import os
import asyncio
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.config import LogConfig, FileConfig
    from src.utils import (
        setup_logger,
        ensure_dir,
        DBConnection,
        table_exists,
        get_latest_date,
    )
    MODULE_IMPORTS_OK = True
    # Define DB_PATH from config if available, needed for fallback DBConnection path
    try: from src.config import DBConfig; DB_PATH = DBConfig.PATH
    except (ImportError, AttributeError): DB_PATH = str(project_root / 'data' / 'pitcher_stats.db') # Fallback if DBConfig not present
except ImportError as e:
    print(f"ERROR: Failed to import required modules from src: {e}. Using fallback logging/config.")
    MODULE_IMPORTS_OK = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('scrape_mlb_boxscores_fallback')
    class FileConfig:
        DATA_DIR = project_root / 'data'
        DEBUG_DIR = DATA_DIR / 'debug_api' / 'mlb_boxscores'
    class LogConfig:
        LOG_DIR = project_root / 'logs'
    DB_PATH = str(project_root / 'data' / 'pitcher_stats.db')
    def setup_logger(name, log_file=None, level=logging.INFO): return logger
    def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)
    class DBConnection:
        def __init__(self, db_path=None):
            self.db_path = Path(db_path or DB_PATH)
        def __enter__(self):
            import sqlite3
            self.conn = sqlite3.connect(str(self.db_path))
            return self.conn
        def __exit__(self, exc_type, exc, tb):
            if self.conn:
                if exc_type:
                    self.conn.rollback()
                else:
                    self.conn.commit()
                self.conn.close()
    def table_exists(conn, table):
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None
    def get_latest_date(conn, table, date_col="game_date"):
        if not table_exists(conn, table):
            return None
        cur = conn.execute(f"SELECT MAX({date_col}) FROM {table}")
        row = cur.fetchone()
        if row and row[0] is not None:
            return pd.to_datetime(row[0])
        return None


# --- Constants ---
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1" # Base API path
MLB_SCHEDULE_ENDPOINT = MLB_STATS_API_BASE + "/schedule"
MLB_GAME_ENDPOINT_FORMAT = MLB_STATS_API_BASE + ".1/game/{game_pk}/feed/live" # v1.1 for live feed

API_HEADERS = {
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36', # Keep UA
     'Accept': 'application/json',
}
MAX_RETRIES = 4
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 10
REQUEST_TIMEOUT = 20

# Offseason Bounds & League IDs
OFFSEASON_START_MONTH = 11; OFFSEASON_START_DAY = 5
OFFSEASON_END_MONTH = 3; OFFSEASON_END_DAY = 7
MLB_LEAGUE_IDS = {103, 104} # AL, NL

# --- Sets to track issues ---
DATES_WITH_ISSUES = set()
PROBLEMATIC_GAME_PKS = set()

# --- Setup Logger ---
log_dir = LogConfig.LOG_DIR if MODULE_IMPORTS_OK else project_root / 'logs'
log_file = log_dir / 'scrape_mlb_boxscores_api.log'
ensure_dir(log_dir)
logger = setup_logger('scrape_mlb_boxscores_api_csv', log_file=log_file, level=logging.INFO)

# --- Define Output and Debug Directories ---
OUTPUT_DIR = FileConfig.DATA_DIR / 'raw' if MODULE_IMPORTS_OK else project_root / 'data' / 'raw'
DEBUG_API_DIR = FileConfig.DEBUG_DIR if hasattr(FileConfig, 'DEBUG_DIR') else OUTPUT_DIR.parent / 'debug_api' / 'mlb_boxscores'
ensure_dir(OUTPUT_DIR)
# Debug dir created later if needed

# --- Retry Logic ---
RETRY_EXCEPTIONS = (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)
def is_retryable_exception(exception):
    if isinstance(exception, httpx.HTTPStatusError): return 500 <= exception.response.status_code < 600
    if isinstance(exception, httpx.RequestError) and hasattr(exception, 'response') and exception.response and exception.response.status_code == 404: return False
    return isinstance(exception, (httpx.TimeoutException, httpx.RequestError))

retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY),
    retry=is_retryable_exception,
    before_sleep=lambda rs: logger.warning(f"Retrying ({rs.attempt_number}/{MAX_RETRIES}): {rs.outcome.exception()}. Waiting {rs.next_action.sleep:.2f}s...")
)

# --- Database Utility Functions ---
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS mlb_boxscores (
    game_pk INTEGER PRIMARY KEY,
    game_date TEXT,
    away_team TEXT,
    home_team TEXT,
    game_number INTEGER,
    double_header TEXT,
    away_pitcher_ids TEXT,
    home_pitcher_ids TEXT,
    hp_umpire TEXT,
    "1b_umpire" TEXT,
    "2b_umpire" TEXT,
    "3b_umpire" TEXT,
    weather TEXT,
    temp REAL,
    wind TEXT,
    elevation REAL,
    dayNight TEXT,
    first_pitch TEXT,
    scraped_timestamp TEXT
)
"""

def ensure_boxscores_table(db_path=DB_PATH):
    """Create the ``mlb_boxscores`` table if it doesn't exist."""
    with DBConnection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)

def get_latest_boxscore_date(db_path=DB_PATH):
    """Return the latest ``game_date`` in ``mlb_boxscores`` if the table exists."""
    with DBConnection(db_path) as conn:
        latest = get_latest_date(conn, "mlb_boxscores", "game_date")
        if latest is not None:
            return latest.strftime("%Y-%m-%d")
    return None

def get_earliest_boxscore_date(db_path=DB_PATH):
    """Return the earliest ``game_date`` in ``mlb_boxscores`` if the table exists."""
    with DBConnection(db_path) as conn:
        if not table_exists(conn, "mlb_boxscores"):
            return None
        cur = conn.execute("SELECT MIN(game_date) FROM mlb_boxscores")
        row = cur.fetchone()
        if row and row[0] is not None:
            return pd.to_datetime(row[0]).strftime("%Y-%m-%d")
    return None

def load_existing_game_pks(db_path=DB_PATH):
    """Return a set of all ``game_pk`` values currently stored."""
    with DBConnection(db_path) as conn:
        if not table_exists(conn, "mlb_boxscores"):
            return set()
        df = pd.read_sql_query("SELECT game_pk FROM mlb_boxscores", conn)
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df.dropna(subset=["game_pk"], inplace=True)
    return set(df["game_pk"].astype(int).tolist())
# --- Core Fetching/Parsing Functions ---

# Removed fetch_html_url as it's no longer needed

@retry_decorator
async def fetch_api_data_direct(client, url, params=None): # Add params optional arg
    logger.debug(f"Fetching API URL: {url} with params: {params}")
    # Pass params to the get request if provided
    response = await client.get(url, timeout=REQUEST_TIMEOUT, params=params)
    if response.status_code == 404: logger.warning(f"API 404 Not Found: {url}"); return None
    elif response.status_code >= 400: logger.error(f"API error {response.status_code} for {url}"); response.raise_for_status()
    logger.debug(f"Successfully fetched API {url}"); return response.json()

# Modified fetch_api_data_wrapper to accept optional params
async def fetch_api_data_wrapper(client, url, params=None): # Add params optional arg
    game_pk_match = re.search(r'/game/(\d+)/', url)
    # Handle schedule URL lacking game_pk differently for logging
    context_id = game_pk_match.group(1) if game_pk_match else f"schedule_{params.get('startDate', 'unknown_date')}" if params else "unknown_context"
    try:
        # Pass params down to the direct fetch function
        result = await fetch_api_data_direct(client, url, params=params)
        if result is None:
             logger.warning(f"API fetch failed definitively for {context_id} after retries.")
             if game_pk_match: # Only add actual game pks to problematic list on fetch failure
                  PROBLEMATIC_GAME_PKS.add(context_id)
        return result
    except Exception as e:
        logger.error(f"API call failed definitively for {url} (context: {context_id}) after retries: {e}")
        if game_pk_match: # Only add actual game pks to problematic list on fetch failure
             PROBLEMATIC_GAME_PKS.add(context_id)
        return None

# Corrected get_game_pks definition - needs client
async def get_game_pks(client, date_str): # <<< Added 'client' back
    """ Gets gamePks for a date using the MLB Stats API schedule endpoint. """
    logger.info(f"Fetching schedule from API for gamePks: {date_str}")
    params = {
        "sportId": 1,
        "startDate": date_str,
        "endDate": date_str,
        "fields": "dates,games,gamePk" # Request only needed fields
    }
    game_pks = []
    try:
        # Use the passed-in client to call the schedule endpoint via the wrapper
        response = await fetch_api_data_wrapper(client, MLB_SCHEDULE_ENDPOINT, params=params) # <<< Pass params here

        if response and 'dates' in response and response['dates']:
            for date_info in response['dates']:
                if 'games' in date_info:
                    for game in date_info['games']:
                        pk = game.get('gamePk')
                        if pk and isinstance(pk, int):
                            if pk not in game_pks: game_pks.append(pk)
                        else: logger.warning(f"Found game entry without valid gamePk for {date_str}: {game}")

        if not game_pks and response is not None: # Check if response existed but no games found
            logger.info(f"No games found via API schedule for {date_str}.")
        elif not game_pks: # Fetch itself likely failed
             logger.warning(f"Failed to fetch or parse schedule for {date_str}, no gamePks obtained.")
             DATES_WITH_ISSUES.add(date_str) # Flag date if schedule fetch fails

        logger.info(f"Extracted {len(game_pks)} unique gamePks via API for {date_str}.")
        return game_pks

    except Exception as e:
        logger.error(f"Unexpected error in get_game_pks for {date_str}: {e}", exc_info=True)
        DATES_WITH_ISSUES.add(date_str)
        return []


def parse_api_data(api_response, game_pk):
    """Parse a single game feed response from the MLB Stats API.

    The MLB API occasionally omits portions of the ``gameData`` or ``liveData``
    trees.  To avoid ``KeyError``/``TypeError`` exceptions we validate the
    presence and type of each nested structure before use.
    """
    try:
        if not isinstance(api_response, dict):
            logger.error(f"Unexpected API response type for gamePk {game_pk}: {type(api_response)}")
            return None

        game_data = api_response.get("gameData") or {}
        live_data = api_response.get("liveData") or {}

        if not isinstance(game_data, dict) or not isinstance(live_data, dict):
            logger.error(f"Malformed gameData/liveData for gamePk {game_pk}")
            return None

        game_info = game_data.get("game") or {}
        venue_info = game_data.get("venue") or {}
        boxscore = live_data.get("boxscore") or {}
        officials = boxscore.get("officials") or []
        teams_box = boxscore.get("teams") or {}
        weather = game_data.get("weather") or {}
        datetime_info = game_data.get("datetime") or {}
        teams_info = game_data.get("teams") or {}
        status = game_data.get("status") or {}

        if status.get("abstractGameState") not in ["Final", "Game Over"]:
            logger.warning(f"Game {game_pk} not final. Skipping.")
            return None

        # AL/NL League Check
        away_league_id = teams_info.get("away", {}).get("league", {}).get("id")
        home_league_id = teams_info.get("home", {}).get("league", {}).get("id")
        if not (away_league_id in MLB_LEAGUE_IDS and home_league_id in MLB_LEAGUE_IDS):
            logger.info(f"Skipping game {game_pk}: Non AL/NL matchup (Leagues: {away_league_id}, {home_league_id}).")
            return None

        data = {"game_pk": game_pk}
        cal_event_id = game_info.get("calendarEventID", "")
        date_parts = cal_event_id.split("-")
        if len(date_parts) >= 3 and all(p.isdigit() for p in date_parts[-3:]):
            data["game_date"] = "-".join(date_parts[-3:])
        else:
            data["game_date"] = game_info.get("gameDate")

        data["away_team"] = teams_info.get("away", {}).get("abbreviation")
        data["home_team"] = teams_info.get("home", {}).get("abbreviation")

        away_pitchers_list = teams_box.get("away", {}).get("pitchers", [])
        home_pitchers_list = teams_box.get("home", {}).get("pitchers", [])
        away_pitchers_list = away_pitchers_list if isinstance(away_pitchers_list, list) else []
        home_pitchers_list = home_pitchers_list if isinstance(home_pitchers_list, list) else []
        data["away_pitcher_ids"] = json.dumps(away_pitchers_list)
        data["home_pitcher_ids"] = json.dumps(home_pitchers_list)

        # Add gameNumber, doubleHeader
        data["game_number"] = game_info.get("gameNumber")
        data["double_header"] = game_info.get("doubleHeader")

        data["hp_umpire"], data["1b_umpire"], data["2b_umpire"], data["3b_umpire"] = None, None, None, None
        if isinstance(officials, list):
            for official in officials:
                if isinstance(official, dict):
                    otype = official.get("officialType")
                    name = official.get("official", {}).get("fullName")
                    if otype == "Home Plate":
                        data["hp_umpire"] = name
                    elif otype == "First Base":
                        data["1b_umpire"] = name
                    elif otype == "Second Base":
                        data["2b_umpire"] = name
                    elif otype == "Third Base":
                        data["3b_umpire"] = name

        # Add temp, dayNight, elevation
        data["weather"] = weather.get("condition")
        data["wind"] = weather.get("wind")
        data["temp"] = weather.get("temp")
        data["dayNight"] = datetime_info.get("dayNight")
        data["elevation"] = venue_info.get("location", {}).get("elevation")

        fp_dt_str = datetime_info.get("firstPitch") or datetime_info.get("time")
        if fp_dt_str:
            try:
                fp_dt = pd.to_datetime(fp_dt_str)
                data["first_pitch"] = fp_dt.strftime("%I:%M %p")
            except Exception:
                data["first_pitch"] = fp_dt_str
        else:
            data["first_pitch"] = None

        data["scraped_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not data.get("away_team") or not data.get("home_team") or not data.get("game_date"):
            logger.error(f"Missing essential data parsing gamePk {game_pk}. Skipping.")
            return None

        logger.debug(f"Parsed API data for gamePk {game_pk}")
        return data
    except Exception as e:
        logger.error(f"Error parsing API response for gamePk {game_pk}: {e}", exc_info=True)
        return None

def save_debug_json(api_response, date_str, game_pk):
    ensure_dir(DEBUG_API_DIR)
    filepath = DEBUG_API_DIR / f"api_fail_{date_str}_{game_pk}.json"
    try:
        with open(filepath, 'w', encoding='utf-8') as f: json.dump(api_response, f, indent=2)
        logger.info(f"Saved debug JSON for failed parse: {filepath.name}")
    except TypeError as te: logger.error(f"Data for gamePk {game_pk} not JSON serializable: {te}")
    except Exception as e: logger.error(f"Failed to save debug JSON to {filepath}: {e}")

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1): yield start_date + timedelta(n)

# --- Main Execution ---
async def main(start_date_str, end_date_str, debug_api):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Ensure DB table exists and determine last processed date
    logger.info(f"Using SQLite DB at: {DB_PATH}")
    ensure_boxscores_table(DB_PATH)

    last_processed_date_str = get_latest_boxscore_date(DB_PATH)
    earliest_date_str = get_earliest_boxscore_date(DB_PATH)
    if earliest_date_str:
        logger.info(f"Earliest game_date in database: {earliest_date_str}")
    if last_processed_date_str:
        logger.info(f"Latest game_date in database: {last_processed_date_str}")
    if not last_processed_date_str and not earliest_date_str:
        logger.info("No records found in mlb_boxscores table; starting from provided start date")

    existing_pks = load_existing_game_pks(DB_PATH)
    logger.info(f"Found {len(existing_pks)} existing gamePks in database.")

    effective_start_date = start_date
    logger.info(f"Processing date range {effective_start_date} to {end_date}")

    # Use API_HEADERS for the main client
    async with httpx.AsyncClient(headers=API_HEADERS, http2=True, timeout=REQUEST_TIMEOUT + 5) as client:
        for current_dt in daterange(effective_start_date, end_date):
            date_str = current_dt.strftime("%Y-%m-%d")
            current_year = current_dt.year; current_month = current_dt.month; current_day = current_dt.day

            if current_year == 2020: logger.info(f"Skipping date {date_str} (Year 2020)."); continue
            is_offseason = False
            if (current_month > OFFSEASON_START_MONTH or (current_month == OFFSEASON_START_MONTH and current_day >= OFFSEASON_START_DAY)) or \
               (current_month < OFFSEASON_END_MONTH or (current_month == OFFSEASON_END_MONTH and current_day < OFFSEASON_END_DAY)): is_offseason = True
            if is_offseason: logger.info(f"Date {date_str} is offseason. Skipping."); continue

            logger.info(f"--- Processing Date: {date_str} ---")
            # --- <<< MODIFIED: Call API schedule endpoint >>> ---
            game_pks = await get_game_pks(client, date_str) # Pass client now

            if not game_pks:
                logger.warning(f"No gamePks found via API schedule for {date_str}.")
                continue

            new_games_for_date = []; processed_pks_in_batch = set(); tasks = []; pks_to_fetch = []
            for pk in game_pks:
                 if pk not in existing_pks:
                      pks_to_fetch.append(pk)
                      api_url = MLB_GAME_ENDPOINT_FORMAT.format(game_pk=pk) # Use correct format string
                      tasks.append(asyncio.ensure_future(fetch_api_data_wrapper(client, api_url))) # Pass client
                 else: logger.debug(f"GamePk {pk} already exists. Skipping API call.")

            if not tasks: logger.info(f"All gamePks for {date_str} already exist."); await asyncio.sleep(0.1); continue

            api_results = await asyncio.gather(*tasks)

            fetch_or_parse_issue_occurred = False
            for i, api_response in enumerate(api_results):
                game_pk = pks_to_fetch[i]
                if game_pk in processed_pks_in_batch: continue
                if api_response is None: fetch_or_parse_issue_occurred = True; continue # PROBLEMATIC_GAME_PKS added in wrapper
                try:
                    parsed_data = parse_api_data(api_response, game_pk)
                    if parsed_data:
                         if parsed_data['game_pk'] not in existing_pks:
                             new_games_for_date.append(parsed_data)
                             existing_pks.add(parsed_data['game_pk'])
                             processed_pks_in_batch.add(parsed_data['game_pk'])
                         else: logger.warning(f"Parsed game {game_pk} but already in existing_pks set.")
                    else: # Parsing returned None (filtered out non-final or non-AL/NL)
                         logger.info(f"Parsing returned None for gamePk {game_pk} (likely filtered).")
                         if debug_api: save_debug_json(api_response, date_str, game_pk)
                         fetch_or_parse_issue_occurred = True # Count filter/parse fail as issue
                except Exception as parse_err:
                     logger.error(f"Unexpected error processing result for gamePk {game_pk}: {parse_err}", exc_info=True)
                     PROBLEMATIC_GAME_PKS.add(game_pk); fetch_or_parse_issue_occurred = True
                     if debug_api and api_response: save_debug_json(api_response, date_str, game_pk)

            # --- Save new games directly to the database ---
            date_save_successful = False
            if new_games_for_date:
                try:
                    df_new = pd.DataFrame(new_games_for_date)
                    if not df_new.empty:
                        ensure_boxscores_table(DB_PATH)
                        with DBConnection(DB_PATH) as conn:
                            df_new.to_sql("mlb_boxscores", conn, if_exists="append", index=False)
                    processed_count = len(new_games_for_date)
                    logger.info(f"Inserted {processed_count} new games for {date_str} into mlb_boxscores.")
                    print(f"Finished processing {date_str}. Added {processed_count} new games.")
                    date_save_successful = True
                except Exception as e:
                    logger.error(f"Error saving games for {date_str} to database: {e}", exc_info=True)
                    print(f"ERROR saving data for {date_str}. Check logs.")
                    DATES_WITH_ISSUES.add(date_str)

            # --- Update issue tracking ---
            if not date_save_successful and fetch_or_parse_issue_occurred and pks_to_fetch:
                 logger.warning(f"Date {date_str} had fetch/parse issues for some attempted gamePks. Adding to issue list.")
                 DATES_WITH_ISSUES.add(date_str)

            await asyncio.sleep(0.5 + random.uniform(0, 0.5)) # Delay between dates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLB box score data via API for a date range and store in SQLite.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=False, help="End date in YYYY-MM-DD format (inclusive).")
    parser.add_argument("--debug-api", action='store_true', help="Save API JSON if parsing fails.")

    args = parser.parse_args()
    try:
        start_dt_arg = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt_str = args.end_date if args.end_date else args.start_date
        end_dt_arg = datetime.strptime(end_dt_str, "%Y-%m-%d")
        if end_dt_arg < start_dt_arg: raise ValueError("End date cannot be before start date.")
    except ValueError as e: print(f"ERROR: Invalid date: {e}. Use YYYY-MM-DD."); sys.exit(1)

    logger.info(f"--- Starting MLB Box Score API Fetcher: {args.start_date} to {end_dt_str} ---")
    logger.info(f"Debug API JSON saving {'enabled' if args.debug_api else 'disabled'}.")

    try:
        asyncio.run(main(args.start_date, end_dt_str, args.debug_api))
    except Exception as e:
        logger.critical(f"Unhandled error during execution: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Script stopped. Check logs: {log_file}")
    finally:
        # --- Report Problematic Items ---
        if DATES_WITH_ISSUES:
            sorted_dates = sorted(list(DATES_WITH_ISSUES)); logger.warning(f"Processing finished. Issues on dates: {', '.join(sorted_dates)}")
            print("\n--- Dates with Potential Issues ---"); [print(d) for d in sorted_dates]
            try:
                 issue_file_d = project_root / 'problematic_scrape_dates.txt'
                 with open(issue_file_d, 'w') as f: f.write("# Dates with errors/warnings during MLB boxscore API fetch\n"); [f.write(f"{d}\n") for d in sorted_dates]
                 logger.info(f"Problematic dates saved to {issue_file_d}"); print(f"\nProblematic dates saved to: {issue_file_d}")
            except Exception as e: logger.error(f"Failed to save problematic dates file: {e}")
        if PROBLEMATIC_GAME_PKS:
             sorted_pks = sorted([int(pk) for pk in PROBLEMATIC_GAME_PKS if isinstance(pk, (int, str)) and str(pk).isdigit()])
             logger.warning(f"Processing finished. Fetch/parse failed for gamePks: {', '.join(map(str, sorted_pks))}")
             print("\n--- Game Pks with Fetch/Parse Issues ---"); [print(p) for p in sorted_pks]
             try:
                  issue_file_pk = project_root / 'problematic_gamepks.txt'
                  with open(issue_file_pk, 'w') as f: f.write("# GamePks that failed API fetch or parsing\n"); [f.write(f"{p}\n") for p in sorted_pks]
                  logger.info(f"Problematic gamePks saved to {issue_file_pk}"); print(f"\nProblematic gamePks saved to: {issue_file_pk}")
             except Exception as e: logger.error(f"Failed to save problematic gamePks file: {e}")
        if not DATES_WITH_ISSUES and not PROBLEMATIC_GAME_PKS: logger.info("Processing finished. No dates or gamePks flagged with critical issues.")
        # --- END Report ---
        logger.info(f"--- API Fetcher finished for Date Range: {args.start_date} to {end_dt_str} ---")