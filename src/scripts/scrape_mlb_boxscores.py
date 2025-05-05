# src/scripts/scrape_mlb_boxscores.py
# (CSV Output Version with API Schedule, AL/NL Filter, New Columns)

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
    from src.data.utils import setup_logger, ensure_dir
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
OUTPUT_CSV_FILE = "mlb_boxscores_combined.csv"
CHECKPOINT_FILE = "mlb_boxscores_checkpoint.txt"

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
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_CSV_FILE
CHECKPOINT_DIR = OUTPUT_DIR.parent / '.checkpoints' if MODULE_IMPORTS_OK else project_root / 'data' / '.checkpoints'
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_FILE
DEBUG_API_DIR = FileConfig.DEBUG_DIR if hasattr(FileConfig, 'DEBUG_DIR') else OUTPUT_DIR.parent / 'debug_api' / 'mlb_boxscores'
ensure_dir(OUTPUT_DIR)
ensure_dir(CHECKPOINT_PATH.parent)
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

# --- Checkpoint Functions ---
def load_last_processed_date(filename=CHECKPOINT_PATH):
    if filename.exists():
        try:
            last_date_str = filename.read_text().strip(); datetime.strptime(last_date_str, "%Y-%m-%d")
            logger.info(f"Checkpoint found. Last successfully processed date: {last_date_str}"); return last_date_str
        except Exception as e: logger.warning(f"Could not read/validate checkpoint '{filename}': {e}"); return None
    return None

def save_checkpoint(date_str, filename=CHECKPOINT_PATH):
    try: ensure_dir(filename.parent); filename.write_text(date_str); logger.info(f"Checkpoint saved for date: {date_str} to {filename}")
    except Exception as e: logger.error(f"Could not write checkpoint file '{filename}': {e}")

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
    # TODO: Thoroughly verify all JSON paths below
    try:
        game_data = api_response.get('gameData', {}); live_data = api_response.get('liveData', {})
        game_info = game_data.get('game', {}); venue_info = game_data.get('venue', {})
        boxscore = live_data.get('boxscore', {}); officials = boxscore.get('officials', [])
        teams_box = boxscore.get('teams', {}); weather = game_data.get('weather', {})
        datetime_info = game_data.get('datetime', {}); teams_info = game_data.get('teams', {})
        status = game_data.get('status', {})
        if status.get('abstractGameState') not in ['Final', 'Game Over']: logger.warning(f"Game {game_pk} not final. Skipping."); return None

        # AL/NL League Check
        away_league_id = teams_info.get('away', {}).get('league', {}).get('id')
        home_league_id = teams_info.get('home', {}).get('league', {}).get('id')
        if not (away_league_id in MLB_LEAGUE_IDS and home_league_id in MLB_LEAGUE_IDS):
            logger.info(f"Skipping game {game_pk}: Non AL/NL matchup (Leagues: {away_league_id}, {home_league_id}).")
            return None

        data = {'game_pk': game_pk}
        cal_event_id = game_info.get('calendarEventID', ''); date_parts = cal_event_id.split('-')
        if len(date_parts) >= 3 and all(p.isdigit() for p in date_parts[-3:]): data['game_date'] = "-".join(date_parts[-3:])
        else: data['game_date'] = game_info.get('gameDate')
        data['away_team'] = teams_info.get('away', {}).get('abbreviation'); data['home_team'] = teams_info.get('home', {}).get('abbreviation')
        away_pitchers_list = teams_box.get('away', {}).get('pitchers', []); home_pitchers_list = teams_box.get('home', {}).get('pitchers', [])
        data['away_pitcher_ids'] = json.dumps(away_pitchers_list if isinstance(away_pitchers_list, list) else []); data['home_pitcher_ids'] = json.dumps(home_pitchers_list if isinstance(home_pitchers_list, list) else [])

        # Add gameNumber, doubleHeader
        data['game_number'] = game_info.get('gameNumber')
        data['double_header'] = game_info.get('doubleHeader')

        data['hp_umpire'], data['1b_umpire'], data['2b_umpire'], data['3b_umpire'] = None, None, None, None
        if isinstance(officials, list):
            for official in officials:
                if isinstance(official, dict):
                    otype = official.get('officialType'); name = official.get('official', {}).get('fullName')
                    if otype == 'Home Plate': data['hp_umpire'] = name
                    elif otype == 'First Base': data['1b_umpire'] = name
                    elif otype == 'Second Base': data['2b_umpire'] = name
                    elif otype == 'Third Base': data['3b_umpire'] = name

        # Add temp, dayNight, elevation
        data['weather'] = weather.get('condition')
        data['wind'] = weather.get('wind')
        data['temp'] = weather.get('temp')
        data['dayNight'] = datetime_info.get('dayNight')
        data['elevation'] = venue_info.get('location', {}).get('elevation')

        fp_dt_str = datetime_info.get('firstPitch') or datetime_info.get('time')
        if fp_dt_str:
             try: fp_dt = pd.to_datetime(fp_dt_str); data['first_pitch'] = fp_dt.strftime('%I:%M %p')
             except Exception: data['first_pitch'] = fp_dt_str
        else: data['first_pitch'] = None
        data['scraped_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not data.get('away_team') or not data.get('home_team') or not data.get('game_date'): logger.error(f"Missing essential data parsing gamePk {game_pk}. Skipping."); return None
        logger.debug(f"Parsed API data for gamePk {game_pk}"); return data
    except Exception as e: logger.error(f"Error parsing API response for gamePk {game_pk}: {e}", exc_info=True); return None

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

    # Load existing data and checkpoint
    all_game_data = []
    existing_pks = set()
    last_processed_date_str = load_last_processed_date()
    resume_from_dt = start_date

    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        logger.info(f"Loading existing data from {OUTPUT_PATH}...")
        try:
            all_game_data_df = pd.read_csv(OUTPUT_PATH, low_memory=False)
            all_game_data_df['game_pk'] = pd.to_numeric(all_game_data_df['game_pk'], errors='coerce').astype('Int64')
            all_game_data_df.dropna(subset=['game_pk'], inplace=True)
            existing_pks = set(all_game_data_df['game_pk'].unique())
            all_game_data = all_game_data_df.to_dict('records')
            logger.info(f"Loaded {len(all_game_data)} existing records with {len(existing_pks)} unique gamePks.")
            if not last_processed_date_str and 'game_date' in all_game_data_df.columns and not all_game_data_df.empty:
                 max_date_in_csv = pd.to_datetime(all_game_data_df['game_date'], errors='coerce').max()
                 if pd.notna(max_date_in_csv):
                      last_processed_date_str = max_date_in_csv.strftime('%Y-%m-%d')
                      logger.info(f"Using max date from CSV ({last_processed_date_str}) as last processed date.")
        except Exception as e: logger.error(f"Error loading existing CSV {OUTPUT_PATH}: {e}. Starting fresh.", exc_info=True); all_game_data = []; existing_pks = set(); last_processed_date_str = None

    if last_processed_date_str:
        try:
            last_processed_dt = datetime.strptime(last_processed_date_str, "%Y-%m-%d").date()
            resume_from_dt = last_processed_dt + timedelta(days=1)
        except ValueError: logger.warning(f"Invalid date in checkpoint '{last_processed_date_str}'."); resume_from_dt = start_date

    effective_start_date = max(start_date, resume_from_dt)
    logger.info(f"Effective processing start date: {effective_start_date.strftime('%Y-%m-%d')}")
    if effective_start_date > end_date: logger.info("Effective start date is after end date."); return

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
                logger.warning(f"No gamePks found via API schedule for {date_str}. Saving checkpoint.")
                save_checkpoint(date_str); continue

            new_games_for_date = []; processed_pks_in_batch = set(); tasks = []; pks_to_fetch = []
            for pk in game_pks:
                 if pk not in existing_pks:
                      pks_to_fetch.append(pk)
                      api_url = MLB_GAME_ENDPOINT_FORMAT.format(game_pk=pk) # Use correct format string
                      tasks.append(asyncio.ensure_future(fetch_api_data_wrapper(client, api_url))) # Pass client
                 else: logger.debug(f"GamePk {pk} already exists. Skipping API call.")

            if not tasks: logger.info(f"All gamePks for {date_str} already exist. Saving checkpoint."); save_checkpoint(date_str); await asyncio.sleep(0.1); continue

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

            # --- Append new data and save entire CSV ---
            date_save_successful = False
            if new_games_for_date:
                 try:
                      all_game_data.extend(new_games_for_date)
                      df_all = pd.DataFrame(all_game_data)
                      # --- <<< MODIFIED: Added new columns to order >>> ---
                      cols_order = [
                           'game_pk', 'game_date', 'away_team', 'home_team',
                           'game_number', 'double_header', # Added new columns
                           'away_pitcher_ids', 'home_pitcher_ids',
                           'hp_umpire', '1b_umpire', '2b_umpire', '3b_umpire',
                           'weather', 'temp', 'wind', # Added temp
                           'elevation', # Added elevation
                           'dayNight', # Added dayNight
                           'first_pitch', 'scraped_timestamp'
                      ]
                      # --- <<< END MODIFIED >>> ---
                      for col in cols_order:
                           if col not in df_all.columns: df_all[col] = None
                      df_all = df_all[cols_order]
                      df_all.drop_duplicates(subset=['game_pk'], keep='last', inplace=True)
                      df_all.sort_values(by=['game_date', 'game_pk'], inplace=True) # Sort before saving
                      df_all.to_csv(OUTPUT_PATH, index=False)
                      processed_count = len(new_games_for_date)
                      logger.info(f"Appended {processed_count} new games for {date_str}. Saved {len(df_all)} total games to {OUTPUT_PATH.name}.")
                      print(f"Finished processing {date_str}. Added {processed_count} new games. Total saved: {len(df_all)}.")
                      date_save_successful = True
                 except Exception as e:
                      logger.error(f"Error saving combined data for {date_str} to {OUTPUT_PATH}: {e}", exc_info=True)
                      print(f"ERROR saving data for {date_str} to CSV. Check logs.")
                      DATES_WITH_ISSUES.add(date_str)

            # --- Update Checkpoint and Issue List ---
            if date_save_successful:
                 save_checkpoint(date_str)
            elif fetch_or_parse_issue_occurred and pks_to_fetch:
                 logger.warning(f"Date {date_str} had fetch/parse issues for some attempted gamePks. Adding to issue list.")
                 DATES_WITH_ISSUES.add(date_str)
            elif game_pks:
                 logger.info(f"No new, valid, parsable games found for {date_str} (likely existed or filtered). Saving checkpoint.")
                 save_checkpoint(date_str)

            await asyncio.sleep(0.5 + random.uniform(0, 0.5)) # Delay between dates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLB box score data via API for date range and save to single CSV.")
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
    logger.info(f"Saving data to CSV: '{OUTPUT_PATH}'.")
    logger.info(f"Using checkpoint file: '{CHECKPOINT_PATH}'.")
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