# src/data/mlb_api.py (Updated for MLB Stats API)

import httpx # Use httpx for consistency with scrape_mlb_boxscores.py
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import time
import re
# Add tenacity for retry logic consistency
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Ensure src directory is in the path if running script directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Attempt to import utils and config
try:
    # DBConfig might not be strictly needed anymore if team_mapping isn't loaded
    from src.config import DBConfig, DataConfig, LogConfig  # Added LogConfig
    from src.utils import setup_logger, ensure_dir, DBConnection
    MODULE_IMPORTS_OK = True
    # Define DB_PATH from config if available, needed for fallback DBConnection path
    try: DB_PATH = DBConfig.PATH
    except (ImportError, AttributeError): DB_PATH = str(project_root / 'data' / 'pitcher_stats.db') # Fallback
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    # Fallback definitions
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mlb_api_fallback')
    class DataConfig: RATE_LIMIT_PAUSE = 1; REQUEST_TIMEOUT = 20; MAX_RETRIES = 3; INITIAL_RETRY_DELAY = 1; MAX_RETRY_DELAY = 10 # Added retry/timeout defaults
    class LogConfig: LOG_DIR = project_root / 'logs'
    DB_PATH = str(project_root / 'data' / 'pitcher_stats.db')
    def setup_logger(name, level=logging.INFO, log_file=None): return logger
    def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)
    class DBConnection: # Basic fallback
        def __init__(self, db_path): self.db_path = db_path
        def __enter__(self): import sqlite3; self.conn = sqlite3.connect(self.db_path); return self.conn
        def __exit__(self,et,ev,tb): self.conn.close()


# Setup logger
log_dir = LogConfig.LOG_DIR if MODULE_IMPORTS_OK else project_root / 'logs'
ensure_dir(log_dir)
logger = setup_logger('mlb_api_module', log_file= log_dir / 'mlb_api.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('mlb_api_fallback')

# --- Constants ---
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_SCHEDULE_ENDPOINT = MLB_STATS_API_BASE + "/schedule"
MLB_TRANSACTIONS_ENDPOINT = MLB_STATS_API_BASE + "/transactions"
# The schedule endpoint can be limited via a ``fields`` parameter.  A previous
# attempt to enumerate nested fields inadvertently omitted required keys,
# resulting in empty responses.  Leaving ``fields`` unset ensures the payload
# contains all game data, including probable pitchers.
SCHEDULE_API_FIELDS = None
# Explicitly request probable pitchers via ``hydrate`` to guarantee the API
# includes this information for historical dates.
SCHEDULE_API_HYDRATE = "probablePitcher"

# Use headers similar to scrape_mlb_boxscores
API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
    'Accept': 'application/json',
}

# Define retry parameters using DataConfig if available
MAX_RETRIES = DataConfig.MAX_RETRIES if hasattr(DataConfig, 'MAX_RETRIES') else 3
INITIAL_RETRY_DELAY = DataConfig.INITIAL_RETRY_DELAY if hasattr(DataConfig, 'INITIAL_RETRY_DELAY') else 1
MAX_RETRY_DELAY = DataConfig.MAX_RETRY_DELAY if hasattr(DataConfig, 'MAX_RETRY_DELAY') else 10
REQUEST_TIMEOUT = DataConfig.REQUEST_TIMEOUT if hasattr(DataConfig, 'REQUEST_TIMEOUT') else 20

# --- Retry Logic (adapted from scrape_mlb_boxscores.py) ---
RETRY_EXCEPTIONS = (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)
def is_retryable_exception(exception):
    """Determine if an exception is retryable."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on server errors (5xx)
        return 500 <= exception.response.status_code < 600
    # Do not retry on 404 Not Found specifically
    if isinstance(exception, httpx.RequestError) and hasattr(exception, 'response') and exception.response and exception.response.status_code == 404:
        return False
    # Retry on other request errors or timeouts
    return isinstance(exception, (httpx.TimeoutException, httpx.RequestError))

retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS), # Use specific exception types
    before_sleep=lambda rs: logger.warning(f"Retrying schedule API call ({rs.attempt_number}/{MAX_RETRIES}): {rs.outcome.exception()}. Waiting {rs.next_action.sleep:.2f}s...")
)

@retry_decorator
def fetch_schedule_api(target_date_str):
    """Fetches schedule data from MLB Stats API for a specific date with retries."""
    params = {
        "sportId": 1,  # MLB
        "startDate": target_date_str,
        "endDate": target_date_str,
    }
    if SCHEDULE_API_HYDRATE:
        params["hydrate"] = SCHEDULE_API_HYDRATE
    if SCHEDULE_API_FIELDS:
        params["fields"] = SCHEDULE_API_FIELDS
    logger.debug(f"Fetching API URL: {MLB_SCHEDULE_ENDPOINT} with params: {params}")
    # Use synchronous httpx client for this non-async script part
    with httpx.Client(headers=API_HEADERS, timeout=REQUEST_TIMEOUT) as client:
        response = client.get(MLB_SCHEDULE_ENDPOINT, params=params)

    if response.status_code == 404:
        logger.warning(f"API 404 Not Found for schedule on {target_date_str}. Likely no games.")
        return None # Treat 404 as no data, not an error to retry indefinitely
    elif response.status_code >= 400:
        logger.error(f"API error {response.status_code} fetching schedule for {target_date_str}")
        response.raise_for_status() # Raise HTTPStatusError for tenacity to catch if retryable

    logger.debug(f"Successfully fetched schedule API for {target_date_str}")
    return response.json()

# --- Function to Load Team Mapping (Kept for potential future use, but not needed by scrape_probable_pitchers anymore) ---
def load_team_mapping(db_path=DB_PATH):
    """Loads the team name/ID/abbreviation mapping from the database."""
    if not MODULE_IMPORTS_OK:
        logger.error("Cannot load team mapping: missing DBConnection/config.")
        return None
    logger.info("Loading team mapping from database (though not currently used by API scraper)...")
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB connection failed.")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_mapping'")
            if not cursor.fetchone():
                logger.error("'team_mapping' table not found.")
                return None
            query = "SELECT team_id, team_name, team_abbr FROM team_mapping"
            mapping_df = pd.read_sql_query(query, conn)
            # Use nullable integer type Int64
            mapping_df['team_id'] = pd.to_numeric(mapping_df['team_id'], errors='coerce').astype('Int64')
            # Use dropna without inplace=True
            mapping_df = mapping_df.dropna(subset=['team_id'])
            logger.info(f"Loaded {len(mapping_df)} teams into mapping.")
            return mapping_df
    except Exception as e:
        logger.error(f"Failed to load team mapping from database: {e}")
        return None

# --- Player ID Extraction (No changes needed) ---
def extract_player_id_from_link(link_href):
    # This function might become obsolete if IDs are always extracted directly from API,
    # but keep it for now in case of fallback or other uses.
    if not link_href or not isinstance(link_href, str): return None
    match = re.search(r'/player/[^/]+-(\d+)', link_href)
    if match:
        try: return int(match.group(1))
        except ValueError: return None
    return None

# --- Main Scraping Function (MODIFIED to use MLB Stats API) ---
def scrape_probable_pitchers(target_date_str):
    """
    Scrapes MLB Stats API for probable pitchers on a specific date.
    Args:
        target_date_str (str): The date in 'YYYY-MM-DD' format.
    Returns:
        list: A list of dictionaries, each containing game and probable pitcher info.
              Keys per dict: game_date, game_pk, home_team_abbr, away_team_abbr,
                             home_probable_pitcher_name, home_probable_pitcher_id,
                             away_probable_pitcher_name, away_probable_pitcher_id
              Returns empty list if no games or error occurs.
    """
    logger.info(f"Attempting to fetch probable pitchers via MLB Stats API for: {target_date_str}")

    try:
        # Add a small delay consistent with DataConfig if defined
        if hasattr(DataConfig, 'RATE_LIMIT_PAUSE'):
            time.sleep(DataConfig.RATE_LIMIT_PAUSE / 2) # Shorter delay before API call

        response_data = fetch_schedule_api(target_date_str)

        # Handle case where API call definitively failed or returned no data (e.g., 404)
        if response_data is None:
            logger.info(f"No schedule data returned from API for {target_date_str}.")
            return []

    except Exception as e:
        # Log error if retry decorator fails definitively
        logger.error(f"API call failed definitively for schedule on {target_date_str} after retries: {e}")
        return []

    results = []
    if response_data and 'dates' in response_data and response_data['dates']:
        # Typically schedule for one date is requested, so dates[0]
        date_info = response_data['dates'][0]
        games = date_info.get('games', [])

        if not games:
            logger.info(f"No games found in API response for {target_date_str}.")
            return []

        for game in games:
            try:
                game_pk = game.get('gamePk')
                status = game.get('status', {}).get('abstractGameState')
                teams_data = game.get('teams', {})
                away_team_info = teams_data.get('away', {}).get('team', {})
                home_team_info = teams_data.get('home', {}).get('team', {})
                away_probable = teams_data.get('away', {}).get('probablePitcher', {})
                home_probable = teams_data.get('home', {}).get('probablePitcher', {})

                # Basic checks
                if not game_pk or not away_team_info.get('id') or not home_team_info.get('id'):
                    logger.warning(f"Skipping game due to missing gamePk or team IDs in API response: {game.get('gamePk', 'N/A')}")
                    continue

                # Get abbreviations directly from API
                away_abbr = away_team_info.get('abbreviation')
                home_abbr = home_team_info.get('abbreviation')

                # Get probable pitcher info - check if the dictionary has content
                away_pid = away_probable.get('id')
                away_name = away_probable.get('fullName')
                home_pid = home_probable.get('id')
                home_name = home_probable.get('fullName')

                # Skip if either probable pitcher is missing (equivalent to TBD)
                if not away_pid or not away_name or not home_pid or not home_name:
                    logger.debug(f"Skipping game {game_pk} ({away_abbr} @ {home_abbr}) due to missing probable pitcher info.")
                    continue

                # Construct the result dictionary matching expected output format
                results.append({
                    "game_date": target_date_str, # Use the requested date
                    "game_pk": game_pk,
                    "home_team_abbr": home_abbr,
                    "away_team_abbr": away_abbr,
                    "home_probable_pitcher_name": home_name,
                    "home_probable_pitcher_id": home_pid,
                    "away_probable_pitcher_name": away_name,
                    "away_probable_pitcher_id": away_pid,
                })

            except Exception as e:
                logger.error(f"Error processing game {game.get('gamePk', 'N/A')} from API response: {e}", exc_info=True)
                continue # Skip this game, proceed to the next

    logger.info(f"Processed {len(results)} games with probable pitchers from API for {target_date_str}")
    return results


@retry_decorator
def _fetch_transactions(start_date: str, end_date: str) -> dict | None:
    params = {"sportId": 1, "startDate": start_date, "endDate": end_date}
    with httpx.Client(headers=API_HEADERS, timeout=REQUEST_TIMEOUT) as client:
        response = client.get(MLB_TRANSACTIONS_ENDPOINT, params=params)
    if response.status_code == 404:
        logger.warning("Transactions API 404 between %s and %s", start_date, end_date)
        return None
    elif response.status_code >= 400:
        logger.error("API error %s fetching transactions", response.status_code)
        response.raise_for_status()
    return response.json()


def update_player_injury_log(start_date: str, end_date: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Fetch IL transactions and append rows to ``player_injury_log`` table."""
    try:
        data = _fetch_transactions(start_date, end_date) or {}
    except Exception as exc:
        logger.error("Failed to fetch transactions: %s", exc)
        return pd.DataFrame()

    transactions = data.get("transactions", [])
    rows = []
    for t in transactions:
        desc = str(t.get("description", "")).lower()
        if "injured list" not in desc:
            continue
        pid = t.get("playerId")
        date = t.get("transactionDate")
        if not pid or not date:
            continue
        if "placed" in desc or "transferred" in desc:
            action = "start"
        elif "reinstated" in desc or "activated" in desc:
            action = "end"
        else:
            continue
        rows.append({"player_id": pid, "date": date, "action": action})

    if not rows:
        logger.info("No IL transactions found between %s and %s", start_date, end_date)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    logs = []
    for pid, grp in df.sort_values("date").groupby("player_id"):
        current_start = None
        for _, row in grp.iterrows():
            if row["action"] == "start":
                current_start = row["date"]
            elif row["action"] == "end" and current_start is not None:
                logs.append({"player_id": pid, "start_date": current_start.date(), "end_date": row["date"].date()})
                current_start = None
        if current_start is not None:
            logs.append({"player_id": pid, "start_date": current_start.date(), "end_date": None})

    log_df = pd.DataFrame(logs)

    with DBConnection(db_path) as conn:
        if log_df.empty:
            return log_df
        if not table_exists(conn, "player_injury_log"):
            log_df.to_sql("player_injury_log", conn, index=False, if_exists="replace")
        else:
            log_df.to_sql("player_injury_log", conn, index=False, if_exists="append")
    logger.info("Added %d injury log rows", len(log_df))
    return log_df

# --- Example Usage Block Modified ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK:
        sys.exit("Exiting: Failed module imports.")

    # Get date from command line argument for testing
    test_date_str = datetime.now().strftime("%Y-%m-%d") # Default to today
    if len(sys.argv) > 1:
        try:
            # Validate input date format
            datetime.strptime(sys.argv[1], "%Y-%m-%d")
            test_date_str = sys.argv[1]
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}. Using today: {test_date_str}")

    print(f"--- Testing MLB API Fetcher for Date: {test_date_str} ---")

    # Team mapping is no longer needed for the API call
    # team_map_df = load_team_mapping() # No longer needed here

    # Pass only the target date to the scraper function
    daily_data = scrape_probable_pitchers(test_date_str)

    if daily_data:
        print(f"\n--- Sample Scraped Data ({test_date_str}) ---")
        # Print first 3 entries for preview
        print(json.dumps(daily_data[:3], indent=2))
        print(f"\nTotal games scraped (with pitchers): {len(daily_data)}")

        # Save test output (optional)
        test_output_dir = project_root / 'data' / 'test_api_output' # Changed folder name
        ensure_dir(test_output_dir)
        test_filename = test_output_dir / f"test_api_probable_pitchers_{test_date_str}.json"
        try:
            with open(test_filename, 'w') as f:
                json.dump(daily_data, f, indent=2)
            print(f"\nTest data saved to: {test_filename}")
        except Exception as e:
            print(f"\nError saving test data: {e}")
    else:
        print(f"\nNo probable pitcher data successfully fetched via API for {test_date_str}.")