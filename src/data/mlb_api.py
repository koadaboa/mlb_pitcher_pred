# src/data/mlb_api.py (Updated for dynamic date URL)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import time
import re

# Ensure src directory is in the path if running script directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Attempt to import utils and config
try:
    from src.config import DBConfig, DataConfig
    from src.data.utils import setup_logger, ensure_dir, normalize_name, DBConnection
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)
    def normalize_name(name): return name.lower().strip() if isinstance(name, str) else ""
    class DBConnection:
        def __init__(self, db_path): self.db_path = db_path
        def __enter__(self): import sqlite3; self.conn = sqlite3.connect(self.db_path); return self.conn
        def __exit__(self,et,ev,tb): self.conn.close()
    class DBConfig: PATH = "data/pitcher_stats.db"
    class DataConfig: RATE_LIMIT_PAUSE = 1

# Setup logger
log_dir = project_root / 'logs'
ensure_dir(log_dir)
logger = setup_logger('mlb_scraper_module', log_file= log_dir / 'mlb_scraper_module.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('mlb_scraper_fallback')

# Base URL part
BASE_URL_PROBABLE = "https://www.mlb.com/probable-pitchers/" # Add trailing slash

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/',
}

# --- Function to Load Team Mapping (Identical to previous version) ---
def load_team_mapping(db_path):
    """Loads the team name/ID/abbreviation mapping from the database."""
    if not MODULE_IMPORTS_OK: logger.error("Cannot load team mapping: missing DBConnection/config."); return None
    logger.info("Loading team mapping from database...")
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB connection failed.")
            cursor = conn.cursor(); cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_mapping'")
            if not cursor.fetchone(): logger.error("'team_mapping' table not found."); return None
            query = "SELECT team_id, team_name, team_abbr FROM team_mapping"
            mapping_df = pd.read_sql_query(query, conn)
            mapping_df['team_id'] = pd.to_numeric(mapping_df['team_id'], errors='coerce').astype('Int64')
            mapping_df.dropna(subset=['team_id'], inplace=True)
            logger.info(f"Loaded {len(mapping_df)} teams into mapping.")
            return mapping_df
    except Exception as e: logger.error(f"Failed to load team mapping from database: {e}"); return None

# --- Player ID Extraction (Identical) ---
def extract_player_id_from_link(link_href):
    if not link_href or not isinstance(link_href, str): return None
    match = re.search(r'/player/[^/]+-(\d+)', link_href)
    if match:
        try: return int(match.group(1))
        except ValueError: return None
    return None

# --- Main Scraping Function (MODIFIED to accept date and build URL) ---
def scrape_probable_pitchers(target_date_str, team_mapping_df):
    """
    Scrapes MLB.com for probable pitchers for a SPECIFIC DATE.
    Uses the provided team mapping to identify team IDs/abbreviations.

    Args:
        target_date_str (str): The target date in 'YYYY-MM-DD' format.
        team_mapping_df (pd.DataFrame): DataFrame loaded from the team_mapping table.

    Returns:
        list: A list of dictionaries, each containing game and probable pitcher info.
              Returns empty list on failure.
    """
    if team_mapping_df is None or team_mapping_df.empty:
         logger.warning("Team mapping data missing/empty. Team names/abbrs will be missing.")
         team_mapping_df = pd.DataFrame(columns=['team_id', 'team_name', 'team_abbr']) # Empty placeholder

    # Construct the target URL using the date
    target_url = f"{BASE_URL_PROBABLE}{target_date_str}"
    logger.info(f"Attempting to scrape probable pitchers from: {target_url}")
    scraped_games = []

    try:
        time.sleep(DataConfig.RATE_LIMIT_PAUSE)
        response = requests.get(target_url, headers=HEADERS, timeout=20)
        # Check if the page for the specific date exists (MLB.com might 404 for dates too far out/past)
        if response.status_code == 404:
             logger.warning(f"Page not found (404) for date {target_date_str} at {target_url}. No games to scrape.")
             return [] # Return empty list, not an error
        response.raise_for_status() # Raise errors for other issues (5xx, etc.)
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Game Container Parsing (Using selectors from previous version) ---
        game_containers = soup.find_all('div', class_=lambda x: x and 'probable-pitchers__matchup' in x.split())
        if not game_containers:
            logger.error(f"Could not find game containers on {target_url}. Scraping failed.")
            return []

        logger.info(f"Found {len(game_containers)} potential game containers for {target_date_str}.")

        for game_container in game_containers:
            game_pk = None; home_team_id = None; away_team_id = None # Ensure defined
            try:
                # Extract gamePk
                game_pk_str = game_container.get('data-gamepk'); game_pk = int(game_pk_str) if game_pk_str and game_pk_str.isdigit() else None
                if not game_pk: logger.warning("No gamePk found. Skipping container."); continue

                # Extract Team IDs from attributes
                game_info_div = game_container.find('div', class_=lambda x: x and 'probable-pitchers__game' in x.split())
                if game_info_div:
                    away_id_str = game_info_div.get('data-team-id-away'); home_id_str = game_info_div.get('data-team-id-home')
                    away_team_id = int(away_id_str) if away_id_str and away_id_str.isdigit() else None
                    home_team_id = int(home_id_str) if home_id_str and home_id_str.isdigit() else None
                else: logger.warning(f"No 'probable-pitchers__game' div for {game_pk}")

                # Lookup Team Names/Abbrs using Scraped IDs
                home_team_name, home_team_abbr = None, None; away_team_name, away_team_abbr = None, None
                if home_team_id is not None:
                    home_match = team_mapping_df[team_mapping_df['team_id'] == home_team_id]
                    if not home_match.empty: home_team_name=home_match.iloc[0]['team_name']; home_team_abbr=home_match.iloc[0]['team_abbr']
                    else: logger.warning(f"Could not map home_team_id: {home_team_id}")
                if away_team_id is not None:
                    away_match = team_mapping_df[team_mapping_df['team_id'] == away_team_id]
                    if not away_match.empty: away_team_name=away_match.iloc[0]['team_name']; away_team_abbr=away_match.iloc[0]['team_abbr']
                    else: logger.warning(f"Could not map away_team_id: {away_team_id}")

                # Extract Pitchers
                pitcher_summaries = game_container.find_all('div', class_='probable-pitchers__pitcher-summary')
                away_pitcher_name, away_pitcher_id = None, None; home_pitcher_name, home_pitcher_id = None, None
                if len(pitcher_summaries) >= 1:
                    name_div = pitcher_summaries[0].find('div', class_='probable-pitchers__pitcher-name'); link_tag = name_div.find('a', href=True) if name_div else None
                    if link_tag: away_pitcher_name=link_tag.get_text(strip=True); away_pitcher_id=extract_player_id_from_link(link_tag['href'])
                    elif name_div: away_pitcher_name=name_div.get_text(strip=True)
                if len(pitcher_summaries) >= 2:
                    name_div = pitcher_summaries[1].find('div', class_='probable-pitchers__pitcher-name'); link_tag = name_div.find('a', href=True) if name_div else None
                    if link_tag: home_pitcher_name=link_tag.get_text(strip=True); home_pitcher_id=extract_player_id_from_link(link_tag['href'])
                    elif name_div: home_pitcher_name=name_div.get_text(strip=True)

                # Handle TBD
                if not away_pitcher_name or "tbd" in away_pitcher_name.lower() or not home_pitcher_name or "tbd" in home_pitcher_name.lower():
                    logger.info(f"Skipping game {game_pk} due to TBD pitcher.")
                    continue

                # Create final data dict (using target_date_str as game_date)
                game_data = {
                    "gamePk": game_pk, "game_date": target_date_str, # Use the input date
                    "home_team_name": home_team_name, "away_team_name": away_team_name,
                    "home_probable_pitcher_name": home_pitcher_name, "home_probable_pitcher_id": home_pitcher_id,
                    "away_probable_pitcher_name": away_pitcher_name, "away_probable_pitcher_id": away_pitcher_id,
                    "home_team_id": home_team_id, "away_team_id": away_team_id,
                    "home_team_abbr": home_team_abbr, "away_team_abbr": away_team_abbr,
                }
                scraped_games.append(game_data)

            except Exception as e: 
                logger.error(f"Error parsing game container (GamePK: {game_pk if game_pk else 'Unknown'}): {e}.")
                continue

        logger.info(f"Successfully scraped {len(scraped_games)} games with non-TBD probable pitchers for {target_date_str}.")
        return scraped_games # Return only the list now

    except requests.exceptions.RequestException as e: logger.error(f"HTTP Error fetching {target_url}: {e}"); return []
    except Exception as e: logger.error(f"Unexpected error scraping {target_date_str}: {e}"); logger.error(traceback.format_exc()); return []

# --- Example Usage Block Modified ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")

    # Get date from command line argument for testing
    test_date_str = datetime.now().strftime("%Y-%m-%d") # Default to today
    if len(sys.argv) > 1:
        try:
            datetime.strptime(sys.argv[1], "%Y-%m-%d")
            test_date_str = sys.argv[1]
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}. Using today: {test_date_str}")

    print(f"--- Testing MLB Scraper for Date: {test_date_str} ---")
    db_path = project_root / DBConfig.PATH
    team_map_df = load_team_mapping(db_path)

    if team_map_df is not None:
        # Pass the target date to the scraper
        daily_data = scrape_probable_pitchers(test_date_str, team_map_df)

        if daily_data:
            print(f"\n--- Sample Scraped Data ({test_date_str}) ---")
            print(json.dumps(daily_data[:3], indent=2))
            print(f"\nTotal games scraped (with pitchers): {len(daily_data)}")
            # Save test output
            test_output_dir = project_root / 'data' / 'test_scraper_output'
            ensure_dir(test_output_dir); test_filename = test_output_dir / f"test_scraped_mapped_pitchers_{test_date_str}.json"
            try:
                with open(test_filename, 'w') as f: json.dump(daily_data, f, indent=2)
                print(f"\nTest data saved to: {test_filename}")
            except Exception as e: print(f"\nError saving test data: {e}")
        else:
            print(f"\nNo probable pitcher data successfully scraped for {test_date_str}.")
    else:
        print(f"\nCould not load team mapping. Scraper test aborted.")