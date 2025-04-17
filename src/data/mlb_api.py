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
    Scrape MLB.com for probable pitchers on a specific date.
    Returns a list of dicts, each with exactly:
      game_date,
      home_team (3‑letter abbr),
      away_team (3‑letter abbr),
      home_pitcher_name,
      home_pitcher_id,
      away_pitcher_name,
      away_pitcher_id
    """
    # fallback empty mapping
    if team_mapping_df is None or team_mapping_df.empty:
        team_mapping_df = pd.DataFrame(columns=['team_id','team_name','team_abbr'])

    # build a quick lookup for team_id → abbr
    id_to_abbr = dict(zip(
        team_mapping_df['team_id'], 
        team_mapping_df['team_abbr']
    ))

    # construct URL
    url = f"{BASE_URL_PROBABLE}{target_date_str}"
    time.sleep(DataConfig.RATE_LIMIT_PAUSE)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            return []  # no games that day
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"HTTP error for {url}: {e}")
        return []

    soup = BeautifulSoup(resp.content, 'html.parser')
    containers = soup.find_all(
        'div',
        class_=lambda c: c and 'probable-pitchers__matchup' in c.split()
    )
    if not containers:
        logger.warning(f"No matchups found on {url}.")
        return []

    results = []
    for mc in containers:
        # --- team IDs & abbrs ---
        game_div = mc.find(
            'div',
            class_=lambda c: c and 'probable-pitchers__game' in c.split()
        )
        if not game_div:
            continue

        try:
            away_tid = int(game_div['data-team-id-away'])
            home_tid = int(game_div['data-team-id-home'])
        except Exception:
            continue

        away_abbr = id_to_abbr.get(away_tid)
        home_abbr = id_to_abbr.get(home_tid)

        # --- pitcher blocks ---
        blocks = mc.find_all('div', class_='probable-pitchers__pitcher-summary')
        if len(blocks) < 2:
            continue

        def parse_pitcher(block):
            name_div = block.find('div', class_='probable-pitchers__pitcher-name')
            link = name_div.find('a', href=True) if name_div else None
            if link:
                nm = link.get_text(strip=True)
                pid = extract_player_id_from_link(link['href'])
            else:
                nm = name_div.get_text(strip=True) if name_div else None
                pid = None
            return nm, pid

        away_name, away_pid = parse_pitcher(blocks[0])
        home_name, home_pid = parse_pitcher(blocks[1])

        # skip if TBD or missing
        if not away_name or 'tbd' in away_name.lower() or not home_name or 'tbd' in home_name.lower():
            continue

        results.append({
            "game_date":          target_date_str,
            "home_team":          home_abbr,
            "away_team":          away_abbr,
            "home_pitcher_name":  home_name,
            "home_pitcher_id":    home_pid,
            "away_pitcher_name":  away_name,
            "away_pitcher_id":    away_pid,
        })

    logger.info(f"Scraped {len(results)} matchups for {target_date_str}")
    return results

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