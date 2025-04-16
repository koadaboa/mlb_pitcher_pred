# Import necessary libraries
import time
import argparse
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# --- BeautifulSoup Import ---
from bs4 import BeautifulSoup

# --- Assume src is in Python path ---
# Add project root to path if needed (adjust relative path if necessary)
# project_root = Path(__file__).resolve().parents[2]
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

try:
    from src.config import DBConfig
    from src.data.utils import DBConnection
except ImportError:
    print("Error: Could not import DBConfig or DBConnection from src.", file=sys.stderr)
    print("Make sure the script is run from a location where 'src' is accessible,", file=sys.stderr)
    print("or adjust the path.", file=sys.stderr)
    class DBConfig: PATH = "data/pitcher_stats.db" # Default path
    class DBConnection:
        def __init__(self, db_name): self.db_name = db_name
        def __enter__(self):
            Path(self.db_name).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_name); return self.conn
        def __exit__(self,t,v,tb):
            if self.conn: self.conn.close()

# --- Selenium Fetching Function ---
def fetch_umpire_game_data_selenium(target_date):
    """
    Fetches umpire names and associated teams for a specific date using Selenium.

    Args:
        target_date (str): Date in YYYY-MM-DD format.

    Returns:
        list: List of tuples (game_date, umpire, home_team, away_team), or None on error.
    """
    url = "https://umpscorecards.com/data/games"
    game_data_list = []
    driver = None # Initialize driver to None

    # --- IMPORTANT: Configure your WebDriver ---
    # You might need to specify the path to your chromedriver or geckodriver
    # Example using Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') # Run headless (no visible browser window)
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Add any other options you need

    try:
        print("Initializing WebDriver...")
        # Make sure chromedriver executable is in your PATH or provide path:
        # driver = webdriver.Chrome(service=Service('/path/to/chromedriver'), options=options)
        driver = webdriver.Chrome(options=options)
        print(f"Navigating to {url}...")
        driver.get(url)

        # --- Wait for the table data to load ---
        # We need to wait for an element *inside* the dynamic table.
        # Let's wait for the table body (tbody) or specifically for table rows (tr) to appear.
        wait_timeout = 20 # Increased timeout
        print(f"Waiting up to {wait_timeout} seconds for table rows to load...")
        try:
            # Wait until at least one table row is present within the tbody of the table inside #table div
            wait = WebDriverWait(driver, wait_timeout)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#table table tbody tr")))
            print("Table content appears loaded.")
        except TimeoutException:
            print(f"Error: Timed out waiting for table content to load after {wait_timeout} seconds.", file=sys.stderr)
            # Optional: Save page source here for debugging
            # with open("timeout_page.html", "w", encoding="utf-8") as f: f.write(driver.page_source)
            return None # Return None indicating failure

        # Short pause just in case rendering is slightly delayed after element presence
        time.sleep(2)

        # Get the page source *after* JavaScript has potentially run
        page_source = driver.page_source

        # Parse the potentially updated HTML source
        print("Parsing loaded page source...")
        soup = BeautifulSoup(page_source, 'html.parser')

        # Find the table and rows (similar logic as before, but applied to rendered HTML)
        table_div = soup.find('div', id='table')
        if not table_div:
            print(f"Error: Could not find table div even after waiting.", file=sys.stderr)
            return None
        table = table_div.find('table')
        if not table:
            print(f"Error: Could not find table element.", file=sys.stderr)
            return None
        tbody = table.find('tbody')
        if not tbody:
             print(f"Warning: Could not find table body (tbody).", file=sys.stderr)
             return []

        rows = tbody.find_all('tr')
        if not rows:
             print(f"Warning: No rows found in table body after waiting.", file=sys.stderr)
             return []


        # Iterate and extract data (same logic as the final requests version)
        print("Extracting data from table rows...")
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 5:
                row_date = cells[1].get_text(strip=True)
                if row_date == target_date:
                    umpire_link = cells[2].find('a')
                    away_team_link = cells[3].find('a')
                    home_team_link = cells[4].find('a')
                    if umpire_link and away_team_link and home_team_link:
                        umpire_name = umpire_link.get_text(strip=True)
                        away_team_abbr = away_team_link.get_text(strip=True)
                        home_team_abbr = home_team_link.get_text(strip=True)
                        game_data_list.append((
                            target_date, umpire_name, home_team_abbr, away_team_abbr
                        ))
                    else:
                         print(f"Warning: Missing link in row for date {row_date}", file=sys.stderr)

        print(f"Found {len(game_data_list)} entries for {target_date}.")
        return game_data_list

    except Exception as e:
        print(f"An unexpected error occurred during Selenium processing: {e}", file=sys.stderr)
        # You might want to print the full traceback here for debugging
        # import traceback
        # print(traceback.format_exc())
        return None
    finally:
        # Make sure to quit the driver to close the browser window/process
        if driver:
            print("Closing WebDriver...")
            driver.quit()

# --- Database Function (Identical to previous version) ---
def append_to_db(game_data_list, db_path):
    """
    Appends fetched game data (date, umpire, home, away) to the SQLite database.
    (No changes needed from the previous version)
    """
    if not game_data_list:
        print("No game data provided to append.")
        return False
    added_count = 0
    sql_insert = """
    INSERT INTO umpire_data (game_date, umpire, home_team, away_team)
    VALUES (?, ?, ?, ?);
    """
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='umpire_data'")
            if not cursor.fetchone():
                print(f"Warning: Table 'umpire_data' does not exist in {db_path}. Creating.", file=sys.stderr)
                try:
                    cursor.execute("""
                        CREATE TABLE umpire_data (
                            game_date TEXT, umpire TEXT, home_team TEXT, away_team TEXT,
                            PRIMARY KEY (game_date, umpire, home_team, away_team)
                        );""")
                    conn.commit()
                    print("Table 'umpire_data' created.")
                except Exception as create_e:
                    print(f"Error: Failed to create 'umpire_data' table: {create_e}", file=sys.stderr)
                    return False
            # Try bulk insert first
            try:
                 cursor.executemany(sql_insert, game_data_list)
                 conn.commit()
                 print(f"Database commit successful after executemany.")
                 return True
            except sqlite3.IntegrityError:
                 conn.rollback()
                 print(f"Warning: Bulk insert failed (likely duplicates). Trying row-by-row.", file=sys.stderr)
                 added_count = 0
                 for record in game_data_list:
                      try:
                           cursor.execute(sql_insert, record)
                           added_count += 1
                      except sqlite3.IntegrityError:
                           pass # Silently ignore duplicates in row-by-row
                      except Exception as row_e:
                           print(f"Error inserting row {record}: {row_e}", file=sys.stderr)
                 conn.commit()
                 print(f"Row-by-row insert complete. Added {added_count} new records.")
                 return True
            except Exception as bulk_e:
                 conn.rollback()
                 print(f"Error during bulk insert: {bulk_e}", file=sys.stderr)
                 return False
    except Exception as e:
        print(f"Database error: {e}", file=sys.stderr)
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLB umpire/game data using Selenium and append to SQLite DB.")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = args.date

    # Validate date format
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Date format is incorrect ({target_date}). Please use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    # Fetch the game data using Selenium
    game_data = fetch_umpire_game_data_selenium(target_date)

    if game_data is None:
        print("An error occurred during fetching/parsing with Selenium.")
        sys.exit(1)
    elif not game_data:
        print(f"No game data found for {target_date}.")
        sys.exit(0)
    else:
        print(f"\nSuccessfully fetched {len(game_data)} games for {target_date} using Selenium.")
        print("Attempting to append to database...")
        db_file_path = DBConfig.PATH
        success = append_to_db(game_data, db_file_path)

        if success:
            print("Database operation finished.")
        else:
            print("Database operation failed.")
            sys.exit(1)