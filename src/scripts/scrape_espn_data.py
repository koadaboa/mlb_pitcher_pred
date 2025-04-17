# scripts/scrape_espn_data.py
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
import logging
import os
import sys
import httpx # For asynchronous HTTP requests
import asyncio # For running async tasks
import nest_asyncio # Handles nested event loops if needed
import random # Added for retry jitter

nest_asyncio.apply() # Apply necessary patch for asyncio

# --- Configuration ---
START_DATE = "20160401"
END_DATE = "20250415" # Inclusive
OUTPUT_CSV = "espn_mlb_umpire_data_optimized.csv" # New output name
CHECKPOINT_FILE = "scraper_checkpoint_optimized.txt" # File to store the last processed date
WEBDRIVER_PATH = None # Set path if needed, e.g., '/path/to/chromedriver'
DEBUG_HTML_DIR = "debug_html" # Directory to save debug HTML files
os.makedirs(DEBUG_HTML_DIR, exist_ok=True) # Create the directory
MAX_CONCURRENT_REQUESTS = 5 # ADJUSTED: Lowered concurrency
REQUEST_TIMEOUT = 25 # Timeout for individual Gamecast page fetches (seconds)
REQUEST_DELAY = 0.5 # ADJUSTED: Increased delay between batches
# Define typical season bounds (Month, Day) - Adjust if needed
SEASON_START_MONTH_DAY = (3, 25) # Late March to be safe
SEASON_END_MONTH_DAY = (11, 5) # Early Nov to include postseason

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Optional: Quieten httpx's own logs

# --- Checkpoint Functions ---
def load_last_processed_date(filename):
    """Loads the last successfully processed date from the checkpoint file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                last_date_str = f.read().strip()
            # Validate the date format
            datetime.strptime(last_date_str, "%Y%m%d")
            logging.info(f"Checkpoint found. Resuming after date: {last_date_str}")
            return last_date_str
        except Exception as e:
            logging.warning(f"Could not read or validate checkpoint file '{filename}': {e}. Starting from beginning.")
            # Optional: Delete or rename the bad checkpoint file here
            # os.remove(filename)
            return None
    return None

def save_checkpoint(filename, date_str):
    """Saves the last successfully processed date to the checkpoint file."""
    try:
        with open(filename, 'w') as f:
            f.write(date_str)
        logging.debug(f"Checkpoint saved for date: {date_str}")
    except Exception as e:
        logging.error(f"Could not write checkpoint file '{filename}': {e}")

# --- Date Generation (Season-Aware & Resumable) ---
def generate_dates_resumable(start_date_str, end_date_str, last_processed_date_str):
    """
    Generates dates ONLY within the typical MLB season (Mar 25 - Nov 5 approx)
    for each year in the range, respecting the checkpoint.
    """
    script_start_dt = datetime.strptime(start_date_str, "%Y%m%d")
    script_end_dt = datetime.strptime(end_date_str, "%Y%m%d")
    start_year = script_start_dt.year
    end_year = script_end_dt.year

    resume_from_dt = None
    if last_processed_date_str:
        try:
            resume_from_dt = datetime.strptime(last_processed_date_str, "%Y%m%d") + timedelta(days=1)
        except ValueError:
            logging.warning("Invalid checkpoint date, starting from script beginning.")
            resume_from_dt = script_start_dt

    start_dt = resume_from_dt if resume_from_dt and resume_from_dt > script_start_dt else script_start_dt

    logging.info(f"Starting date generation from: {start_dt.strftime('%Y-%m-%d')} within seasonal bounds.")

    for year in range(start_year, end_year + 1):
        # Define season start/end for THIS year
        try:
            year_season_start = datetime(year, SEASON_START_MONTH_DAY[0], SEASON_START_MONTH_DAY[1])
            year_season_end = datetime(year, SEASON_END_MONTH_DAY[0], SEASON_END_MONTH_DAY[1])
        except ValueError: # Handle potential date issues
            logging.warning(f"Could not create season bounds for year {year}, skipping year.")
            continue

        # Determine the actual iteration start/end for this year, respecting script bounds and resume point
        current_year_start = max(year_season_start, script_start_dt)
        current_year_end = min(year_season_end, script_end_dt)

        # Adjust start based on resume point for the current year
        if year == start_dt.year:
             current_date = max(current_year_start, start_dt)
        elif year > start_dt.year:
             current_date = current_year_start
        else: # year < start_dt.year (already processed or before script start)
             continue

        # Iterate only within the valid range for this year
        if current_date > current_year_end:
             logging.debug(f"Skipping year {year}: effective start {current_date.strftime('%Y-%m-%d')} > effective end {current_year_end.strftime('%Y-%m-%d')}")
             continue

        logging.debug(f"Generating dates for {year} from {current_date.strftime('%Y-%m-%d')} to {current_year_end.strftime('%Y-%m-%d')}")
        while current_date <= current_year_end:
            yield current_date.strftime("%Y%m%d")
            current_date += timedelta(days=1)


# --- HTML Parsing Functions ---
def parse_umpire_info(soup, game_url):
    """Extracts umpire names from the Gamecast page soup using a refined search."""
    umpires = {'home_plate_umpire': None, 'first_base_umpire': None, 'second_base_umpire': None, 'third_base_umpire': None}
    umpire_list_ul = None
    try:
        # Refined search for umpire list
        list_wrapper = soup.find('div', class_='GameInfo__List__Wrapper')
        if list_wrapper:
            potential_uls = list_wrapper.find_all('ul', class_='GameInfo__List')
            for ul in potential_uls:
                first_header = ul.find('li', class_='GameInfo__List__ItemHeader')
                if first_header and 'Umpires' in first_header.get_text():
                    umpire_list_ul = ul
                    break
        # Fallback search if wrapper not found (might be needed for very old pages)
        if not umpire_list_ul:
             headers = soup.find_all('li', class_='GameInfo__List__ItemHeader', string=re.compile(r'Umpires:'))
             if headers: umpire_list_ul = headers[0].find_parent('ul')

        if umpire_list_ul:
            items = umpire_list_ul.find_all('li', class_='GameInfo__List__Item')
            for item in items:
                text = item.get_text(strip=True)
                hp_match = re.search(r"Home Plate Umpire\s*-\s*(.*)", text)
                fb_match = re.search(r"First Base Umpire\s*-\s*(.*)", text)
                sb_match = re.search(r"Second Base Umpire\s*-\s*(.*)", text)
                tb_match = re.search(r"Third Base Umpire\s*-\s*(.*)", text)
                if hp_match: umpires['home_plate_umpire'] = hp_match.group(1).strip()
                elif fb_match: umpires['first_base_umpire'] = fb_match.group(1).strip()
                elif sb_match: umpires['second_base_umpire'] = sb_match.group(1).strip()
                elif tb_match: umpires['third_base_umpire'] = tb_match.group(1).strip()
        else:
            logging.warning(f"Could not find umpire list for {game_url}")
            # Optionally save HTML if needed for debugging umpire parsing failures
            # try:
            #     game_id = game_url.split('gameId/')[-1].split('/')[0] if 'gameId/' in game_url else "unknown_game"
            #     filename = os.path.join(DEBUG_HTML_DIR, f"umpire_ul_not_found_{game_id}.html")
            #     with open(filename, "w", encoding="utf-8") as f: f.write(soup.prettify())
            #     logging.warning(f"Saved HTML source for umpire debug: {filename}")
            # except Exception as save_e: logging.error(f"Could not save debug umpire HTML: {save_e}")

    except Exception as e:
        logging.error(f"Error parsing umpires for {game_url}: {e}")
    return umpires

def parse_team_info(soup, game_url):
    """Extracts home and away team names from the Gamecast page soup."""
    home_team, away_team = None, None
    try:
        competitors_div = soup.find('div', class_='Gamestrip__Competitors')
        if competitors_div:
            away_team_div = competitors_div.find('div', class_='Gamestrip__Team--left')
            if away_team_div:
                away_name_tag = away_team_div.find('h2', class_='ScoreCell__TeamName')
                if away_name_tag: away_team = away_name_tag.get_text(strip=True)

            home_team_div = competitors_div.find('div', class_='Gamestrip__Team--right')
            if home_team_div:
                home_name_tag = home_team_div.find('h2', class_='ScoreCell__TeamName')
                if home_name_tag: home_team = home_name_tag.get_text(strip=True)
        if not home_team or not away_team:
             logging.warning(f"Could not extract both team names from {game_url}")
    except Exception as e:
        logging.error(f"Error parsing teams for {game_url}: {e}")
    return home_team, away_team

# --- Asynchronous Fetching and Processing ---
async def fetch_and_parse_gamecast(client, url, semaphore):
    """
    Fetches a single Gamecast URL, parses it, returns structured data or None.
    Includes retry logic for request errors.
    """
    max_retries = 3
    base_delay = 0.5 # seconds

    async with semaphore: # Control concurrency
        for attempt in range(max_retries):
            try:
                logging.debug(f"Fetching (Attempt {attempt+1}/{max_retries}): {url}")
                response = await client.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status() # Raise error for bad status codes (4xx, 5xx)
                logging.debug(f"Successfully fetched: {url} (Status: {response.status_code})")

                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'lxml') # Use lxml parser
                umpires = parse_umpire_info(soup, url)
                home_team, away_team = parse_team_info(soup, url)

                if home_team and away_team:
                    # Return data including the URL for reference if needed
                    # Add a small sleep *after* successful processing before returning
                    await asyncio.sleep(0.1) # Small delay even on success
                    return {'url': url, 'home_team': home_team, 'away_team': away_team, **umpires}
                else:
                    logging.warning(f"Could not parse required team names from {url}, skipping result.")
                    return None # Indicate failure to parse essentials

            except (httpx.RequestError, httpx.TimeoutException) as e: # Catch broader request errors including ConnectionTerminated and Timeouts
                if attempt == max_retries - 1: # Last attempt failed
                    logging.error(f"Request error fetching {url} after {max_retries} attempts: {e}")
                    return None # Failed after retries
                else:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                    logging.warning(f"Request error fetching {url} (Attempt {attempt+1}): {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay) # Wait before retrying

            except Exception as e: # Catch other unexpected errors during parsing etc.
                logging.error(f"Unexpected error processing gamecast {url}: {e}")
                # import traceback # Uncomment for detailed traceback
                # logging.error(traceback.format_exc()) # Uncomment for detailed traceback
                return None # Non-retryable error for this function

        # Should not be reached if loop completes normally, but ensures None return on failure
        return None


async def process_daily_gamecasts(urls, date_str):
    """Fetches and parses a list of Gamecast URLs asynchronously for a given date."""
    if not urls:
        return []

    daily_results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    # Define headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36', # Example User Agent
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
        }


    # Use an httpx AsyncClient for connection pooling and header management
    # Consider adding http1=True if HTTP/2 issues persist
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, http2=True, timeout=REQUEST_TIMEOUT + 5) as client:
        tasks = []
        for url in urls:
            # Schedule the fetch and parse task
            task = asyncio.ensure_future(fetch_and_parse_gamecast(client, url, semaphore))
            tasks.append(task)

        # Wait for all tasks to complete using asyncio.gather
        # return_exceptions=True allows the loop to continue even if some tasks fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process the results
    valid_results_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Exception occurred within the asyncio.gather handling itself (less common)
            logging.error(f"Task wrapper exception for URL {urls[i]}: {result}")
        elif result is None:
            # fetch_and_parse_gamecast returned None, indicating failure (retries exhausted or parsing error)
            logging.debug(f"Task for URL {urls[i]} failed or returned no data after processing.")
        elif result is not None:
            # Successfully parsed result
            result_with_date = {
                'game_date': datetime.strptime(date_str, "%Y%m%d").strftime('%Y-%m-%d'),
                **result # Unpack the dictionary from fetch_and_parse
            }
            daily_results.append(result_with_date)
            valid_results_count += 1
        # else: Should not happen if result is neither Exception nor None

    logging.info(f"Finished processing {len(urls)} URLs for {date_str}. Successfully parsed: {valid_results_count}")
    # Add a small delay after processing a batch of URLs for a date
    await asyncio.sleep(REQUEST_DELAY)
    return daily_results

# --- Main Scraping Logic ---
def scrape_espn_data_optimized(start_date, end_date, webdriver_path=None):
    """Main function using Selenium for scoreboard and httpx/asyncio for gamecasts."""
    driver = None
    all_game_data = [] # Holds data across the entire run

    # Load checkpoint and existing data
    last_processed_date = load_last_processed_date(CHECKPOINT_FILE)
    if last_processed_date and os.path.exists(OUTPUT_CSV):
        logging.info(f"Loading existing data from {OUTPUT_CSV}...")
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            # Keep data types consistent if possible, especially dates
            if 'game_date' in existing_df.columns:
                 existing_df['game_date'] = pd.to_datetime(existing_df['game_date']).dt.strftime('%Y-%m-%d')
            all_game_data = existing_df.to_dict('records')
            logging.info(f"Loaded {len(all_game_data)} existing records.")
        except Exception as e:
            logging.error(f"Error loading {OUTPUT_CSV}: {e}. Starting fresh.")
            all_game_data = []
            last_processed_date = None # Reset checkpoint if load fails
    else:
         all_game_data = []
         logging.info("No valid checkpoint or existing data file found. Starting fresh.")

    # Get list of dates to process, respecting checkpoints and season bounds
    date_iterator = list(generate_dates_resumable(start_date, end_date, last_processed_date))

    if not date_iterator:
        logging.info("All required dates seem to be processed or fall outside season bounds.")
        if all_game_data: return pd.DataFrame(all_game_data) # Return loaded data
        else: return None # Nothing loaded or scraped

    # --- Setup WebDriver (Only if needed) ---
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless') # Enable headless for optimized run
        prefs = {"profile.managed_default_content_settings.images": 2} # Disable images
        options.add_experimental_option("prefs", prefs)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36") # Example user agent
        options.add_argument('--log-level=3') # Suppress browser logs

        if webdriver_path:
            service = Service(executable_path=webdriver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)
        logging.info("WebDriver initialized successfully (Headless Mode).")
    except Exception as e:
        logging.error(f"Failed WebDriver init: {e}")
        # Decide whether to exit or proceed without Selenium (won't work for scoreboard)
        if all_game_data: return pd.DataFrame(all_game_data)
        else: return None
    # --- End WebDriver Setup ---


    # --- Date Loop ---
    processed_dates_count = 0
    try:
        for date_str in date_iterator:
            scoreboard_url = f"https://www.espn.com/mlb/scoreboard/_/date/{date_str}"
            logging.info(f"Processing date: {date_str} - URL: {scoreboard_url}")
            gamecast_links_to_visit = []
            page_source_scoreboard = ""
            daily_game_data = [] # Store results just for this day

            # --- Selenium for Scoreboard Links ---
            try:
                driver.get(scoreboard_url)
                # Wait for scoreboard sections to appear
                WebDriverWait(driver, 20).until( EC.presence_of_element_located((By.CSS_SELECTOR, "section.Scoreboard[id]")) )
                time.sleep(1.5) # Small delay for JS rendering
                page_source_scoreboard = driver.page_source
                soup_scoreboard = BeautifulSoup(page_source_scoreboard, 'lxml')
                game_sections = soup_scoreboard.find_all('section', class_='Scoreboard')
                logging.info(f"Found {len(game_sections)} game sections.")

                # Robust Link Finding Logic
                for section in game_sections:
                    # Check for Spring Training or Postponed/Canceled first
                    st_note = section.find(lambda tag: tag.name == 'div' and ('ScoreboardScoreCell__Note' in tag.get('class', []) or 'ScoreCell__Time' in tag.get('class', [])) and re.search(r'Spring Training|Postponed|Canceled|Suspended', tag.get_text(), re.IGNORECASE))
                    if st_note:
                        logging.debug(f"Skipping section due to note: {st_note.get_text(strip=True)}")
                        continue

                    # Try finding the Gamecast link via common patterns
                    gamecast_link_tag = None
                    # Pattern 1: Explicit 'Gamecast' button/link
                    gamecast_link_tag = section.find('a', href=re.compile(r'/mlb/game/_/gameId/'), string=re.compile(r'Gamecast', re.I))
                    # Pattern 2: Link within callouts section (often contains Gamecast or Box Score)
                    if not gamecast_link_tag:
                        callouts_div = section.find('div', class_='Scoreboard__Callouts')
                        if callouts_div:
                             links = callouts_div.find_all('a', href=re.compile(r'/mlb/game/_/gameId/'))
                             if links: gamecast_link_tag = links[0] # Take the first likely relevant link

                    # Fallback: Link on the team names/scores container itself (less reliable)
                    if not gamecast_link_tag:
                        score_link = section.find('a', class_='Scoreboard__Link', href=re.compile(r'/mlb/game/_/gameId/'))
                        if score_link: gamecast_link_tag = score_link


                    if gamecast_link_tag and gamecast_link_tag.has_attr('href'):
                        link = gamecast_link_tag['href']
                        if not link.startswith('http'):
                             link = "https://www.espn.com" + link
                        # Basic validation of the game ID format
                        if re.search(r'gameId/\d+', link):
                             if link not in gamecast_links_to_visit:
                                  gamecast_links_to_visit.append(link)
                                  logging.debug(f"Found Gamecast link: {link}")
                        else:
                             logging.warning(f"Found link with potentially invalid gameId format, skipping: {link}")
                    # else:
                        # logging.debug(f"No Gamecast link found in section for {date_str}")


            except TimeoutException: logging.warning(f"Timeout waiting for scoreboard elements on {date_str}. Skipping."); continue
            except Exception as e: logging.error(f"Error processing scoreboard page {date_str}: {e}"); continue
            # --- End Selenium for Scoreboard ---

            num_links_found = len(gamecast_links_to_visit)
            logging.info(f"Found {num_links_found} valid Gamecast links for {date_str}.")

            # Debug HTML Saving (Keep as before)
            if len(game_sections) > 0 and num_links_found <= 0 and page_source_scoreboard: # Adjusted condition slightly
                 try:
                     filename = os.path.join(DEBUG_HTML_DIR, f"scoreboard_no_links_{date_str}.html")
                     with open(filename, "w", encoding="utf-8") as f: f.write(BeautifulSoup(page_source_scoreboard, 'html.parser').prettify())
                     logging.warning(f"No links found on {date_str} despite game sections. Saved HTML: {filename}")
                 except Exception as save_e: logging.error(f"Could not save debug scoreboard HTML: {save_e}")


            # --- Asyncio/httpx for Gamecasts ---
            if gamecast_links_to_visit:
                 logging.info(f"Fetching {num_links_found} Gamecast pages asynchronously...")
                 start_async = time.time()
                 # Run the async processing function
                 daily_game_data = asyncio.run(process_daily_gamecasts(gamecast_links_to_visit, date_str))
                 duration_async = time.time() - start_async
                 logging.info(f"Async fetch/parse for {date_str} took {duration_async:.2f}s.")
            # --- End Asyncio/httpx ---

            # --- Checkpoint Save ---
            if daily_game_data: # Only extend if new data was fetched
                 all_game_data.extend(daily_game_data) # Add new data (if any)

            try:
                if not all_game_data: # Handle case where no data exists yet
                     logging.info(f"No data collected yet after processing {date_str}. Skipping save/checkpoint.")
                     continue # Move to the next date

                current_df = pd.DataFrame(all_game_data) # Create DF from potentially updated list
                cols_order = ['game_date', 'home_team', 'away_team', 'home_plate_umpire', 'first_base_umpire', 'second_base_umpire', 'third_base_umpire', 'url']
                for col in cols_order:
                    if col not in current_df.columns: current_df[col] = None # Add missing columns

                # Drop duplicates based on essential game info before saving
                # Keep the 'last' entry in case retries fetched the same game multiple times successfully
                key_cols = ['game_date', 'home_team', 'away_team']
                current_df_unique = current_df.drop_duplicates(subset=key_cols, keep='last')

                # Reorder columns for CSV output (drop URL before saving)
                final_cols_order = [col for col in cols_order if col != 'url']
                current_df_to_save = current_df_unique[final_cols_order]

                # Save the potentially updated, ordered, and deduplicated data
                current_df_to_save.to_csv(OUTPUT_CSV, index=False)
                save_checkpoint(CHECKPOINT_FILE, date_str) # Save checkpoint ONLY after successful save
                logging.info(f"Checkpoint saved for {date_str}. Total unique records: {len(current_df_to_save)}")

                # Update in-memory list to only contain the unique records to prevent memory bloat
                all_game_data = current_df_unique.to_dict('records')

            except Exception as e:
                logging.error(f"CRITICAL: Failed to save/checkpoint for {date_str}: {e}. Stopping.")
                break # Stop processing if saving fails
            # --- End Checkpoint Save ---
            processed_dates_count += 1

    except KeyboardInterrupt: logging.warning("KeyboardInterrupt received.")
    except Exception as e: logging.error(f"Error in main loop: {e}"); import traceback; logging.error(traceback.format_exc())
    finally:
        if driver: driver.quit(); logging.info("WebDriver closed.")

    # --- Final Processing ---
    logging.info(f"Scraping loop finished. Processed {processed_dates_count} dates in this run.")
    if not all_game_data: logging.warning("No data scraped or loaded."); return None

    # Reload the final saved CSV to ensure consistency and proper deduplication one last time
    final_df = None
    try:
        if os.path.exists(OUTPUT_CSV):
             final_df = pd.read_csv(OUTPUT_CSV)
             logging.info(f"Final dataset loaded from {OUTPUT_CSV}: {len(final_df)} records.")
        else:
             logging.warning(f"Output file {OUTPUT_CSV} not found at the end, returning in-memory data.")
             # Fallback to potentially non-deduplicated in-memory data if file doesn't exist
             final_df = pd.DataFrame(all_game_data)
             if not final_df.empty:
                  # Apply final ordering/deduplication again if using in-memory data
                  cols_order = ['game_date', 'home_team', 'away_team', 'home_plate_umpire', 'first_base_umpire', 'second_base_umpire', 'third_base_umpire']
                  for col in cols_order:
                       if col not in final_df.columns: final_df[col] = None
                  final_df['game_date'] = pd.to_datetime(final_df['game_date']).dt.strftime('%Y-%m-%d')
                  final_df = final_df.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='last')
                  final_df = final_df[[col for col in cols_order if col in final_df.columns]] # Ensure final cols exist

    except Exception as e:
         logging.error(f"Error loading/processing final CSV {OUTPUT_CSV}: {e}")
         final_df = pd.DataFrame(all_game_data) # Fallback again


    return final_df


# --- Execution ---
if __name__ == "__main__":
    # Ensure all helper/async functions are defined above this block
    logging.info("--- Starting Optimized ESPN MLB Scraper with Seasonal Skipping ---")
    final_data = scrape_espn_data_optimized(START_DATE, END_DATE, WEBDRIVER_PATH)

    if final_data is not None and not final_data.empty:
        logging.info(f"--- Script finished. {len(final_data)} total unique records found in {OUTPUT_CSV}. ---")
        print(f"\nData potentially saved to {OUTPUT_CSV}") # Changed message as final save might differ
        print("\nFinal data sample (tail):")
        print(final_data.tail())
    elif final_data is not None and final_data.empty:
         logging.info("--- Script finished. No data collected or loaded (check date range/season bounds/checkpoints). ---")
         print(f"\nNo data collected. Check {OUTPUT_CSV} for any previously saved data.")
    else:
        logging.error("--- Script finished with errors or no data. ---")
        print(f"\nScript encountered errors or collected no data. Check logs and {OUTPUT_CSV} for partial data.")