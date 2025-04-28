# src/data/utils.py
import re
import logging
from pathlib import Path
import pandas as pd
from src.config import DBConfig, LogConfig
import sqlite3
from datetime import datetime

def normalize_name(name):
    """
    Normalize player names for better matching
    
    Args:
        name (str): Player name to normalize
        
    Returns:
        str: Normalized player name
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove all accented characters
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace('ñ', 'n').replace('ç', 'c').replace('ü', 'u')
    
    # Remove suffixes like Jr., Sr., III
    name = re.sub(r'\b(jr|jr\.|sr|sr\.|iii|ii|iv)\b', '', name)
    
    # Remove all punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Handle lastname, firstname format
    if ", " in name:
        parts = name.split(", ", 1)
        if len(parts) == 2:
            last, first = parts
            name = f"{first} {last}"
    
    return name

def setup_logger(name, log_file=None, level=logging.DEBUG):
    """Set up a logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling NA values"""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def ensure_dir(path):
    """Ensure a directory exists and return Path object"""
    p = Path(path)
    p.mkdir(exist_ok=True, parents=True)
    return p

class DBConnection:
    """Context manager for database connections"""
    def __init__(self, db_name=DBConfig.PATH):
        self.db_name = db_name

    def __enter__(self):
        ensure_dir(Path(self.db_name).parent)
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "conn"):
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()

logger = setup_logger('utils', LogConfig.LOG_DIR / 'utils.log')

def find_latest_file(directory, pattern):
    """
    Finds the most recent file in a directory matching a glob pattern,
    based on timestamp in the filename or modification time.

    Args:
        directory (str or Path): The directory to search in.
        pattern (str): The glob pattern to match files (e.g., "*.pkl").

    Returns:
        Path or None: The Path object of the latest file, or None if no match found.
    """
    try:
        search_dir = Path(directory)
        if not search_dir.is_dir():
            logger.error(f"Directory not found: {search_dir}")
            return None

        files = list(search_dir.glob(pattern))
        if not files:
            # Use logging.info or logging.debug for non-critical missing files
            logger.info(f"No files found matching pattern '{pattern}' in {search_dir}")
            return None

        latest_file, latest_timestamp = None, 0
        # Matches _YYYYMMDD_HHMMSS. before the extension (e.g., _20250416_091259.)
        ts_pattern = re.compile(r"_(\d{8}_\d{6})\.")
        parsed_successfully = False

        for f in files:
            match = ts_pattern.search(f.name)
            if match:
                try:
                    # Extract timestamp string and parse it
                    ts_str = match.group(1)
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").timestamp()
                    if ts > latest_timestamp:
                        latest_timestamp = ts
                        latest_file = f
                    parsed_successfully = True
                except ValueError:
                    # Log parsing errors, but don't stop the process
                    logger.warning(f"Could not parse timestamp from filename: {f.name}")
                except Exception as e:
                    logger.warning(f"Error processing timestamp for file {f.name}: {e}")


        # Fallback to modification time if no timestamps were parsed or found
        if not parsed_successfully and files:
            logger.warning(f"Could not determine latest file by timestamp pattern '{ts_pattern.pattern}' for '{pattern}'. Falling back to modification time.")
            try:
                # Get the file with the maximum modification time
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                parsed_successfully = True # Mark as successful using mtime
                logger.info(f"Determined latest file by modification time: {latest_file.name}")
            except Exception as e:
                logger.error(f"Error finding latest file by modification time for pattern '{pattern}': {e}")
                return None # Return None if mtime fallback also fails

        # Final logging based on outcome
        if latest_file:
            logger.info(f"Found latest file for pattern '{pattern}': {latest_file.name}")
        elif parsed_successfully: # Should not happen if max() worked, but belt-and-suspenders
             logger.warning(f"Parsed successfully but latest_file is None for '{pattern}'.")
        else:
             logger.error(f"Could not find latest file for pattern '{pattern}' using any method.")

        return latest_file

    except Exception as e:
        logger.error(f"Unexpected error in find_latest_file: {e}", exc_info=True)
        return None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_nan_counts(df, df_name="DataFrame", columns=None):
    """Logs NaN counts and percentages for specified columns or all columns."""
    if columns is None:
        columns = df.columns
    
    logging.info(f"--- NaN Report for {df_name} ---")
    total_rows = len(df)
    if total_rows == 0:
        logging.info("DataFrame is empty.")
        return

    nan_info = df[columns].isna().sum()
    nan_info = nan_info[nan_info > 0] # Only report columns with NaNs

    if nan_info.empty:
        logging.info("No NaNs found in the specified columns.")
    else:
        nan_percentage = (nan_info / total_rows) * 100
        nan_summary = pd.DataFrame({'NaN Count': nan_info, 'NaN Percentage': nan_percentage.round(2)})
        logging.info(f"\nTotal Rows: {total_rows}\nNaN Summary (Columns with NaNs):\n{nan_summary.to_string()}")
    logging.info(f"--- End NaN Report for {df_name} ---")