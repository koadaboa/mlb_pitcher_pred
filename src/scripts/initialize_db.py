# src/scripts/initialize_db.py
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

from src.data.db import init_database, clear_database

def setup_database(clear_existing=False):
    """
    Set up the database structure only (no data fetching or processing)
    
    Args:
        clear_existing (bool): Whether to clear existing database before setup
    """
    try:
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        # Optionally clear the database
        if clear_existing:
            logger.info("Clearing existing database...")
            clear_database()
        
        # Initialize database structure
        logger.info("Initializing database structure...")
        init_database()
        logger.info("Database structure initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    setup_database(clear_existing=False)