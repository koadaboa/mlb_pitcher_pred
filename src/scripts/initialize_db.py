# src/scripts/initialize_db.py

import logging
import os
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import necessary functions
from src.data.db import init_database, get_db_connection

def ensure_database_initialization():
    """Ensure database is properly initialized with all required tables"""
    db_path = Path("data/pitcher_stats.db")
    logger.info(f"Checking database at {db_path.absolute()}")
    
    # Check if database file exists
    if db_path.exists():
        logger.info(f"Database file exists, size: {db_path.stat().st_size / 1024:.2f} KB")
    else:
        logger.info("Database file does not exist, will create it")
    
    # Initialize database (creates tables if they don't exist)
    init_database()
    
    # Verify tables were created
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    logger.info(f"Tables in database: {tables}")
    
    # Check if all required tables exist
    required_tables = ["pitchers", "game_stats", "traditional_stats", "pitch_mix", "prediction_features"]
    missing_tables = [table for table in required_tables if table not in tables]
    
    if missing_tables:
        logger.error(f"Missing tables: {missing_tables}")
        return False
    
    # Check table structures
    for table in required_tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        logger.info(f"Table {table} has {len(columns)} columns")
    
    conn.close()
    logger.info("Database initialization verified successfully")
    return True

if __name__ == "__main__":
    ensure_database_initialization()