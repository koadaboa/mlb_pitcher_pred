# src/scripts/update_schema.py
import argparse
from src.data.db import update_database_schema
from src.data.utils import setup_logger

logger = setup_logger(__name__)

def main():
    """Add new columns to database for enhanced features"""
    logger.info("Updating database schema for enhanced features...")
    update_database_schema()
    logger.info("Schema update complete!")
    return 0

if __name__ == "__main__":
    main()