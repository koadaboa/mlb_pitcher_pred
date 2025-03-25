import sqlite3
import pandas as pd
from pathlib import Path

def get_db_connection(db_name='pitcher_stats.db'):
    """Get a connection to the SQLite database"""
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    db_path = data_dir / db_name
    return sqlite3.connect(db_path)

def execute_query(query, params=None):
    """Execute a query and return results as a DataFrame"""
    conn = get_db_connection()
    try:
        if params:
            return pd.read_sql_query(query, conn, params=params)
        else:
            return pd.read_sql_query(query, conn)
    finally:
        conn.close()