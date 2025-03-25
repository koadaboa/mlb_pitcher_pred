# Pitcher Performance Analysis: Statcast and Traditional Stats
# Purpose: Create datasets for predicting strikeouts and ERA at the pitcher-game level
# Uses SQLite for data storage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import date, timedelta
import warnings
import os
import pickle
from pathlib import Path
import sqlite3

# Import pybaseball library
import pybaseball
from pybaseball import statcast, statcast_pitcher, pitching_stats
from pybaseball import cache

# Enable pybaseball cache for large data requests
cache.enable()
print("PyBaseball cache enabled.")

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Global variables
SEASONS = [2019, 2021,2022, 2023, 2024]
RATE_LIMIT_PAUSE = 5  # seconds to wait between API calls
DB_PATH = "data/pitcher_stats.db"  # SQLite database path

# Initialize the SQLite database and create schema
def init_database():
    """
    Initialize the SQLite database with the necessary tables
    """
    print("Initializing SQLite database...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    
    # Pitchers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pitchers (
        pitcher_id INTEGER PRIMARY KEY,
        player_name TEXT,
        statcast_id INTEGER,
        traditional_id INTEGER
    )
    ''')
    
    # Traditional season-level stats
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS traditional_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pitcher_id INTEGER,
        season INTEGER,
        team TEXT,
        era REAL,
        k_per_9 REAL,
        bb_per_9 REAL,
        k_bb_ratio REAL,
        whip REAL,
        babip REAL,
        lob_pct REAL,
        fip REAL,
        xfip REAL,
        war REAL,
        FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id)
    )
    ''')
    
    # Game-level Statcast data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS game_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pitcher_id INTEGER,
        game_id TEXT,
        game_date TEXT,
        season INTEGER,
        strikeouts INTEGER,
        hits INTEGER,
        walks INTEGER,
        home_runs INTEGER,
        release_speed_mean REAL,
        release_speed_max REAL,
        release_spin_rate_mean REAL,
        swinging_strike_pct REAL,
        called_strike_pct REAL,
        zone_rate REAL,
        FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id)
    )
    ''')
    
    # Pitch mix table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pitch_mix (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_stats_id INTEGER,
        pitch_type TEXT,
        percentage REAL,
        FOREIGN KEY (game_stats_id) REFERENCES game_stats(id)
    )
    ''')
    
    # Features table for ML models
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pitcher_id INTEGER,
        game_id TEXT,
        game_date TEXT,
        season INTEGER,
        last_3_games_strikeouts_avg REAL,
        last_5_games_strikeouts_avg REAL,
        last_3_games_k9_avg REAL,
        last_5_games_k9_avg REAL,
        last_3_games_era_avg REAL,
        last_5_games_era_avg REAL,
        last_3_games_fip_avg REAL,
        last_5_games_fip_avg REAL,
        last_3_games_velo_avg REAL,
        last_5_games_velo_avg REAL,
        last_3_games_swinging_strike_pct_avg REAL,
        last_5_games_swinging_strike_pct_avg REAL,
        days_rest INTEGER,
        team_changed BOOLEAN,
        FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database initialization complete.")

# Function to safely fetch statcast data with error handling and rate limiting
def fetch_statcast_safely(start_date, end_date, max_retries=3):
    """
    Fetch statcast data with error handling and rate limiting
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        max_retries (int): Maximum number of retries in case of failure
    Returns:
        pandas.DataFrame: Statcast data
    """
    retries = 0
    while retries < max_retries:
        try:
            print(f"Fetching data from {start_date} to {end_date}...")
            data = statcast(start_dt=start_date, end_dt=end_date)
            time.sleep(RATE_LIMIT_PAUSE)  # Respect rate limits
            return data
        except pd.errors.ParserError as e:
            print(f"Parser error encountered: {e}")
            print("Attempting to handle parser error by manually adjusting request parameters...")
            # Try smaller date range as a workaround
            start_dt_obj = pd.to_datetime(start_date)
            end_dt_obj = pd.to_datetime(end_date)
            mid_dt_obj = start_dt_obj + (end_dt_obj - start_dt_obj) / 2
            mid_dt = mid_dt_obj.strftime('%Y-%m-%d')
            
            print(f"Splitting request into two: {start_date} to {mid_dt} and {mid_dt} to {end_date}")
            try:
                df1 = statcast(start_dt=start_date, end_dt=mid_dt)
                time.sleep(RATE_LIMIT_PAUSE)
                df2 = statcast(start_dt=mid_dt, end_dt=end_date)
                time.sleep(RATE_LIMIT_PAUSE)
                return pd.concat([df1, df2], ignore_index=True)
            except Exception as nested_error:
                print(f"Nested error: {nested_error}")
                retries += 1
                print(f"Retrying ({retries}/{max_retries})...")
                time.sleep(RATE_LIMIT_PAUSE * 2)  # Longer pause before retry
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            print(f"Retrying ({retries}/{max_retries})...")
            time.sleep(RATE_LIMIT_PAUSE * 2)  # Longer pause before retry
    
    print(f"Failed to fetch data after {max_retries} retries")
    return pd.DataFrame()  # Return empty DataFrame if all retries fail

# Function to fetch data for a season in chunks
def fetch_season_in_chunks(season, chunk_size=14):
    """
    Fetch statcast data for a whole season in smaller chunks to avoid timeout/memory issues
    Args:
        season (int): MLB season year
        chunk_size (int): Number of days per chunk
    Returns:
        pandas.DataFrame: Season's statcast data
    """
    # Define season start and end dates (approximate MLB season)
    if season == 2024:
        # 2024 season started on March 28
        season_start = f"{season}-03-28"
        # Use current date as end if we're in 2024
        if date.today().year == 2024:
            season_end = date.today().strftime('%Y-%m-%d')
        else:
            # Otherwise go through end of regular season (approximate)
            season_end = f"{season}-10-01"
    else:
        # Regular seasons (approximate dates)
        season_start = f"{season}-04-01"
        season_end = f"{season}-10-01"
    
    # Convert to datetime for easier manipulation
    start_dt = pd.to_datetime(season_start)
    end_dt = pd.to_datetime(season_end)
    
    all_data = []
    
    # Create chunks of dates
    current_start = start_dt
    total_chunks = (end_dt - start_dt).days // chunk_size + 1
    
    with tqdm(total=total_chunks, desc=f"Season {season}") as pbar:
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size), end_dt)
            
            chunk_data = fetch_statcast_safely(
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            )
            
            if not chunk_data.empty:
                chunk_data['season'] = season
                all_data.append(chunk_data)
            
            current_start = current_end + timedelta(days=1)
            pbar.update(1)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

# Function to fetch traditional pitching stats with error handling
def fetch_pitching_stats_safely(season, max_retries=3):
    """
    Fetch traditional pitching stats with error handling and rate limiting
    Args:
        season (int): MLB season year
        max_retries (int): Maximum number of retries
    Returns:
        pandas.DataFrame: Traditional pitching stats
    """
    retries = 0
    while retries < max_retries:
        try:
            print(f"Fetching traditional pitching stats for {season}...")
            data = pitching_stats(season)
            time.sleep(RATE_LIMIT_PAUSE)  # Respect rate limits
            return data
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            print(f"Retrying ({retries}/{max_retries})...")
            time.sleep(RATE_LIMIT_PAUSE * 2)
    
    print(f"Failed to fetch pitching stats for {season} after {max_retries} retries")
    return pd.DataFrame()

# Function to check if a table is populated in the database
def is_table_populated(table_name):
    """
    Check if a table in the database has data
    Args:
        table_name (str): Table name to check
    Returns:
        bool: True if the table has data
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    
    conn.close()
    
    return count > 0

# Function to store statcast data in the database
def store_statcast_data(statcast_df, force_refresh=False):
    """
    Process and store statcast data in SQLite database
    Args:
        statcast_df (pandas.DataFrame): Raw statcast data
        force_refresh (bool): Whether to force refresh existing data
    """
    if statcast_df.empty:
        print("No statcast data to store.")
        return
    
    # Check if we need to refresh the data
    if not force_refresh and is_table_populated('game_stats'):
        print("Game stats table already populated and force_refresh is False. Skipping.")
        return
    
    print("Processing and storing statcast data...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data if force_refresh
    if force_refresh:
        cursor.execute("DELETE FROM pitch_mix")
        cursor.execute("DELETE FROM game_stats")
        conn.commit()
    
    # Process statcast data to game level
    game_level = aggregate_statcast_to_game_level(statcast_df)
    
    # Add pitchers to the pitchers table if they don't exist
    for _, row in game_level[['pitcher', 'player_name']].drop_duplicates().iterrows():
        cursor.execute(
            "INSERT OR IGNORE INTO pitchers (statcast_id, player_name) VALUES (?, ?)",
            (int(row['pitcher']), row['player_name'])
        )
    
    conn.commit()
    
    # Get pitcher IDs mapping
    cursor.execute("SELECT pitcher_id, statcast_id FROM pitchers WHERE statcast_id IS NOT NULL")
    pitcher_map = {statcast_id: pitcher_id for pitcher_id, statcast_id in cursor.fetchall()}
    
    # Insert game stats
    game_stats_ids = {}  # Map to store game_id -> database_id for pitch mix data
    
    for _, row in game_level.iterrows():
        # Get pitcher_id from mapping
        statcast_id = row['pitcher']
        if statcast_id not in pitcher_map:
            print(f"Warning: No pitcher_id found for statcast_id {statcast_id}")
            continue
        
        pitcher_id = pitcher_map[statcast_id]
        
        # Insert game stats
        game_date = row['game_date'].strftime('%Y-%m-%d')
        game_id = row['game_id']
        season = row['game_date'].year
        
        # Check if this game already exists for this pitcher
        cursor.execute(
            "SELECT id FROM game_stats WHERE pitcher_id = ? AND game_id = ?",
            (pitcher_id, game_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            game_stats_id = existing[0]
        else:
            # Insert new game stats
            cursor.execute('''
                INSERT INTO game_stats (
                    pitcher_id, game_id, game_date, season, 
                    strikeouts, hits, walks, home_runs,
                    release_speed_mean, release_speed_max, 
                    release_spin_rate_mean, swinging_strike_pct,
                    called_strike_pct, zone_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pitcher_id, game_id, game_date, season,
                row.get('strikeouts', 0), row.get('hits', 0), row.get('walks', 0), row.get('home_runs', 0),
                row.get('release_speed_mean', 0), row.get('release_speed_max', 0),
                row.get('release_spin_rate_mean', 0), row.get('swinging_strike_pct', 0),
                row.get('called_strike_pct', 0), row.get('zone', 0)
            ))
            
            game_stats_id = cursor.lastrowid
        
        # Store ID for pitch mix data
        game_key = (pitcher_id, game_id)
        game_stats_ids[game_key] = game_stats_id
        
        # Store pitch mix data if available
        pitch_cols = [col for col in row.index if col.startswith('pitch_pct_')]
        
        for col in pitch_cols:
            pitch_type = col.replace('pitch_pct_', '')
            percentage = row[col]
            
            if percentage > 0:
                cursor.execute(
                    "INSERT OR REPLACE INTO pitch_mix (game_stats_id, pitch_type, percentage) VALUES (?, ?, ?)",
                    (game_stats_id, pitch_type, percentage)
                )
    
    conn.commit()
    conn.close()
    
    print("Statcast data stored in database.")

# Function to store traditional stats in the database
def store_traditional_stats(trad_df, force_refresh=False):
    """
    Process and store traditional stats in SQLite database
    Args:
        trad_df (pandas.DataFrame): Traditional stats data
        force_refresh (bool): Whether to force refresh existing data
    """
    if trad_df.empty:
        print("No traditional stats to store.")
        return
    
    # Check if we need to refresh the data
    if not force_refresh and is_table_populated('traditional_stats'):
        print("Traditional stats table already populated and force_refresh is False. Skipping.")
        return
    
    print("Processing and storing traditional stats...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data if force_refresh
    if force_refresh:
        cursor.execute("DELETE FROM traditional_stats")
        conn.commit()
    
    # Process traditional stats
    processed_trad = process_traditional_stats(trad_df)
    
    # Add pitchers to the pitchers table if they don't exist
    for _, row in processed_trad[['pitcher_id', 'Name']].drop_duplicates().iterrows():
        cursor.execute(
            "INSERT OR IGNORE INTO pitchers (traditional_id, player_name) VALUES (?, ?)",
            (int(row['pitcher_id']), row['Name'])
        )
    
    conn.commit()
    
    # Get pitcher IDs mapping
    cursor.execute("SELECT pitcher_id, traditional_id FROM pitchers WHERE traditional_id IS NOT NULL")
    trad_id_map = {trad_id: pitcher_id for pitcher_id, trad_id in cursor.fetchall()}
    
    # Insert traditional stats
    for _, row in processed_trad.iterrows():
        trad_id = row['pitcher_id']
        if trad_id not in trad_id_map:
            print(f"Warning: No internal pitcher_id found for traditional_id {trad_id}")
            continue
        
        pitcher_id = trad_id_map[trad_id]
        season = row['Season']
        
        # Check if stats for this pitcher and season already exist
        cursor.execute(
            "SELECT id FROM traditional_stats WHERE pitcher_id = ? AND season = ?",
            (pitcher_id, season)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE traditional_stats
                SET team = ?, era = ?, k_per_9 = ?, bb_per_9 = ?, k_bb_ratio = ?,
                    whip = ?, babip = ?, lob_pct = ?, fip = ?, xfip = ?, war = ?
                WHERE id = ?
            ''', (
                row.get('Team', ''),
                row.get('ERA', 0.0),
                row.get('K/9', 0.0),
                row.get('BB/9', 0.0),
                row.get('K/BB', 0.0),
                row.get('WHIP', 0.0),
                row.get('BABIP', 0.0),
                row.get('LOB%', 0.0),
                row.get('FIP', 0.0),
                row.get('xFIP', 0.0),
                row.get('WAR', 0.0),
                existing[0]
            ))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO traditional_stats (
                    pitcher_id, season, team, era, k_per_9, bb_per_9, k_bb_ratio,
                    whip, babip, lob_pct, fip, xfip, war
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pitcher_id, season,
                row.get('Team', ''),
                row.get('ERA', 0.0),
                row.get('K/9', 0.0),
                row.get('BB/9', 0.0),
                row.get('K/BB', 0.0),
                row.get('WHIP', 0.0),
                row.get('BABIP', 0.0),
                row.get('LOB%', 0.0),
                row.get('FIP', 0.0),
                row.get('xFIP', 0.0),
                row.get('WAR', 0.0)
            ))
    
    conn.commit()
    conn.close()
    
    print("Traditional stats stored in database.")

# Function to map pitcher IDs between Statcast and traditional stats
def update_pitcher_mapping():
    """
    Create mapping between Statcast and traditional stats pitcher IDs in the database
    """
    print("Updating pitcher ID mappings...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First, try to map based on exact name matches
    cursor.execute('''
        UPDATE pitchers AS p1
        SET traditional_id = (
            SELECT traditional_id FROM pitchers AS p2
            WHERE p2.player_name = p1.player_name
            AND p2.traditional_id IS NOT NULL
            LIMIT 1
        )
        WHERE p1.statcast_id IS NOT NULL
        AND p1.traditional_id IS NULL
    ''')
    
    # Check how many pitchers were mapped
    cursor.execute('''
        SELECT COUNT(*) FROM pitchers 
        WHERE statcast_id IS NOT NULL 
        AND traditional_id IS NOT NULL
    ''')
    mapped_count = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(*) FROM pitchers 
        WHERE statcast_id IS NOT NULL
    ''')
    total_statcast = cursor.fetchone()[0]
    
    # If less than 20% were mapped, try more aggressive name matching (last name only)
    if mapped_count < total_statcast * 0.2:
        print(f"Only {mapped_count}/{total_statcast} pitchers mapped with exact names. Using more aggressive matching...")
        
        # Get all pitchers with statcast_id but no traditional_id
        cursor.execute('''
            SELECT pitcher_id, player_name, statcast_id
            FROM pitchers
            WHERE statcast_id IS NOT NULL
            AND traditional_id IS NULL
        ''')
        unmatched_statcast = cursor.fetchall()
        
        # Get all pitchers with traditional_id
        cursor.execute('''
            SELECT pitcher_id, player_name, traditional_id
            FROM pitchers
            WHERE traditional_id IS NOT NULL
        ''')
        trad_pitchers = cursor.fetchall()
        
        # Create dictionaries for last name matching
        trad_last_names = {}
        for pid, name, trad_id in trad_pitchers:
            last_name = name.split()[-1].lower() if name and ' ' in name else name.lower()
            if last_name not in trad_last_names:
                trad_last_names[last_name] = []
            trad_last_names[last_name].append((pid, trad_id))
        
        # Match by last name
        for pid, name, statcast_id in unmatched_statcast:
            last_name = name.split()[-1].lower() if name and ' ' in name else name.lower()
            if last_name in trad_last_names and len(trad_last_names[last_name]) == 1:
                trad_pid, trad_id = trad_last_names[last_name][0]
                cursor.execute(
                    "UPDATE pitchers SET traditional_id = ? WHERE pitcher_id = ?",
                    (trad_id, pid)
                )
    
    conn.commit()
    
    # Check how many pitchers were mapped
    cursor.execute('''
        SELECT COUNT(*) FROM pitchers 
        WHERE statcast_id IS NOT NULL 
        AND traditional_id IS NOT NULL
    ''')
    mapped_count = cursor.fetchone()[0]
    
    print(f"Mapped {mapped_count}/{total_statcast} pitchers between Statcast and traditional stats.")
    
    conn.close()

# Function to retrieve data from the database for feature engineering
def get_pitcher_data():
    """
    Retrieve joined data from the database for feature engineering
    Returns:
        pandas.DataFrame: Joined data from game_stats and traditional_stats
    """
    print("Retrieving pitcher data from database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Join game_stats with traditional_stats using pitcher_id and season
    query = '''
    SELECT 
        g.id as game_db_id,
        g.pitcher_id,
        p.player_name,
        g.game_id,
        g.game_date,
        g.season,
        g.strikeouts,
        g.hits,
        g.walks,
        g.home_runs,
        g.release_speed_mean,
        g.release_speed_max,
        g.release_spin_rate_mean,
        g.swinging_strike_pct,
        g.called_strike_pct,
        g.zone_rate,
        t.team,
        t.era,
        t.k_per_9,
        t.bb_per_9,
        t.k_bb_ratio,
        t.whip,
        t.babip,
        t.lob_pct,
        t.fip,
        t.xfip,
        t.war
    FROM 
        game_stats g
    JOIN 
        pitchers p ON g.pitcher_id = p.pitcher_id
    LEFT JOIN 
        traditional_stats t ON g.pitcher_id = t.pitcher_id AND g.season = t.season
    ORDER BY
        g.pitcher_id, g.game_date
    '''
    
    df = pd.read_sql_query(query, conn)
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Get pitch mix data for each game
    query_pitch_mix = '''
    SELECT 
        gs.id as game_db_id,
        pm.pitch_type,
        pm.percentage
    FROM 
        game_stats gs
    JOIN 
        pitch_mix pm ON gs.id = pm.game_stats_id
    '''
    
    pitch_mix_df = pd.read_sql_query(query_pitch_mix, conn)
    
    conn.close()
    
    # Add pitch mix columns to the main dataframe
    if not pitch_mix_df.empty:
        # Pivot the pitch mix data
        pitch_mix_pivot = pitch_mix_df.pivot(
            index='game_db_id', 
            columns='pitch_type', 
            values='percentage'
        ).reset_index()
        
        # Rename columns with 'pitch_pct_' prefix
        pitch_mix_pivot.columns = ['game_db_id'] + [f'pitch_pct_{col}' for col in pitch_mix_pivot.columns[1:]]
        
        # Merge with the main dataframe
        df = pd.merge(df, pitch_mix_pivot, on='game_db_id', how='left')
    
    print(f"Retrieved {len(df)} rows of pitcher data.")
    return df

# Function to create prediction features and store in database
def create_prediction_features(force_refresh=False):
    """
    Create and store prediction features in the database
    Args:
        force_refresh (bool): Whether to force refresh existing features
    """
    # Check if we need to refresh the data
    if not force_refresh and is_table_populated('prediction_features'):
        print("Prediction features table already populated and force_refresh is False. Skipping.")
        return
    
    print("Creating prediction features...")
    
    # Get the data from database
    df = get_pitcher_data()
    
    if df.empty:
        print("No data available for feature engineering.")
        return
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing features if force_refresh
    if force_refresh:
        cursor.execute("DELETE FROM prediction_features")
        conn.commit()
    
    # Create a dataframe to store the features
    features = []
    
    # Process each pitcher separately
    for pitcher_id, pitcher_data in df.groupby('pitcher_id'):
        # Sort by game date
        pitcher_data = pitcher_data.sort_values('game_date')
        
        for window in [3, 5]:
            # Use shift to create lagged features properly - this prevents data leakage
            # by only using past data for each observation
            pitcher_data[f'last_{window}_games_strikeouts_avg'] = pitcher_data['strikeouts'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_k9_avg'] = pitcher_data['k_per_9'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_era_avg'] = pitcher_data['era'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_fip_avg'] = pitcher_data['fip'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_velo_avg'] = pitcher_data['release_speed_mean'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_swinging_strike_pct_avg'] = pitcher_data['swinging_strike_pct'].rolling(
                window=window, min_periods=1).mean().shift(1)
        
        # Calculate days of rest
        pitcher_data['prev_game_date'] = pitcher_data['game_date'].shift(1)
        pitcher_data['days_rest'] = (pitcher_data['game_date'] - pitcher_data['prev_game_date']).dt.days
        pitcher_data['days_rest'] = pitcher_data['days_rest'].fillna(5)  # Default to 5 days for first appearance
        
        # Create team changed flag
        pitcher_data['team_changed'] = pitcher_data['team'].shift(1) != pitcher_data['team']
        pitcher_data['team_changed'] = pitcher_data['team_changed'].fillna(False).astype(int)
        
        # Add to features dataset
        features.append(pitcher_data)
    
    # Combine all pitcher features
    if features:
        all_features = pd.concat(features, ignore_index=True)
        
        # Fill NA values
        all_features = all_features.fillna(0)
        
        # Insert into database
        for _, row in all_features.iterrows():
            # Check if this game already has features
            cursor.execute(
                "SELECT id FROM prediction_features WHERE pitcher_id = ? AND game_id = ?",
                (row['pitcher_id'], row['game_id'])
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE prediction_features
                    SET 
                        last_3_games_strikeouts_avg = ?,
                        last_5_games_strikeouts_avg = ?,
                        last_3_games_k9_avg = ?,
                        last_5_games_k9_avg = ?,
                        last_3_games_era_avg = ?,
                        last_5_games_era_avg = ?,
                        last_3_games_fip_avg = ?,
                        last_5_games_fip_avg = ?,
                        last_3_games_velo_avg = ?,
                        last_5_games_velo_avg = ?,
                        last_3_games_swinging_strike_pct_avg = ?,
                        last_5_games_swinging_strike_pct_avg = ?,
                        days_rest = ?,
                        team_changed = ?
                    WHERE id = ?
                ''', (
                    row['last_3_games_strikeouts_avg'],
                    row['last_5_games_strikeouts_avg'],
                    row['last_3_games_k9_avg'],
                    row['last_5_games_k9_avg'],
                    row['last_3_games_era_avg'],
                    row['last_5_games_era_avg'],
                    row['last_3_games_fip_avg'],
                    row['last_5_games_fip_avg'],
                    row['last_3_games_velo_avg'],
                    row['last_5_games_velo_avg'],
                    row['last_3_games_swinging_strike_pct_avg'],
                    row['last_5_games_swinging_strike_pct_avg'],
                    row['days_rest'],
                    row['team_changed'],
                    existing[0]
                ))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO prediction_features (
                        pitcher_id, game_id, game_date, season,
                        last_3_games_strikeouts_avg, last_5_games_strikeouts_avg,
                        last_3_games_k9_avg, last_5_games_k9_avg,
                        last_3_games_era_avg, last_5_games_era_avg,
                        last_3_games_fip_avg, last_5_games_fip_avg,
                        last_3_games_velo_avg, last_5_games_velo_avg,
                        last_3_games_swinging_strike_pct_avg, last_5_games_swinging_strike_pct_avg,
                        days_rest, team_changed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['pitcher_id'],
                    row['game_id'],
                    row['game_date'].strftime('%Y-%m-%d'),
                    row['season'],
                    row['last_3_games_strikeouts_avg'],
                    row['last_5_games_strikeouts_avg'],
                    row['last_3_games_k9_avg'],
                    row['last_5_games_k9_avg'],
                    row['last_3_games_era_avg'],
                    row['last_5_games_era_avg'],
                    row['last_3_games_fip_avg'],
                    row['last_5_games_fip_avg'],
                    row['last_3_games_velo_avg'],
                    row['last_5_games_velo_avg'],
                    row['last_3_games_swinging_strike_pct_avg'],
                    row['last_5_games_swinging_strike_pct_avg'],
                    row['days_rest'],
                    row['team_changed']
                ))
        
        conn.commit()
        
        print(f"Stored prediction features for {len(all_features)} game records.")
    else:
        print("No features created.")
    
    conn.close()

# Function to export the final dataset to CSV (for compatibility)
def export_dataset_to_csv():
    """
    Export the final dataset (with features) to CSV
    """
    print("Exporting final dataset to CSV...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Join all necessary tables
    query = '''
    SELECT 
        p.player_name,
        g.game_id,
        g.game_date,
        g.season,
        g.strikeouts,
        g.hits,
        g.walks,
        g.home_runs,
        g.release_speed_mean,
        g.release_speed_max,
        g.release_spin_rate_mean,
        g.swinging_strike_pct,
        g.called_strike_pct,
        g.zone_rate,
        t.team,
        t.era,
        t.k_per_9,
        t.bb_per_9,
        t.k_bb_ratio,
        t.whip,
        t.babip,
        t.lob_pct,
        t.fip,
        t.xfip,
        t.war,
        f.last_3_games_strikeouts_avg,
        f.last_5_games_strikeouts_avg,
        f.last_3_games_k9_avg,
        f.last_5_games_k9_avg,
        f.last_3_games_era_avg,
        f.last_5_games_era_avg,
        f.last_3_games_fip_avg,
        f.last_5_games_fip_avg,
        f.last_3_games_velo_avg,
        f.last_5_games_velo_avg,
        f.last_3_games_swinging_strike_pct_avg,
        f.last_5_games_swinging_strike_pct_avg,
        f.days_rest,
        f.team_changed
    FROM 
        game_stats g
    JOIN 
        pitchers p ON g.pitcher_id = p.pitcher_id
    LEFT JOIN 
        traditional_stats t ON g.pitcher_id = t.pitcher_id AND g.season = t.season
    LEFT JOIN
        prediction_features f ON g.pitcher_id = f.pitcher_id AND g.game_id = f.game_id
    ORDER BY
        p.player_name, g.game_date
    '''
    
    df = pd.read_sql_query(query, conn)
    
    # Get pitch mix data
    query_pitch_mix = '''
    SELECT 
        gs.id as game_db_id,
        gs.game_id,
        gs.pitcher_id,
        pm.pitch_type,
        pm.percentage
    FROM 
        game_stats gs
    JOIN 
        pitch_mix pm ON gs.id = pm.game_stats_id
    '''
    
    pitch_mix_df = pd.read_sql_query(query_pitch_mix, conn)
    
    conn.close()
    
    # Add pitch mix columns to the main dataframe
    if not pitch_mix_df.empty:
        # Create a pivot table for pitch mix
        pivot_df = pitch_mix_df.pivot_table(
            index=['pitcher_id', 'game_id'],
            columns='pitch_type',
            values='percentage',
            aggfunc='first'
        ).reset_index()
        
        # Rename columns to match the format used in the rest of the code
        pivot_df.columns.name = None
        pitch_cols = [col for col in pivot_df.columns if col not in ['pitcher_id', 'game_id']]
        for col in pitch_cols:
            pivot_df.rename(columns={col: f'pitch_pct_{col}'}, inplace=True)
        
        # Merge with main dataset
        df = pd.merge(df, pivot_df, on=['pitcher_id', 'game_id'], how='left')
    
    # Save to CSV
    df.to_csv('data/pitcher_game_level_data.csv', index=False)
    
    print(f"Exported {len(df)} rows to data/pitcher_game_level_data.csv")
    return df

# Function to aggregate statcast data to pitcher-game level
def aggregate_statcast_to_game_level(statcast_df):
    """
    Aggregate statcast data to pitcher-game level
    Args:
        statcast_df (pandas.DataFrame): Raw statcast data
    Returns:
        pandas.DataFrame: Aggregated pitcher-game level data
    """
    print("Aggregating statcast data to pitcher-game level...")
    
    # Ensure we have required columns
    required_cols = ['game_date', 'pitcher', 'player_name']
    if not all(col in statcast_df.columns for col in required_cols):
        print(f"Missing required columns. Available columns: {statcast_df.columns.tolist()}")
        return pd.DataFrame()
    
    # Convert game_date to datetime
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    
    # Create unique game ID
    statcast_df['game_id'] = statcast_df['game_pk'].astype(str)
    
    # Group by pitcher and game
    grouped = statcast_df.groupby(['pitcher', 'game_id', 'game_date', 'player_name'])
    
    # Calculate pitcher-game level metrics
    agg_dict = {
        # Pitch counts
        'pitch_type': ['count', lambda x: x.value_counts().to_dict()],
        'release_speed': ['mean', 'std', 'max'],
        'release_pos_x': ['mean', 'std'],
        'release_pos_z': ['mean', 'std'],
        'pfx_x': ['mean', 'std'],
        'pfx_z': ['mean', 'std'],
        'plate_x': ['mean', 'std'],
        'plate_z': ['mean', 'std'],
        'effective_speed': ['mean', 'max'],
        'release_spin_rate': ['mean', 'std'],
        'release_extension': ['mean'],
        'zone': lambda x: (x == 1).mean(),  # Zone percentage
        'type': lambda x: (x == 'S').mean(),  # Strike percentage
        'events': lambda x: x.value_counts().to_dict(),  # Outcomes
        'description': lambda x: x.value_counts().to_dict(),  # Detailed outcomes
    }
    
    # Apply aggregation
    game_level = grouped.agg(agg_dict).reset_index()
    
    # Flatten multi-level columns
    game_level.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in game_level.columns.values]
    
    # Calculate additional metrics
    if 'events_lambda' in game_level.columns:
        # Extract strikeouts
        game_level['strikeouts'] = game_level['events_lambda'].apply(
            lambda x: x.get('strikeout', 0) if isinstance(x, dict) else 0
        )
        
        # Extract other key events
        game_level['hits'] = game_level['events_lambda'].apply(
            lambda x: sum([x.get(e, 0) for e in ['single', 'double', 'triple', 'home_run']])
            if isinstance(x, dict) else 0
        )
        
        game_level['walks'] = game_level['events_lambda'].apply(
            lambda x: x.get('walk', 0) if isinstance(x, dict) else 0
        )
        
        game_level['home_runs'] = game_level['events_lambda'].apply(
            lambda x: x.get('home_run', 0) if isinstance(x, dict) else 0
        )
    else:
        # If events_lambda column is missing, add default columns
        game_level['strikeouts'] = 0
        game_level['hits'] = 0
        game_level['walks'] = 0
        game_level['home_runs'] = 0
    
    # Extract pitch mix percentages
    if 'pitch_type_lambda' in game_level.columns:
        pitch_types = set()
        for pitch_dict in game_level['pitch_type_lambda']:
            if isinstance(pitch_dict, dict):
                pitch_types.update(pitch_dict.keys())
        
        for pitch in pitch_types:
            game_level[f'pitch_pct_{pitch}'] = game_level['pitch_type_lambda'].apply(
                lambda x: x.get(pitch, 0) / sum(x.values()) if isinstance(x, dict) and sum(x.values()) > 0 else 0
            )
    
    # Extract swinging strike percentage
    if 'description_lambda' in game_level.columns:
        game_level['swinging_strike_pct'] = game_level['description_lambda'].apply(
            lambda x: x.get('swinging_strike', 0) / sum(x.values()) if isinstance(x, dict) and sum(x.values()) > 0 else 0
        )
        
        game_level['called_strike_pct'] = game_level['description_lambda'].apply(
            lambda x: x.get('called_strike', 0) / sum(x.values()) if isinstance(x, dict) and sum(x.values()) > 0 else 0
        )
    else:
        # Default values if missing
        game_level['swinging_strike_pct'] = 0
        game_level['called_strike_pct'] = 0
    
    # Clean up dictionary columns that we've extracted
    cols_to_drop = [col for col in game_level.columns if col.endswith('_lambda')]
    game_level = game_level.drop(columns=cols_to_drop)
    
    # Ensure all numeric columns have sensible values
    for col in game_level.select_dtypes(include=[np.number]).columns:
        game_level[col] = game_level[col].fillna(0)
    
    return game_level

# Function to process traditional pitching stats to game level
def process_traditional_stats(trad_df):
    """
    Process traditional pitching stats to prepare for merging
    Args:
        trad_df (pandas.DataFrame): Traditional pitching stats
    Returns:
        pandas.DataFrame: Processed traditional pitching stats
    """
    print("Processing traditional pitching stats...")
    
    # Ensure we have required columns
    required_cols = ['Season', 'Name', 'IDfg']
    if not all(col in trad_df.columns for col in required_cols):
        print(f"Missing required columns. Available columns: {trad_df.columns.tolist()}")
        return pd.DataFrame()
    
    # Create a pitcher_id column from IDfg for merging
    trad_df['pitcher_id'] = trad_df['IDfg']
    
    # Process columns for easier merging
    trad_df['Name'] = trad_df['Name'].str.strip()
    
    # Select relevant columns
    key_stats = ['pitcher_id', 'Name', 'Team', 'Season', 'ERA', 'W', 'L', 'G', 'GS', 'CG', 'ShO', 
                'SV', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 
                'SO', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 
                'FIP', 'xFIP', 'WAR']
    
    # Use only available columns
    available_cols = [col for col in key_stats if col in trad_df.columns]
    processed_df = trad_df[available_cols].copy()
    
    return processed_df

# Function to create exploratory visualizations
# Create a file with just the fixed visualization function to add to your script

def create_visualizations(df):
    """
    Create visualizations for the dataset with improved column name handling
    Args:
        df (pandas.DataFrame): Dataset to visualize
    """
    print("Creating visualizations...")
    
    # Print columns for debugging
    print("Available columns for visualization:")
    print(df.columns.tolist())
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (12, 8)})
    
    # 1. Distribution of strikeouts per game
    if 'strikeouts' in df.columns:
        plt.figure()
        sns.histplot(df['strikeouts'], bins=20, kde=True)
        plt.title('Distribution of Strikeouts per Game')
        plt.xlabel('Strikeouts')
        plt.ylabel('Frequency')
        plt.savefig('data/strikeout_distribution.png')
        plt.close()  # Close figure to prevent display in non-interactive environment
    
    # 2. Correlation between statcast metrics and strikeouts
    if 'strikeouts' in df.columns:
        plt.figure(figsize=(14, 10))
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        strikeout_corr = df[numeric_cols].corr()['strikeouts'].sort_values(ascending=False)
        
        # Plot top 20 correlations (or fewer if not enough)
        corr_count = min(20, len(strikeout_corr) - 1)
        if corr_count > 0:
            top_corr = strikeout_corr.iloc[1:corr_count+1]  # Skip self-correlation
            sns.barplot(x=top_corr.values, y=top_corr.index)
            plt.title('Top Features Correlated with Strikeouts')
            plt.tight_layout()
            plt.savefig('data/strikeout_correlations.png')
            plt.close()
    
    # 3. Velocity vs Strikeouts
    if all(col in df.columns for col in ['release_speed_mean', 'strikeouts']):
        plt.figure()
        sns.scatterplot(x='release_speed_mean', y='strikeouts', data=df)
        plt.title('Velocity vs Strikeouts')
        plt.xlabel('Average Release Speed (mph)')
        plt.ylabel('Strikeouts')
        plt.savefig('data/velocity_vs_strikeouts.png')
        plt.close()
    
    # 4. ERA distribution - check different possible column names
    era_col = None
    for possible_col in ['era', 'ERA', 'era_x', 'era_y']:
        if possible_col in df.columns:
            era_col = possible_col
            break
    
    if era_col:
        print(f"Using '{era_col}' column for ERA visualizations")
        plt.figure()
        # Limit to reasonable ERA values (0-10) to avoid extreme outliers
        era_data = df[df[era_col] < 10]
        sns.histplot(era_data[era_col], bins=20, kde=True)
        plt.title('Distribution of ERA')
        plt.xlabel('ERA')
        plt.ylabel('Frequency')
        plt.savefig('data/era_distribution.png')
        plt.close()
        
        # 5. Correlation between statcast metrics and ERA
        plt.figure(figsize=(14, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if era_col in numeric_cols:
            era_corr = df[numeric_cols].corr()[era_col].sort_values()
            
            # Plot top 20 correlations (both positive and negative)
            top_count = min(10, len(era_corr) // 2)
            if top_count > 0:
                top_era_corr = pd.concat([era_corr.iloc[:top_count], era_corr.iloc[-top_count:]])
                sns.barplot(x=top_era_corr.values, y=top_era_corr.index)
                plt.title('Features Most Correlated with ERA')
                plt.tight_layout()
                plt.savefig('data/era_correlations.png')
                plt.close()
        else:
            print(f"Column '{era_col}' not found in numeric columns for correlation")
    else:
        print("No ERA column found for visualization")
    
    # 6. Pitch Mix visualization for top strikeout pitchers
    pitch_cols = [col for col in df.columns if col.startswith('pitch_pct_')]
    if pitch_cols and 'player_name' in df.columns and 'strikeouts' in df.columns:
        plt.figure(figsize=(14, 10))
        
        # Get top 10 pitchers by strikeout count (or fewer if not enough data)
        top_pitcher_count = min(10, len(df['player_name'].unique()))
        top_k_pitchers = df.groupby('player_name')['strikeouts'].mean().sort_values(ascending=False).head(top_pitcher_count)
        
        if not top_k_pitchers.empty:
            # Create pitch mix dataset for top pitchers
            pitch_mix_data = []
            for pitcher in top_k_pitchers.index:
                pitcher_data = df[df['player_name'] == pitcher][pitch_cols].mean()
                pitcher_data['pitcher'] = pitcher
                pitch_mix_data.append(pitcher_data)
            
            if pitch_mix_data:
                try:
                    pitch_mix_df = pd.DataFrame(pitch_mix_data).set_index('pitcher')
                    
                    # Plot pitch mix
                    pitch_mix_df.plot(kind='bar', stacked=True)
                    plt.title(f'Pitch Mix for Top {top_pitcher_count} Strikeout Pitchers')
                    plt.ylabel('Percentage')
                    plt.legend(title='Pitch Type')
                    plt.tight_layout()
                    plt.savefig('data/top_pitcher_pitch_mix.png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating pitch mix visualization: {e}")
    
    print("Visualizations saved to data directory")

# Main function to fetch and save statcast data
def get_statcast_data(force_refresh=False):
    """
    Fetch statcast data for multiple seasons
    Args:
        force_refresh (bool): Whether to force refresh cached data
    Returns:
        pandas.DataFrame: Combined statcast data
    """
    cache_file = "data/statcast_pitcher_data.pkl"
    
    # Check if we have a recent cached version
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached statcast data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    all_statcast_data = []
    
    for season in SEASONS:
        season_data = fetch_season_in_chunks(season)
        
        if not season_data.empty:
            # Filter to only include pitcher-relevant data
            pitcher_data = season_data[season_data['pitcher'].notna()].copy()
            pitcher_data['season'] = season
            all_statcast_data.append(pitcher_data)
            
            # Save season data separately as backup
            season_cache = f"data/statcast_pitcher_{season}.pkl"
            with open(season_cache, 'wb') as f:
                pickle.dump(pitcher_data, f)
            print(f"Saved {season} data to {season_cache}")
    
    if not all_statcast_data:
        print("No statcast data retrieved")
        return pd.DataFrame()
    
    # Combine all season data
    combined_data = pd.concat(all_statcast_data, ignore_index=True)
    
    # Save combined data
    with open(cache_file, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"Saved combined statcast data to {cache_file}")
    
    return combined_data

# Main function to fetch and save traditional pitching stats
def get_traditional_stats(force_refresh=False):
    """
    Fetch traditional pitching stats for multiple seasons
    Args:
        force_refresh (bool): Whether to force refresh cached data
    Returns:
        pandas.DataFrame: Combined traditional pitching stats
    """
    cache_file = "data/traditional_pitcher_data.pkl"
    
    # Check if we have a recent cached version
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached traditional stats from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    all_traditional_data = []
    
    for season in SEASONS:
        season_data = fetch_pitching_stats_safely(season)
        
        if not season_data.empty:
            season_data['Season'] = season
            all_traditional_data.append(season_data)
            
            # Save season data separately as backup
            season_cache = f"data/traditional_pitcher_{season}.pkl"
            with open(season_cache, 'wb') as f:
                pickle.dump(season_data, f)
            print(f"Saved {season} traditional data to {season_cache}")
    
    if not all_traditional_data:
        print("No traditional pitching data retrieved")
        return pd.DataFrame()
    
    # Combine all season data
    combined_data = pd.concat(all_traditional_data, ignore_index=True)
    
    # Save combined data
    with open(cache_file, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"Saved combined traditional data to {cache_file}")
    
    return combined_data

# Main execution flow
def main():
    """
    Main function to run the entire pipeline
    """
    print("Starting pitcher performance analysis pipeline...")
    
    # Initialize the database
    init_database()
    
    # 1. Fetch statcast data
    statcast_data = get_statcast_data(force_refresh=False)
    if statcast_data.empty:
        print("No statcast data available. Exiting.")
        return
    
    # 2. Fetch traditional pitching stats
    traditional_data = get_traditional_stats(force_refresh=False)
    if traditional_data.empty:
        print("No traditional pitching data available. Exiting.")
        return
    
    # 3. Store data in the database
    store_statcast_data(statcast_data, force_refresh=False)
    store_traditional_stats(traditional_data, force_refresh=False)
    
    # 4. Update pitcher ID mappings
    update_pitcher_mapping()
    
    # 5. Create prediction features
    create_prediction_features(force_refresh=False)
    
    # 6. Export final dataset to CSV
    final_data = export_dataset_to_csv()
    
    # 7. Create visualizations
    create_visualizations(final_data)
    
    print("Pipeline completed successfully!")

# Run the main function
if __name__ == "__main__":
    main()