# Database utilities for the MLB pitcher prediction project
import sqlite3
import pandas as pd
from pathlib import Path
from src.data.utils import setup_logger, ensure_dir
from functools import lru_cache
import hashlib
from datetime import datetime
from config import DBConfig
import pybaseball as pb
import os
from src.data.db import DBConnection

logger = setup_logger(__name__)

_cache = {}
_cache_timeout = 300 # 5 minutes

DB_PATH = DBConfig.PATH

def execute_query(query, params=None):
    """Execute a query and return results as a DataFrame"""
    with DBConnection() as conn:
        if params:
            return pd.read_sql_query(query, conn, params=params)
        else:
            return pd.read_sql_query(query, conn)

def is_table_populated(table_name):
    """
    Check if a table in the database has data
    
    Args:
        table_name (str): Table name to check
        
    Returns:
        bool: True if the table has data
    """
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
    
    return count > 0

def create_database_schema():
    """ create database tables """ 
    ensure_dir(Path(DB_PATH).parent)
    
    # Connect to database
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        # Create tables
        logger.info("Creating database schema...")

        # Drop existing tables to ensure clean schema
        cursor.execute("DROP TABLE IF EXISTS teams")
        cursor.execute("DROP TABLE IF EXISTS pitcher_ids")
        cursor.execute("DROP TABLE IF EXISTS games")
        cursor.execute("DROP TABLE IF EXISTS statcast_pitches")
        cursor.execute("DROP TABLE IF EXISTS game_stats")
        cursor.execute("DROP TABLE IF EXISTS pitch_mix")
        cursor.execute("DROP TABLE IF EXISTS team_batting_stats")
        cursor.execute("DROP TABLE IF EXISTS batter_profiles")
        cursor.execute("DROP TABLE IF EXISTS sequence_patterns")
        cursor.execute("DROP TABLE IF EXISTS prediction_features")
        cursor.execute("DROP TABLE IF EXISTS starter_probability")

        # 1. Pitcher ID Mapping table (to handle different ID systems)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pitcher_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_mlbam INTEGER UNIQUE,
            key_fangraphs INTEGER,
            name TEXT,
            is_starter INTEGER,
            first_seen_date TEXT,
            last_seen_date TEXT
        )
        ''')
        
        # 2. Teams table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT UNIQUE,
            team_name TEXT,
            key_fangraphs INTEGER
        )
        ''')
        
        # 3. Games table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER UNIQUE,
            game_date TEXT,
            season INTEGER,
            home_team TEXT,
            away_team TEXT,
            ballpark TEXT,
            game_type TEXT,
            FOREIGN KEY (home_team) REFERENCES teams(team_id),
            FOREIGN KEY (away_team) REFERENCES teams(team_id)
        )
        ''')
        
        # 4. Raw Statcast pitch data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS statcast_pitches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitcher_id INTEGER,
            game_id INTEGER,
            pitch_type TEXT,
            game_date TEXT,
            release_speed REAL,
            release_pos_x REAL,
            release_pos_z REAL,
            release_spin_rate INTEGER,
            pfx_x REAL,
            pfx_z REAL,
            plate_x REAL,
            plate_z REAL,
            batter_id INTEGER,
            batter_stands TEXT,
            events TEXT,
            description TEXT,
            zone INTEGER,
            balls INTEGER,
            strikes INTEGER,
            at_bat_number INTEGER,
            pitch_number INTEGER,
            inning INTEGER,
            inning_topbot TEXT,
            outs_when_up INTEGER,
            on_1b INTEGER,
            on_2b INTEGER,
            on_3b INTEGER,
            FOREIGN KEY (pitcher_id) REFERENCES pitcher_ids(key_mlbam),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
        ''')
        
        # 5. Aggregated game-level pitcher stats
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitcher_id INTEGER,
            game_id INTEGER,
            game_date TEXT,
            season INTEGER,
            opponent_team TEXT,
            strikeouts INTEGER,
            batters_faced INTEGER,
            pitches_thrown INTEGER,
            strikes_thrown INTEGER,
            called_strikes INTEGER,
            swinging_strikes INTEGER,
            foul_strikes INTEGER,
            csw INTEGER,
            csw_rate REAL,
            avg_release_speed REAL,
            max_release_speed REAL,
            avg_spin_rate REAL,
            zone_rate REAL,
            first_pitch_strike_rate REAL,
            chase_rate REAL,
            contact_rate REAL,
            whiff_rate REAL,
            FOREIGN KEY (pitcher_id) REFERENCES pitcher_ids(key_mlbam),
            FOREIGN KEY (game_id) REFERENCES games(game_id),
            FOREIGN KEY (opponent_team) REFERENCES teams(team_id),
            UNIQUE(pitcher_id, game_id)
        )
        ''')
        
        # 6. Pitch mix by game
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pitch_mix (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_stats_id INTEGER,
            pitch_type TEXT,
            count INTEGER,
            percentage REAL,
            velocity_avg REAL,
            movement_x_avg REAL,
            movement_z_avg REAL,
            whiff_rate REAL,
            FOREIGN KEY (game_stats_id) REFERENCES game_stats(id)
        )
        ''')
        
        # 7. Team batting stats
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_batting_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT,
            season INTEGER,
            k_percent REAL,
            bb_percent REAL,
            avg REAL,
            obp REAL,
            slg REAL,
            ops REAL,
            iso REAL,
            babip REAL,
            o_swing_percent REAL,
            z_contact_percent REAL,
            contact_percent REAL,
            zone_percent REAL,
            swstr_percent REAL,
            hard_hit_percent REAL,
            pull_percent REAL,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            UNIQUE(team_id, season)
        )
        ''')
        
        # 8. Batter profiles (selective aggregation from statcast_batter)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batter_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT,
            season INTEGER,
            batter_hand TEXT,
            zone_whiff_rate REAL,
            chase_rate REAL,
            strikeout_rate REAL,
            z1_whiff_rate REAL,
            z2_whiff_rate REAL,
            z3_whiff_rate REAL,
            z4_whiff_rate REAL,
            z5_whiff_rate REAL,
            z6_whiff_rate REAL,
            z7_whiff_rate REAL,
            z8_whiff_rate REAL,
            z9_whiff_rate REAL,
            fb_whiff_rate REAL,
            breaking_whiff_rate REAL,
            offspeed_whiff_rate REAL,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            UNIQUE(team_id, season, batter_hand)
        )
        ''')
        
        # 9. Sequence patterns (for pitcher tendencies)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sequence_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitcher_id INTEGER,
            season INTEGER,
            count_state TEXT,
            first_pitch TEXT,
            ahead_count_pitch TEXT,
            behind_count_pitch TEXT,
            two_strike_pitch TEXT,
            strikeout_pitch TEXT,
            fb_percent_first_pitch REAL,
            breaking_percent_two_strike REAL, 
            fb_to_breaking_rate REAL,
            breaking_to_fb_rate REAL,
            FOREIGN KEY (pitcher_id) REFERENCES pitcher_ids(key_mlbam),
            UNIQUE(pitcher_id, season, count_state)
        )
        ''')
        
        # 10. Prediction features (pregame)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitcher_id INTEGER,
            game_id TEXT,
            game_date TEXT,
            season INTEGER,
            opponent_team TEXT,
            
            last_3_games_strikeouts_avg REAL,
            last_5_games_strikeouts_avg REAL,
            last_10_games_strikeouts_avg REAL,
            last_3_games_velo_avg REAL,
            last_5_games_velo_avg REAL,
            last_10_games_velo_avg REAL,
            last_3_games_swinging_strike_pct REAL,
            last_5_games_swinging_strike_pct REAL,
            last_10_games_swinging_strike_pct REAL,
            last_3_games_csw_rate REAL,
            last_5_games_csw_rate REAL,
            last_10_games_csw_rate REAL,
            
            last_3_games_strikeouts_std REAL,
            last_5_games_strikeouts_std REAL,
            last_10_games_strikeouts_std REAL,
            strikeout_consistency REAL,
            
            days_rest INTEGER,
            team_changed INTEGER,
            
            opponent_k_rate REAL,
            opponent_contact_rate REAL,
            opponent_chase_rate REAL,
            opponent_zone_contact_rate REAL,
            opponent_k_vs_pitch_type REAL,
            
            pitcher_handedness TEXT,
            opponent_k_rate_vs_hand REAL,
            ballpark_k_factor REAL,
            historical_k_rate_vs_opponent REAL,
            
            expected_fb_usage REAL,
            expected_breaking_usage REAL,
            expected_offspeed_usage REAL,
            
            fb_percent_first_pitch REAL,
            breaking_percent_two_strike REAL,
            
            actual_strikeouts INTEGER,
            
            FOREIGN KEY (pitcher_id) REFERENCES pitcher_ids(key_mlbam),
            FOREIGN KEY (game_id) REFERENCES games(game_id),
            FOREIGN KEY (opponent_team) REFERENCES teams(team_id),
            UNIQUE(pitcher_id, game_id)
        )
        ''')
        
        # 11. Pitcher starting probability
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS starter_probability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitcher_id INTEGER,
            season INTEGER,
            games_started INTEGER,
            total_games INTEGER,
            is_known_starter INTEGER,
            estimated_starter_probability REAL, 
            last_role TEXT,
            FOREIGN KEY (pitcher_id) REFERENCES pitcher_ids(key_mlbam),
            UNIQUE(pitcher_id, season)
        )
        ''')
        
        conn.commit()

    logger.info("Database schema created successfully")

def initialize_team_data():

    """Initialize team data in the database"""

    with DBConnection() as conn:

        # Default team mapping
        teams = [
            # Team ID, Team Name, FanGraphs ID
            ('LAD', 'Los Angeles Dodgers', 22),
            ('NYY', 'New York Yankees', 9),
            ('BOS', 'Boston Red Sox', 3),
            ('CHC', 'Chicago Cubs', 17),
            ('SFG', 'San Francisco Giants', 30),
            ('NYM', 'New York Mets', 25),
            ('HOU', 'Houston Astros', 21),
            ('ATL', 'Atlanta Braves', 16),
            ('PHI', 'Philadelphia Phillies', 26),
            ('OAK', 'Oakland Athletics', 10),
            ('CLE', 'Cleveland Guardians', 5),
            ('SEA', 'Seattle Mariners', 11),
            ('STL', 'St. Louis Cardinals', 28),
            ('TBR', 'Tampa Bay Rays', 12),
            ('TEX', 'Texas Rangers', 13),
            ('TOR', 'Toronto Blue Jays', 14),
            ('MIN', 'Minnesota Twins', 8),
            ('ARI', 'Arizona Diamondbacks', 15),
            ('MIL', 'Milwaukee Brewers', 23),
            ('LAA', 'Los Angeles Angels', 1),
            ('CWS', 'Chicago White Sox', 4),
            ('COL', 'Colorado Rockies', 19),
            ('WSN', 'Washington Nationals', 24),
            ('DET', 'Detroit Tigers', 6),
            ('KCR', 'Kansas City Royals', 7),
            ('PIT', 'Pittsburgh Pirates', 27),
            ('SDP', 'San Diego Padres', 29),
            ('BAL', 'Baltimore Orioles', 2),
            ('CIN', 'Cincinnati Reds', 18),
            ('MIA', 'Miami Marlins', 20)
        ]
        
        cursor = conn.cursor()
        
        for team_id, team_name, key_fangraphs in teams:
            cursor.execute(
                "INSERT OR IGNORE INTO teams (team_id, team_name, key_fangraphs) VALUES (?, ?, ?)",
                (team_id, team_name, key_fangraphs)
            )
        
        conn.commit()
        logger.info("Team data initialized")
        
        return True

def fetch_pitcher_id_mapping(seasons=None):
    """
    Fetch pitcher ID mappings between MLBAM and FanGraphs with flexible starter criteria
    """
    if seasons is None:
        seasons = list(range(2023, 2026))  # Default: Recent seasons
    
    logger.info(f"Fetching pitcher ID mappings for seasons: {seasons}")
    
    # More comprehensive column mapping
    column_mapping = {
        'fangraphs_id': ['IDfg', 'playerid', 'key_fangraphs', 'FG ID'],
        'name': ['Name', 'name', 'player_name', 'Player Name'],
        'team': ['Team', 'team', 'team_abbrev', 'Tm'],
        'games': ['G', 'games_played', 'Games Played'],
        'games_started': ['GS', 'games_started', 'Games Started']
    }
    
    # This will store our mapping data
    all_starters = []
    all_pitchers = []
    
    # Get current year
    current_year = datetime.now().year
    
    try:
        # Get Chadwick Register for additional ID mapping
        logger.info("Fetching Chadwick Register data...")
        player_lookup = pb.chadwick_register()
        
        # Process each season to find starters
        for season in seasons:
            logger.info(f"Processing season {season}")
            
            try:
                # Get pitching stats for the season
                pitching_stats = pb.pitching_stats(season, season, qual=0)
                
                # Function to find the first matching column
                def find_column(possible_names):
                    for name in possible_names:
                        if name in pitching_stats.columns:
                            return name
                    return None
                
                # Dynamically find column names
                id_col = find_column(column_mapping['fangraphs_id'])
                name_col = find_column(column_mapping['name'])
                team_col = find_column(column_mapping['team'])
                games_col = find_column(column_mapping['games'])
                games_started_col = find_column(column_mapping['games_started'])
                
                # Comprehensive logging for debugging
                logger.info(f"Found columns:")
                logger.info(f"  ID Column: {id_col}")
                logger.info(f"  Name Column: {name_col}")
                logger.info(f"  Team Column: {team_col}")
                logger.info(f"  Games Column: {games_col}")
                logger.info(f"  Games Started Column: {games_started_col}")
                
                # Verify all required columns are found
                if not all([id_col, name_col, team_col, games_col, games_started_col]):
                    logger.error(f"Missing required columns in season {season}")
                    logger.error(f"Available columns: {list(pitching_stats.columns)}")
                    continue
                
                # Select and rename columns
                season_pitchers = pitching_stats[[
                    id_col, name_col, team_col, games_col, games_started_col
                ]].copy()
                
                season_pitchers.columns = [
                    'fangraphs_id', 'name', 'team', 'games', 'games_started'
                ]
                
                # Convert columns to appropriate types
                season_pitchers['fangraphs_id'] = pd.to_numeric(season_pitchers['fangraphs_id'], errors='coerce')
                season_pitchers['games'] = pd.to_numeric(season_pitchers['games'], errors='coerce')
                season_pitchers['games_started'] = pd.to_numeric(season_pitchers['games_started'], errors='coerce')
                
                season_pitchers['season'] = season
                
                # Drop rows with invalid data
                season_pitchers = season_pitchers.dropna(subset=['fangraphs_id', 'games', 'games_started'])
                
                # Add to master pitcher list
                all_pitchers.append(season_pitchers)
                
                # Adjust starter criteria based on the season
                if season == current_year:
                    # For current year, be more lenient
                    # Consider a starter if they've started at least 1 game
                    # or have a games_started/games ratio suggesting they might be a starter
                    starters = season_pitchers[
                        (season_pitchers['games_started'] >= 1) | 
                        ((season_pitchers['games_started']/season_pitchers['games'] >= 0.3) & 
                         (season_pitchers['games'] >= 3))
                    ].copy()
                else:
                    # Previous seasons use original criteria
                    starters = season_pitchers[
                        (season_pitchers['games_started'] >= 5) | 
                        ((season_pitchers['games_started']/season_pitchers['games'] >= 0.5) & 
                         (season_pitchers['games'] >= 8))
                    ].copy()
                
                starters['is_known_starter'] = 1
                
                # Add to master starter list
                all_starters.append(starters)
                
                logger.info(f"Season {season}: {len(starters)} starters identified out of {len(season_pitchers)} pitchers")
                
            except Exception as e:
                logger.error(f"Error processing season {season}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Combine all pitcher data
        if all_pitchers:
            all_pitchers_df = pd.concat(all_pitchers, ignore_index=True)
            
            # Get unique pitchers
            unique_pitchers = all_pitchers_df[['fangraphs_id', 'name']].drop_duplicates()
            
            # Check which ones are starters
            if all_starters:
                all_starters_df = pd.concat(all_starters, ignore_index=True)
                # Get unique FG IDs that were starters in any season
                starter_ids = all_starters_df['fangraphs_id'].unique()
                unique_pitchers['is_starter'] = unique_pitchers['fangraphs_id'].isin(starter_ids).astype(int)
            else:
                unique_pitchers['is_starter'] = 0
            
            logger.info(f"Total unique pitchers: {len(unique_pitchers)}")
            logger.info(f"Total unique starters: {unique_pitchers['is_starter'].sum()}")
            
            # Modify player lookup and merging
            player_lookup_filtered = player_lookup[['key_fangraphs', 'key_mlbam']].dropna()
            
            # Convert to appropriate types for joining
            player_lookup_filtered['key_fangraphs'] = player_lookup_filtered['key_fangraphs'].astype(int)
            unique_pitchers['fangraphs_id'] = unique_pitchers['fangraphs_id'].astype(int)
            
            # Join on key_fangraphs
            merged_df = pd.merge(
                unique_pitchers,
                player_lookup_filtered,
                left_on='fangraphs_id',
                right_on='key_fangraphs',
                how='inner'
            )
            
            if not merged_df.empty:
                # Create final mapping DataFrame
                mapping_df = pd.DataFrame({
                    'key_fangraphs': merged_df['fangraphs_id'],
                    'key_mlbam': merged_df['key_mlbam'],
                    'name': merged_df['name'],
                    'is_starter': merged_df['is_starter']
                })
                
                logger.info(f"Created mapping for {len(mapping_df)} pitchers")
                return mapping_df
            else:
                logger.warning("No matches found between pitcher IDs and player lookup")
                return pd.DataFrame()
        else:
            logger.error("No pitcher data found for any season")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error fetching pitcher ID mappings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def initialize_pitcher_ids(mapping_df=None):
    """Initialize pitcher ID mapping in the database"""
    if mapping_df is None or mapping_df.empty:
        # Fetch the mapping if not provided
        mapping_df = fetch_pitcher_id_mapping()
    
    if mapping_df.empty:
        logger.error("No pitcher mapping data available")
        return False
    
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        # Insert pitcher ID mappings
        for _, row in mapping_df.iterrows():
            cursor.execute(
            """
            INSERT OR IGNORE INTO pitcher_ids 
            (key_mlbam, key_fangraphs, name, is_starter)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(row['key_mlbam']),
                int(row['key_fangraphs']),
                row['name'],
                int(row['is_starter'])
            )
        )
    
        conn.commit()
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM pitcher_ids WHERE is_starter = 1")
        starter_count = cursor.fetchone()[0]
    
    logger.info(f"Inserted {starter_count} starters into pitcher_ids table")
    
    return True

def extract_statcast_for_starters(seasons, output_dir="data/statcast"):
    """
    Fetch Statcast data for all identified starting pitchers
    
    Args:
        seasons (list): List of seasons to fetch
        output_dir (str): Directory to save raw Statcast files
    
    Returns:
        bool: Success status
    """
    # Create the output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Connect to the database
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        # Get all starting pitchers
        cursor.execute("SELECT key_mlbam FROM pitcher_ids WHERE is_starter = 1")
        starter_ids = [row[0] for row in cursor.fetchall()]
        
        if not starter_ids:
            logger.error("No starters found in the database")
            return False
        
        logger.info(f"Found {len(starter_ids)} starters to fetch Statcast data for")
        
        # Process starters in batches to avoid overwhelming the API
        batch_size = 10
        success_count = 0
        
        for i in range(0, len(starter_ids), batch_size):
            batch = starter_ids[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} / {(len(starter_ids) + batch_size - 1)//batch_size}")
            
            for pitcher_id in batch:
                try:
                    # Get pitcher name for logging
                    cursor.execute("SELECT name FROM pitcher_ids WHERE mlbam_id = ?", (pitcher_id,))
                    name = cursor.fetchone()[0]
                    
                    logger.info(f"Fetching Statcast data for {name} (ID: {pitcher_id})")
                    
                    # Fetch data for each season
                    for season in seasons:
                        try:
                            # Define date range for the season
                            if season == 2025:  # Current season
                                start_date = f"{season}-03-01"
                                end_date = "2025-04-01"  # Current date in our scenario
                            else:
                                start_date = f"{season}-03-01"
                                end_date = f"{season}-11-01"
                            
                            # Fetch the data using pybaseball's statcast_pitcher function
                            logger.info(f"  Fetching {name} for {start_date} to {end_date}")
                            
                            pitcher_data = pb.statcast_pitcher(start_date, end_date, pitcher_id)
                            
                            if pitcher_data.empty:
                                logger.warning(f"  No data found for {name} in {season}")
                                continue
                            
                            # Save to CSV
                            output_file = f"{output_dir}/pitcher_{pitcher_id}_{season}.csv"
                            pitcher_data.to_csv(output_file, index=False)
                            logger.info(f"  Saved {len(pitcher_data)} rows to {output_file}")
                            
                            success_count += 1
                        
                        except Exception as e:
                            logger.error(f"  Error fetching {name} for {season}: {e}")
                
                except Exception as e:
                    logger.error(f"Error processing pitcher {pitcher_id}: {e}")
        
    logger.info(f"Successfully fetched data for {success_count} pitcher-seasons")
    
    return success_count > 0

def process_team_batting_data(batting_csv):
    """
    Process team batting data from CSV and store in database
    
    Args:
        batting_csv (str): Path to team batting CSV file
        conn (sqlite3.Connection): Database connection
    
    Returns:
        bool: Success status
    """
    try:
        # Load the CSV file
        team_batting = pd.read_csv(batting_csv)
        
        if team_batting.empty:
            logger.error(f"No data found in {batting_csv}")
            return False
        
        logger.info(f"Loaded {len(team_batting)} rows from {batting_csv}")

        with DBConnection() as conn:
            cursor = conn.cursor()
        
            # Create a database connection if not provided
            if conn is None:
                conn = sqlite3.connect(DB_PATH)
            
            cursor = conn.cursor()
            
            # Process each team-season
            for _, row in team_batting.iterrows():
                try:
                    # Check if we have a team ID mapping for this team
                    cursor.execute("SELECT team_id FROM teams WHERE key_fangraphs = ?", (row['teamIDfg'],))
                    result = cursor.fetchone()
                    
                    if not result:
                        logger.warning(f"No team mapping found for FanGraphs ID (key_fangraphs): {row['teamIDfg']}")
                        continue
                    
                    team_id = result[0]
                    season = row['Season']
                    
                    # Insert team batting stats
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO team_batting_stats 
                        (team_id, season, k_percent, bb_percent, avg, obp, slg, ops, iso, babip, 
                        o_swing_percent, z_contact_percent, contact_percent, zone_percent, 
                        swstr_percent, hard_hit_percent, pull_percent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            team_id,
                            season,
                            row.get('K%', 0),
                            row.get('BB%', 0),
                            row.get('AVG', 0),
                            row.get('OBP', 0),
                            row.get('SLG', 0),
                            row.get('OPS', 0),
                            row.get('ISO', 0),
                            row.get('BABIP', 0),
                            row.get('O-Swing%', 0),
                            row.get('Z-Contact%', 0),
                            row.get('Contact%', 0),
                            row.get('Zone%', 0),
                            row.get('SwStr%', 0),
                            row.get('Hard%', 0),
                            row.get('Pull%', 0)
                        )
                    )
                
                except Exception as e:
                    logger.error(f"Error processing team {row.get('Team', 'unknown')}: {e}")
            
            conn.commit()
        logger.info("Team batting data processed successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing team batting data: {e}")
        return False

def process_statcast_batter_data(batter_csv):
    """
    Process selective features from statcast batter data
    
    Args:
        batter_csv (str): Path to statcast batter CSV file
        conn (sqlite3.Connection): Database connection
    
    Returns:
        bool: Success status
    """
    try:
        # Load the CSV file
        batter_data = pd.read_csv(batter_csv)
        
        if batter_data.empty:
            logger.error(f"No data found in {batter_csv}")
            return False
        
        logger.info(f"Loaded {len(batter_data)} rows from {batter_csv}")
        
        
        with DBConnection() as conn:
            cursor = conn.cursor()
            
            # Extract team ID from data
            if 'home_team' in batter_data.columns:
                team_id = batter_data['home_team'].iloc[0]
            else:
                logger.error("No home_team column found in batter data")
                return False
            
            # Extract season from data
            if 'game_year' in batter_data.columns:
                season = batter_data['game_year'].iloc[0]
            elif 'game_date' in batter_data.columns:
                # Extract year from first game date
                game_date = pd.to_datetime(batter_data['game_date'].iloc[0])
                season = game_date.year
            else:
                logger.error("No season information found in batter data")
                return False
            
            # Aggregate data by batter handedness
            for hand in ['L', 'R']:
                # Filter to batters with this handedness
                batter_hand_data = batter_data[batter_data['stand'] == hand]
                
                if batter_hand_data.empty:
                    logger.warning(f"No {hand}-handed batter data found")
                    continue
                
                # Calculate whiff rates for each zone
                zone_whiff_rates = {}
                for zone in range(1, 10):
                    zone_pitches = batter_hand_data[batter_hand_data['zone'] == zone]
                    if not zone_pitches.empty:
                        swings = zone_pitches[zone_pitches['description'].isin(
                            ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
                        )]
                        
                        if not swings.empty:
                            whiffs = swings[swings['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
                            zone_whiff_rates[f'z{zone}_whiff_rate'] = len(whiffs) / len(swings)
                        else:
                            zone_whiff_rates[f'z{zone}_whiff_rate'] = 0
                    else:
                        zone_whiff_rates[f'z{zone}_whiff_rate'] = 0
                
                # Calculate pitch type whiff rates
                # Group pitch types
                fastballs = ['FF', 'FT', 'FC', 'SI', 'FS']
                breaking = ['SL', 'CU', 'KC', 'EP']
                offspeed = ['CH', 'SC', 'KN', 'FO']
                
                # Fastball whiff rate
                fb_data = batter_hand_data[batter_hand_data['pitch_type'].isin(fastballs)]
                if not fb_data.empty:
                    fb_swings = fb_data[fb_data['description'].isin(
                        ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
                    )]
                    if not fb_swings.empty:
                        fb_whiffs = fb_swings[fb_swings['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
                        fb_whiff_rate = len(fb_whiffs) / len(fb_swings)
                    else:
                        fb_whiff_rate = 0
                else:
                    fb_whiff_rate = 0
                
                # Breaking ball whiff rate
                breaking_data = batter_hand_data[batter_hand_data['pitch_type'].isin(breaking)]
                if not breaking_data.empty:
                    breaking_swings = breaking_data[breaking_data['description'].isin(
                        ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
                    )]
                    if not breaking_swings.empty:
                        breaking_whiffs = breaking_swings[breaking_swings['description'].isin(
                            ['swinging_strike', 'swinging_strike_blocked']
                        )]
                        breaking_whiff_rate = len(breaking_whiffs) / len(breaking_swings)
                    else:
                        breaking_whiff_rate = 0
                else:
                    breaking_whiff_rate = 0
                
                # Offspeed whiff rate
                offspeed_data = batter_hand_data[batter_hand_data['pitch_type'].isin(offspeed)]
                if not offspeed_data.empty:
                    offspeed_swings = offspeed_data[offspeed_data['description'].isin(
                        ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
                    )]
                    if not offspeed_swings.empty:
                        offspeed_whiffs = offspeed_swings[offspeed_swings['description'].isin(
                            ['swinging_strike', 'swinging_strike_blocked']
                        )]
                        offspeed_whiff_rate = len(offspeed_whiffs) / len(offspeed_swings)
                    else:
                        offspeed_whiff_rate = 0
                else:
                    offspeed_whiff_rate = 0
                
                # Chase rate (swings on pitches outside zone / pitches outside zone)
                outside_zone = batter_hand_data[batter_hand_data['zone'] == 0]
                if not outside_zone.empty:
                    chases = outside_zone[outside_zone['description'].isin(
                        ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
                    )]
                    chase_rate = len(chases) / len(outside_zone)
                else:
                    chase_rate = 0
                
                # Strikeout rate
                strikeouts = batter_hand_data[batter_hand_data['events'] == 'strikeout']
                at_bats = batter_hand_data['at_bat_number'].nunique()
                strikeout_rate = len(strikeouts) / at_bats if at_bats > 0 else 0
                
                # Zone whiff rate (for pitches in zone)
                in_zone = batter_hand_data[batter_hand_data['zone'] > 0]
                if not in_zone.empty:
                    zone_swings = in_zone[in_zone['description'].isin(
                        ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
                    )]
                    if not zone_swings.empty:
                        zone_whiffs = zone_swings[zone_swings['description'].isin(
                            ['swinging_strike', 'swinging_strike_blocked']
                        )]
                        zone_whiff_rate = len(zone_whiffs) / len(zone_swings)
                    else:
                        zone_whiff_rate = 0
                else:
                    zone_whiff_rate = 0
                
                # Store in database
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO batter_profiles
                    (team_id, season, batter_hand, zone_whiff_rate, chase_rate, strikeout_rate,
                    z1_whiff_rate, z2_whiff_rate, z3_whiff_rate, z4_whiff_rate, z5_whiff_rate,
                    z6_whiff_rate, z7_whiff_rate, z8_whiff_rate, z9_whiff_rate,
                    fb_whiff_rate, breaking_whiff_rate, offspeed_whiff_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        team_id,
                        season,
                        hand,
                        zone_whiff_rate,
                        chase_rate,
                        strikeout_rate,
                        zone_whiff_rates.get('z1_whiff_rate', 0),
                        zone_whiff_rates.get('z2_whiff_rate', 0),
                        zone_whiff_rates.get('z3_whiff_rate', 0),
                        zone_whiff_rates.get('z4_whiff_rate', 0),
                        zone_whiff_rates.get('z5_whiff_rate', 0),
                        zone_whiff_rates.get('z6_whiff_rate', 0),
                        zone_whiff_rates.get('z7_whiff_rate', 0),
                        zone_whiff_rates.get('z8_whiff_rate', 0),
                        zone_whiff_rates.get('z9_whiff_rate', 0),
                        fb_whiff_rate,
                        breaking_whiff_rate,
                        offspeed_whiff_rate
                    )
                )
            
            conn.commit()
        logger.info(f"Batter profile data for {team_id} in {season} processed successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing batter data: {e}")
        return False

def get_starting_pitchers_alternative(seasons=None):
    """
    Alternative approach to identifying starting pitchers using game logs
    
    Args:
        seasons (list): List of seasons to process
        
    Returns:
        dict: Dictionary mapping seasons to lists of starting pitcher IDs
    """
    if seasons is None:
        seasons = list(range(2015, 2026))
    
    logger.info(f"Identifying starting pitchers for seasons: {seasons}")
    
    starters_by_season = {}
    
    try:
        # Process each season
        for season in seasons:
            try:
                logger.info(f"Processing season {season}")
                
                # Get starting pitcher IDs from team game logs
                # For each team, look at who was the first pitcher in each game
                team_ids = list(range(1, 31))  # MLB team IDs 1-30
                
                season_starters = set()
                
                for team_id in team_ids:
                    try:
                        # Fetch team game logs
                        team_schedules = pb.schedule_and_record(season, team_id)
                        
                        if team_schedules.empty:
                            logger.warning(f"No game logs found for team {team_id} in {season}")
                            continue
                        
                        # Look for starting pitcher information
                        if 'Starting.Pitcher' in team_schedules.columns:
                            # Extract unique starting pitchers
                            starters = team_schedules['Starting.Pitcher'].dropna().unique()
                            logger.info(f"Team {team_id}: Found {len(starters)} starting pitchers")
                            
                            # For each starter name, map to MLBAM ID
                            for starter_name in starters:
                                # Look up player ID using name
                                players = pb.playerid_lookup(
                                    last=starter_name.split(',')[0].strip() if ',' in starter_name else '',
                                    first=starter_name.split(',')[1].strip() if ',' in starter_name else starter_name
                                )
                                
                                if not players.empty:
                                    # Get MLBAM ID
                                    mlbam_id = players['key_mlbam'].iloc[0]
                                    season_starters.add(mlbam_id)
                                else:
                                    logger.warning(f"Could not find ID for {starter_name}")
                            
                    except Exception as e:
                        logger.error(f"Error processing team {team_id} in {season}: {e}")
                
                starters_by_season[season] = list(season_starters)
                logger.info(f"Season {season}: Identified {len(season_starters)} starting pitchers")
                
            except Exception as e:
                logger.error(f"Error processing season {season}: {e}")
                starters_by_season[season] = []
        
        return starters_by_season
    
    except Exception as e:
        logger.error(f"Error identifying starting pitchers: {e}")
        return {}

def main():
    """Main function"""
    create_database_schema()
    
    # Initialize team data
    initialize_team_data()
    
    # Initialize pitcher ID mappings
    initialize_pitcher_ids()
    
    # Test processing team batting data
    team_batting_path = 'team_batting.csv'
    if os.path.exists(team_batting_path):
        process_team_batting_data(team_batting_path)
    
    # Test processing statcast batter data
    batter_csv = 'statcast_batter_shohei.csv'
    if os.path.exists(batter_csv):
        process_statcast_batter_data(batter_csv)
    
    # Extract statcast data for starters (commented out to avoid accidental API calls)
    # extract_statcast_for_starters([2024, 2025])
    
    logger.info("Database setup complete")

if __name__ == "__main__":
    main()