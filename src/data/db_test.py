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

def create_tables_from_csv_structure(csv_files):
    """
    Create database tables with column names directly from CSV headers
    
    Args:
        csv_files (dict): Dictionary mapping table names to sample CSV files
    
    Returns:
        bool: Success status
    """
    success = True
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        # Create essential tables we know the structure of
        logger.info("Creating essential tables...")
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
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT UNIQUE,
            team_name TEXT,
            key_fangraphs INTEGER
        )
        ''')
        
        # Create tables based on CSV structure
        for table_name, csv_file in csv_files.items():
            try:
                if not os.path.exists(csv_file):
                    logger.error(f"CSV file not found: {csv_file}")
                    success = False
                    continue
                    
                logger.info(f"Creating table {table_name} from {csv_file}")
                
                # Read CSV headers
                df = pd.read_csv(csv_file)
                
                # Drop unnamed index column if present
                if "" in df.columns or "Unnamed: 0" in df.columns:
                    unnamed_col = "" if "" in df.columns else "Unnamed: 0"
                    df = df.drop(columns=[unnamed_col])
                
                # Get column data types
                dtype_map = {
                    'int64': 'INTEGER',
                    'float64': 'REAL',
                    'object': 'TEXT',
                    'bool': 'INTEGER',
                    'datetime64[ns]': 'TEXT',
                }
                
                # Build column definitions
                column_defs = []
                column_defs.append("id INTEGER PRIMARY KEY AUTOINCREMENT")
                
                # Add custom fields that aren't in the CSV
                if table_name == 'statcast_pitches':
                    column_defs.append("pitcher_id INTEGER")
                    column_defs.append("season INTEGER")
                elif table_name == 'statcast_batters':
                    column_defs.append("team_id TEXT")
                    column_defs.append("season INTEGER")
                
                # Add columns from CSV
                for col in df.columns:
                    # Create a SQL-safe column name
                    sql_col = col.replace(" ", "_").replace("%", "pct").replace("-", "_")
                    dtype = dtype_map.get(str(df[col].dtype), 'TEXT')
                    column_defs.append(f'"{sql_col}" {dtype}')
                
                # Check if table exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    logger.info(f"Table {table_name} already exists. Dropping it to recreate.")
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Create the table
                create_query = f"""
                CREATE TABLE {table_name} (
                    {', '.join(column_defs)}
                )
                """
                
                cursor.execute(create_query)
                logger.info(f"Successfully created table {table_name} with columns from {csv_file}")
                
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                success = False
        
        conn.commit()
    
    return success

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
        seasons = list(range(2025, 2026))  # Default: Recent seasons
    
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

def extract_statcast_for_starters_direct_to_db(seasons):
    """
    Fetch Statcast data for all identified starting pitchers and insert directly into database
    using pandas to_sql to preserve column names
    
    Args:
        seasons (list): List of seasons to fetch
    
    Returns:
        bool: Success status
    """
    import pybaseball as pb
    
    with DBConnection() as conn:
        # Get all starting pitchers
        query = "SELECT key_mlbam, name FROM pitcher_ids WHERE is_starter = 1"
        starter_df = pd.read_sql_query(query, conn)
        
        if starter_df.empty:
            logger.error("No starters found in the database")
            return False
        
        starter_ids = list(zip(starter_df['key_mlbam'], starter_df['name']))
        logger.info(f"Found {len(starter_ids)} starters to fetch Statcast data for")
        
        # Process starters in batches to avoid overwhelming the API
        batch_size = DBConfig.BATCH_SIZE
        success_count = 0
        total_pitches = 0
        
        for i in range(0, len(starter_ids), batch_size):
            batch = starter_ids[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} / {(len(starter_ids) + batch_size - 1)//batch_size}")
            
            for pitcher_id, name in batch:
                try:
                    logger.info(f"Fetching Statcast data for {name} (ID: {pitcher_id})")
                    
                    # Fetch data for each season
                    for season in seasons:
                        try:
                            # Define date range for the season
                            if season == 2025:  # Current season
                                start_date = f"{season}-03-30"
                                end_date = "2025-04-06"  # Current date in our scenario
                            else:
                                start_date = f"{season}-03-30"
                                end_date = f"{season}-11-01"
                            
                            # Fetch the data using pybaseball's statcast_pitcher function
                            logger.info(f"  Fetching {name} for {start_date} to {end_date}")
                            
                            pitcher_data = pb.statcast_pitcher(start_date, end_date, pitcher_id)
                            
                            if pitcher_data.empty:
                                logger.warning(f"  No data found for {name} in {season}")
                                continue
                            
                            # Add pitcher_id and season columns if not already there
                            pitcher_data['pitcher_id'] = pitcher_id
                            pitcher_data['season'] = season
                            
                            # Convert date columns to string to avoid SQLite issues
                            if 'game_date' in pitcher_data.columns:
                                pitcher_data['game_date'] = pd.to_datetime(pitcher_data['game_date']).dt.strftime('%Y-%m-%d')
                            
                            # Insert directly into database using to_sql
                            # The if_exists='append' option adds to the table instead of replacing it
                            pitcher_data.to_sql('statcast_pitches', conn, if_exists='append', index=False)
                            
                            logger.info(f"  Inserted {len(pitcher_data)} rows for {name} in {season}")
                            total_pitches += len(pitcher_data)
                            success_count += 1
                            
                        except Exception as e:
                            logger.error(f"  Error fetching {name} for {season}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                
                except Exception as e:
                    logger.error(f"Error processing pitcher {pitcher_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    # Log total insertions
    logger.info(f"Successfully inserted {total_pitches} pitch records for {success_count} pitcher-seasons")
    
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

def load_statcast_pitcher_to_database(csv_file):
    """
    Load a single Statcast pitcher CSV file into the database preserving column names
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        int: Number of rows inserted
    """
    try:
        # Extract pitcher_id and season from filename
        filename = Path(csv_file).name
        parts = filename.replace(".csv", "").split("_")
        
        if len(parts) < 3:
            logger.error(f"Invalid filename format: {filename}. Expected format: pitcher_ID_SEASON.csv")
            return 0
            
        pitcher_id = int(parts[1])
        season = int(parts[2])
        
        logger.info(f"Processing pitcher ID {pitcher_id} from season {season}")
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        if df.empty:
            logger.warning(f"Empty CSV file: {csv_file}")
            return 0
            
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Handling unnamed index column if present
        if "" in df.columns or "Unnamed: 0" in df.columns:
            unnamed_col = "" if "" in df.columns else "Unnamed: 0"
            df = df.drop(columns=[unnamed_col])
        
        # Ensure game_date is in proper format
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
        
        # Create a new DataFrame with pitcher_id and season
        df['pitcher_id'] = pitcher_id
        df['season'] = season
        
        # Get SQL-safe column names by replacing special characters
        df.columns = [col.replace(" ", "_").replace("%", "pct").replace("-", "_") for col in df.columns]
        
        rows_inserted = 0
        
        with DBConnection() as conn:
            cursor = conn.cursor()
            
            # Process in batches to improve performance
            batch_size = 1000
            total_rows = len(df)
            
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    # Filter out NaN values
                    filtered_row = {k: v for k, v in row.items() if not pd.isna(v)}
                    
                    # Build query
                    columns = list(filtered_row.keys())
                    placeholders = ', '.join(['?'] * len(columns))
                    query = f"INSERT INTO statcast_pitches ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    # Execute query
                    cursor.execute(query, list(filtered_row.values()))
                    rows_inserted += 1
                
                # Commit batch
                conn.commit()
                logger.info(f"Inserted {rows_inserted}/{total_rows} rows...")
        
        logger.info(f"Successfully inserted {rows_inserted} rows from {csv_file}")
        return rows_inserted
        
    except Exception as e:
        logger.error(f"Error processing {csv_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

# Specialized function for loading Statcast batter data
def load_statcast_batter_to_database(csv_file):
    """
    Load a single Statcast batter CSV file into the database with proper column handling
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        int: Number of rows inserted
    """
    try:
        # Extract batter_id and season from filename (e.g., batter_XXXXX_YYYY.csv)
        filename = Path(csv_file).name
        parts = filename.replace(".csv", "").split("_")
        
        if len(parts) < 3:
            logger.error(f"Invalid filename format: {filename}. Expected format: batter_ID_SEASON.csv")
            return 0
            
        team_id = parts[1]  # This might be a team ID rather than a player ID
        season = int(parts[2])
        
        logger.info(f"Processing batter/team data {team_id} from season {season}")
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        if df.empty:
            logger.warning(f"Empty CSV file: {csv_file}")
            return 0
            
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Handling unnamed index column if present
        if "" in df.columns or "Unnamed: 0" in df.columns:
            unnamed_col = "" if "" in df.columns else "Unnamed: 0"
            df = df.drop(columns=[unnamed_col])
        
        # Ensure game_date is in proper format
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
        
        # Add season column if not present
        if 'season' not in df.columns:
            df['season'] = season
        
        # Create the table if it doesn't exist
        with DBConnection() as conn:
            cursor = conn.cursor()
            
            # Check if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='statcast_batters'")
            if not cursor.fetchone():
                # Create the table with key columns
                create_query = """
                CREATE TABLE statcast_batters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    season INTEGER,
                    game_pk INTEGER,
                    game_date TEXT,
                    batter INTEGER,
                    pitcher INTEGER,
                    stand TEXT,
                    p_throws TEXT,
                    pitch_type TEXT,
                    events TEXT,
                    description TEXT,
                    zone INTEGER,
                    release_speed REAL,
                    release_spin_rate REAL,
                    pfx_x REAL,
                    pfx_z REAL,
                    plate_x REAL,
                    plate_z REAL,
                    home_team TEXT,
                    away_team TEXT,
                    at_bat_number INTEGER,
                    pitch_number INTEGER,
                    inning INTEGER,
                    inning_topbot TEXT,
                    launch_speed REAL,
                    launch_angle REAL
                )
                """
                cursor.execute(create_query)
                conn.commit()
        
        rows_inserted = 0
        
        with DBConnection() as conn:
            cursor = conn.cursor()
            
            # Process each row
            for _, row in df.iterrows():
                # Collect values for insertion
                values = {
                    'team_id': team_id,
                    'season': season,
                    'game_pk': row.get('game_pk', None),
                    'game_date': row.get('game_date', None),
                    'batter': row.get('batter', None),
                    'pitcher': row.get('pitcher', None),
                    'stand': row.get('stand', None),
                    'p_throws': row.get('p_throws', None),
                    'pitch_type': row.get('pitch_type', None),
                    'events': row.get('events', None),
                    'description': row.get('description', None),
                    'zone': row.get('zone', None),
                    'release_speed': safe_float(row.get('release_speed', None)),
                    'release_spin_rate': safe_float(row.get('release_spin_rate', None)),
                    'pfx_x': safe_float(row.get('pfx_x', None)),
                    'pfx_z': safe_float(row.get('pfx_z', None)),
                    'plate_x': safe_float(row.get('plate_x', None)),
                    'plate_z': safe_float(row.get('plate_z', None)),
                    'home_team': row.get('home_team', None),
                    'away_team': row.get('away_team', None),
                    'at_bat_number': row.get('at_bat_number', None),
                    'pitch_number': row.get('pitch_number', None),
                    'inning': row.get('inning', None),
                    'inning_topbot': row.get('inning_topbot', None),
                    'launch_speed': safe_float(row.get('launch_speed', None)),
                    'launch_angle': safe_float(row.get('launch_angle', None))
                }
                
                # Filter out None values
                filtered_values = {k: v for k, v in values.items() if v is not None}
                
                # Build the insert query
                columns = list(filtered_values.keys())
                placeholders = ', '.join(['?'] * len(columns))
                query = f"INSERT INTO statcast_batters ({', '.join(columns)}) VALUES ({placeholders})"
                
                # Execute the query
                cursor.execute(query, list(filtered_values.values()))
                rows_inserted += 1
                
                # Commit periodically
                if rows_inserted % 1000 == 0:
                    conn.commit()
                    logger.info(f"Inserted {rows_inserted} rows so far...")
            
            conn.commit()
        
        logger.info(f"Successfully inserted {rows_inserted} rows from {csv_file}")
        return rows_inserted
        
    except Exception as e:
        logger.error(f"Error processing {csv_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

# Specialized function for loading team batting data
def load_team_batting_to_database(csv_file):
    """
    Load a team batting CSV file into the database with proper column handling
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        int: Number of rows inserted
    """
    try:
        logger.info(f"Processing team batting data from {csv_file}")
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        if df.empty:
            logger.warning(f"Empty CSV file: {csv_file}")
            return 0
            
        logger.info(f"Loaded {len(df)} rows from {csv_file}")
        
        # Handling unnamed index column if present
        if "" in df.columns or "Unnamed: 0" in df.columns:
            unnamed_col = "" if "" in df.columns else "Unnamed: 0"
            df = df.drop(columns=[unnamed_col])
        
        # Create the table if it doesn't exist
        with DBConnection() as conn:
            cursor = conn.cursor()
            
            # Check if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_batting_stats'")
            if not cursor.fetchone():
                # Create the table with essential columns
                create_query = """
                CREATE TABLE team_batting_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    season INTEGER,
                    teamIDfg INTEGER,
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
                    hard_hit_percent REAL
                )
                """
                cursor.execute(create_query)
                conn.commit()
        
        rows_inserted = 0
        
        with DBConnection() as conn:
            cursor = conn.cursor()
            
            # Get existing team mappings
            cursor.execute("SELECT team_id, key_fangraphs FROM teams")
            team_mappings = {fangraphs_id: team_id for team_id, fangraphs_id in cursor.fetchall()}
            
            # Process each row
            for _, row in df.iterrows():
                # Get team_id from mapping
                fangraphs_id = row.get('teamIDfg', None)
                team_id = team_mappings.get(fangraphs_id)
                
                if not team_id:
                    logger.warning(f"No team mapping found for FanGraphs ID: {fangraphs_id}")
                    team_id = f"TEAM_{fangraphs_id}"  # Create a temporary mapping
                
                season = row.get('Season', 0)
                
                # Handle % signs in column names
                kpct = safe_float(row.get('K%', 0))
                bbpct = safe_float(row.get('BB%', 0))
                ospct = safe_float(row.get('O-Swing%', 0))
                zcpct = safe_float(row.get('Z-Contact%', 0))
                cpct = safe_float(row.get('Contact%', 0))
                zpct = safe_float(row.get('Zone%', 0))
                sspct = safe_float(row.get('SwStr%', 0))
                hhpct = safe_float(row.get('Hard%', 0))
                
                # Collect values for insertion
                values = {
                    'team_id': team_id,
                    'season': season,
                    'teamIDfg': fangraphs_id,
                    'k_percent': kpct,
                    'bb_percent': bbpct,
                    'avg': safe_float(row.get('AVG', 0)),
                    'obp': safe_float(row.get('OBP', 0)),
                    'slg': safe_float(row.get('SLG', 0)),
                    'ops': safe_float(row.get('OPS', 0)),
                    'iso': safe_float(row.get('ISO', 0)),
                    'babip': safe_float(row.get('BABIP', 0)),
                    'o_swing_percent': ospct,
                    'z_contact_percent': zcpct,
                    'contact_percent': cpct,
                    'zone_percent': zpct,
                    'swstr_percent': sspct,
                    'hard_hit_percent': hhpct
                }
                
                # Filter out None values
                filtered_values = {k: v for k, v in values.items() if v is not None}
                
                # Build the insert query
                columns = list(filtered_values.keys())
                placeholders = ', '.join(['?'] * len(columns))
                query = f"INSERT INTO team_batting_stats ({', '.join(columns)}) VALUES ({placeholders})"
                
                # Execute the query
                cursor.execute(query, list(filtered_values.values()))
                rows_inserted += 1
            
            conn.commit()
        
        logger.info(f"Successfully inserted {rows_inserted} rows from {csv_file}")
        return rows_inserted
        
    except Exception as e:
        logger.error(f"Error processing {csv_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

def load_all_data_to_database(csv_directory):
    """
    Load all CSV files from a directory into the database
    
    Args:
        csv_directory (str): Directory containing CSV files
        
    Returns:
        dict: Count of rows inserted by data type
    """
    directory = Path(csv_directory)
    
    # Find all CSV files by type
    pitcher_files = list(directory.glob("pitcher_*.csv"))
    batter_files = list(directory.glob("batter_*.csv"))
    team_batting_files = list(directory.glob("team_batting*.csv"))
    
    logger.info(f"Found {len(pitcher_files)} pitcher files, {len(batter_files)} batter files, "
               f"and {len(team_batting_files)} team batting files")
    
    # Verify tables exist before loading data
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        # Check if required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='statcast_pitches'")
        pitcher_table_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='statcast_batters'")
        batter_table_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_batting_stats'")
        team_table_exists = cursor.fetchone() is not None
        
        if not pitcher_table_exists:
            logger.error("statcast_pitches table does not exist! Cannot load pitcher data.")
        
        if not batter_table_exists:
            logger.error("statcast_batters table does not exist! Cannot load batter data.")
            
        if not team_table_exists:
            logger.error("team_batting_stats table does not exist! Cannot load team data.")
    
    results = {
        'pitcher': 0,
        'batter': 0,
        'team_batting': 0
    }
    
    # Process each file type if tables exist
    if pitcher_table_exists:
        for file_path in pitcher_files:
            if str(file_path).endswith('pitcher_example.csv'):
                logger.info(f"Skipping example file: {file_path}")
                continue
            rows = load_statcast_pitcher_to_database(file_path)
            results['pitcher'] += rows
    
    if batter_table_exists:
        for file_path in batter_files:
            if str(file_path).endswith('batter_example.csv'):
                logger.info(f"Skipping example file: {file_path}")
                continue
            rows = load_statcast_batter_to_database(file_path)
            results['batter'] += rows
    
    if team_table_exists:
        for file_path in team_batting_files:
            if str(file_path).endswith('team_batting_example.csv'):
                logger.info(f"Skipping example file: {file_path}")
                continue
            rows = load_team_batting_to_database(file_path)
            results['team_batting'] += rows
    
    logger.info(f"Successfully loaded {results['pitcher']} pitcher records, {results['batter']} batter records, "
               f"and {results['team_batting']} team batting records")
    
    return results

# Utility function for safe float conversion
def safe_float(value, default=None):
    """Safely convert a value to float, handling NA values"""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def main():
    """Initialize database and create/fetch data"""
    # Initialize essential schema and team data 
    logger.info("Setting up database...")
    
    # Create tables based on sample CSV files
    statcast_dir = "data/statcast"
    pitcher_example = os.path.join(statcast_dir, "pitcher_example.csv")
    batter_example = os.path.join(statcast_dir, "batter_example.csv")
    team_batting_example = os.path.join(statcast_dir, "team_batting_example.csv")
    
    # Sample CSV files to infer table structure
    sample_csv_files = {
        'statcast_pitches': pitcher_example,
        'statcast_batters': batter_example,
        'team_batting_stats': team_batting_example
    }
    
    # Create tables based on CSV structure
    create_tables_from_csv_structure(sample_csv_files)
    
    # Initialize team data
    initialize_team_data()
    
    # Initialize pitcher ID mappings
    initialize_pitcher_ids()
    
    # Fetch Statcast data DIRECTLY to database
    seasons = [2024, 2025]  # Current year and last year
    logger.info(f"Fetching Statcast data for seasons: {seasons}")
    
    # Extract data directly to database (no CSV files)
    success = extract_statcast_for_starters_direct_to_db(seasons)
    
    if success:
        logger.info("Successfully fetched Statcast data and inserted into database")
    else:
        logger.error("Failed to fetch/insert Statcast data")
    
    # Log database stats
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM statcast_pitches")
        pitch_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT pitcher_id) FROM statcast_pitches")
        pitcher_count = cursor.fetchone()[0]
        
        logger.info(f"Database now contains {pitch_count} pitch records from {pitcher_count} pitchers")
    
    logger.info("Database setup complete")

if __name__ == "__main__":
    main()