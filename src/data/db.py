# Database utilities for the MLB pitcher prediction project
import sqlite3
import pandas as pd
from pathlib import Path
from src.data.utils import safe_float, normalize_name
from config import DBConfig
from src.data.utils import setup_logger, ensure_dir
from functools import lru_cache
import hashlib
import time

logger = setup_logger(__name__)

# Database path
DB_PATH = DBConfig.PATH

_cache = {}
_cache_timeout = 300 # 5 minutes

class DBConnection:
    """Context manager for database connections"""
    def __init__(self, db_name=DB_PATH):
        self.db_name = db_name

    def __enter__(self):
        ensure_dir(Path(self.db_name).parent)
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

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

# Update the init_database function in src/data/db.py
def init_database():
    """
    Initialize the SQLite database with the necessary tables
    focusing only on strikeout prediction
    """
    logger.info("Initializing SQLite database...")
    
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        # Create tables
        
        # Teams table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id TEXT PRIMARY KEY,
            team_name TEXT,
            team_abbrev TEXT
        )
        ''')
        
        # Game context table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_context (
            game_id TEXT PRIMARY KEY,
            game_date TEXT,
            season INTEGER,
            home_team_id TEXT,
            away_team_id TEXT,
            stadium TEXT,
            day_night TEXT,
            FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
        )
        ''')
        
        # Pitchers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pitchers (
            pitcher_id INTEGER PRIMARY KEY,
            player_name TEXT,
            normalized_name TEXT,
            statcast_id INTEGER
        )
        ''')
        
        # Game-level Statcast data
        # Check if opponent_team_id column exists
        cursor.execute("PRAGMA table_info(game_stats)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if not columns:  # Table doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitcher_id INTEGER,
                game_id TEXT,
                game_date TEXT,
                season INTEGER,
                opponent_team_id TEXT,
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
                FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id),
                FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id)
            )
            ''')
        elif 'opponent_team_id' not in column_names:  # Table exists but needs column
            cursor.execute('ALTER TABLE game_stats ADD COLUMN opponent_team_id TEXT')
        
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
            last_10_games_strikeouts_avg REAL,
            last_3_games_velo_avg REAL,
            last_5_games_velo_avg REAL,
            last_10_games_velo_avg REAL,
            last_3_games_swinging_strike_pct_avg REAL,
            last_5_games_swinging_strike_pct_avg REAL,
            last_10_games_swinging_strike_pct_avg REAL,
            days_rest INTEGER,
            team_changed BOOLEAN,
            FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id)
        )
        ''')
        
        conn.commit()
    
    logger.info("Database initialization complete.")

def store_statcast_data(statcast_df, force_refresh=False):
    """
    Process and store statcast data in SQLite database
    
    Args:
        statcast_df (pandas.DataFrame): Raw statcast data
        force_refresh (bool): Whether to force refresh existing data
    """
    
    if statcast_df.empty:
        logger.warning("No statcast data to store.")
        return
    
    # Check if we need to refresh the data
    if not force_refresh and is_table_populated('game_stats'):
        logger.info("Game stats table already populated and force_refresh is False. Skipping.")
        return
    
    logger.info("Processing and storing statcast data...")
    
    # Connect to database
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        # Clear existing data if force_refresh
        if force_refresh:
            cursor.execute("DELETE FROM pitch_mix")
            cursor.execute("DELETE FROM game_stats")
            conn.commit()
        
        # Process statcast data to game level - import inside function to avoid circular import
        from src.data.process import aggregate_to_game_level
        game_level = aggregate_to_game_level(statcast_df)
        game_level = game_level.fillna(0)
        
        if game_level.empty:
            logger.warning("No game-level data to store after aggregation.")
            return
        
        # Check if 'pitcher' column exists in the dataframe
        if 'pitcher' not in game_level.columns:
            logger.error("Required 'pitcher' column missing from game_level dataframe.")
            logger.info(f"Available columns: {game_level.columns.tolist()}")
            return
        
        # Add pitchers to the pitchers table if they don't exist
        pitcher_cols = ['pitcher', 'player_name']
        if all(col in game_level.columns for col in pitcher_cols):
            for _, row in game_level[pitcher_cols].drop_duplicates().iterrows():
                try:
                    pitcher_id = int(row['pitcher'])
                    player_name = row['player_name']
                    norm_name = normalize_name(player_name)
                    
                    cursor.execute(
                        "INSERT OR IGNORE INTO pitchers (statcast_id, player_name, normalized_name) VALUES (?, ?, ?)",
                        (pitcher_id, player_name, norm_name)
                    )
                except (ValueError, TypeError) as e:
                    logger.error(f"Error inserting pitcher {row['player_name']}: {e}")
                    continue
        else:
            logger.error(f"Missing required columns. Available: {game_level.columns.tolist()}")
            return
    
        conn.commit()
    
        # Get pitcher IDs mapping
        cursor.execute("SELECT pitcher_id, statcast_id FROM pitchers WHERE statcast_id IS NOT NULL")
        pitcher_map = {int(statcast_id): pitcher_id for pitcher_id, statcast_id in cursor.fetchall()}
        
        # Insert game stats
        game_stats_ids = {}  # Map to store game_id -> database_id for pitch mix data
        inserted_count = 0

        for _, row in game_level.iterrows():
            # Get pitcher_id from mapping
            try:
                statcast_id = int(row['pitcher'])
                if statcast_id not in pitcher_map:
                    logger.warning(f"No pitcher_id found for statcast_id {statcast_id}")
                    continue
                
                pitcher_id = pitcher_map[statcast_id]
                
                # Insert game stats
                game_date = row['game_date'].strftime('%Y-%m-%d')
                game_id = row['game_id']
                season = row['season'] if 'season' in row else row['game_date'].year
                
                # Check if this game already exists for this pitcher
                cursor.execute(
                    "SELECT id FROM game_stats WHERE pitcher_id = ? AND game_id = ?",
                    (pitcher_id, game_id)
                )
                existing = cursor.fetchone()
                
                if existing:
                    game_stats_id = existing[0]
                    # Update existing record
                    cursor.execute('''
                        UPDATE game_stats
                        SET 
                            strikeouts = ?,
                            hits = ?,
                            walks = ?,
                            home_runs = ?,
                            release_speed_mean = ?,
                            release_speed_max = ?,
                            release_spin_rate_mean = ?,
                            swinging_strike_pct = ?,
                            called_strike_pct = ?,
                            zone_rate = ?
                        WHERE id = ?
                    ''', (
                        int(row.get('strikeouts', 0)), 
                        int(row.get('hits', 0)), 
                        int(row.get('walks', 0)), 
                        int(row.get('home_runs', 0)),
                        safe_float(row.get('release_speed_mean', 0)), 
                        safe_float(row.get('release_speed_max', 0)),
                        safe_float(row.get('release_spin_rate_mean', 0)), 
                        safe_float(row.get('swinging_strike_pct', 0)),
                        safe_float(row.get('called_strike_pct', 0)), 
                        safe_float(row.get('zone_rate', 0)),
                        game_stats_id
                    ))
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
                        int(row.get('strikeouts', 0)), 
                        int(row.get('hits', 0)), 
                        int(row.get('walks', 0)), 
                        int(row.get('home_runs', 0)),
                        safe_float(row.get('release_speed_mean', 0)), 
                        safe_float(row.get('release_speed_max', 0)),
                        safe_float(row.get('release_spin_rate_mean', 0)), 
                        safe_float(row.get('swinging_strike_pct', 0)),
                        safe_float(row.get('called_strike_pct', 0)), 
                        safe_float(row.get('zone_rate', 0))
                    ))
                    
                    game_stats_id = cursor.lastrowid
                    inserted_count += 1
                
                # Store ID for pitch mix data
                game_key = (pitcher_id, game_id)
                game_stats_ids[game_key] = game_stats_id
                
                # Delete existing pitch mix data for this game
                cursor.execute("DELETE FROM pitch_mix WHERE game_stats_id = ?", (game_stats_id,))
                
                # Store pitch mix data if available
                pitch_mix_count = 0
                pitch_cols = [col for col in row.index if col.startswith('pitch_pct_')]
                
                for col in pitch_cols:
                    pitch_type = col.replace('pitch_pct_', '')
                    percentage = row[col]
                    
                    if percentage > 0:
                        cursor.execute(
                            "INSERT INTO pitch_mix (game_stats_id, pitch_type, percentage) VALUES (?, ?, ?)",
                            (game_stats_id, pitch_type, percentage)
                        )
                        pitch_mix_count += 1
                
                # Commit periodically to avoid large transactions
                if inserted_count % 1000 == 0:
                    conn.commit()
                    logger.info(f"Processed {inserted_count} games so far...")
            
            except Exception as e:
                logger.error(f"Error processing game record: {e}")
                continue
        
        conn.commit()
        
        # Verify pitch mix data was stored
        cursor.execute("SELECT COUNT(*) FROM pitch_mix")
        pitch_mix_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM game_stats")
        game_stats_count = cursor.fetchone()[0]
        
        logger.info(f"Stored {game_stats_count} game records with {pitch_mix_count} pitch mix records.")
        
        if pitch_mix_count == 0:
            # Print some diagnostic info about the pitch columns
            pitch_cols = [col for col in game_level.columns if col.startswith('pitch_pct_')]
            if pitch_cols:
                logger.warning(f"Found {len(pitch_cols)} pitch columns in the data, but no records were stored.")
                logger.warning(f"Pitch columns: {', '.join(pitch_cols)}")
                
                # Check for non-zero values in pitch columns
                for col in pitch_cols:
                    non_zero = (game_level[col] > 0).sum()
                    if non_zero > 0:
                        logger.warning(f"Column {col} has {non_zero} non-zero values but none were stored.")
            else:
                logger.warning("No pitch_pct_* columns found in the data. Check the aggregation process.")
    
    
    logger.info("Statcast data stored in database.")

def get_pitcher_data(force_refresh=False):
    """
    Retrieve joined data from the database for feature engineering with caching
    
    Args:
        force_refresh (bool): Whether to force refresh the cache
        
    Returns:
        pandas.DataFrame: Joined data with all features
    """
    # Generate cache key based on current database state (mtime of db file)
    db_path = Path(DBConfig.PATH)
    if db_path.exists():
        last_modified = int(db_path.stat().st_mtime)
        cache_key = f"pitcher_data_{last_modified}"
    else:
        cache_key = "pitcher_data_default"
    
    current_time = time.time()
    
    # Check if we have a valid cached result
    if not force_refresh and cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if current_time - cached_time < _cache_timeout:
            logger.info(f"Using cached pitcher data ({len(cached_data)} rows)")
            return cached_data
    
    logger.info("Retrieving pitcher data from database...")
    
    # Original database query code...
    with DBConnection() as conn:
        cursor = conn.cursor()

        # First, get all column names from prediction_features table
        cursor.execute("PRAGMA table_info(prediction_features)")
        pf_columns = [row[1] for row in cursor.fetchall()]
        
        # Remove columns that would cause duplicates (pitcher_id, game_id, game_date, season)
        pf_columns = [col for col in pf_columns if col not in ('id', 'pitcher_id', 'game_id', 'game_date', 'season')]
        
        # Construct the SELECT part of the query
        base_columns = """
            g.id as game_db_id,
            g.pitcher_id,
            p.player_name,
            p.statcast_id,
            g.game_id,
            g.game_date,
            g.season,
            g.opponent_team_id,
            g.strikeouts,
            g.hits,
            g.walks,
            g.home_runs,
            g.release_speed_mean,
            g.release_speed_max,
            g.release_spin_rate_mean,
            g.swinging_strike_pct,
            g.called_strike_pct,
            g.zone_rate
        """
        
        # Add prediction feature columns
        feature_columns = ",\n        ".join([f"f.{col}" for col in pf_columns])
        select_clause = base_columns
        if feature_columns:
            select_clause += ",\n        " + feature_columns
        
        # Construct the full query
        query = f"""
        SELECT 
            {select_clause}
        FROM 
            game_stats g
        JOIN 
            pitchers p ON g.pitcher_id = p.pitcher_id
        LEFT JOIN
            prediction_features f ON g.pitcher_id = f.pitcher_id AND g.game_id = f.game_id
        ORDER BY
            g.pitcher_id, g.game_date
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Get pitch mix data if pitch_mix_features table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pitch_mix_features'")
        has_pitch_mix = cursor.fetchone() is not None
        
        if has_pitch_mix:
            pitch_mix_query = '''
            SELECT 
                pf.pitcher_id,
                pf.game_id,
                pmf.pitch_type,
                pmf.percentage
            FROM 
                pitch_mix_features pmf
            JOIN 
                prediction_features pf ON pmf.prediction_feature_id = pf.id
            '''
            
            try:
                pitch_mix = pd.read_sql_query(pitch_mix_query, conn)
                
                # Add pitch mix data if available
                if not pitch_mix.empty:
                    # Pivot the pitch mix data
                    pitch_pivot = pitch_mix.pivot_table(
                        index=['pitcher_id', 'game_id'],
                        columns='pitch_type',
                        values='percentage',
                        fill_value=0
                    ).reset_index()
                    
                    # Fix column names after pivot
                    if isinstance(pitch_pivot.columns, pd.MultiIndex):
                        pitch_pivot.columns = [
                            'pitcher_id' if col[0] == 'pitcher_id' else 
                            'game_id' if col[0] == 'game_id' else
                            f'prev_game_pitch_pct_{col[1]}' for col in pitch_pivot.columns
                        ]
                    
                    # Merge with main DataFrame
                    df = pd.merge(df, pitch_pivot, on=['pitcher_id', 'game_id'], how='left')
                    logger.info(f"Added {len(pitch_pivot.columns)-2} pitch mix columns")
            except Exception as e:
                logger.warning(f"Error retrieving pitch mix data: {e}")
    
    logger.info(f"Retrieved {len(df)} rows of pitcher data with {len(df.columns)} columns")

    current_time = time.time()
    for k in list(_cache.keys()):
        if current_time - _cache[k][0] > _cache_timeout * 2:
            del _cache[k]
    
    # Check for enhanced features
    enhanced_patterns = ['_std', 'trend_', 'momentum_', 'entropy']
    enhanced_cols = [col for col in df.columns if any(pattern in col for pattern in enhanced_patterns)]
    if enhanced_cols:
        logger.info(f"Retrieved {len(enhanced_cols)} enhanced feature columns: {', '.join(enhanced_cols[:5])}...")
    else:
        logger.warning("No enhanced feature columns found!")

    # Cache the result
    _cache[cache_key] = (current_time, df)
    
    logger.info(f"Retrieved and cached {len(df)} rows of pitcher data")
    
    return df

def clear_database():
    """Clear all data from the database for fresh start"""
    logger.info("Clearing all data from the database...")
    
    with DBConnection() as conn:
        cursor = conn.cursor()
        
        # Delete data from all tables
        cursor.execute("DELETE FROM pitch_mix")
        cursor.execute("DELETE FROM prediction_features")
        cursor.execute("DELETE FROM game_stats")
        cursor.execute("DELETE FROM pitchers")
        
        # Reset auto-increment counters
        cursor.execute("DELETE FROM sqlite_sequence")
        
        conn.commit()
    
    logger.info("Database cleared successfully")

def update_database_schema():
    """Update the database schema to support enhanced features"""
    logger.info("Updating database schema for enhanced features...")
    
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        # Check for existing columns
        cursor.execute("PRAGMA table_info(prediction_features)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        # New columns to add
        new_columns = [
            # Standard deviation features
            ('last_3_games_strikeouts_std', 'REAL'),
            ('last_5_games_strikeouts_std', 'REAL'),
            ('last_3_games_velo_std', 'REAL'),
            ('last_5_games_velo_std', 'REAL'),
            ('last_3_games_swinging_strike_pct_std', 'REAL'),
            ('last_5_games_swinging_strike_pct_std', 'REAL'),
            ('last_3_games_called_strike_pct_avg', 'REAL'),
            ('last_5_games_called_strike_pct_avg', 'REAL'),
            ('last_3_games_zone_rate_avg', 'REAL'),
            ('last_5_games_zone_rate_avg', 'REAL'),
            
            # Trend indicators
            ('trend_3_strikeouts', 'REAL'),
            ('trend_5_strikeouts', 'REAL'),
            ('trend_3_release_speed_mean', 'REAL'),
            ('trend_5_release_speed_mean', 'REAL'),
            ('trend_3_swinging_strike_pct', 'REAL'),
            ('trend_5_swinging_strike_pct', 'REAL'),
            
            # Momentum indicators
            ('momentum_3_strikeouts', 'REAL'),
            ('momentum_5_strikeouts', 'REAL'),
            ('momentum_3_release_speed_mean', 'REAL'),
            ('momentum_5_release_speed_mean', 'REAL'),
            ('momentum_3_swinging_strike_pct', 'REAL'),
            ('momentum_5_swinging_strike_pct', 'REAL'),
            
            # Pitch mix entropy
            ('pitch_entropy', 'REAL'),
            ('prev_game_pitch_entropy', 'REAL'),
            
            # NEW: Pitcher-specific baseline features
            ('career_so_avg', 'REAL'),
            ('career_so_std', 'REAL'),
            ('career_so_per_batter', 'REAL'),
            ('career_so_consistency', 'REAL'),
            ('prev_so_deviation', 'REAL'),
            ('so_deviation_3g_avg', 'REAL'),
            ('so_deviation_5g_avg', 'REAL'),
            ('is_home_game', 'INTEGER'),
            ('home_away_so_exp', 'REAL'),
            
            # NEW: Enhanced matchup features
            ('opponent_whiff_rate', 'REAL'),
            ('opponent_chase_rate', 'REAL'),
            ('opponent_zone_contact_rate', 'REAL'),
            ('opponent_k_vs_avg', 'REAL'),
            ('opponent_whiff_vs_avg', 'REAL'),
            ('matchup_advantage', 'REAL'),
            ('recency_weighted_matchup', 'REAL')
        ]
        
        # Add columns if they don't exist
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE prediction_features ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added column {col_name} to prediction_features table")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not add column {col_name}: {e}")
        
        # Create pitch_mix_features table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pitch_mix_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_feature_id INTEGER,
            pitch_type TEXT,
            percentage REAL,
            FOREIGN KEY (prediction_feature_id) REFERENCES prediction_features(id)
        )
        ''')
        logger.info("Created pitch_mix_features table if it didn't exist")
        
        conn.commit()
    
    logger.info("Database schema update complete")

def store_game_context(game_context_df):
    """Store game context information in the database"""
    if game_context_df.empty:
        logger.warning("No game context data to store")
        return
    
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        # First, ensure teams exist
        teams_to_add = set()
        if 'home_team' in game_context_df.columns:
            teams_to_add.update(game_context_df['home_team'].dropna().unique())
        if 'away_team' in game_context_df.columns:
            teams_to_add.update(game_context_df['away_team'].dropna().unique())
        
        for team in teams_to_add:
            cursor.execute(
                "INSERT OR IGNORE INTO teams (team_id, team_name) VALUES (?, ?)",
                (team, team)
            )
        
        # Insert game context records
        inserted = 0
        for _, row in game_context_df.iterrows():
            try:
                # Convert timestamp to string if needed
                game_date = row.get('game_date', '')
                if hasattr(game_date, 'strftime'):
                    game_date = game_date.strftime('%Y-%m-%d')
                    
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO game_context 
                    (game_id, game_date, season, home_team_id, away_team_id, stadium, day_night)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get('game_id', ''),
                        game_date,
                        row.get('season', 0),
                        row.get('home_team', ''),
                        row.get('away_team', ''),
                        row.get('stadium', ''),
                        row.get('day_night', '')
                    )
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting game context: {e}")
                continue
        
        conn.commit()

    logger.info(f"Stored {inserted} game context records")

def store_team_data(team_data_df):
    """Store team data in the database"""
    if team_data_df.empty:
        logger.warning("No team data to store")
        return
    
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        # Create team stats table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT,
            season INTEGER,
            strikeout_rate REAL,
            walk_rate REAL,
            batting_avg REAL,
            ops REAL,
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
        ''')
        
        # Insert team stats records
        inserted = 0
        for _, row in team_data_df.iterrows():
            try:
                # First ensure team exists
                cursor.execute(
                    "INSERT OR IGNORE INTO teams (team_id, team_name) VALUES (?, ?)",
                    (row['team_id'], row['team_id'])
                )
                
                # Insert stats
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO team_stats 
                    (team_id, season, strikeout_rate, walk_rate, batting_avg, ops)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row['team_id'],
                        row['season'],
                        row.get('strikeout_rate', 0),
                        row.get('walk_rate', 0),
                        row.get('batting_avg', 0),
                        row.get('ops', 0)
                    )
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting team stats: {e}")
                continue
        
        conn.commit()

    logger.info(f"Stored {inserted} team stat records")

def store_opponent_data(opponent_df):
    """Store opponent mapping data in the database"""
    if opponent_df.empty:
        logger.warning("No opponent data to store")
        return
    
    with DBConnection() as conn:
        cursor = conn.cursor()
    
        updated = 0
        for _, row in opponent_df.iterrows():
            try:
                # Get pitcher_id from statcast_id
                cursor.execute(
                    "SELECT pitcher_id FROM pitchers WHERE statcast_id = ?",
                    (row['pitcher'],)
                )
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"No pitcher_id found for statcast_id {row['pitcher']}")
                    continue
                    
                pitcher_id = result[0]
                
                # Update game_stats with opponent_team_id
                cursor.execute(
                    "UPDATE game_stats SET opponent_team_id = ? WHERE pitcher_id = ? AND game_id = ?",
                    (row['opponent_team_id'], pitcher_id, row['game_id'])
                )
                
                if cursor.rowcount > 0:
                    updated += 1
            except Exception as e:
                logger.error(f"Error storing opponent data: {e}")
                continue
        
        conn.commit()

    logger.info(f"Updated opponent data for {updated} game records")