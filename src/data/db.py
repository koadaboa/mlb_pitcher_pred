# Database utilities for the MLB pitcher prediction project
import sqlite3
import pandas as pd
import logging
from difflib import SequenceMatcher
from pathlib import Path

from src.data.utils import normalize_name  # Use the new utils module

logger = logging.getLogger(__name__)

# Database path
DB_PATH = "data/pitcher_stats.db"

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling NA values"""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_db_connection(db_name=DB_PATH):
    """
    Get a connection to the SQLite database
    
    Args:
        db_name (str): Path to the SQLite database file
        
    Returns:
        sqlite3.Connection: Connection to the database
    """
    data_dir = Path(db_name).parent
    data_dir.mkdir(exist_ok=True, parents=True)
    
    return sqlite3.connect(db_name)

def execute_query(query, params=None):
    """
    Execute a query and return results as a DataFrame
    
    Args:
        query (str): SQL query to execute
        params (dict, optional): Parameters for the query
        
    Returns:
        pandas.DataFrame: Query results
    """
    conn = get_db_connection()
    try:
        if params:
            return pd.read_sql_query(query, conn, params=params)
        else:
            return pd.read_sql_query(query, conn)
    finally:
        conn.close()

def execute_update(query, params=None):
    """
    Execute a SQL update query
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query
        
    Returns:
        int: Number of rows affected
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()

def is_table_populated(table_name):
    """
    Check if a table in the database has data
    
    Args:
        table_name (str): Table name to check
        
    Returns:
        bool: True if the table has data
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    
    conn.close()
    
    return count > 0

# Updated init_database function in src/data/db.py

def init_database():
    """
    Initialize the SQLite database with the necessary tables
    """
    logger.info("Initializing SQLite database...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables
    
    # Pitchers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pitchers (
        pitcher_id INTEGER PRIMARY KEY,
        player_name TEXT,
        statcast_id INTEGER
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
        outs INTEGER,
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
        last_3_games_outs_avg REAL,
        last_5_games_outs_avg REAL,
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
    conn = get_db_connection()
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
        conn.close()
        return
    
    # Check if 'pitcher' column exists in the dataframe
    if 'pitcher' not in game_level.columns:
        logger.error("Required 'pitcher' column missing from game_level dataframe.")
        logger.info(f"Available columns: {game_level.columns.tolist()}")
        conn.close()
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
        conn.close()
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
    
    conn.close()
    
    logger.info("Statcast data stored in database.")

def store_traditional_stats(trad_df, force_refresh=False):
    """
    Process and store traditional stats in SQLite database
    
    Args:
        trad_df (pandas.DataFrame): Traditional stats data
        force_refresh (bool): Whether to force refresh existing data
    """
    from src.data.process import process_traditional_stats, normalize_name
    
    if trad_df.empty:
        logger.warning("No traditional stats to store.")
        return
    
    # Check if we need to refresh the data
    if not force_refresh and is_table_populated('traditional_stats'):
        logger.info("Traditional stats table already populated and force_refresh is False. Skipping.")
        return
    
    logger.info("Processing and storing traditional stats...")
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing data if force_refresh
    if force_refresh:
        cursor.execute("DELETE FROM traditional_stats")
        conn.commit()
    
    # Process traditional stats
    processed_trad = process_traditional_stats(trad_df)
    
    if processed_trad.empty:
        logger.warning("No processed traditional stats to store.")
        conn.close()
        return
    
    # Ensure required columns exist
    required_cols = ['pitcher_id', 'player_name', 'season']
    if not all(col in processed_trad.columns for col in required_cols):
        logger.error(f"Missing required columns in traditional stats. Available: {processed_trad.columns.tolist()}")
        conn.close()
        return
    
    # Add pitchers to the pitchers table if they don't exist
    for _, row in processed_trad[['pitcher_id', 'player_name']].drop_duplicates().iterrows():
        try:
            trad_id = int(row['pitcher_id'])
            player_name = row['player_name']
            norm_name = normalize_name(player_name)
            
            cursor.execute(
                "INSERT OR IGNORE INTO pitchers (traditional_id, player_name, normalized_name) VALUES (?, ?, ?)",
                (trad_id, player_name, norm_name)
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error inserting traditional pitcher {row['player_name']}: {e}")
            continue
    
    conn.commit()
    
    # Get pitcher IDs mapping
    cursor.execute("SELECT pitcher_id, traditional_id FROM pitchers WHERE traditional_id IS NOT NULL")
    trad_id_map = {trad_id: pitcher_id for pitcher_id, trad_id in cursor.fetchall()}
    
    # Insert traditional stats
    stats_inserted = 0
    for _, row in processed_trad.iterrows():
        try:
            trad_id = int(row['pitcher_id'])
            if trad_id not in trad_id_map:
                logger.warning(f"No internal pitcher_id found for traditional_id {trad_id}")
                continue
            
            pitcher_id = trad_id_map[trad_id]
            season = row['season']
            
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
                    str(row.get('team', '')),
                    safe_float(row.get('era', 0.0)),
                    safe_float(row.get('k_per_9', 0.0)),
                    safe_float(row.get('bb_per_9', 0.0)),
                    safe_float(row.get('k_bb_ratio', 0.0)),
                    safe_float(row.get('whip', 0.0)),
                    safe_float(row.get('babip', 0.0)),
                    safe_float(row.get('lob_pct', 0.0)),
                    safe_float(row.get('fip', 0.0)),
                    safe_float(row.get('xfip', 0.0)),
                    safe_float(row.get('war', 0.0)),
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
                    pitcher_id, 
                    season,
                    str(row.get('team', '')),
                    safe_float(row.get('era', 0.0)),
                    safe_float(row.get('k_per_9', 0.0)),
                    safe_float(row.get('bb_per_9', 0.0)),
                    safe_float(row.get('k_bb_ratio', 0.0)),
                    safe_float(row.get('whip', 0.0)),
                    safe_float(row.get('babip', 0.0)),
                    safe_float(row.get('lob_pct', 0.0)),
                    safe_float(row.get('fip', 0.0)),
                    safe_float(row.get('xfip', 0.0)),
                    safe_float(row.get('war', 0.0))
                ))
            stats_inserted += 1
            
            # Commit periodically to avoid large transactions
            if stats_inserted % 100 == 0:
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting traditional stats for pitcher {row.get('player_name', 'unknown')}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"Traditional stats stored in database. Inserted/updated {stats_inserted} records.")

def calculate_similarity(name1, name2):
    """
    Calculate string similarity between two names
    
    Args:
        name1 (str): First name
        name2 (str): Second name
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Use SequenceMatcher for similarity
    return SequenceMatcher(None, name1, name2).ratio()

def update_pitcher_mapping():
    """
    Create mapping between Statcast and traditional stats pitcher IDs in the database
    using Chadwick Bureau crosswalk data
    """
    logger.info("Updating pitcher ID mappings...")
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check current mapping status
    cursor.execute("SELECT COUNT(*) FROM pitchers WHERE statcast_id IS NOT NULL")
    total_statcast = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM pitchers WHERE traditional_id IS NOT NULL")
    total_traditional = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM pitchers WHERE statcast_id IS NOT NULL AND traditional_id IS NOT NULL")
    already_mapped = cursor.fetchone()[0]
    
    logger.info(f"Current status: {already_mapped} pitchers mapped out of {total_statcast} statcast and {total_traditional} traditional")
    
    # Get list of statcast IDs that need mapping
    cursor.execute("""
        SELECT statcast_id FROM pitchers 
        WHERE statcast_id IS NOT NULL AND traditional_id IS NULL
    """)
    statcast_ids = [row[0] for row in cursor.fetchall() if row[0] is not None]
    
    # Get list of traditional IDs that need mapping
    cursor.execute("""
        SELECT traditional_id FROM pitchers 
        WHERE traditional_id IS NOT NULL AND statcast_id IS NULL
    """)
    traditional_ids = [row[0] for row in cursor.fetchall() if row[0] is not None]
    
    mappings_found = 0
    
    # First try mapping from statcast to traditional
    if statcast_ids:
        logger.info(f"Attempting to map {len(statcast_ids)} Statcast IDs to FanGraphs IDs")
        
        # Get mappings from Chadwick Bureau data
        mapping_df = get_player_id_map(statcast_ids=statcast_ids)
        
        if not mapping_df.empty:
            # Update database with mappings
            for _, player in mapping_df.iterrows():
                if pd.notna(player['statcast_id']) and pd.notna(player['traditional_id']):
                    try:
                        cursor.execute("""
                            UPDATE pitchers
                            SET traditional_id = ?
                            WHERE statcast_id = ? AND traditional_id IS NULL
                        """, (int(player['traditional_id']), int(player['statcast_id'])))
                        
                        if cursor.rowcount > 0:
                            mappings_found += 1
                    except Exception as e:
                        logger.error(f"Error updating mapping for player {player['player_name']}: {e}")
    
    # Then try mapping from traditional to statcast
    if traditional_ids:
        logger.info(f"Attempting to map {len(traditional_ids)} FanGraphs IDs to Statcast IDs")
        
        # Get mappings from Chadwick Bureau data
        mapping_df = get_player_id_map(fangraphs_ids=traditional_ids)
        
        if not mapping_df.empty:
            # Update database with mappings
            for _, player in mapping_df.iterrows():
                if pd.notna(player['statcast_id']) and pd.notna(player['traditional_id']):
                    try:
                        cursor.execute("""
                            UPDATE pitchers
                            SET statcast_id = ?
                            WHERE traditional_id = ? AND statcast_id IS NULL
                        """, (int(player['statcast_id']), int(player['traditional_id'])))
                        
                        if cursor.rowcount > 0:
                            mappings_found += 1
                    except Exception as e:
                        logger.error(f"Error updating mapping for player {player['player_name']}: {e}")
    
    conn.commit()
    
    # Check final mapping status
    cursor.execute("""
        SELECT COUNT(*) FROM pitchers 
        WHERE statcast_id IS NOT NULL 
        AND traditional_id IS NOT NULL
    """)
    final_mapped_count = cursor.fetchone()[0]
    
    logger.info(f"Mapped {mappings_found} additional pitchers using Chadwick Bureau data")
    logger.info(f"Total: {final_mapped_count}/{total_statcast} pitchers mapped between Statcast and traditional stats")
    
    # For any remaining unmapped pitchers, we can fall back to the name-based matching
    # as a last resort (keeping the existing code here)
    
    # Check for pitchers that have multiple entries (existing code)
    cursor.execute('''
        SELECT GROUP_CONCAT(pitcher_id), normalized_name, COUNT(*) as cnt
        FROM pitchers
        GROUP BY normalized_name
        HAVING cnt > 1
    ''')
    
    duplicate_players = cursor.fetchall()
    if duplicate_players:
        logger.info(f"Found {len(duplicate_players)} players with multiple entries - merging data...")
        
        # Existing code for merging duplicate entries...
    
    conn.close()

def get_pitcher_data():
    """
    Retrieve joined data from the database for feature engineering
    
    Returns:
        pandas.DataFrame: Joined data from game_stats and traditional_stats
    """
    logger.info("Retrieving pitcher data from database...")
    
    conn = get_db_connection()
    
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
        try:
            pitch_mix_pivot = pitch_mix_df.pivot(
                index='game_db_id', 
                columns='pitch_type', 
                values='percentage'
            ).reset_index()
            
            # Rename columns with 'pitch_pct_' prefix
            pitch_mix_pivot.columns = ['game_db_id'] + [f'pitch_pct_{col}' for col in pitch_mix_pivot.columns[1:]]
            
            # Merge with the main dataframe
            df = pd.merge(df, pitch_mix_pivot, on='game_db_id', how='left')
            
            logger.info(f"Added {len(pitch_mix_pivot.columns) - 1} pitch mix columns to the dataset")
        except Exception as e:
            logger.error(f"Error adding pitch mix data: {e}")
    else:
        logger.warning("No pitch mix data found in the database")
    
    logger.info(f"Retrieved {len(df)} rows of pitcher data with {len(df.columns)} columns")
    
    # Convert numeric columns with null values
    numeric_cols = ['era', 'k_per_9', 'bb_per_9', 'k_bb_ratio', 'whip', 'babip', 
                    'lob_pct', 'fip', 'xfip', 'war']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Analyze null values
    null_counts = df.isnull().sum()
    if null_counts.max() > 0:
        logger.info(f"Null value counts in key columns:\n{null_counts[null_counts > 0]}")
    
    return df

def export_dataset_to_csv():
    """
    Export the final dataset (with features) to CSV
    
    Returns:
        pandas.DataFrame: The exported dataset
    """
    logger.info("Exporting final dataset to CSV...")
    
    conn = get_db_connection()
    
    # Join all necessary tables
    query = '''
    SELECT 
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
    
    # Get the list of game_stats_id to game_id mappings
    query_game_ids = '''
    SELECT id, pitcher_id, game_id
    FROM game_stats
    '''
    
    game_id_map = pd.read_sql_query(query_game_ids, conn)
    
    conn.close()
    
    # Add pitch mix columns to the main dataframe
    if not pitch_mix_df.empty:
        # Map game_db_id to (pitcher_id, game_id) pairs
        game_dict = game_id_map.set_index('id').apply(lambda row: (row['pitcher_id'], row['game_id']), axis=1).to_dict()
        
        # Add these keys to the pitch_mix_df
        pitch_mix_df['pitcher_id'] = pitch_mix_df['game_db_id'].map(lambda x: game_dict.get(x, (None, None))[0])
        pitch_mix_df['game_id'] = pitch_mix_df['game_db_id'].map(lambda x: game_dict.get(x, (None, None))[1])
        
        # Create a pivot table for pitch mix
        try:
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
            logger.info(f"Added {len(pitch_cols)} pitch mix columns to the exported dataset")
        except Exception as e:
            logger.error(f"Error adding pitch mix data to export: {e}")
    else:
        logger.warning("No pitch mix data to add to the exported dataset")
    
    # Convert null values in numeric columns to zeros
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Convert strings to proper data types
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    df.to_csv('data/pitcher_game_level_data.csv', index=False)
    
    logger.info(f"Exported {len(df)} rows to data/pitcher_game_level_data.csv")
    return df

def clear_database():
    """Clear all data from the database for fresh start"""
    logger.info("Clearing all data from the database...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete data from all tables
    cursor.execute("DELETE FROM pitch_mix")
    cursor.execute("DELETE FROM prediction_features")
    cursor.execute("DELETE FROM game_stats")
    cursor.execute("DELETE FROM traditional_stats")
    cursor.execute("DELETE FROM pitchers")
    
    # Reset auto-increment counters
    cursor.execute("DELETE FROM sqlite_sequence")
    
    conn.commit()
    conn.close()
    
    logger.info("Database cleared successfully")

def examine_data_structure(statcast_data):
    """
    Examine the structure of the raw data
    
    Args:
        statcast_data (pandas.DataFrame): Raw statcast data
    """
    logger.info("Examining raw statcast data structure...")
    
    # Check data shape
    logger.info(f"Data shape: {statcast_data.shape}")
    
    # Print available columns
    logger.info(f"Available columns: {', '.join(statcast_data.columns[:20])}...")
    
    # Check for pitch type information
    if 'pitch_type' in statcast_data.columns:
        pitch_types = statcast_data['pitch_type'].dropna().unique()
        logger.info(f"Pitch types: {', '.join(pitch_types[:20])}...")
        
        # Count of each pitch type
        pitch_counts = statcast_data['pitch_type'].value_counts()
        logger.info(f"Top 5 pitch types by frequency:")
        for pitch, count in pitch_counts.head(5).items():
            logger.info(f"  {pitch}: {count} ({count/len(statcast_data)*100:.1f}%)")
    else:
        logger.warning("No 'pitch_type' column found in the data")
    
    # Check for game events
    if 'events' in statcast_data.columns:
        events = statcast_data['events'].dropna().unique()
        logger.info(f"Event types: {', '.join(events[:20])}...")
    else:
        logger.warning("No 'events' column found in the data")
    
    # Check for key metrics
    key_metrics = ['release_speed', 'release_spin_rate', 'zone']
    for metric in key_metrics:
        if metric in statcast_data.columns:
            logger.info(f"{metric} stats: min={statcast_data[metric].min()}, max={statcast_data[metric].max()}, mean={statcast_data[metric].mean()}")
        else:
            logger.warning(f"No '{metric}' column found in the data")
    
    # Check player name formats
    if 'player_name' in statcast_data.columns:
        sample_names = statcast_data['player_name'].dropna().unique()[:10]
        logger.info(f"Sample player names: {', '.join(sample_names)}")
    else:
        logger.warning("No 'player_name' column found in the data")
    
    # Check for nulls in key columns
    null_pcts = statcast_data.isnull().mean() * 100
    high_null_cols = null_pcts[null_pcts > 50].sort_values(ascending=False)
    if not high_null_cols.empty:
        logger.warning(f"Columns with >50% null values:")
        for col, pct in high_null_cols.items():
            logger.warning(f"  {col}: {pct:.1f}% null")
    
    logger.info("Data structure examination complete")

def troubleshoot_database():
    """
    Run diagnostics on the database
    """
    logger.info("Running database diagnostics...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check all tables and their row counts
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        logger.info(f"Table {table}: {count} rows")
    
    # Check for schema issues
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        logger.info(f"Table {table} schema: {len(columns)} columns")
        
        # Check for potential foreign key issues
        if table in ['game_stats', 'traditional_stats', 'prediction_features']:
            cursor.execute(f"SELECT COUNT(*) FROM {table} t LEFT JOIN pitchers p ON t.pitcher_id = p.pitcher_id WHERE p.pitcher_id IS NULL")
            orphans = cursor.fetchone()[0]
            if orphans > 0:
                logger.warning(f"Found {orphans} rows in {table} with no matching pitcher record")
        
        if table == 'pitch_mix':
            cursor.execute(f"SELECT COUNT(*) FROM {table} pm LEFT JOIN game_stats gs ON pm.game_stats_id = gs.id WHERE gs.id IS NULL")
            orphans = cursor.fetchone()[0]
            if orphans > 0:
                logger.warning(f"Found {orphans} rows in {table} with no matching game_stats record")
    
    # Check for mapping issues
    cursor.execute("""
        SELECT 
            COUNT(*) as total_pitchers,
            SUM(CASE WHEN statcast_id IS NOT NULL THEN 1 ELSE 0 END) as with_statcast,
            SUM(CASE WHEN traditional_id IS NOT NULL THEN 1 ELSE 0 END) as with_traditional,
            SUM(CASE WHEN statcast_id IS NOT NULL AND traditional_id IS NOT NULL THEN 1 ELSE 0 END) as fully_mapped
        FROM pitchers
    """)
    
    mapping_stats = cursor.fetchone()
    logger.info(f"Pitcher mapping: {mapping_stats[3]}/{mapping_stats[0]} fully mapped, {mapping_stats[1]} with statcast ID, {mapping_stats[2]} with traditional ID")
    
    if mapping_stats[3] < min(mapping_stats[1], mapping_stats[2]) * 0.75:
        logger.warning("Low mapping rate - fewer than 75% of potential matches were established")
    
    # Check for traditional stats coverage
    cursor.execute("""
        SELECT 
            g.season,
            COUNT(DISTINCT g.pitcher_id) as game_pitchers,
            COUNT(DISTINCT CASE WHEN t.pitcher_id IS NOT NULL THEN g.pitcher_id ELSE NULL END) as trad_pitchers,
            COUNT(DISTINCT gs.pitcher_id) as game_stats_pitchers
        FROM 
            game_stats g
        LEFT JOIN 
            traditional_stats t ON g.pitcher_id = t.pitcher_id AND g.season = t.season
        LEFT JOIN
            game_stats gs ON g.pitcher_id = gs.pitcher_id
        GROUP BY 
            g.season
        ORDER BY 
            g.season
    """)
    
    season_stats = cursor.fetchall()
    logger.info("Seasonal coverage:")
    for season, game_pitchers, trad_pitchers, game_stats_pitchers in season_stats:
        logger.info(f"  Season {season}: {trad_pitchers}/{game_pitchers} pitchers have traditional stats ({trad_pitchers/game_pitchers*100 if game_pitchers else 0:.1f}%)")
    
    # Check for potential data type issues
    cursor.execute("""
        SELECT 
            COUNT(*) as total_trad_stats,
            SUM(CASE WHEN era = 0 THEN 1 ELSE 0 END) as zero_era,
            SUM(CASE WHEN k_per_9 = 0 THEN 1 ELSE 0 END) as zero_k9,
            SUM(CASE WHEN fip = 0 THEN 1 ELSE 0 END) as zero_fip
        FROM traditional_stats
    """)
    
    trad_stats = cursor.fetchone()
    if trad_stats[0] > 0:
        logger.info(f"Traditional stats zero values: ERA: {trad_stats[1]}/{trad_stats[0]} ({trad_stats[1]/trad_stats[0]*100:.1f}%), K/9: {trad_stats[2]}/{trad_stats[0]} ({trad_stats[2]/trad_stats[0]*100:.1f}%), FIP: {trad_stats[3]}/{trad_stats[0]} ({trad_stats[3]/trad_stats[0]*100:.1f}%)")
        
        if trad_stats[1]/trad_stats[0] > 0.5 or trad_stats[2]/trad_stats[0] > 0.5 or trad_stats[3]/trad_stats[0] > 0.5:
            logger.warning("High percentage of zero values in traditional stats - possible data type or conversion issue")
    
    # Check for pitch mix data
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT game_stats_id) as games_with_mix,
            COUNT(*) as total_pitch_records,
            COUNT(DISTINCT pitch_type) as unique_pitch_types
        FROM pitch_mix
    """)
    
    pitch_stats = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM game_stats")
    total_games = cursor.fetchone()[0]
    
    logger.info(f"Pitch mix: {pitch_stats[0]}/{total_games} games have pitch mix data ({pitch_stats[0]/total_games*100 if total_games else 0:.1f}%)")
    logger.info(f"  {pitch_stats[1]} total pitch records with {pitch_stats[2]} unique pitch types")
    
    if pitch_stats[0] == 0 and total_games > 0:
        logger.error("No pitch mix data found despite having game records - see store_statcast_data function for issues")
    
    conn.close()
    logger.info("Database diagnostics complete")

def store_processed_data(processed_df, force_refresh=False):
    """
    Store processed data in the database
    
    Args:
        processed_df (pd.DataFrame): Processed pitcher data
        force_refresh (bool): Whether to force refresh existing data
    """
    # Check if we need to refresh the data
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if processed_data table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='processed_data'
    """)
    
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create the table if it doesn't exist
        logger.info("Creating processed_data table...")
        cursor.execute("""
            CREATE TABLE processed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitcher_id INTEGER,
                statcast_id INTEGER,
                traditional_id INTEGER,
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
                era REAL,
                fip REAL,
                xfip REAL,
                k_per_9 REAL,
                bb_per_9 REAL,
                whip REAL,
                babip REAL,
                lob_pct REAL,
                war REAL,
                team TEXT,
                FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id)
            )
        """)
        conn.commit()
    elif force_refresh:
        # Clear existing data if force_refresh is True
        logger.info("Clearing existing processed data...")
        cursor.execute("DELETE FROM processed_data")
        conn.commit()
    else:
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM processed_data")
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info(f"Processed data table already contains {count} records and force_refresh is False. Skipping.")
            conn.close()
            return
    
    # Get pitcher mappings
    cursor.execute("""
        SELECT pitcher_id, statcast_id, traditional_id
        FROM pitchers
        WHERE statcast_id IS NOT NULL
    """)
    pitcher_map = {row[1]: (row[0], row[2]) for row in cursor.fetchall()}
    
    # Prepare for insertion
    processed_count = 0
    
    # Define which columns to store in the database
    db_columns = [
        'pitcher_id', 'statcast_id', 'traditional_id', 'game_id', 'game_date', 
        'season', 'strikeouts', 'hits', 'walks', 'home_runs', 
        'release_speed_mean', 'release_speed_max', 'release_spin_rate_mean',
        'swinging_strike_pct', 'called_strike_pct', 'zone_rate',
        'era', 'fip', 'xfip', 'k_per_9', 'bb_per_9',
        'whip', 'babip', 'lob_pct', 'war', 'team'
    ]
    
    # Define a function to safely convert values
    def safe_value(val, default=None):
        """Convert value safely handling NA values"""
        if pd.isna(val):
            return default
        return val
    
    # Insert data
    for _, row in processed_df.iterrows():
        try:
            # Get pitcher IDs
            statcast_id = safe_value(row.get('pitcher'))
            if statcast_id and int(statcast_id) in pitcher_map:
                pitcher_id, traditional_id = pitcher_map[int(statcast_id)]
            else:
                # Skip records without mapping
                continue
            
            # Prepare values for insertion
            values = []
            for col in db_columns:
                if col == 'pitcher_id':
                    values.append(pitcher_id)
                elif col == 'statcast_id':
                    values.append(statcast_id)
                elif col == 'traditional_id':
                    values.append(traditional_id)
                elif col in ['strikeouts', 'hits', 'walks', 'home_runs', 'season']:
                    values.append(int(safe_value(row.get(col), 0)))
                elif col in ['game_id', 'game_date', 'team']:
                    values.append(str(safe_value(row.get(col), '')))
                else:
                    values.append(float(safe_value(row.get(col), 0.0)))
            
            # Create placeholders for SQL query
            placeholders = ', '.join(['?'] * len(db_columns))
            columns = ', '.join(db_columns)
            
            # Insert into database
            cursor.execute(f"""
                INSERT INTO processed_data ({columns})
                VALUES ({placeholders})
            """, values)
            
            processed_count += 1
            
            # Commit every 1000 records
            if processed_count % 1000 == 0:
                logger.info(f"Processed {processed_count} records...")
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            continue
    
    # Final commit
    conn.commit()
    logger.info(f"Stored {processed_count} processed records in the database")
    
    conn.close()