# Database utilities for the MLB pitcher prediction project
import sqlite3
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Database path
DB_PATH = "data/pitcher_stats.db"

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
    
    logger.info("Database initialization complete.")

def store_statcast_data(statcast_df, force_refresh=False):
    """
    Process and store statcast data in SQLite database
    
    Args:
        statcast_df (pandas.DataFrame): Raw statcast data
        force_refresh (bool): Whether to force refresh existing data
    """
    from src.data.process import aggregate_statcast_to_game_level
    
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
    
    # Process statcast data to game level
    game_level = aggregate_statcast_to_game_level(statcast_df)
    
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
                cursor.execute(
                    "INSERT OR IGNORE INTO pitchers (statcast_id, player_name) VALUES (?, ?)",
                    (pitcher_id, row['player_name'])
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
    pitcher_map = {statcast_id: pitcher_id for pitcher_id, statcast_id in cursor.fetchall()}
    
    # Insert game stats
    game_stats_ids = {}  # Map to store game_id -> database_id for pitch mix data
    
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
                    row.get('called_strike_pct', 0), row.get('zone_rate', 0)
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
        except Exception as e:
            logger.error(f"Error processing game record: {e}")
            continue
    
    conn.commit()
    
    # Verify pitch mix data was stored
    cursor.execute("SELECT COUNT(*) FROM pitch_mix")
    pitch_mix_count = cursor.fetchone()[0]
    logger.info(f"Stored {pitch_mix_count} pitch mix records.")
    
    conn.close()
    
    logger.info("Statcast data stored in database.")

def store_traditional_stats(trad_df, force_refresh=False):
    """
    Process and store traditional stats in SQLite database
    
    Args:
        trad_df (pandas.DataFrame): Traditional stats data
        force_refresh (bool): Whether to force refresh existing data
    """
    from src.data.process import process_traditional_stats
    
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
    required_cols = ['pitcher_id', 'Name', 'Season']
    if not all(col in processed_trad.columns for col in required_cols):
        logger.error(f"Missing required columns in traditional stats. Available: {processed_trad.columns.tolist()}")
        conn.close()
        return
    
    # Add pitchers to the pitchers table if they don't exist
    for _, row in processed_trad[['pitcher_id', 'Name']].drop_duplicates().iterrows():
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO pitchers (traditional_id, player_name) VALUES (?, ?)",
                (int(row['pitcher_id']), row['Name'])
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error inserting traditional pitcher {row['Name']}: {e}")
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
            stats_inserted += 1
        except Exception as e:
            logger.error(f"Error inserting traditional stats for pitcher {row.get('Name', 'unknown')}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"Traditional stats stored in database. Inserted/updated {stats_inserted} records.")

def update_pitcher_mapping():
    """
    Create mapping between Statcast and traditional stats pitcher IDs in the database
    """
    from src.data.process import normalize_name
    
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
    
    # First, try to map based on exact name matches
    cursor.execute('''
        UPDATE pitchers AS p1
        SET traditional_id = (
            SELECT p2.traditional_id FROM pitchers AS p2
            WHERE LOWER(p2.player_name) = LOWER(p1.player_name)
            AND p2.traditional_id IS NOT NULL
            AND p1.pitcher_id != p2.pitcher_id
            LIMIT 1
        )
        WHERE p1.statcast_id IS NOT NULL
        AND p1.traditional_id IS NULL
    ''')
    
    # Use the opposite mapping as well
    cursor.execute('''
        UPDATE pitchers AS p1
        SET statcast_id = (
            SELECT p2.statcast_id FROM pitchers AS p2
            WHERE LOWER(p2.player_name) = LOWER(p1.player_name)
            AND p2.statcast_id IS NOT NULL
            AND p1.pitcher_id != p2.pitcher_id
            LIMIT 1
        )
        WHERE p1.traditional_id IS NOT NULL
        AND p1.statcast_id IS NULL
    ''')
    
    # Check how many pitchers were mapped with exact matches
    cursor.execute('''
        SELECT COUNT(*) FROM pitchers 
        WHERE statcast_id IS NOT NULL 
        AND traditional_id IS NOT NULL
    ''')
    exact_mapped_count = cursor.fetchone()[0]
    
    logger.info(f"Mapped {exact_mapped_count - already_mapped} additional pitchers with exact name matches.")
    
    # Get unmatched pitchers with statcast_id
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
    
    # Create dictionaries for better name matching
    trad_names = {}
    for pid, name, trad_id in trad_pitchers:
        norm_name = normalize_name(name)
        if norm_name not in trad_names:
            trad_names[norm_name] = []
        trad_names[norm_name].append((pid, trad_id))
        
        # Also index by last name for fallback matching
        last_name = norm_name.split()[-1] if norm_name and ' ' in norm_name else norm_name
        if last_name not in trad_names:
            trad_names[last_name] = []
        trad_names[last_name].append((pid, trad_id))
    
    # Match with normalized names
    matches_count = 0
    for pid, name, statcast_id in unmatched_statcast:
        norm_name = normalize_name(name)
        
        # Try full normalized name
        if norm_name in trad_names and len(trad_names[norm_name]) == 1:
            trad_pid, trad_id = trad_names[norm_name][0]
            cursor.execute(
                "UPDATE pitchers SET traditional_id = ? WHERE pitcher_id = ?",
                (trad_id, pid)
            )
            matches_count += 1
            continue
        
        # If full name didn't match uniquely, try last name as fallback
        last_name = norm_name.split()[-1] if norm_name and ' ' in norm_name else norm_name
        if last_name in trad_names and len(trad_names[last_name]) == 1:
            trad_pid, trad_id = trad_names[last_name][0]
            cursor.execute(
                "UPDATE pitchers SET traditional_id = ? WHERE pitcher_id = ?",
                (trad_id, pid)
            )
            matches_count += 1
    
    conn.commit()
    
    # Check final mapping status
    cursor.execute('''
        SELECT COUNT(*) FROM pitchers 
        WHERE statcast_id IS NOT NULL 
        AND traditional_id IS NOT NULL
    ''')
    final_mapped_count = cursor.fetchone()[0]
    
    logger.info(f"Mapped {final_mapped_count - exact_mapped_count} additional pitchers with enhanced name matching.")
    logger.info(f"Total: {final_mapped_count}/{total_statcast} pitchers mapped between Statcast and traditional stats.")
    
    # Check for pitchers that have multiple entries
    cursor.execute('''
        SELECT GROUP_CONCAT(pitcher_id), player_name, COUNT(*) as cnt
        FROM pitchers
        GROUP BY LOWER(player_name)
        HAVING cnt > 1
    ''')
    
    duplicate_players = cursor.fetchall()
    if duplicate_players:
        logger.info(f"Found {len(duplicate_players)} players with multiple entries - merging data...")
        
        for pid_group, name, count in duplicate_players:
            pids = pid_group.split(',')
            
            # Find the "best" record - one with both IDs if possible
            cursor.execute('''
                SELECT pitcher_id, statcast_id, traditional_id
                FROM pitchers
                WHERE LOWER(player_name) = LOWER(?)
                ORDER BY (statcast_id IS NOT NULL) + (traditional_id IS NOT NULL) DESC
            ''', (name,))
            
            records = cursor.fetchall()
            if not records:
                continue
                
            main_pid, main_statcast, main_trad = records[0]
            
            # Collect all IDs from duplicate records
            all_statcast = [r[1] for r in records if r[1] is not None]
            all_trad = [r[2] for r in records if r[2] is not None]
            
            # Update main record with any missing IDs
            if main_statcast is None and all_statcast:
                cursor.execute("UPDATE pitchers SET statcast_id = ? WHERE pitcher_id = ?", 
                             (all_statcast[0], main_pid))
                
            if main_trad is None and all_trad:
                cursor.execute("UPDATE pitchers SET traditional_id = ? WHERE pitcher_id = ?", 
                             (all_trad[0], main_pid))
            
            # Update foreign keys in other tables to point to main record
            for other_pid in pids:
                if other_pid == str(main_pid):
                    continue
                    
                # Update game_stats
                cursor.execute("UPDATE game_stats SET pitcher_id = ? WHERE pitcher_id = ?", 
                             (main_pid, other_pid))
                
                # Update traditional_stats  
                cursor.execute("UPDATE traditional_stats SET pitcher_id = ? WHERE pitcher_id = ?", 
                             (main_pid, other_pid))
                
                # Update prediction_features
                cursor.execute("UPDATE prediction_features SET pitcher_id = ? WHERE pitcher_id = ?", 
                             (main_pid, other_pid))
                
            # Delete the duplicate records
            for other_pid in pids:
                if other_pid == str(main_pid):
                    continue
                cursor.execute("DELETE FROM pitchers WHERE pitcher_id = ?", (other_pid,))
    
    conn.commit()
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
        pitch_mix_pivot = pitch_mix_df.pivot(
            index='game_db_id', 
            columns='pitch_type', 
            values='percentage'
        ).reset_index()
        
        # Rename columns with 'pitch_pct_' prefix
        pitch_mix_pivot.columns = ['game_db_id'] + [f'pitch_pct_{col}' for col in pitch_mix_pivot.columns[1:]]
        
        # Merge with the main dataframe
        df = pd.merge(df, pitch_mix_pivot, on='game_db_id', how='left')
    
    logger.info(f"Retrieved {len(df)} rows of pitcher data.")
    # Log data stats
    null_counts = df.isnull().sum()
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
    
    logger.info(f"Exported {len(df)} rows to data/pitcher_game_level_data.csv")
    return df