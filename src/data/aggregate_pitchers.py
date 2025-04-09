"""
Module for aggregating pitch-level data to game-level for pitchers.
"""

import sqlite3
import logging
import pandas as pd
from src.data.utils import setup_logger, DBConnection, ensure_dir
from config import DBConfig

logger = setup_logger(__name__)

def aggregate_pitchers_to_game_level(force_rebuild=False):
    """
    Aggregate pitch-level data to game-level for pitchers and store in game_level_pitchers table.
    
    Args:
        force_rebuild (bool): If True, drop existing table and rebuild
    
    Returns:
        bool: Success status
    """
    try:
        # Check if table already exists
        with DBConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_level_pitchers'")
            table_exists = cursor.fetchone() is not None
        
        # If table exists and force_rebuild is False, return success
        if table_exists and not force_rebuild:
            logger.info("game_level_pitchers table already exists, use force_rebuild=True to recreate")
            return True
        
        # SQL script to aggregate data
        sql_script = ""
        
        # Drop table if it exists and force_rebuild is True
        if table_exists and force_rebuild:
            sql_script += '''
            DROP TABLE IF EXISTS game_level_pitchers;
            '''
            logger.info("Dropping existing game_level_pitchers table")
        
        # Create new table
        sql_script += '''
        -- Create game_level_pitchers table from statcast_pitchers data
        CREATE TABLE IF NOT EXISTS game_level_pitchers AS
        SELECT
            pitcher_id,
            player_name,
            game_date,
            game_pk,
            home_team,
            away_team,
            p_throws,
            season,
            -- Strikeout metrics
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts,
            COUNT(DISTINCT at_bat_number) AS batters_faced,
            -- Pitch metrics
            COUNT(*) AS total_pitches,
            AVG(release_speed) AS avg_velocity,
            MAX(release_speed) AS max_velocity,
            AVG(release_spin_rate) AS avg_spin_rate,
            -- Movement metrics
            AVG(pfx_x) AS avg_horizontal_break,
            AVG(pfx_z) AS avg_vertical_break,
            -- Zone metrics
            AVG(CASE WHEN zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS zone_percent,
            SUM(CASE WHEN description = 'swinging_strike' OR description = 'swinging_strike_blocked' THEN 1 ELSE 0 END) / 
                CAST(COUNT(*) AS REAL) AS swinging_strike_percent,
            -- Innings estimation (based on outs recorded)
            SUM(CASE 
                WHEN events = 'field_out' OR events = 'strikeout' OR events = 'grounded_into_double_play' 
                     OR events = 'force_out' OR events = 'sac_fly' OR events = 'sac_bunt' OR events = 'double_play' 
                THEN 1 ELSE 0 END) / 3.0 AS innings_pitched
        FROM 
            statcast_pitchers
        GROUP BY 
            pitcher_id, game_date, game_pk, player_name, home_team, away_team, p_throws, season;

        -- Add additional derived metrics
        ALTER TABLE game_level_pitchers ADD COLUMN k_per_9 REAL;
        ALTER TABLE game_level_pitchers ADD COLUMN k_percent REAL;

        UPDATE game_level_pitchers
        SET 
            k_per_9 = CASE WHEN innings_pitched > 0 THEN (strikeouts * 9.0 / innings_pitched) ELSE 0 END,
            k_percent = CASE WHEN batters_faced > 0 THEN (strikeouts * 1.0 / batters_faced) ELSE 0 END;

        -- Create an index for better performance
        CREATE INDEX IF NOT EXISTS idx_game_level_pitchers_pitcher_game 
        ON game_level_pitchers (pitcher_id, game_date, game_pk);

        -- Add pitch mix percentages
        -- First, add a temporary table to calculate pitch type counts
        CREATE TEMPORARY TABLE pitcher_pitch_counts AS
        SELECT
            pitcher_id,
            game_date,
            game_pk,
            pitch_type,
            COUNT(*) as pitch_count
        FROM
            statcast_pitchers
        GROUP BY
            pitcher_id, game_date, game_pk, pitch_type;

        -- Then add columns for major pitch types
        ALTER TABLE game_level_pitchers ADD COLUMN fastball_percent REAL DEFAULT 0;
        ALTER TABLE game_level_pitchers ADD COLUMN breaking_percent REAL DEFAULT 0;
        ALTER TABLE game_level_pitchers ADD COLUMN offspeed_percent REAL DEFAULT 0;

        -- Update with fastball percentages (FF: 4-seam, FT: 2-seam, FC: cutter, SI: sinker)
        UPDATE game_level_pitchers
        SET fastball_percent = (
            SELECT SUM(pitch_count) * 1.0 / game_level_pitchers.total_pitches
            FROM pitcher_pitch_counts
            WHERE pitcher_pitch_counts.pitcher_id = game_level_pitchers.pitcher_id
            AND pitcher_pitch_counts.game_date = game_level_pitchers.game_date
            AND pitcher_pitch_counts.game_pk = game_level_pitchers.game_pk
            AND pitcher_pitch_counts.pitch_type IN ('FF', 'FT', 'FC', 'SI')
        );

        -- Update with breaking ball percentages (SL: slider, CU: curveball, KC: knuckle curve)
        UPDATE game_level_pitchers
        SET breaking_percent = (
            SELECT SUM(pitch_count) * 1.0 / game_level_pitchers.total_pitches
            FROM pitcher_pitch_counts
            WHERE pitcher_pitch_counts.pitcher_id = game_level_pitchers.pitcher_id
            AND pitcher_pitch_counts.game_date = game_level_pitchers.game_date
            AND pitcher_pitch_counts.game_pk = game_level_pitchers.game_pk
            AND pitcher_pitch_counts.pitch_type IN ('SL', 'CU', 'KC', 'KN')
        );

        -- Update with offspeed percentages (CH: changeup, FS: splitter, FO: forkball)
        UPDATE game_level_pitchers
        SET offspeed_percent = (
            SELECT SUM(pitch_count) * 1.0 / game_level_pitchers.total_pitches
            FROM pitcher_pitch_counts
            WHERE pitcher_pitch_counts.pitcher_id = game_level_pitchers.pitcher_id
            AND pitcher_pitch_counts.game_date = game_level_pitchers.game_date
            AND pitcher_pitch_counts.game_pk = game_level_pitchers.game_pk
            AND pitcher_pitch_counts.pitch_type IN ('CH', 'FS', 'FO', 'SC')
        );

        -- Drop the temporary table
        DROP TABLE pitcher_pitch_counts;
        '''

        # Execute the SQL script using DBConnection
        with DBConnection() as conn:
            conn.executescript(sql_script)
            
        logger.info("Successfully aggregated pitcher data to game level")
        return True
        
    except Exception as e:
        logger.error(f"Error aggregating pitcher data: {e}")
        return False

def get_game_level_pitcher_stats(limit=None, pitcher_id=None, seasons=None):
    """
    Retrieve game-level pitcher stats from the database.
    
    Args:
        limit (int): Optional limit on number of rows
        pitcher_id (int): Optional filter for specific pitcher
        seasons (list): Optional list of seasons to include
        
    Returns:
        pandas.DataFrame: Game-level pitcher statistics
    """
    try:
        with DBConnection() as conn:
            # Build query
            query = "SELECT * FROM game_level_pitchers"
            conditions = []
            
            if pitcher_id:
                conditions.append(f"pitcher_id = {pitcher_id}")
            
            if seasons:
                season_list = ', '.join(str(s) for s in seasons)
                conditions.append(f"season IN ({season_list})")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            if limit:
                query += f" LIMIT {limit}"
                
            # Execute query
            df = pd.read_sql_query(query, conn)
            
            logger.info(f"Retrieved {len(df)} game-level pitcher records")
            return df
            
    except Exception as e:
        logger.error(f"Error retrieving game-level pitcher data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # This can be called directly for testing
    aggregate_pitchers_to_game_level()