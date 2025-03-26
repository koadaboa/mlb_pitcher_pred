# Functions for processing data after fetching from external sources
import pandas as pd
import numpy as np
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

def aggregate_to_game_level(statcast_df):
    """
    Aggregate statcast pitch-level data to pitcher-game level
    """
    import numpy as np
    import pandas as pd
    
    logger.info("Aggregating statcast data to pitcher-game level...")
    
    # Basic validation
    if statcast_df.empty:
        logger.warning("Empty dataframe provided for aggregation.")
        return pd.DataFrame()
    
    # Ensure required columns exist
    required_cols = ['game_date', 'pitcher']
    missing = [col for col in required_cols if col not in statcast_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return pd.DataFrame()
    
    # Convert game_date to datetime
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'], errors='coerce')
    
    # Create game_id column if it doesn't exist
    if 'game_id' not in statcast_df.columns:
        if 'game_pk' in statcast_df.columns:
            statcast_df['game_id'] = statcast_df['game_pk'].astype(str)
        else:
            # Create synthetic game ID from date and home team
            logger.warning("Creating synthetic game IDs")
            if 'home_team' in statcast_df.columns:
                statcast_df['game_id'] = statcast_df['game_date'].dt.strftime('%Y%m%d') + '_' + statcast_df['home_team']
            else:
                # Last resort - create IDs based on date only
                statcast_df['game_id'] = statcast_df['game_date'].dt.strftime('%Y%m%d') + '_' + \
                                        statcast_df.groupby(['game_date']).ngroup().astype(str)
    
    # Define grouping columns
    group_cols = ['pitcher', 'game_id', 'game_date']
    if 'player_name' in statcast_df.columns:
        group_cols.append('player_name')
    if 'season' in statcast_df.columns:
        group_cols.append('season')
    elif 'game_date' in statcast_df.columns:
        # Add season column based on year
        statcast_df['season'] = statcast_df['game_date'].dt.year
        group_cols.append('season')
        
    # Log grouping info
    logger.info(f"Grouping by columns: {group_cols}")
    
    # Group by pitcher and game
    grouped = statcast_df.groupby(group_cols)
    
    # Count pitches per game
    pitch_counts = grouped.size().reset_index(name='pitch_count')
    game_level = pitch_counts.copy()
    
    # Try to aggregate metrics
    try:
        # Pitch velocity and spin rate
        metric_cols = ['release_speed', 'release_spin_rate']
        for col in metric_cols:
            if col in statcast_df.columns:
                means = grouped[col].mean().reset_index(name=f"{col}_mean")
                maxes = grouped[col].max().reset_index(name=f"{col}_max")
                game_level = pd.merge(game_level, means, on=group_cols, how='left')
                game_level = pd.merge(game_level, maxes, on=group_cols, how='left')
        
        # Zone rate
        if 'zone' in statcast_df.columns:
            zone_rate = grouped['zone'].apply(lambda x: (x > 0).mean()).reset_index(name='zone_rate')
            game_level = pd.merge(game_level, zone_rate, on=group_cols, how='left')
        
        # Process events (strikeouts, walks, etc.)
        if 'events' in statcast_df.columns:
            # Filter to rows with events
            events_df = statcast_df[statcast_df['events'].notna()]
            
            if not events_df.empty:
                # Track key events
                for event, new_name in [('strikeout', 'strikeouts'), ('walk', 'walks'), 
                                       ('home_run', 'home_runs')]:
                    # Count occurrences
                    event_counts = events_df[events_df['events'] == event].groupby(group_cols).size().reset_index(name=new_name)
                    if not event_counts.empty:
                        game_level = pd.merge(game_level, event_counts, on=group_cols, how='left')
                
                # Hits (combination of single, double, triple, home_run)
                hit_events = ['single', 'double', 'triple', 'home_run']
                hits_df = events_df[events_df['events'].isin(hit_events)].groupby(group_cols).size().reset_index(name='hits')
                if not hits_df.empty:
                    game_level = pd.merge(game_level, hits_df, on=group_cols, how='left')
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
    
    # Fill NAs with zeros
    for col in game_level.columns:
        if col not in group_cols and pd.api.types.is_numeric_dtype(game_level[col]):
            game_level[col] = game_level[col].fillna(0)
    
    # Check for columns with all zeros (only numeric columns)
    numeric_cols = game_level.select_dtypes(include=[np.number]).columns
    zero_cols = [col for col in numeric_cols if game_level[col].sum() == 0]
    if zero_cols:
        logger.warning(f"The following columns contain all zeros: {zero_cols}")
    
    logger.info(f"Aggregated data to {len(game_level)} pitcher-game records with {len(game_level.columns)} columns")
    return game_level

def process_traditional_stats(trad_df):
    """
    Process traditional pitching stats to prepare for merging
    
    Args:
        trad_df (pandas.DataFrame): Traditional pitching stats
        
    Returns:
        pandas.DataFrame: Processed traditional pitching stats
    """
    logger.info("Processing traditional pitching stats...")
    logger.info(f"Input shape: {trad_df.shape} with columns: {', '.join(trad_df.columns[:10])}...")
    
    # Common column name variations to handle different data sources
    column_mappings = {
        'Season': ['season', 'SEASON', 'year', 'YEAR'],
        'Name': ['name', 'NAME', 'player_name', 'PLAYER_NAME', 'PlayerName', 'full_name'],
        'IDfg': ['playerid', 'PLAYERID', 'player_id', 'mlbam_id', 'mlbam', 'MLBAM_ID', 'FG_ID', 'fg_id']
    }
    
    # Try to standardize column names
    for standard_name, alternatives in column_mappings.items():
        if standard_name not in trad_df.columns:
            for alt in alternatives:
                if alt in trad_df.columns:
                    trad_df[standard_name] = trad_df[alt]
                    logger.info(f"Mapped column {alt} to {standard_name}")
                    break
    
    # Check if required columns exist now
    required_cols = ['Season', 'Name']
    missing_cols = [col for col in required_cols if col not in trad_df.columns]
    
    if missing_cols:
        logger.error(f"Still missing required columns after mapping: {missing_cols}")
        return pd.DataFrame()
    
    # We also need a player ID column - check if we have one
    id_col = None
    for col in ['IDfg', 'playerid', 'PLAYERID', 'player_id', 'mlbam_id', 'mlbam']:
        if col in trad_df.columns:
            id_col = col
            break
    
    if id_col is None:
        logger.error("No player ID column found in traditional stats")
        return pd.DataFrame()
    
    # Create a pitcher_id column from the ID column
    trad_df['pitcher_id'] = trad_df[id_col]
    
    # Process columns for easier merging
    if 'Name' in trad_df.columns:
        trad_df['Name'] = trad_df['Name'].astype(str).str.strip()
    
    # Map column names to our standard format
    stat_mappings = {
        'ERA': ['era', 'ERA', 'earned_run_avg'],
        'K/9': ['k_per_9', 'so9', 'so/9', 'k9', 'strikeouts_per_9'],
        'BB/9': ['bb_per_9', 'bb9', 'bb/9', 'walks_per_9'],
        'K/BB': ['k_bb_ratio', 'so_bb', 'so/bb', 'k/bb', 'strikeout_to_walk'],
        'WHIP': ['whip', 'WHIP', 'walks_hits_per_ip'],
        'BABIP': ['babip', 'BABIP', 'batting_avg_on_balls_in_play'],
        'LOB%': ['lob_pct', 'lob%', 'LOB%', 'left_on_base_pct'],
        'FIP': ['fip', 'FIP', 'fielding_independent_pitching'],
        'xFIP': ['xfip', 'xFIP', 'expected_fielding_independent_pitching'],
        'WAR': ['war', 'WAR', 'wins_above_replacement'],
        'Team': ['team', 'TEAM', 'team_name', 'TEAM_NAME', 'organization']
    }
    
    # Standardize stat column names
    processed_df = trad_df.copy()
    for standard_name, alternatives in stat_mappings.items():
        if standard_name not in processed_df.columns:
            for alt in alternatives:
                if alt in processed_df.columns:
                    processed_df[standard_name] = processed_df[alt]
                    logger.info(f"Mapped stat column {alt} to {standard_name}")
                    break
    
    # Ensure the required columns exist
    required_stats = ['ERA', 'Team']
    for stat in required_stats:
        if stat not in processed_df.columns:
            logger.warning(f"Required stat {stat} not found in traditional stats")
            processed_df[stat] = np.nan
    
    # Convert percentage strings to floats if needed
    pct_cols = ['LOB%', 'BABIP']
    for col in pct_cols:
        if col in processed_df.columns and processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].astype(str).str.rstrip('%').astype('float') / 100
    
    # Ensure numeric data types
    numeric_cols = ['ERA', 'K/9', 'BB/9', 'K/BB', 'WHIP', 'BABIP', 'LOB%', 'FIP', 'xFIP', 'WAR']
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Standardize column names to lowercase for database consistency
    column_standardization = {
        'Name': 'player_name',
        'Season': 'season',
        'Team': 'team',
        'ERA': 'era', 
        'K/9': 'k_per_9', 
        'BB/9': 'bb_per_9',
        'K/BB': 'k_bb_ratio', 
        'WHIP': 'whip',
        'BABIP': 'babip',
        'LOB%': 'lob_pct',
        'FIP': 'fip',
        'xFIP': 'xfip',
        'WAR': 'war'
    }
    
    # Rename columns
    for old_name, new_name in column_standardization.items():
        if old_name in processed_df.columns:
            processed_df.rename(columns={old_name: new_name}, inplace=True)
    
    logger.info(f"Processed {len(processed_df)} traditional stat records with {len(processed_df.columns)} columns")
    return processed_df

def normalize_name(name):
    """
    Normalize player names for better matching
    
    Args:
        name (str): Player name to normalize
        
    Returns:
        str: Normalized player name
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove all accented characters
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace('ñ', 'n').replace('ç', 'c').replace('ü', 'u')
    
    # Remove suffixes like Jr., Sr., III
    name = re.sub(r'\b(jr|jr\.|sr|sr\.|iii|ii|iv)\b', '', name)
    
    # Remove all punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Handle lastname, firstname format
    if ", " in name:
        parts = name.split(", ", 1)
        if len(parts) == 2:
            last, first = parts
            name = f"{first} {last}"
    
    return name

def merge_traditional_stats(game_level_df, traditional_df):
    """
    Merge game-level data with traditional stats based on pitcher ID and season
    
    Args:
        game_level_df (pd.DataFrame): Game-level statcast data
        traditional_df (pd.DataFrame): Season-level traditional stats
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    import pandas as pd
    
    logger.info(f"Merging {len(game_level_df)} game records with traditional stats")
    
    # Ensure we have the necessary columns
    if 'pitcher' not in game_level_df.columns or 'season' not in game_level_df.columns:
        logger.error("Game level data missing required columns for merging")
        return game_level_df
    
    # Process traditional stats to prepare for merging
    if 'IDfg' in traditional_df.columns:
        # If it's raw traditional data
        trad_processed = traditional_df.copy()
        
        # Rename ID columns
        trad_processed['traditional_id'] = trad_processed['IDfg']
        
        if 'Season' in trad_processed.columns:
            trad_processed['season'] = trad_processed['Season']
        
        # Select relevant columns for merging
        stat_cols = ['ERA', 'FIP', 'xFIP', 'K/9', 'BB/9', 'WHIP', 'BABIP', 'LOB%', 'WAR', 'Team']
        relevant_cols = ['traditional_id', 'season'] + [col for col in stat_cols if col in trad_processed.columns]
        trad_processed = trad_processed[relevant_cols]
        
        # Convert column names to lowercase
        trad_processed.columns = [col.lower() if col != 'LOB%' else 'lob_pct' for col in trad_processed.columns]
        
    else:
        # If it's already processed from the database
        trad_processed = traditional_df.copy()
    
    # Get pitcher mappings from the database
    from src.data.db import get_db_connection
    conn = get_db_connection()
    pitcher_map = pd.read_sql("""
        SELECT pitcher_id, statcast_id, traditional_id 
        FROM pitchers 
        WHERE statcast_id IS NOT NULL AND traditional_id IS NOT NULL
    """, conn)
    conn.close()
    
    if pitcher_map.empty:
        logger.error("No pitcher ID mappings found in database")
        return game_level_df
    
    # Add traditional_id to game_level data using the mapping
    pitcher_dict = dict(zip(pitcher_map['statcast_id'], pitcher_map['traditional_id']))
    game_level_df['traditional_id'] = game_level_df['pitcher'].astype(int).map(pitcher_dict)
    
    # Merge with traditional stats
    merged_df = pd.merge(
        game_level_df,
        trad_processed,
        on=['traditional_id', 'season'],
        how='left'
    )
    
    # Report merge stats
    merged_count = merged_df.dropna(subset=['era']).shape[0]
    merge_pct = merged_count / len(game_level_df) * 100
    logger.info(f"Successfully merged traditional stats for {merged_count} games ({merge_pct:.1f}%)")
    
    return merged_df