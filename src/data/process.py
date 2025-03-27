# Functions for processing data after fetching from external sources
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

from src.data.utils import normalize_name  # Use the new utils module

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

def export_data_to_csv():
    """Export game stats and traditional stats to separate CSV files"""
    import pandas as pd
    from pathlib import Path
    from src.data.db import execute_query
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Export game-level stats
    logger.info("Exporting game-level stats...")
    
    # Join pitchers table to get player names
    game_query = """
    SELECT 
        g.pitcher_id,
        p.player_name,
        p.statcast_id,
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
        g.zone_rate
    FROM 
        game_stats g
    JOIN 
        pitchers p ON g.pitcher_id = p.pitcher_id
    ORDER BY 
        g.game_date, p.player_name
    """
    
    game_stats = execute_query(game_query)
    
    # Add pitch mix data if available
    pitch_mix_query = """
    SELECT 
        gs.id as game_stats_id,
        gs.pitcher_id,
        gs.game_id,
        pm.pitch_type,
        pm.percentage
    FROM 
        pitch_mix pm
    JOIN 
        game_stats gs ON pm.game_stats_id = gs.id
    """
    
    pitch_mix = execute_query(pitch_mix_query)
    
    # If we have pitch mix data, pivot it to add to game stats
    if not pitch_mix.empty:
        logger.info(f"Adding {len(pitch_mix)} pitch mix records to game stats...")
        
        # Create a unique key to join on
        pitch_mix['join_key'] = pitch_mix['pitcher_id'].astype(str) + '_' + pitch_mix['game_id'].astype(str)
        
        # Pivot to get one column per pitch type
        pitch_pivot = pitch_mix.pivot_table(
            index='join_key', 
            columns='pitch_type', 
            values='percentage',
            aggfunc='mean'
        ).reset_index()
        
        # Rename columns to avoid confusion
        pitch_types = [col for col in pitch_pivot.columns if col != 'join_key']
        for pitch in pitch_types:
            pitch_pivot.rename(columns={pitch: f'pitch_pct_{pitch}'}, inplace=True)
        
        # Add join key to game stats
        game_stats['join_key'] = game_stats['pitcher_id'].astype(str) + '_' + game_stats['game_id'].astype(str)
        
        # Merge
        game_stats = pd.merge(game_stats, pitch_pivot, on='join_key', how='left')
        game_stats.drop('join_key', axis=1, inplace=True)
    
    # Save to CSV
    game_file = output_dir / "game_level_stats.csv"
    game_stats.to_csv(game_file, index=False)
    
    logger.info(f"Exported {len(game_stats)} game-level records to {game_file}")
    logger.info(f"Game stats include {len(game_stats.columns)} columns")
    
    # 2. Export traditional (season-level) stats
    logger.info("Exporting traditional (season-level) stats...")
    
    trad_query = """
    SELECT 
        t.pitcher_id,
        p.player_name,
        p.statcast_id,
        p.traditional_id,
        t.season,
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
        traditional_stats t
    JOIN 
        pitchers p ON t.pitcher_id = p.pitcher_id
    ORDER BY 
        t.season, p.player_name
    """
    
    trad_stats = execute_query(trad_query)
    trad_file = output_dir / "traditional_season_stats.csv"
    trad_stats.to_csv(trad_file, index=False)
    
    logger.info(f"Exported {len(trad_stats)} traditional stat records to {trad_file}")
    
    # 3. Export metadata about the dataset
    metadata = {
        "game_stats_count": len(game_stats),
        "game_stats_seasons": sorted(game_stats['season'].unique().tolist()),
        "traditional_stats_count": len(trad_stats),
        "traditional_stats_seasons": sorted(trad_stats['season'].unique().tolist()),
        "unique_pitchers_in_game_stats": game_stats['pitcher_id'].nunique(),
        "unique_pitchers_in_traditional_stats": trad_stats['pitcher_id'].nunique(),
        "unique_pitchers_in_both": len(set(game_stats['pitcher_id']).intersection(set(trad_stats['pitcher_id'])))
    }
    
    # Calculate some statistics for context
    if not game_stats.empty:
        metadata["avg_games_per_pitcher"] = len(game_stats) / game_stats['pitcher_id'].nunique()
        metadata["avg_strikeouts_per_game"] = game_stats['strikeouts'].mean()
    
    if not trad_stats.empty:
        metadata["avg_era"] = trad_stats['era'].mean()
        metadata["avg_k_per_9"] = trad_stats['k_per_9'].mean()
    
    # Save metadata
    meta_df = pd.DataFrame([metadata])
    meta_file = output_dir / "dataset_metadata.csv"
    meta_df.to_csv(meta_file, index=False)
    
    logger.info(f"Exported dataset metadata to {meta_file}")
    
    # 4. Export mapping between pitchers
    mapping_query = """
    SELECT 
        pitcher_id,
        player_name,
        statcast_id,
        traditional_id
    FROM 
        pitchers
    WHERE 
        statcast_id IS NOT NULL OR traditional_id IS NOT NULL
    """
    
    pitcher_map = execute_query(mapping_query)
    map_file = output_dir / "pitcher_id_mapping.csv"
    pitcher_map.to_csv(map_file, index=False)
    
    logger.info(f"Exported {len(pitcher_map)} pitcher mappings to {map_file}")
    
    return {
        "game_stats": game_file,
        "traditional_stats": trad_file,
        "metadata": meta_file,
        "pitcher_mapping": map_file
    }