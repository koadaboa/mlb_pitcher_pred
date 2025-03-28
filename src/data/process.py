# Functions for processing data after fetching from external sources
import pandas as pd
import numpy as np
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

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

def aggregate_to_game_level(statcast_df):
    """
    Aggregate statcast pitch-level data to pitcher-game level
    """
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
        
        # Swinging strike percentage
        if 'description' in statcast_df.columns:
            swinging_descriptions = ['swinging_strike', 'swinging_strike_blocked']
            swinging_df = statcast_df[statcast_df['description'].isin(swinging_descriptions)]
            
            if not swinging_df.empty:
                total_pitches = grouped.size()
                swinging_counts = swinging_df.groupby(group_cols).size()
                swinging_pct = (swinging_counts / total_pitches * 100).reset_index(name='swinging_strike_pct')
                game_level = pd.merge(game_level, swinging_pct, on=group_cols, how='left')
            
            # Called strike percentage
            called_strike_df = statcast_df[statcast_df['description'] == 'called_strike']
            if not called_strike_df.empty:
                called_counts = called_strike_df.groupby(group_cols).size()
                called_pct = (called_counts / total_pitches * 100).reset_index(name='called_strike_pct')
                game_level = pd.merge(game_level, called_pct, on=group_cols, how='left')
                
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

def export_data_to_csv():
    """Export game stats to CSV file"""
    import pandas as pd
    from pathlib import Path
    from src.data.db import execute_query
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export game-level stats
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
    
    # Export metadata about the dataset
    metadata = {
        "game_stats_count": len(game_stats),
        "game_stats_seasons": sorted(game_stats['season'].unique().tolist()),
        "unique_pitchers_in_game_stats": game_stats['pitcher_id'].nunique(),
        "avg_games_per_pitcher": len(game_stats) / game_stats['pitcher_id'].nunique() if game_stats['pitcher_id'].nunique() > 0 else 0,
        "avg_strikeouts_per_game": game_stats['strikeouts'].mean() if 'strikeouts' in game_stats.columns else 0
    }
    
    # Save metadata
    meta_df = pd.DataFrame([metadata])
    meta_file = output_dir / "dataset_metadata.csv"
    meta_df.to_csv(meta_file, index=False)
    
    logger.info(f"Exported dataset metadata to {meta_file}")
    
    # Export mapping between pitchers
    mapping_query = """
    SELECT 
        pitcher_id,
        player_name,
        statcast_id
    FROM 
        pitchers
    WHERE 
        statcast_id IS NOT NULL
    """
    
    pitcher_map = execute_query(mapping_query)
    map_file = output_dir / "pitcher_id_mapping.csv"
    pitcher_map.to_csv(map_file, index=False)
    
    logger.info(f"Exported {len(pitcher_map)} pitcher mappings to {map_file}")
    
    return {
        "game_stats": game_file,
        "metadata": meta_file,
        "pitcher_mapping": map_file
    }