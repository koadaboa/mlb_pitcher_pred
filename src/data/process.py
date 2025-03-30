# Functions for processing data after fetching from external sources
import pandas as pd
import numpy as np
from src.data.utils import setup_logger
import re
from collections import defaultdict

logger = setup_logger(__name__)

# Add helper functions for each processing step
def _validate_statcast_data(statcast_df):
    """Validate input statcast data"""
    if statcast_df.empty:
        logger.warning("Empty dataframe provided for aggregation.")
        return False
    
    required_cols = ['game_date', 'pitcher']
    missing = [col for col in required_cols if col not in statcast_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    
    return True

def _prepare_statcast_data(statcast_df):
    """Prepare statcast data for aggregation"""
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
    
    return statcast_df

def _get_grouping_columns(statcast_df):
    """
    Get the grouping columns for statcast data
    """
    group_cols = ['pitcher', 'game_id', 'game_date']
    if 'player_name' in statcast_df.columns:
        group_cols.append('player_name')
    if 'season' in statcast_df.columns:
        group_cols.append('season')
    elif 'game_date' in statcast_df.columns:
        # Add season column based on year
        statcast_df['season'] = statcast_df['game_date'].dt.year
        group_cols.append('season')
    return group_cols

def _add_pitch_mix_data(statcast_df, game_level, grouped, group_cols):
    """
    Extract pitch mix data from statcast data
    """
    logger.info("Pitch type column found - extracting pitch mix data")
        
    # Get unique pitchers and games for logging
    unique_pitchers = statcast_df['pitcher'].nunique()
    unique_games = statcast_df['game_id'].nunique()
    unique_combos = statcast_df.groupby(['pitcher', 'game_id']).ngroups
    
    logger.info(f"Unique pitchers: {unique_pitchers}")
    logger.info(f"Unique games: {unique_games}")
    logger.info(f"Unique pitcher-game combinations: {unique_combos}")
    
    try:
        # Count pitches per type per game
        pitch_type_counts = statcast_df.groupby(group_cols + ['pitch_type']).size().reset_index(name='type_count')
        
        # Merge with total pitch counts
        pitch_type_counts = pd.merge(
            pitch_type_counts, 
            grouped['pitch_count'].reset_index(), 
            on=group_cols,
            how='left'
        )
        
        # Calculate percentage
        pitch_type_counts['percentage'] = (pitch_type_counts['type_count'] / pitch_type_counts['pitch_count']) * 100
        
        # Get unique pitch types
        pitch_types = pitch_type_counts['pitch_type'].unique()
        logger.info(f"Found {len(pitch_types)} unique pitch types: {', '.join(pitch_types)}")
        
        # Pivot to wide format (one column per pitch type)
        pitch_pivot = pitch_type_counts.pivot_table(
            index=group_cols,
            columns='pitch_type',
            values='percentage',
            fill_value=0
        ).reset_index()
        
        # Fix MultiIndex columns after pivot
        if isinstance(pitch_pivot.columns, pd.MultiIndex):
            pitch_pivot.columns = [
                col[0] if col[0] in group_cols else f'pitch_pct_{col[1]}' 
                for col in pitch_pivot.columns
            ]
        else:
            # Rename pivot columns
            pitch_types = [col for col in pitch_pivot.columns if col not in group_cols]
            new_cols = list(group_cols).copy()
            for col in pitch_types:
                new_cols.append(f'pitch_pct_{col}')
            pitch_pivot.columns = new_cols
        
        # Merge with game level data
        game_level = pd.merge(game_level, pitch_pivot, on=group_cols, how='left')
        
    except Exception as e:
        logger.error(f"Error during pitch mix extraction: {e}")
        logger.info("Falling back to basic aggregation")
        
    return game_level

def _add_pitch_metrics(statcast_df, game_level, grouped, group_cols):
    """
    Add pitch metrics to game level data
    """
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
    except Exception as e:
        logger.error(f"Error during pitch metrics extraction: {e}")
        logger.info("Falling back to basic aggregation")

    return game_level

def _add_event_metrics(statcast_df, game_level, grouped, group_cols):
    """
    Add event metrics to game level data
    """
    try:
        
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

    return game_level

def _finalize_game_level_data(game_level, group_cols):
    """
    Finalize game level data by filling NAs with zeros and removing duplicates
    """
    # Fill NAs with zeros
    for col in game_level.columns:
        if col not in group_cols and pd.api.types.is_numeric_dtype(game_level[col]):
            game_level[col] = game_level[col].fillna(0)
    
    # Check for columns with all zeros (only numeric columns)
    numeric_cols = game_level.select_dtypes(include=[np.number]).columns
    zero_cols = [col for col in numeric_cols if game_level[col].sum() == 0]
    if zero_cols:
        logger.warning(f"The following columns contain all zeros: {zero_cols}")
    
    return game_level

def aggregate_to_game_level(statcast_df):
    """
    Aggregate statcast pitch-level data to pitcher-game level
    """
    logger.info("Aggregating statcast data to pitcher-game level...")
    logger.info(f"Input data shape: {statcast_df.shape}")
    
    if 'pitch_type' in statcast_df.columns:
        logger.info(f"Available columns: {', '.join(statcast_df.columns[:20])}...")
    
    if not _validate_statcast_data(statcast_df):
        return pd.DataFrame()
    
    statcast_df = _prepare_statcast_data(statcast_df)
    
    # Define grouping columns
    group_cols = _get_grouping_columns(statcast_df)
        
    # Log grouping info
    logger.info(f"Grouping by columns: {group_cols}")
    
     # Group by pitcher and game
    grouped = statcast_df.groupby(group_cols)
    
    # Count pitches per game
    pitch_counts = grouped.size().reset_index(name='pitch_count')
    game_level = pitch_counts.copy()
    
    # Extract pitch mix data if available
    if 'pitch_type' in statcast_df.columns:
        game_level = _add_pitch_mix_data(statcast_df, game_level, grouped, group_cols)

    game_level = _add_pitch_metrics(statcast_df, game_level, grouped, group_cols)
    game_level = _add_event_metrics(statcast_df, game_level, grouped, group_cols)

    game_level = _finalize_game_level_data(game_level, group_cols)
    
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

# Extract game context from statcast data
def extract_game_context(statcast_df):
    """Extract game context information from statcast data"""
    if statcast_df.empty:
        return pd.DataFrame()
    
    # Fields we need from statcast data
    context_fields = ['game_date', 'game_pk', 'home_team', 'away_team', 
                     'stadium', 'game_type', 'home_score', 'away_score']
    
    # Check if fields exist
    missing = [f for f in context_fields if f not in statcast_df.columns]
    if missing:
        logger.warning(f"Missing context fields: {missing}")
        # Use available fields
        context_fields = [f for f in context_fields if f in statcast_df.columns]
    
    if not context_fields:
        return pd.DataFrame()
    
    # Group by game to get unique game context
    game_context = statcast_df[context_fields].drop_duplicates('game_pk')
    
    # Add season information
    if 'game_date' in game_context.columns:
        game_context['season'] = pd.to_datetime(game_context['game_date']).dt.year
    
    # Rename columns to match database schema
    game_context = game_context.rename(columns={
        'game_pk': 'game_id',
        'game_type': 'day_night'  # This may need adjustment
    })
    
    return game_context

def map_opponents_to_games(statcast_df, pitcher_team_map):
    """Map opponent teams to pitcher game records"""
    logger.info("Starting opponent mapping process...")
    
    # Fields needed
    needed_fields = ['pitcher', 'game_pk', 'home_team', 'away_team']
    
    # Check if fields exist
    if not all(field in statcast_df.columns for field in needed_fields):
        logger.warning(f"Missing required fields for opponent mapping")
        return pd.DataFrame()
    
    # Process in batches to avoid memory issues
    batch_size = 5000
    total_rows = len(statcast_df)
    pitcher_games = pd.DataFrame()
    
    for i in range(0, total_rows, batch_size):
        logger.info(f"Processing batch {i//batch_size + 1} of {total_rows//batch_size + 1}...")
        batch = statcast_df.iloc[i:i+batch_size]
        
        # Group by pitcher and game
        batch_games = batch[needed_fields].drop_duplicates(['pitcher', 'game_pk'])
        pitcher_games = pd.concat([pitcher_games, batch_games], ignore_index=True)
    
    logger.info(f"Processing {len(pitcher_games)} unique pitcher-game combinations...")
    
    # Prepare for opponents
    pitcher_games['opponent_team_id'] = None
    
    # Process in smaller batches
    processed = 0
    for i in range(0, len(pitcher_games), 1000):
        batch = pitcher_games.iloc[i:i+1000].copy()
        
        # Determine opponent for each record
        for idx, row in batch.iterrows():
            pitcher_id = row['pitcher']
            pitcher_team = pitcher_team_map.get(pitcher_id)
            
            if not pitcher_team:
                continue
                
            # If pitcher is on home team, opponent is away team
            if pitcher_team == row['home_team']:
                batch.at[idx, 'opponent_team_id'] = row['away_team']
            # If pitcher is on away team, opponent is home team
            elif pitcher_team == row['away_team']:
                batch.at[idx, 'opponent_team_id'] = row['home_team']
        
        # Update original dataframe
        pitcher_games.iloc[i:i+1000] = batch
        
        processed += len(batch)
        logger.info(f"Processed {processed}/{len(pitcher_games)} opponent mappings...")
    
    # Filter to records with valid opponent
    result = pitcher_games[pitcher_games['opponent_team_id'].notna()]
    result = result[['pitcher', 'game_pk', 'opponent_team_id']].rename(
        columns={'game_pk': 'game_id'}
    )
    
    # Add this line to convert game_id to string
    result['game_id'] = result['game_id'].astype(str)
    
    logger.info(f"Mapped {len(result)} pitcher-game combinations to opponents")
    return result