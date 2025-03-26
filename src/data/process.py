# Functions for processing data after fetching from external sources
import pandas as pd
import numpy as np
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

def aggregate_statcast_to_game_level(statcast_df):
    """
    Aggregate statcast data to pitcher-game level
    
    Args:
        statcast_df (pandas.DataFrame): Raw statcast data
        
    Returns:
        pandas.DataFrame: Aggregated pitcher-game level data
    """
    logger.info("Aggregating statcast data to pitcher-game level...")
    
    # Analyze input data structure for debugging
    logger.info(f"Input data shape: {statcast_df.shape}")
    logger.info(f"Available columns: {', '.join(statcast_df.columns[:10])}...")
    
    # Check for required columns
    required_cols = ['game_date', 'pitcher']
    missing_cols = [col for col in required_cols if col not in statcast_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        
        # Try to recover if possible - map common column name variations
        column_alternatives = {
            'pitcher': ['pitcher_id', 'pitcher_name', 'player_id'],
            'game_date': ['game_day', 'date']
        }
        
        for missing_col in missing_cols.copy():
            for alt in column_alternatives.get(missing_col, []):
                if alt in statcast_df.columns:
                    logger.info(f"Using {alt} column as {missing_col}")
                    statcast_df[missing_col] = statcast_df[alt]
                    missing_cols.remove(missing_col)
                    break
        
        if missing_cols:
            logger.error("Cannot continue with data aggregation due to missing required columns")
            return pd.DataFrame()
    
    # Ensure we have player_name
    if 'player_name' not in statcast_df.columns:
        if 'pitcher_name' in statcast_df.columns:
            statcast_df['player_name'] = statcast_df['pitcher_name']
        elif 'pitcher' in statcast_df.columns and 'player_name' not in statcast_df.columns:
            # Create placeholder player names from pitcher IDs
            statcast_df['player_name'] = 'Pitcher_' + statcast_df['pitcher'].astype(str)
            logger.warning("Created placeholder player names from pitcher IDs")
    
    # Convert game_date to datetime
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'], errors='coerce')
    
    # Create unique game ID
    if 'game_pk' in statcast_df.columns:
        statcast_df['game_id'] = statcast_df['game_pk'].astype(str)
    else:
        # If game_pk is not available, create a synthetic game ID
        logger.warning("game_pk column not found, creating synthetic game IDs")
        statcast_df['game_id'] = statcast_df['game_date'].dt.strftime('%Y%m%d') + '_' + statcast_df.groupby(['game_date', 'pitcher']).ngroup().astype(str)
    
    # Add season if not present
    if 'season' not in statcast_df.columns:
        statcast_df['season'] = statcast_df['game_date'].dt.year
    
    # Ensure the pitcher column is string type for grouping
    statcast_df['pitcher'] = statcast_df['pitcher'].astype(str)
    
    # Log the number of unique pitchers, games, and pitcher-game combinations
    logger.info(f"Unique pitchers: {statcast_df['pitcher'].nunique()}")
    logger.info(f"Unique games: {statcast_df['game_id'].nunique()}")
    logger.info(f"Unique pitcher-game combinations: {statcast_df.groupby(['pitcher', 'game_id']).ngroups}")
    
    # Process pitch types directly from raw data before aggregation
    pitch_type_present = False
    if 'pitch_type' in statcast_df.columns:
        pitch_type_present = True
        # Get counts of each pitch type for each pitcher-game
        logger.info("Pitch type column found - extracting pitch mix data")
        
        # Create empty dictionaries to store pitch counts and percentages
        pitch_counts = defaultdict(lambda: defaultdict(int))
        pitch_percentages = defaultdict(dict)
        
        # Count pitches by type for each pitcher-game
        for _, row in statcast_df.iterrows():
            if pd.notna(row['pitch_type']):
                pitcher = row['pitcher']
                game_id = row['game_id']
                pitch_type = row['pitch_type']
                pitch_counts[(pitcher, game_id)]['total'] += 1
                pitch_counts[(pitcher, game_id)][pitch_type] += 1
        
        # Calculate percentages
        for (pitcher, game_id), counts in pitch_counts.items():
            total = counts['total']
            if total > 0:
                for pitch_type, count in counts.items():
                    if pitch_type != 'total':
                        pitch_percentages[(pitcher, game_id)][pitch_type] = count / total
    else:
        logger.warning("No pitch_type column found - unable to extract pitch mix data")
    
    # Group by pitcher and game
    group_cols = ['pitcher', 'game_id', 'game_date']
    if 'player_name' in statcast_df.columns:
        group_cols.append('player_name')
    if 'season' in statcast_df.columns:
        group_cols.append('season')
    
    grouped = statcast_df.groupby(group_cols)
    
    # Count total pitches per pitcher per game
    pitch_counts_df = grouped.size().reset_index(name='pitch_count')
    
    # Calculate basic stats
    try:
        # Create aggregation functions for game-level metrics
        agg_dict = {'pitch_count': 'first'}  # Already calculated
        
        # Add optional columns to the aggregation if they exist
        optional_cols = [
            'release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'effective_speed', 'release_spin_rate', 'release_extension',
            'zone', 'description'
        ]
        
        for col in optional_cols:
            if col in statcast_df.columns:
                if col in ['release_speed', 'effective_speed', 'release_spin_rate']:
                    agg_dict[col] = ['mean', 'max']
                elif col in ['release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z']:
                    agg_dict[col] = 'mean'
                elif col == 'zone':
                    agg_dict[col] = lambda x: (x == 1).mean()  # Zone percentage
        
        # Apply aggregation
        game_level = grouped.agg(agg_dict)
        
        # Flatten multi-level columns if they exist
        if isinstance(game_level.columns, pd.MultiIndex):
            game_level.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                for col in game_level.columns.values]
        
        # Reset index to convert grouped columns to regular columns
        game_level = game_level.reset_index()
        
    except Exception as e:
        logger.error(f"Error during statcast aggregation: {e}")
        # Fallback to basic aggregation
        logger.info("Falling back to basic aggregation")
        game_level = pitch_counts_df  # Just use pitch counts
    
    # Extract game outcome data
    # Look for columns that might contain event outcomes
    outcome_col = None
    for col_name in ['events', 'event', 'outcome', 'play_outcome']:
        if col_name in statcast_df.columns:
            outcome_col = col_name
            break
    
    if outcome_col:
        logger.info(f"Found outcome data in column: {outcome_col}")
        outcome_agg = statcast_df.groupby(['pitcher', 'game_id'])[outcome_col].value_counts().unstack(fill_value=0)
        
        # Check if event types exist for strikeouts, hits, walks, home runs
        for outcome_type, col_name in [
            ('strikeout', 'strikeouts'), 
            ('walk', 'walks'),
            ('home_run', 'home_runs')
        ]:
            if outcome_type in outcome_agg.columns:
                # Map outcome to game_level
                outcome_map = outcome_agg[outcome_type].to_dict()
                game_level[col_name] = game_level.set_index(['pitcher', 'game_id']).index.map(
                    lambda x: outcome_map.get(x, 0)
                )
            else:
                game_level[col_name] = 0
        
        # For hits, we may need to check multiple event types
        hit_types = ['single', 'double', 'triple', 'home_run']
        hit_cols = [ht for ht in hit_types if ht in outcome_agg.columns]
        
        if hit_cols:
            # Sum all hit types
            hits_map = outcome_agg[hit_cols].sum(axis=1).to_dict()
            game_level['hits'] = game_level.set_index(['pitcher', 'game_id']).index.map(
                lambda x: hits_map.get(x, 0)
            )
        else:
            game_level['hits'] = 0
    else:
        # If no event column found, add default outcome columns
        logger.warning("No outcome/event column found - using zeros for game outcomes")
        game_level['strikeouts'] = 0
        game_level['hits'] = 0
        game_level['walks'] = 0
        game_level['home_runs'] = 0
    
    # Calculate swing and miss metrics
    if 'description' in statcast_df.columns:
        # Get swinging strike percentage
        swing_miss = statcast_df.groupby(['pitcher', 'game_id'])['description'].apply(
            lambda x: (x == 'swinging_strike').mean() if len(x) > 0 else 0
        ).to_dict()
        
        game_level['swinging_strike_pct'] = game_level.set_index(['pitcher', 'game_id']).index.map(
            lambda x: swing_miss.get(x, 0)
        )
        
        # Get called strike percentage
        called_strike = statcast_df.groupby(['pitcher', 'game_id'])['description'].apply(
            lambda x: (x == 'called_strike').mean() if len(x) > 0 else 0
        ).to_dict()
        
        game_level['called_strike_pct'] = game_level.set_index(['pitcher', 'game_id']).index.map(
            lambda x: called_strike.get(x, 0)
        )
    else:
        # Default values if not available
        game_level['swinging_strike_pct'] = 0
        game_level['called_strike_pct'] = 0
    
    # Calculate zone rate if zone column exists
    if 'zone' in statcast_df.columns:
        zone_rate = statcast_df.groupby(['pitcher', 'game_id'])['zone'].apply(
            lambda x: (x != 0).mean() if len(x) > 0 else 0
        ).to_dict()
        
        game_level['zone_rate'] = game_level.set_index(['pitcher', 'game_id']).index.map(
            lambda x: zone_rate.get(x, 0)
        )
    else:
        game_level['zone_rate'] = 0
    
    # Add pitch mix percentages to game_level
    if pitch_type_present:
        # Get unique pitch types
        all_pitch_types = set()
        for percentages in pitch_percentages.values():
            all_pitch_types.update(percentages.keys())
        
        logger.info(f"Found {len(all_pitch_types)} unique pitch types: {', '.join(sorted(all_pitch_types))}")
        
        # Add columns for each pitch type
        for pitch_type in all_pitch_types:
            column_name = f'pitch_pct_{pitch_type}'
            game_level[column_name] = game_level.apply(
                lambda row: pitch_percentages.get((row['pitcher'], row['game_id']), {}).get(pitch_type, 0),
                axis=1
            )
    
    # Ensure all numeric columns have sensible values
    for col in game_level.select_dtypes(include=[np.number]).columns:
        game_level[col] = game_level[col].fillna(0)
    
    logger.info(f"Aggregated data to {len(game_level)} pitcher-game records with {len(game_level.columns)} columns")
    
    # List columns with all zero values - these might indicate problems
    zero_cols = [col for col in game_level.columns if game_level[col].sum() == 0]
    if zero_cols:
        logger.warning(f"The following columns contain all zeros and may need attention: {', '.join(zero_cols)}")
    
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