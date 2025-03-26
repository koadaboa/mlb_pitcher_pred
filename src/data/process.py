# Functions for processing data after fetching from external sources
import pandas as pd
import numpy as np
import logging

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
    
    # Ensure we have required columns
    required_cols = ['game_date', 'pitcher', 'player_name']
    missing_cols = [col for col in required_cols if col not in statcast_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}. Available columns: {statcast_df.columns.tolist()}")
        # Try to recover from this error if possible
        if 'player_name' in missing_cols and 'pitcher_name' in statcast_df.columns:
            statcast_df['player_name'] = statcast_df['pitcher_name']
            missing_cols.remove('player_name')
        
        if missing_cols:
            logger.error("Cannot continue with data aggregation due to missing columns")
            return pd.DataFrame()
    
    # Convert game_date to datetime
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    
    # Create unique game ID
    if 'game_pk' in statcast_df.columns:
        statcast_df['game_id'] = statcast_df['game_pk'].astype(str)
    else:
        # If game_pk is not available, create a synthetic game ID
        logger.warning("game_pk column not found, creating synthetic game IDs")
        statcast_df['game_id'] = statcast_df['game_date'].dt.strftime('%Y%m%d') + '_' + statcast_df.groupby(['game_date', 'pitcher']).ngroup().astype(str)
    
    # Group by pitcher and game
    grouped = statcast_df.groupby(['pitcher', 'game_id', 'game_date', 'player_name'])
    
    # Calculate pitcher-game level metrics
    try:
        agg_dict = {
            # Pitch counts
            'pitch_type': ['count', lambda x: x.value_counts().to_dict()],
        }
        
        # Add optional columns to the aggregation if they exist
        optional_cols = [
            'release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'effective_speed', 'release_spin_rate', 'release_extension',
            'zone', 'type', 'events', 'description'
        ]
        
        for col in optional_cols:
            if col in statcast_df.columns:
                if col == 'release_speed':
                    agg_dict[col] = ['mean', 'std', 'max']
                elif col in ['release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z']:
                    agg_dict[col] = ['mean', 'std']
                elif col == 'effective_speed':
                    agg_dict[col] = ['mean', 'max']
                elif col == 'release_spin_rate':
                    agg_dict[col] = ['mean', 'std']
                elif col == 'release_extension':
                    agg_dict[col] = ['mean']
                elif col == 'zone':
                    agg_dict[col] = lambda x: (x == 1).mean()  # Zone percentage
                elif col == 'type':
                    agg_dict[col] = lambda x: (x == 'S').mean()  # Strike percentage
                elif col in ['events', 'description']:
                    agg_dict[col] = lambda x: x.value_counts().to_dict()  # Outcomes
        
        # Apply aggregation
        game_level = grouped.agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        game_level.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                              for col in game_level.columns.values]
        
    except Exception as e:
        logger.error(f"Error during statcast aggregation: {e}")
        # Try a more basic aggregation as fallback
        logger.info("Attempting simpler aggregation as fallback")
        agg_dict = {
            'pitch_type': 'count',  # Just count pitches
        }
        
        game_level = grouped.agg(agg_dict).reset_index()
    
    # Calculate additional metrics
    if 'events_lambda' in game_level.columns:
        # Extract strikeouts
        game_level['strikeouts'] = game_level['events_lambda'].apply(
            lambda x: x.get('strikeout', 0) if isinstance(x, dict) else 0
        )
        
        # Extract other key events
        game_level['hits'] = game_level['events_lambda'].apply(
            lambda x: sum([x.get(e, 0) for e in ['single', 'double', 'triple', 'home_run']])
            if isinstance(x, dict) else 0
        )
        
        game_level['walks'] = game_level['events_lambda'].apply(
            lambda x: x.get('walk', 0) if isinstance(x, dict) else 0
        )
        
        game_level['home_runs'] = game_level['events_lambda'].apply(
            lambda x: x.get('home_run', 0) if isinstance(x, dict) else 0
        )
    else:
        # If events_lambda column is missing, add default columns
        logger.warning("events_lambda column missing, using zeros for game outcomes")
        game_level['strikeouts'] = 0
        game_level['hits'] = 0
        game_level['walks'] = 0
        game_level['home_runs'] = 0
    
    # Extract pitch mix percentages
    if 'pitch_type_lambda' in game_level.columns:
        pitch_types = set()
        for pitch_dict in game_level['pitch_type_lambda']:
            if isinstance(pitch_dict, dict):
                pitch_types.update(pitch_dict.keys())
        
        for pitch in pitch_types:
            game_level[f'pitch_pct_{pitch}'] = game_level['pitch_type_lambda'].apply(
                lambda x: x.get(pitch, 0) / sum(x.values()) if isinstance(x, dict) and sum(x.values()) > 0 else 0
            )
    
    # Extract swinging strike percentage
    if 'description_lambda' in game_level.columns:
        game_level['swinging_strike_pct'] = game_level['description_lambda'].apply(
            lambda x: x.get('swinging_strike', 0) / sum(x.values()) if isinstance(x, dict) and sum(x.values()) > 0 else 0
        )
        
        game_level['called_strike_pct'] = game_level['description_lambda'].apply(
            lambda x: x.get('called_strike', 0) / sum(x.values()) if isinstance(x, dict) and sum(x.values()) > 0 else 0
        )
    else:
        # Default values if missing
        game_level['swinging_strike_pct'] = 0
        game_level['called_strike_pct'] = 0
    
    # Clean up dictionary columns that we've extracted
    cols_to_drop = [col for col in game_level.columns if col.endswith('_lambda')]
    game_level = game_level.drop(columns=cols_to_drop)
    
    # Ensure all numeric columns have sensible values
    for col in game_level.select_dtypes(include=[np.number]).columns:
        game_level[col] = game_level[col].fillna(0)
    
    logger.info(f"Aggregated data to {len(game_level)} pitcher-game records")
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
    
    # Ensure we have required columns
    required_cols = ['Season', 'Name', 'IDfg']
    missing_cols = [col for col in required_cols if col not in trad_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}. Available columns: {trad_df.columns.tolist()}")
        
        # Try to recover if possible
        if 'Season' in missing_cols and 'season' in trad_df.columns:
            trad_df['Season'] = trad_df['season']
            missing_cols.remove('Season')
            
        if 'Name' in missing_cols and 'name' in trad_df.columns:
            trad_df['Name'] = trad_df['name']
            missing_cols.remove('Name')
            
        if 'IDfg' in missing_cols and 'playerid' in trad_df.columns:
            trad_df['IDfg'] = trad_df['playerid']
            missing_cols.remove('IDfg')
        
        if missing_cols:
            logger.error("Cannot continue with traditional stats processing due to missing columns")
            return pd.DataFrame()
    
    # Create a pitcher_id column from IDfg for merging
    trad_df['pitcher_id'] = trad_df['IDfg']
    
    # Process columns for easier merging
    trad_df['Name'] = trad_df['Name'].str.strip()
    
    # Select relevant columns
    key_stats = ['pitcher_id', 'Name', 'Team', 'Season', 'ERA', 'W', 'L', 'G', 'GS', 'CG', 'ShO', 
                'SV', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 
                'SO', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 
                'FIP', 'xFIP', 'WAR']
    
    # Use only available columns
    available_cols = [col for col in key_stats if col in trad_df.columns]
    processed_df = trad_df[available_cols].copy()

    # Convert percentage strings to floats if needed
    pct_cols = ['LOB%']
    for col in pct_cols:
        if col in processed_df.columns and processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].str.rstrip('%').astype('float') / 100
    
    logger.info(f"Processed {len(processed_df)} traditional stat records")
    return processed_df

def normalize_name(name):
    """
    Normalize player names for better matching
    
    Args:
        name (str): Player name to normalize
        
    Returns:
        str: Normalized player name
    """
    if not name:
        return ""
    # Remove suffixes like Jr., Sr., III
    name = name.replace("Jr.", "").replace("Sr.", "").replace("III", "").replace("II", "").replace("IV", "")
    # Remove periods and commas
    name = name.replace(".", "").replace(",", "")
    # Convert to lowercase and strip whitespace
    name = name.lower().strip()
    # Handle lastname, firstname format
    if ", " in name:
        last, first = name.split(", ", 1)
        name = f"{first} {last}"
    return name