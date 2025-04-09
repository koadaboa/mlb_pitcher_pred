"""
Feature engineering module for pitcher strikeout prediction.

This module provides functions to extract and transform features from pitch-level data
to create predictive features for modeling pitcher strikeouts.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from src.data.utils import setup_logger, DBConnection, ensure_dir, safe_float
from config import DBConfig, StrikeoutModelConfig

logger = setup_logger(__name__)

def load_pitch_data(limit=None, pitcher_id=None, seasons=None, batch_size=None):
    """
    Load pitch data from the database, optionally in batches.
    
    Args:
        limit (int): Optional limit on number of rows
        pitcher_id (int): Optional filter for specific pitcher
        seasons (list): Optional list of seasons to include
        batch_size (int): Optional batch size for processing large datasets
        
    Returns:
        pandas.DataFrame or generator: Pitch data
    """
    if seasons is None:
        seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    
    with DBConnection() as conn:
        # Get total count for batching
        count_query = "SELECT COUNT(*) FROM statcast_pitchers"
        conditions = []
        
        if pitcher_id:
            conditions.append(f"pitcher_id = {pitcher_id}")
        
        if seasons:
            season_list = ', '.join(str(s) for s in seasons)
            conditions.append(f"season IN ({season_list})")
        
        if conditions:
            count_query += " WHERE " + " AND ".join(conditions)
        
        total_count = pd.read_sql_query(count_query, conn).iloc[0, 0]
        
        # If batching is not requested or not needed, load all at once
        if batch_size is None or total_count <= batch_size or limit is not None:
            query = "SELECT * FROM statcast_pitchers"
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(df)} pitch records at once")
            return df
        
        # Otherwise, use batching
        else:
            logger.info(f"Loading {total_count} records in batches of {batch_size}")
            
            def batch_generator():
                for offset in range(0, total_count, batch_size):
                    batch_query = "SELECT * FROM statcast_pitchers"
                    
                    if conditions:
                        batch_query += " WHERE " + " AND ".join(conditions)
                    
                    batch_query += f" LIMIT {batch_size} OFFSET {offset}"
                    
                    batch_df = pd.read_sql_query(batch_query, conn)
                    logger.info(f"Loaded batch of {len(batch_df)} records (offset {offset})")
                    
                    yield batch_df
            
            return batch_generator()

def extract_game_data(df):
    """
    Extract unique games from pitch data and create game-level identifiers.
    
    Args:
        df (pandas.DataFrame): Pitch-level data
        
    Returns:
        pandas.DataFrame: Game-level data with basic aggregated statistics
    """
    # Convert game_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Group by pitcher, game_date, and game_pk to identify unique games
    game_groups = df.groupby(['pitcher_id', 'game_date', 'game_pk'])
    
    # Calculate strikeouts and outs for innings pitched estimation
    game_data = game_groups.agg({
        'player_name': 'first',
        'events': lambda x: sum(x == 'strikeout'),  # Count of strikeouts
        'home_team': 'first',
        'away_team': 'first',
        'p_throws': 'first',
        'season': 'first',
        'pitcher_days_since_prev_game': lambda x: safe_float(x.iloc[0]) if len(x) > 0 else None
    }).reset_index()
    
    # Count total pitches and batters faced
    game_data['total_pitches'] = game_groups.size()
    game_data['batters_faced'] = game_groups['at_bat_number'].nunique()
    
    # Calculate outs for innings pitched
    outs_events = ['field_out', 'strikeout', 'grounded_into_double_play', 
                   'force_out', 'sac_fly', 'sac_bunt', 'double_play']
    outs = game_groups['events'].apply(
        lambda x: sum(pd.notna(x) & x.isin(outs_events))
    )
    game_data['innings_pitched'] = (outs / 3).round(1)  # Round to 1 decimal for partial innings
    game_data['innings_pitched'] = game_data['innings_pitched'].fillna(0)
    
    # Calculate K/9 and K% metrics
    game_data['K_per_9'] = 9 * game_data['strikeouts'] / game_data['innings_pitched'].replace(0, np.nan)
    game_data['K_per_9'] = game_data['K_per_9'].fillna(0)
    
    game_data['K_pct'] = game_data['strikeouts'] / game_data['batters_faced'].replace(0, np.nan)
    game_data['K_pct'] = game_data['K_pct'].fillna(0)
    
    return game_data

def extract_pitch_features(df, game_data):
    """
    Extract pitch velocity, movement, and other pitch characteristics.
    
    Args:
        df (pandas.DataFrame): Pitch-level data
        game_data (pandas.DataFrame): Game-level data
        
    Returns:
        pandas.DataFrame: Enhanced game data with pitch features
    """
    # Group by pitcher, game_date, and game_pk
    pitch_features = df.groupby(['pitcher_id', 'game_date', 'game_pk']).agg({
        # Velocity features
        'release_speed': ['mean', 'std', 'max'],
        
        # Movement features
        'pfx_x': ['mean', 'std'],
        'pfx_z': ['mean', 'std'],
        
        # Location features
        'plate_x': ['std'],  # Horizontal location variation
        'plate_z': ['std'],  # Vertical location variation
        
        # Spin rate
        'release_spin_rate': ['mean', 'std'],
        
        # Extension
        'release_extension': ['mean'],
    })
    
    # Flatten the column names
    pitch_features.columns = ['_'.join(col).strip() for col in pitch_features.columns.values]
    
    # Reset index to merge with game_data
    pitch_features = pitch_features.reset_index()
    
    # Merge with game_data
    enhanced_game_data = pd.merge(
        game_data,
        pitch_features,
        on=['pitcher_id', 'game_date', 'game_pk'],
        how='left'
    )
    
    return enhanced_game_data

def calculate_pitch_mix_features(df, game_data):
    """
    Calculate pitch mix features (percentage of each pitch type thrown).
    
    Args:
        df (pandas.DataFrame): Pitch-level data
        game_data (pandas.DataFrame): Game-level data
        
    Returns:
        pandas.DataFrame: Enhanced game data with pitch mix features
    """
    # Get total pitches by type for each game
    pitch_counts = df.groupby(['pitcher_id', 'game_date', 'game_pk', 'pitch_type']).size().reset_index(name='count')
    
    # Get total pitches for each game
    total_pitches = pitch_counts.groupby(['pitcher_id', 'game_date', 'game_pk'])['count'].sum().reset_index(name='total')
    
    # Merge to calculate percentages
    pitch_mix = pd.merge(pitch_counts, total_pitches, on=['pitcher_id', 'game_date', 'game_pk'])
    pitch_mix['percentage'] = pitch_mix['count'] / pitch_mix['total']
    
    # Pivot to get pitch type percentages as separate columns
    pitch_mix_wide = pitch_mix.pivot_table(
        index=['pitcher_id', 'game_date', 'game_pk'],
        columns='pitch_type',
        values='percentage',
        fill_value=0
    )
    
    # Flatten column names
    pitch_mix_wide.columns = [f'pct_{col}' for col in pitch_mix_wide.columns]
    
    # Reset index to merge with game_data
    pitch_mix_wide = pitch_mix_wide.reset_index()
    
    # Merge with game_data
    enhanced_game_data = pd.merge(
        game_data,
        pitch_mix_wide,
        on=['pitcher_id', 'game_date', 'game_pk'],
        how='left'
    )
    
    # Calculate aggregate pitch categories
    # Breaking balls (SL: slider, CU: curveball, KC: knuckle curve, KN: knuckleball)
    breaking_cols = [col for col in enhanced_game_data.columns if col.startswith('pct_') and 
                    any(pitch in col for pitch in ['SL', 'CU', 'KC', 'KN'])]
    
    # Fastballs (FF: four-seam, FT: two-seam, FC: cutter, SI: sinker)
    fastball_cols = [col for col in enhanced_game_data.columns if col.startswith('pct_') and 
                    any(pitch in col for pitch in ['FF', 'FT', 'FC', 'SI'])]
    
    # Offspeed (CH: changeup, FS: splitter, FO: forkball, SC: screwball)
    offspeed_cols = [col for col in enhanced_game_data.columns if col.startswith('pct_') and 
                    any(pitch in col for pitch in ['CH', 'FS', 'FO', 'SC'])]
    
    # Calculate aggregated percentages
    if breaking_cols:
        enhanced_game_data['pct_breaking'] = enhanced_game_data[breaking_cols].sum(axis=1)
    else:
        enhanced_game_data['pct_breaking'] = 0
    
    if fastball_cols:
        enhanced_game_data['pct_fastball'] = enhanced_game_data[fastball_cols].sum(axis=1)
    else:
        enhanced_game_data['pct_fastball'] = 0
    
    if offspeed_cols:
        enhanced_game_data['pct_offspeed'] = enhanced_game_data[offspeed_cols].sum(axis=1)
    else:
        enhanced_game_data['pct_offspeed'] = 0
    
    return enhanced_game_data

def extract_strike_zone_features(df, game_data):
    """
    Extract features related to the strike zone performance.
    
    Args:
        df (pandas.DataFrame): Pitch-level data
        game_data (pandas.DataFrame): Game-level data
        
    Returns:
        pandas.DataFrame: Enhanced game data with strike zone features
    """
    # Calculate strike zone features
    zone_features = []
    
    # Define descriptions for different pitch outcomes
    strike_descriptions = ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul']
    swinging_strike_descriptions = ['swinging_strike', 'swinging_strike_blocked']
    
    # Process each game
    for (pitcher_id, game_date, game_pk), group in df.groupby(['pitcher_id', 'game_date', 'game_pk']):
        # Total pitches
        total_pitches = len(group)
        if total_pitches == 0:
            continue
        
        # Strike percentage
        strikes = sum(group['description'].isin(strike_descriptions))
        strike_pct = strikes / total_pitches
        
        # First pitch strike percentage
        first_pitches = group[group['pitch_number'] == 1]
        first_pitch_strikes = sum(first_pitches['description'].isin(strike_descriptions))
        first_pitch_strike_pct = first_pitch_strikes / len(first_pitches) if len(first_pitches) > 0 else 0
        
        # Swinging strike percentage
        swinging_strikes = sum(group['description'].isin(swinging_strike_descriptions))
        swinging_strike_pct = swinging_strikes / total_pitches
        
        # Two strike count performance
        two_strike_counts = group[(group['balls'] <= 3) & (group['strikes'] == 2)]
        two_strike_pitches = len(two_strike_counts)
        
        # K% in two strike counts
        two_strike_Ks = sum(two_strike_counts['events'] == 'strikeout')
        two_strike_K_pct = two_strike_Ks / two_strike_pitches if two_strike_pitches > 0 else 0
        
        # Called Strike to Ball Ratio
        called_strikes = sum(group['description'] == 'called_strike')
        balls = sum(group['description'] == 'ball')
        called_strike_to_ball_ratio = called_strikes / balls if balls > 0 else 0
        
        # Zone percentage (percentage of pitches in the strike zone)
        # Zones 1-9 represent the strike zone
        in_zone_pitches = sum(group['zone'].between(1, 9, inclusive='both'))
        zone_pct = in_zone_pitches / total_pitches
        
        # Edge percentage (zones at the edge of the strike zone)
        edge_zones = [1, 3, 7, 9]  # Corner zones
        edge_pitches = sum(group['zone'].isin(edge_zones))
        edge_pct = edge_pitches / total_pitches
        
        # Add features to results
        zone_features.append({
            'pitcher_id': pitcher_id,
            'game_date': game_date,
            'game_pk': game_pk,
            'strike_pct': strike_pct,
            'first_pitch_strike_pct': first_pitch_strike_pct,
            'swinging_strike_pct': swinging_strike_pct,
            'two_strike_K_pct': two_strike_K_pct,
            'called_strike_to_ball_ratio': called_strike_to_ball_ratio,
            'zone_pct': zone_pct,
            'edge_pct': edge_pct
        })
    
    # Convert to DataFrame
    zone_df = pd.DataFrame(zone_features) if zone_features else pd.DataFrame(
        columns=['pitcher_id', 'game_date', 'game_pk', 'strike_pct', 'first_pitch_strike_pct', 
                'swinging_strike_pct', 'two_strike_K_pct', 'called_strike_to_ball_ratio', 
                'zone_pct', 'edge_pct'])
    
    # Merge with game_data
    enhanced_game_data = pd.merge(
        game_data,
        zone_df,
        on=['pitcher_id', 'game_date', 'game_pk'],
        how='left'
    )
    
    return enhanced_game_data

def calculate_rolling_features(game_data, window_sizes=None):
    """
    Calculate rolling features based on past games (trailing averages).
    
    Args:
        game_data (pandas.DataFrame): Game-level data
        window_sizes (list): List of window sizes for rolling calculations
        
    Returns:
        pandas.DataFrame: Enhanced game data with rolling features
    """
    if window_sizes is None:
        window_sizes = StrikeoutModelConfig.WINDOW_SIZES
    
    # Ensure data is sorted by pitcher and date
    game_data = game_data.sort_values(['pitcher_id', 'game_date'])
    
    # Metrics to calculate rolling averages for
    metrics = [
        'strikeouts', 'innings_pitched', 'K_per_9', 'K_pct', 
        'release_speed_mean', 'release_spin_rate_mean',
        'strike_pct', 'first_pitch_strike_pct', 'swinging_strike_pct'
    ]
    
    # Add pitch mix metrics if they exist
    pitch_mix_cols = [col for col in game_data.columns if col.startswith('pct_')]
    metrics.extend([col for col in pitch_mix_cols if col in game_data.columns])
    
    # Create rolling features for each window size
    for window in window_sizes:
        for metric in metrics:
            if metric in game_data.columns:
                # Create rolling average (avoid future data leakage by using shift later)
                col_name = f'rolling_{window}g_{metric}'
                game_data[col_name] = game_data.groupby('pitcher_id')[metric].rolling(
                    window=window, min_periods=1).mean().reset_index(0, drop=True)
    
    # Additional rolling features
    for window in window_sizes:
        # Strikeout consistency (standard deviation)
        game_data[f'rolling_{window}g_K_std'] = game_data.groupby('pitcher_id')['strikeouts'].rolling(
            window=window, min_periods=2).std().reset_index(0, drop=True)
        
        # Fill NaN values (when there's only one game in window)
        game_data[f'rolling_{window}g_K_std'] = game_data[f'rolling_{window}g_K_std'].fillna(0)
        
        # Velocity trend (is velocity increasing or decreasing?)
        if 'release_speed_mean' in game_data.columns:
            game_data[f'rolling_{window}g_velocity_trend'] = game_data.groupby('pitcher_id')['release_speed_mean'].apply(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda vals: 1 if (np.diff(vals) > 0).sum() > len(vals)/2 else
                               -1 if (np.diff(vals) < 0).sum() > len(vals)/2 else 0,
                    raw=True
                )
            ).reset_index(0, drop=True)
            
            # Fill NaN values
            game_data[f'rolling_{window}g_velocity_trend'] = game_data[f'rolling_{window}g_velocity_trend'].fillna(0)
    
    return game_data

def create_days_rest_features(game_data):
    """
    Create features related to days of rest between starts.
    
    Args:
        game_data (pandas.DataFrame): Game-level data
        
    Returns:
        pandas.DataFrame: Enhanced game data with rest features
    """
    # Convert game_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(game_data['game_date']):
        game_data['game_date'] = pd.to_datetime(game_data['game_date'])
    
    # Ensure data is sorted by pitcher and date
    game_data = game_data.sort_values(['pitcher_id', 'game_date'])
    
    # Calculate days since last start
    game_data['days_since_last_game'] = game_data.groupby('pitcher_id')['game_date'].diff().dt.days
    
    # Fill NA for first game of each pitcher
    game_data['days_since_last_game'] = game_data['days_since_last_game'].fillna(5)  # Assume 5 days for first start
    
    # Create categorical rest day features
    game_data['rest_days_4_less'] = (game_data['days_since_last_game'] <= 4).astype(int)
    game_data['rest_days_5'] = (game_data['days_since_last_game'] == 5).astype(int)
    game_data['rest_days_6_more'] = (game_data['days_since_last_game'] >= 6).astype(int)
    
    return game_data

def create_handedness_features(game_data):
    """
    Create features related to pitcher handedness.
    
    Args:
        game_data (pandas.DataFrame): Game-level data
        
    Returns:
        pandas.DataFrame: Enhanced game data with handedness features
    """
    # Simple feature based on pitcher's throwing hand
    game_data['throws_right'] = (game_data['p_throws'] == 'R').astype(int)
    return game_data

def create_home_away_features(game_data):
    """
    Create features related to home/away status.
    
    Args:
        game_data (pandas.DataFrame): Game-level data
        
    Returns:
        pandas.DataFrame: Enhanced game data with home/away features
    """
    # Determine if home or away pitcher
    for idx, row in game_data.iterrows():
        player_team = None
        # Use p_throws as a proxy for team identification
        if 'p_throws' in row and pd.notna(row['p_throws']):
            if row['home_team'] == row['p_throws']:
                player_team = 'home'
            elif row['away_team'] == row['p_throws']:
                player_team = 'away'
            
        game_data.at[idx, 'is_home'] = 1 if player_team == 'home' else 0
        
    # Get opponent team
    game_data['opponent_team'] = np.where(
        game_data['is_home'] == 1,
        game_data['away_team'],
        game_data['home_team']
    )
    
    return game_data

def combine_features(game_data):
    """
    Combine features to create interaction terms.
    
    Args:
        game_data (pandas.DataFrame): Game-level data with engineered features
        
    Returns:
        pandas.DataFrame: Enhanced game data with interaction features
    """
    # Create combinations of important features
    
    # Velocity + Movement
    if all(col in game_data.columns for col in ['release_speed_mean', 'pfx_z_mean']):
        game_data['velocity_movement_z'] = game_data['release_speed_mean'] * game_data['pfx_z_mean']
    
    # Velocity + Spin
    if all(col in game_data.columns for col in ['release_speed_mean', 'release_spin_rate_mean']):
        game_data['velocity_spin'] = game_data['release_speed_mean'] * game_data['release_spin_rate_mean'] / 1000
    
    # Swinging strike % + Breaking ball %
    if all(col in game_data.columns for col in ['swinging_strike_pct', 'pct_breaking']):
        game_data['swinging_breaking_interaction'] = game_data['swinging_strike_pct'] * game_data['pct_breaking']
    
    # Rest days + Velocity
    if all(col in game_data.columns for col in ['days_since_last_game', 'release_speed_mean']):
        game_data['rest_velocity_interaction'] = (game_data['days_since_last_game'] >= 5).astype(int) * game_data['release_speed_mean']
    
    return game_data

def prepare_final_features(game_data):
    """
    Prepare final feature set by shifting features to avoid data leakage.
    
    Args:
        game_data (pandas.DataFrame): Game-level data with all features
        
    Returns:
        pandas.DataFrame: Final features dataset ready for modeling
    """
    # Ensure data is sorted by pitcher and date
    game_data = game_data.sort_values(['pitcher_id', 'game_date'])
    
    # Identify target column and ID columns
    target_col = 'strikeouts'
    id_cols = ['pitcher_id', 'game_date', 'game_pk', 'player_name', 'season', 'home_team', 'away_team']
    
    # Identify feature columns (all columns except target and IDs)
    feature_cols = [col for col in game_data.columns if col not in id_cols + [target_col]]
    
    # Create a copy of the dataset
    final_data = game_data.copy()
    
    # Shift features by pitcher to avoid data leakage
    # Each prediction will use only data available before the game
    for col in feature_cols:
        final_data[col] = final_data.groupby('pitcher_id')[col].shift(1)
    
    # Drop rows with NA features (first game for each pitcher)
    # Only check a few critical features to determine if we should drop
    critical_features = [col for col in feature_cols if 'rolling' in col][:3]
    if critical_features:
        final_data = final_data.dropna(subset=critical_features)
    
    # Replace remaining NAs with appropriate values
    final_data = final_data.fillna({
        # Fill numeric columns with 0 or appropriate defaults
        col: 0 for col in final_data.select_dtypes(include=['number']).columns
        if col not in id_cols + [target_col]
    })
    
    # For categorical columns, fill with most common value
    for col in final_data.select_dtypes(include=['object']).columns:
        if col not in id_cols and col in final_data.columns:
            final_data[col] = final_data[col].fillna(final_data[col].mode()[0] if not final_data[col].mode().empty else 'Unknown')
    
    return final_data

def store_predictive_features(features_df):
    """
    Store predictive features in the database.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with predictive features
        
    Returns:
        bool: Success status
    """
    try:
        with DBConnection() as conn:
            # Store the features
            features_df.to_sql('predictive_pitch_features', conn, if_exists='replace', index=False)
            
        logger.info(f"Successfully stored {len(features_df)} records to predictive_pitch_features")
        return True
        
    except Exception as e:
        logger.error(f"Error storing predictive features: {e}")
        return False

def process_pitch_data_batch(batch_df):
    """
    Process a batch of pitch data to extract game-level features.
    
    Args:
        batch_df (pandas.DataFrame): Batch of pitch data
        
    Returns:
        pandas.DataFrame: Processed game-level features
    """
    # Extract game-level data
    game_data = extract_game_data(batch_df)
    
    # Process features
    game_data = extract_pitch_features(batch_df, game_data)
    game_data = calculate_pitch_mix_features(batch_df, game_data)
    game_data = extract_strike_zone_features(batch_df, game_data)
    game_data = create_home_away_features(game_data)
    
    return game_data

def create_predictive_features(limit=None, seasons=None, batch_size=None):
    """
    Create predictive features from pitch data.
    
    Args:
        limit (int): Optional limit on rows to process
        seasons (list): Optional list of seasons to include
        batch_size (int): Optional batch size for processing large datasets
        
    Returns:
        pandas.DataFrame: Predictive features at the game level
    """
    # Load data with batching if needed
    data = load_pitch_data(limit=limit, seasons=seasons, batch_size=batch_size)
    
    # If batch processing is enabled
    if batch_size is not None and not isinstance(data, pd.DataFrame):
        all_game_data = []
        
        # Process each batch
        for batch_df in data:
            # Extract basic game features
            batch_game_data = process_pitch_data_batch(batch_df)
            all_game_data.append(batch_game_data)
        
        # Combine all batches
        game_data = pd.concat(all_game_data, ignore_index=True)
        
        # Deduplicate in case there are overlapping games
        game_data = game_data.drop_duplicates(subset=['pitcher_id', 'game_date', 'game_pk'])
        
    else:
        # Process all data at once
        game_data = process_pitch_data_batch(data)
    
    # Calculate additional features requiring all data
    game_data = calculate_rolling_features(game_data)
    game_data = create_days_rest_features(game_data)
    game_data = create_handedness_features(game_data)
    game_data = combine_features(game_data)
    
    # Prepare final feature set
    final_features = prepare_final_features(game_data)
    
    return final_features

def main(limit=None, seasons=None, batch_size=None):
    """
    Main function to create and store predictive features.
    
    Args:
        limit (int): Optional limit on rows to process
        seasons (list): Optional list of seasons to include
        batch_size (int): Optional batch size for processing large datasets
        
    Returns:
        bool: Success status
    """
    try:
        logger.info("Starting feature engineering process")
        
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = DBConfig.BATCH_SIZE
        
        # Create predictive features
        features_df = create_predictive_features(limit=limit, seasons=seasons, batch_size=batch_size)
        
        # Store features
        success = store_predictive_features(features_df)
        
        if success:
            logger.info("Feature engineering completed successfully")
        else:
            logger.error("Feature engineering failed during storage")
            
        return success
        
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create predictive features for strikeout prediction')
    parser.add_argument('--limit', type=int, help='Limit on number of rows to process')
    parser.add_argument('--seasons', nargs='+', type=int, help='Seasons to include')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing')
    
    args = parser.parse_args()
    
    main(limit=args.limit, seasons=args.seasons, batch_size=args.batch_size)