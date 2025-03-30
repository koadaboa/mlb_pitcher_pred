# Feature engineering functions for pitcher strikeout prediction model
import pandas as pd
import numpy as np
from pathlib import Path
import pybaseball
from src.data.db import get_db_connection, get_pitcher_data
from config import StrikeoutModelConfig
from src.data.utils import setup_logger

logger = setup_logger(__name__)

def create_prediction_features(force_refresh=False):
    """
    Create and store prediction features for strikeout prediction in the database
    
    Args:
        force_refresh (bool): Whether to force refresh existing features
    """
    # Check if we need to refresh the data
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM prediction_features")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count > 0 and not force_refresh:
        logger.info("Prediction features table already populated and force_refresh is False. Skipping.")
        return
    
    logger.info("Creating prediction features...")
    
    # Get the data from database
    df = get_pitcher_data()
    
    if df.empty:
        logger.warning("No data available for feature engineering.")
        return
    
    # Apply enhanced feature engineering
    enhanced_df = create_enhanced_features(df)
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing features if force_refresh
    if force_refresh:
        cursor.execute("DELETE FROM prediction_features")
        conn.commit()
    
    # Insert into database
    features_inserted = 0
    
    for _, row in enhanced_df.iterrows():
        try:
            # Check if this game already has features
            cursor.execute(
                "SELECT id FROM prediction_features WHERE pitcher_id = ? AND game_id = ?",
                (row['pitcher_id'], row['game_id'])
            )
            existing = cursor.fetchone()
            
            # List of standard features we want to store
            std_features = [
                'last_3_games_strikeouts_avg', 'last_5_games_strikeouts_avg',
                'last_3_games_velo_avg', 'last_5_games_velo_avg',
                'last_3_games_swinging_strike_pct_avg', 'last_5_games_swinging_strike_pct_avg',
                'last_3_games_called_strike_pct_avg', 'last_5_games_called_strike_pct_avg', 
                'last_3_games_zone_rate_avg', 'last_5_games_zone_rate_avg',
                'days_rest', 'team_changed'
            ]
            
            # List of new features we're adding
            new_features = [
                'last_3_games_strikeouts_std', 'last_5_games_strikeouts_std',
                'last_3_games_velo_std', 'last_5_games_velo_std',
                'last_3_games_swinging_strike_pct_std', 'last_5_games_swinging_strike_pct_std',
                'trend_3_strikeouts', 'trend_5_strikeouts',
                'trend_3_release_speed_mean', 'trend_5_release_speed_mean',
                'trend_3_swinging_strike_pct', 'trend_5_swinging_strike_pct',
                'momentum_3_strikeouts', 'momentum_5_strikeouts', 
                'momentum_3_release_speed_mean', 'momentum_5_release_speed_mean',
                'momentum_3_swinging_strike_pct', 'momentum_5_swinging_strike_pct',
                'pitch_entropy', 'prev_game_pitch_entropy'
            ]
            
            # Get values for standard features
            std_values = [row.get(feature, 0) for feature in std_features]
            
            # Get values for new features
            new_values = [row.get(feature, 0) for feature in new_features]
            
            # Combine all feature values
            all_features = std_features + new_features
            all_values = std_values + new_values
            
            if existing:
                # Update existing record
                set_clause = ", ".join([f"{feat} = ?" for feat in all_features])
                sql = f"UPDATE prediction_features SET {set_clause} WHERE id = ?"
                cursor.execute(sql, all_values + [existing[0]])
            else:
                # Insert new record
                columns = ["pitcher_id", "game_id", "game_date", "season"] + all_features
                placeholders = ", ".join(["?"] * len(columns))
                
                values = [
                    row['pitcher_id'],
                    row['game_id'],
                    row['game_date'].strftime('%Y-%m-%d'),
                    row['season']
                ] + all_values
                
                sql = f"INSERT INTO prediction_features ({', '.join(columns)}) VALUES ({placeholders})"
                cursor.execute(sql, values)
            
            # Handle pitch mix features in a separate table
            prev_pitch_mix_cols = [col for col in row.index if col.startswith('prev_game_pitch_pct_')]
            if prev_pitch_mix_cols:
                # Create or update pitch mix features
                feature_id = existing[0] if existing else cursor.lastrowid
                
                # Delete existing pitch mix features for this prediction feature
                cursor.execute("DELETE FROM pitch_mix_features WHERE prediction_feature_id = ?", (feature_id,))
                
                # Insert new pitch mix features
                for col in prev_pitch_mix_cols:
                    pitch_type = col.replace('prev_game_pitch_pct_', '')
                    percentage = row[col]
                    
                    if percentage > 0:
                        cursor.execute(
                            "INSERT INTO pitch_mix_features (prediction_feature_id, pitch_type, percentage) VALUES (?, ?, ?)",
                            (feature_id, pitch_type, percentage)
                        )
            
            features_inserted += 1
            
            # Commit periodically to avoid large transactions
            if features_inserted % 1000 == 0:
                conn.commit()
                logger.info(f"Processed {features_inserted} features so far...")
                
        except Exception as e:
            logger.error(f"Error inserting features for game {row['game_id']}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"Stored prediction features for {features_inserted} game records.")

def create_enhanced_features(df):
    """
    Create enhanced prediction features including rolling window statistics,
    trends, momentum indicators, volatility metrics, and pitch mix features.
    
    Args:
        df (pandas.DataFrame): DataFrame with pitcher data
        
    Returns:
        pandas.DataFrame: DataFrame with enhanced features
    """
    logger.info("Creating enhanced prediction features...")
    
    if df.empty:
        logger.warning("No data available for feature engineering.")
        return pd.DataFrame()
    
    # Add opponent features
    df = add_opponent_features(df)
    
    # Process each pitcher separately
    features = []
    
    for pitcher_id, pitcher_data in df.groupby('pitcher_id'):
        # Sort by game date
        pitcher_data = pitcher_data.sort_values('game_date')
        
        # Add feature groups sequentially
        pitcher_data = _add_rolling_window_features(pitcher_data)
        pitcher_data = _add_trend_features(pitcher_data)
        pitcher_data = _add_momentum_features(pitcher_data)
        pitcher_data = _add_volatility_features(pitcher_data)
        pitcher_data = _add_pitch_mix_features(pitcher_data)
        pitcher_data = _add_rest_day_features(pitcher_data)
        
        # Add to features dataset
        features.append(pitcher_data)
    
    # Combine all pitcher features
    if features:
        all_features = pd.concat(features, ignore_index=True)
        all_features = all_features.fillna(0)
        logger.info(f"Created enhanced features for {len(all_features)} game records")
        return all_features
    else:
        logger.warning("No features created.")
        return pd.DataFrame()

def _add_rolling_window_features(pitcher_data):
    """
    Add rolling window statistics (mean, std) for various metrics
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with rolling window features added
    """
    # Base metrics to calculate rolling window stats for
    mean_metrics = [
        'strikeouts', 'release_speed_mean', 'swinging_strike_pct', 
        'called_strike_pct', 'zone_rate'
    ]
    
    # Metrics that also get standard deviation calculations
    std_metrics = [
        'strikeouts', 'release_speed_mean', 'swinging_strike_pct'
    ]
    
    # Calculate for each window size defined in config
    for window in StrikeoutModelConfig.WINDOW_SIZES:
        # Calculate mean for all metrics
        for metric in mean_metrics:
            if metric in pitcher_data.columns:
                # Average over window (shifted by 1 to avoid data leakage)
                pitcher_data[f'last_{window}_games_{metric}_avg'] = pitcher_data[metric].rolling(
                    window=window, min_periods=1).mean().shift(1)
        
        # Calculate standard deviation for selected metrics
        for metric in std_metrics:
            if metric in pitcher_data.columns:
                # Standard deviation (shifted by 1 to avoid data leakage)
                pitcher_data[f'last_{window}_games_{metric}_std'] = pitcher_data[metric].rolling(
                    window=window, min_periods=2).std().shift(1)
    
    return pitcher_data

def _add_trend_features(pitcher_data):
    """
    Add trend indicators showing directional changes over time
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with trend features added
    """
    trend_metrics = [
        'strikeouts', 'release_speed_mean', 'swinging_strike_pct'
    ]
    
    for window in StrikeoutModelConfig.WINDOW_SIZES:
        for metric in trend_metrics:
            if metric in pitcher_data.columns:
                # Recent window average
                recent_avg = pitcher_data[metric].rolling(window=window, min_periods=1).mean().shift(1)
                # Previous window average (shifted by window+1 to get the window before the recent one)
                prev_avg = pitcher_data[metric].rolling(window=window, min_periods=1).mean().shift(window+1)
                # Trend = recent - previous
                pitcher_data[f'trend_{window}_{metric}'] = recent_avg - prev_avg
                
        # Add increasing streak indicator (percentage of recent games with increase)
        if 'strikeouts' in pitcher_data.columns:
            pitcher_data[f'increasing_so_streak_{window}'] = (
                pitcher_data['strikeouts'].rolling(window=window, min_periods=2)
                .apply(lambda x: (np.diff(x) > 0).sum() / (len(x)-1) * 100)
                .shift(1)
            )
    
    return pitcher_data

def _add_momentum_features(pitcher_data):
    """
    Add momentum indicators using weighted averages (recent games more important)
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with momentum features added
    """
    momentum_metrics = [
        'strikeouts', 'release_speed_mean', 'swinging_strike_pct'
    ]
    
    for window in StrikeoutModelConfig.WINDOW_SIZES:
        # Define weights (more recent games have higher weights)
        weights = np.arange(1, window+1)
        
        for metric in momentum_metrics:
            if metric in pitcher_data.columns:
                # Apply weighted average
                pitcher_data[f'momentum_{window}_{metric}'] = pitcher_data[metric].rolling(
                    window=window, min_periods=1).apply(
                        lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):]), raw=True
                    ).shift(1)
    
    return pitcher_data

def _add_volatility_features(pitcher_data):
    """
    Add volatility metrics to measure consistency
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with volatility features added
    """
    for window in StrikeoutModelConfig.WINDOW_SIZES:
        # Coefficient of variation for strikeouts
        std_col = f'last_{window}_games_strikeouts_std' 
        avg_col = f'last_{window}_games_strikeouts_avg'
        
        if all(col in pitcher_data.columns for col in [std_col, avg_col]):
            pitcher_data[f'strikeout_volatility_{window}'] = (
                pitcher_data[std_col] / 
                pitcher_data[avg_col].clip(lower=0.1)  # Avoid division by zero
            ).shift(1)
        
        # Recovery pattern after poor performance
        if all(col in pitcher_data.columns for col in ['strikeouts', avg_col]):
            pitcher_data[f'post_poor_performance_{window}'] = (
                pitcher_data['strikeouts'].shift(1) < 
                pitcher_data[avg_col].shift(1)
            ).astype(int)
    
    return pitcher_data

def _add_pitch_mix_features(pitcher_data):
    """
    Add pitch mix features including diversity measures
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with pitch mix features added
    """
    # Identify pitch mix columns
    pitch_mix_cols = [col for col in pitcher_data.columns if col.startswith('pitch_pct_')]
    
    if not pitch_mix_cols:
        return pitcher_data
    
    # Add previous game pitch mix
    for col in pitch_mix_cols:
        pitcher_data[f'prev_game_{col}'] = pitcher_data[col].shift(1)
    
    # Calculate pitch entropy (diversity measure)
    def calc_entropy(row):
        # Get non-zero percentages and convert to probabilities
        probs = [row[col]/100 for col in pitch_mix_cols if row[col] > 0]
        if not probs:
            return 0
        # Calculate entropy: -sum(p * log2(p))
        return -sum(p * np.log2(p) for p in probs)
    
    pitcher_data['pitch_entropy'] = pitcher_data.apply(calc_entropy, axis=1)
    pitcher_data['prev_game_pitch_entropy'] = pitcher_data['pitch_entropy'].shift(1)
    
    # Calculate pitch mix similarity (how much pitch selection changed)
    def pitch_similarity(current, previous):
        if pd.isna(previous).any() or pd.isna(current).any():
            return np.nan
        
        current_vec = [current[col] for col in pitch_mix_cols]
        previous_vec = [previous[col] for col in pitch_mix_cols]
        
        # If all zeros, return 1 (perfect similarity)
        if sum(current_vec) == 0 or sum(previous_vec) == 0:
            return 1.0
            
        return 1.0 - cosine(current_vec, previous_vec)
    
    # Calculate similarity for each row with the previous row
    pitch_similarities = []
    
    for i in range(len(pitcher_data)):
        if i == 0:
            pitch_similarities.append(np.nan)
        else:
            current = pitcher_data.iloc[i]
            previous = pitcher_data.iloc[i-1]
            pitch_similarities.append(pitch_similarity(current, previous))
    
    pitcher_data['pitch_mix_similarity'] = pitch_similarities
    
    return pitcher_data

def _add_rest_day_features(pitcher_data):
    """
    Add features related to rest days between appearances
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with rest day features added
    """
    # Calculate days of rest
    pitcher_data['prev_game_date'] = pitcher_data['game_date'].shift(1)
    pitcher_data['days_rest'] = (pitcher_data['game_date'] - pitcher_data['prev_game_date']).dt.days
    pitcher_data['days_rest'] = pitcher_data['days_rest'].fillna(5)  # Default to 5 days for first appearance
    
    # Create team changed flag (placeholder - actual logic would need team info)
    pitcher_data['team_changed'] = 0
    
    return pitcher_data

# Add to create_enhanced_features in src/features/engineering.py
def add_opponent_features(df):
    """Add opponent team-specific features"""
    if 'opponent_team_id' not in df.columns:
        logger.warning("No opponent_team_id column found for opponent features")
        return df
    
    # Get unique combinations of opponent and season
    opponent_seasons = df[['opponent_team_id', 'season']].drop_duplicates()
    
    # Fetch opponent team stats
    try:
        team_stats = pybaseball.team_batting_stats(min(df['season']), max(df['season']))
        
        # Process team stats by season
        team_stats_by_season = {}
        for season in df['season'].unique():
            season_stats = team_stats[team_stats['Season'] == season]
            # Create lookup dictionary
            team_stats_by_season[season] = season_stats.set_index('Team').to_dict('index')
        
        # Add opponent features
        df['opponent_strikeout_rate'] = df.apply(
            lambda row: team_stats_by_season.get(row['season'], {}).get(
                row['opponent_team_id'], {}).get('K%', 0),
            axis=1
        )
        
        df['opponent_walk_rate'] = df.apply(
            lambda row: team_stats_by_season.get(row['season'], {}).get(
                row['opponent_team_id'], {}).get('BB%', 0),
            axis=1
        )
        
        df['opponent_ops'] = df.apply(
            lambda row: team_stats_by_season.get(row['season'], {}).get(
                row['opponent_team_id'], {}).get('OPS', 0),
            axis=1
        )
        
        # Calculate normalized features (how this opponent compares to average)
        for season in df['season'].unique():
            season_mask = df['season'] == season
            season_avg_k = df.loc[season_mask, 'opponent_strikeout_rate'].mean()
            if season_avg_k > 0:
                df.loc[season_mask, 'opponent_k_vs_avg'] = (
                    df.loc[season_mask, 'opponent_strikeout_rate'] / season_avg_k
                )
        
        logger.info(f"Added opponent features for {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error adding opponent features: {e}")
    
    return df