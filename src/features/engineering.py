# Feature engineering functions for pitcher strikeout prediction model
import pandas as pd
import numpy as np
import datetime
import sqlite3
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
    
    # Check for existing columns
    cursor.execute("PRAGMA table_info(prediction_features)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    # Find opponent-specific columns we need to add
    opponent_columns = [col for col in enhanced_df.columns if col.startswith('so_vs_')]
    new_opponent_columns = [col for col in opponent_columns if col not in existing_columns]
    
    # Add any new opponent-specific columns
    for col in new_opponent_columns:
        try:
            cursor.execute(f"ALTER TABLE prediction_features ADD COLUMN {col} REAL")
            logger.info(f"Added opponent-specific column {col} to prediction_features table")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add column {col}: {e}")
    
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
            
            # Get all available columns excluding database system columns
            feature_cols = [col for col in row.index if col in existing_columns]
            
            # Convert values to SQLite-compatible types
            feature_vals = []
            for col in feature_cols:
                val = row.get(col)
                
                # Convert pandas Timestamp to string
                if isinstance(val, pd.Timestamp):
                    val = val.strftime('%Y-%m-%d')
                # Convert numpy types to Python types
                elif isinstance(val, np.integer):
                    val = int(val)
                elif isinstance(val, np.floating):
                    val = float(val)
                elif isinstance(val, np.bool_):
                    val = bool(val)
                elif val is None or pd.isna(val):
                    val = None
                
                feature_vals.append(val)
            
            # Skip if no features to update
            if not feature_cols:
                continue
            
            # Handle game_date properly
            game_date = row.get('game_date')
            if isinstance(game_date, (pd.Timestamp, datetime.datetime)):
                game_date_str = game_date.strftime('%Y-%m-%d')
            else:
                # If not a datetime, convert to string directly
                game_date_str = str(game_date)
            
            if existing:
                # Update existing record
                set_clause = ", ".join([f"{col} = ?" for col in feature_cols])
                sql = f"UPDATE prediction_features SET {set_clause} WHERE id = ?"
                cursor.execute(sql, feature_vals + [existing[0]])
            else:
                # Insert new record
                columns = ["pitcher_id", "game_id", "game_date", "season"] + feature_cols
                placeholders = ", ".join(["?"] * len(columns))
                
                # Convert core values to proper types
                pitcher_id = int(row['pitcher_id']) if not pd.isna(row['pitcher_id']) else None
                game_id = str(row['game_id']) if not pd.isna(row['game_id']) else None
                season = int(row['season']) if not pd.isna(row['season']) else None
                
                values = [
                    pitcher_id,
                    game_id,
                    game_date_str,
                    season
                ] + feature_vals
                
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
                    percentage = float(row[col]) if not pd.isna(row[col]) else 0.0
                    
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
            logger.error(f"Error inserting features for game {row.get('game_id', 0)}: {e}")
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
        pitcher_data = pitcher_data.sort_values('game_date').reset_index(drop=True)
        
        # Add feature groups sequentially
        pitcher_data = _add_rolling_window_features(pitcher_data)
        pitcher_data = _add_trend_features(pitcher_data)
        pitcher_data = _add_momentum_features(pitcher_data)
        pitcher_data = _add_volatility_features(pitcher_data)
        pitcher_data = _add_pitch_mix_features(pitcher_data)
        pitcher_data = _add_rest_day_features(pitcher_data)
        
        # Add new enhanced features
        pitcher_data = _add_pitcher_specific_baselines(pitcher_data)
        pitcher_data = _add_enhanced_matchup_features(pitcher_data)
        
        # Add to features dataset
        features.append(pitcher_data)
    
    # Combine all pitcher features with reset index to avoid issues
    if features:
        # Make sure to reset indices to avoid duplicates
        for i in range(len(features)):
            features[i] = features[i].reset_index(drop=True)
            
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

def _add_pitcher_specific_baselines(pitcher_data):
    """
    Add pitcher-specific baseline features to capture individual tendencies
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with pitcher-specific baseline features added
    """
    # Ensure we have a clean index
    pitcher_data = pitcher_data.reset_index(drop=True)
    
    # Ensure game_date is a datetime object
    if 'game_date' in pitcher_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(pitcher_data['game_date']):
            try:
                pitcher_data['game_date'] = pd.to_datetime(pitcher_data['game_date'])
            except:
                logger.warning("Could not convert game_date to datetime, baseline features may be affected")
    
    # Calculate career statistics for this pitcher (before current game)
    career_stats = []
    
    for i in range(len(pitcher_data)):
        current_date = pitcher_data.iloc[i]['game_date']
        
        # Handle non-datetime objects
        if not isinstance(current_date, pd.Timestamp) and not isinstance(current_date, datetime.datetime):
            # Use row index as a proxy for chronological order if date is invalid
            prior_games = pitcher_data.iloc[:i]
        else:
            prior_games = pitcher_data[pitcher_data['game_date'] < current_date]
        
        if prior_games.empty:
            # No prior games, use zeros
            career_stats.append({
                'career_so_avg': 0,
                'career_so_std': 0,
                'career_so_per_batter': 0,
                'career_so_consistency': 0
            })
        else:
            # Calculate career averages before current game
            career_so_avg = prior_games['strikeouts'].mean()
            career_so_std = prior_games['strikeouts'].std() if len(prior_games) > 1 else 0
            
            # Calculate more advanced metrics if data is available
            if 'pitch_count' in prior_games.columns:
                batters_faced = prior_games['pitch_count'] / 3.8  # Approximate batters faced
                so_per_batter = prior_games['strikeouts'] / batters_faced
                career_so_per_batter = so_per_batter.mean()
            else:
                career_so_per_batter = 0
            
            # Calculate consistency score (lower std relative to mean = more consistent)
            if career_so_avg > 0:
                career_so_consistency = 1 - (career_so_std / (career_so_avg + 1))
            else:
                career_so_consistency = 0
            
            career_stats.append({
                'career_so_avg': career_so_avg,
                'career_so_std': career_so_std,
                'career_so_per_batter': career_so_per_batter,
                'career_so_consistency': career_so_consistency
            })
    
    # Create a new DataFrame for all the new columns
    new_columns = pd.DataFrame(career_stats, index=range(len(pitcher_data)))
    
    # Calculate deviation from expected strikeouts
    # This shows how a pitcher performs relative to their own baseline
    if len(pitcher_data) > 1:
        # Shift by 1 to avoid data leakage
        new_columns['prev_so_deviation'] = (
            (pitcher_data['strikeouts'] - new_columns['career_so_avg']) / 
            (new_columns['career_so_std'].clip(lower=1))
        ).shift(1)
        
        # Rolling average of deviations (captures hot/cold streaks)
        for window in [3, 5]:
            new_columns[f'so_deviation_{window}g_avg'] = (
                new_columns['prev_so_deviation']
                .rolling(window=window, min_periods=1)
                .mean()
            )
    
    # Add home/away splits
    if 'home_team' in pitcher_data.columns and 'statcast_id' in pitcher_data.columns:
        pitcher_id = pitcher_data['statcast_id'].iloc[0]
        
        # Mark home games (where pitcher's team matches home_team)
        is_home_game = pitcher_data.apply(
            lambda row: row.get('home_team', '') == pitcher_data['home_team'].mode().iloc[0], 
            axis=1
        ).astype(int)
        
        new_columns['is_home_game'] = is_home_game.values
        
        # Calculate home/away splits
        home_so_avg = pitcher_data[is_home_game == 1]['strikeouts'].mean()
        away_so_avg = pitcher_data[is_home_game == 0]['strikeouts'].mean()
        
        # Fill NaN values with overall average
        overall_avg = pitcher_data['strikeouts'].mean()
        home_so_avg = home_so_avg if not pd.isna(home_so_avg) else overall_avg
        away_so_avg = away_so_avg if not pd.isna(away_so_avg) else overall_avg
        
        # Add home/away expectations
        new_columns['home_away_so_exp'] = is_home_game.apply(
            lambda x: home_so_avg if x == 1 else away_so_avg
        ).values
    
    # Join with original data - create a new DataFrame to avoid issues
    result = pitcher_data.copy()
    for col in new_columns.columns:
        result[col] = new_columns[col].values
    
    return result

def _add_enhanced_matchup_features(pitcher_data):
    """
    Add enhanced matchup-specific features to capture pitcher vs. opponent dynamics
    
    Args:
        pitcher_data (pandas.DataFrame): Data for a single pitcher
        
    Returns:
        pandas.DataFrame: Data with enhanced matchup features added
    """
    # Ensure we have a clean index
    pitcher_data = pitcher_data.reset_index(drop=True)
    
    # Ensure game_date is a datetime object
    if 'game_date' in pitcher_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(pitcher_data['game_date']):
            try:
                pitcher_data['game_date'] = pd.to_datetime(pitcher_data['game_date'])
            except:
                logger.warning("Could not convert game_date to datetime, matchup features may be affected")
                # Sort by index as a fallback
                pitcher_data = pitcher_data.sort_index()
                
    # Check if opponent data is available
    if 'opponent_team_id' not in pitcher_data.columns:
        logger.warning("No opponent team information available for matchup features")
        return pitcher_data
    
    # Get unique opponents faced
    unique_opponents = pitcher_data['opponent_team_id'].unique()
    
    # Create a dictionary to store all the new columns we'll add
    new_columns = {}
    
    # Calculate historical performance vs each opponent
    for opponent in unique_opponents:
        # Skip if opponent is missing
        if pd.isna(opponent):
            continue
        
        # Create column name for this opponent
        col_name = f"so_vs_{opponent}"
        
        # For each game, calculate historical stats vs this opponent
        historical_stats = []
        
        for i in range(len(pitcher_data)):
            current_date = pitcher_data.iloc[i]['game_date']
            current_opponent = pitcher_data.iloc[i]['opponent_team_id']
            
            # Handle date comparison safely
            if isinstance(current_date, (pd.Timestamp, datetime.datetime)):
                # Use date comparison
                prior_matchups = pitcher_data[
                    (pitcher_data['game_date'] < current_date) & 
                    (pitcher_data['opponent_team_id'] == opponent)
                ]
            else:
                # Use index as proxy for chronological order
                prior_matchups = pitcher_data.iloc[:i][
                    pitcher_data.iloc[:i]['opponent_team_id'] == opponent
                ]
            
            if current_opponent == opponent:
                if prior_matchups.empty:
                    # No prior matchups with this opponent
                    historical_stats.append(0)
                else:
                    # Calculate average strikeouts vs this opponent
                    historical_stats.append(prior_matchups['strikeouts'].mean())
            else:
                # Not playing this opponent in current game
                historical_stats.append(np.nan)
        
        # Store the historical stats for this opponent
        new_columns[col_name] = historical_stats
    
    # Calculate matchup advantage score
    if 'opponent_strikeout_rate' in pitcher_data.columns:
        # Higher score = better matchup for strikeouts
        new_columns['matchup_advantage'] = (
            pitcher_data['opponent_strikeout_rate'] / 
            pitcher_data['opponent_strikeout_rate'].mean()
        ).values
    
    # Add recency-weighted matchup feature
    recency_weighted = np.full(len(pitcher_data), np.nan)
    
    for i in range(len(pitcher_data)):
        current_date = pitcher_data.iloc[i]['game_date']
        current_opponent = pitcher_data.iloc[i]['opponent_team_id']
        
        if pd.isna(current_opponent):
            continue
        
        # Handle date comparison safely
        if isinstance(current_date, (pd.Timestamp, datetime.datetime)):
            # Use date comparison
            prior_matchups = pitcher_data[
                (pitcher_data['game_date'] < current_date) & 
                (pitcher_data['opponent_team_id'] == current_opponent)
            ]
        else:
            # Use index as proxy for chronological order
            prior_matchups = pitcher_data.iloc[:i][
                pitcher_data.iloc[:i]['opponent_team_id'] == current_opponent
            ]
            
        if prior_matchups.empty:
            continue
            
        # Sort by date if possible
        if isinstance(current_date, (pd.Timestamp, datetime.datetime)):
            prior_matchups = prior_matchups.sort_values('game_date', ascending=False)
            
            # Calculate days since each matchup
            days_since = [(current_date - date).days for date in prior_matchups['game_date']]
        else:
            # Use row index difference as a proxy for recency
            days_since = [i - idx for idx in prior_matchups.index]
        
        # Apply exponential decay weights based on recency
        # More recent games have higher weight
        weights = np.exp(-np.array(days_since) / 365)  # 1-year half-life
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate weighted average
        weighted_avg = np.sum(prior_matchups['strikeouts'].values * weights)
        
        # Store weighted average
        recency_weighted[i] = weighted_avg
    
    # Store the recency weighted matchup data
    new_columns['recency_weighted_matchup'] = recency_weighted
    
    # Fill NaN values with overall average for recency weighted matchup
    overall_avg = pitcher_data['strikeouts'].mean()
    if 'recency_weighted_matchup' in new_columns:
        new_columns['recency_weighted_matchup'] = np.where(
            np.isnan(new_columns['recency_weighted_matchup']),
            overall_avg,
            new_columns['recency_weighted_matchup']
        )
    
    # Add the new columns to the original DataFrame
    result = pitcher_data.copy()
    for col, values in new_columns.items():
        if len(values) == len(result):
            result[col] = values
    
    return result

def add_opponent_features(df):
    """Add opponent team-specific features"""
    if 'opponent_team_id' not in df.columns:
        logger.warning("No opponent_team_id column found for opponent features")
        return df
    
    # Get unique combinations of opponent and season
    opponent_seasons = df[['opponent_team_id', 'season']].drop_duplicates()
    
    # Fetch opponent team stats
    try:
        team_stats = pybaseball.team_batting(min(df['season']), max(df['season']))
        
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
        
        # Add new enhanced opponent features
        df['opponent_whiff_rate'] = df.apply(
            lambda row: team_stats_by_season.get(row['season'], {}).get(
                row['opponent_team_id'], {}).get('SwStr%', 0),
            axis=1
        )
        
        df['opponent_chase_rate'] = df.apply(
            lambda row: team_stats_by_season.get(row['season'], {}).get(
                row['opponent_team_id'], {}).get('O-Swing%', 0),
            axis=1
        )
        
        df['opponent_zone_contact_rate'] = df.apply(
            lambda row: team_stats_by_season.get(row['season'], {}).get(
                row['opponent_team_id'], {}).get('Z-Contact%', 0),
            axis=1
        )
        
        # Calculate normalized features (how this opponent compares to average)
        for season in df['season'].unique():
            season_mask = df['season'] == season
            
            # Strikeout rate vs average
            season_avg_k = df.loc[season_mask, 'opponent_strikeout_rate'].mean()
            if season_avg_k > 0:
                df.loc[season_mask, 'opponent_k_vs_avg'] = (
                    df.loc[season_mask, 'opponent_strikeout_rate'] / season_avg_k
                )
            
            # Whiff rate vs average
            season_avg_whiff = df.loc[season_mask, 'opponent_whiff_rate'].mean()
            if season_avg_whiff > 0:
                df.loc[season_mask, 'opponent_whiff_vs_avg'] = (
                    df.loc[season_mask, 'opponent_whiff_rate'] / season_avg_whiff
                )
        
        logger.info(f"Added enhanced opponent features for {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error adding opponent features: {e}")
    
    return df