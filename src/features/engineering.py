# Feature engineering functions for pitcher prediction models
import pandas as pd
import numpy as np
import logging
import sqlite3
from pathlib import Path

from src.data.db import get_db_connection, get_pitcher_data

logger = logging.getLogger(__name__)

def create_prediction_features(force_refresh=False):
    """
    Create and store prediction features in the database
    
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
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing features if force_refresh
    if force_refresh:
        cursor.execute("DELETE FROM prediction_features")
        conn.commit()
    
    # Create a dataframe to store the features
    features = []
    
    # Process each pitcher separately
    for pitcher_id, pitcher_data in df.groupby('pitcher_id'):
        # Sort by game date
        pitcher_data = pitcher_data.sort_values('game_date')
        
        for window in [3, 5]:
            # Use shift to create lagged features properly - this prevents data leakage
            # by only using past data for each observation
            pitcher_data[f'last_{window}_games_strikeouts_avg'] = pitcher_data['strikeouts'].rolling(
                window=window, min_periods=1).mean().shift(1)

            pitcher_data[f'last_{window}_games_outs_avg'] = pitcher_data['outs'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_k9_avg'] = pitcher_data['k_per_9'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_era_avg'] = pitcher_data['era'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_fip_avg'] = pitcher_data['fip'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_velo_avg'] = pitcher_data['release_speed_mean'].rolling(
                window=window, min_periods=1).mean().shift(1)
            
            pitcher_data[f'last_{window}_games_swinging_strike_pct_avg'] = pitcher_data['swinging_strike_pct'].rolling(
                window=window, min_periods=1).mean().shift(1)
        
        # Calculate days of rest
        pitcher_data['prev_game_date'] = pitcher_data['game_date'].shift(1)
        pitcher_data['days_rest'] = (pitcher_data['game_date'] - pitcher_data['prev_game_date']).dt.days
        pitcher_data['days_rest'] = pitcher_data['days_rest'].fillna(5)  # Default to 5 days for first appearance
        
        # Create team changed flag
        pitcher_data['team_changed'] = pitcher_data['team'].shift(1) != pitcher_data['team']
        pitcher_data['team_changed'] = pitcher_data['team_changed'].fillna(False).astype(int)
        
        # Add to features dataset
        features.append(pitcher_data)
    
    # Combine all pitcher features
    if features:
        all_features = pd.concat(features, ignore_index=True)
        
        # Fill NA values
        all_features = all_features.fillna(0)
        
        # Insert into database
        features_inserted = 0
        
        for _, row in all_features.iterrows():
            try:
                # Check if this game already has features
                cursor.execute(
                    "SELECT id FROM prediction_features WHERE pitcher_id = ? AND game_id = ?",
                    (row['pitcher_id'], row['game_id'])
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute('''
                        UPDATE prediction_features
                        SET 
                            last_3_games_strikeouts_avg = ?,
                            last_5_games_strikeouts_avg = ?,
                            last_3_games_k9_avg = ?,
                            last_5_games_k9_avg = ?,
                            last_3_games_era_avg = ?,
                            last_5_games_era_avg = ?,
                            last_3_games_fip_avg = ?,
                            last_5_games_fip_avg = ?,
                            last_3_games_velo_avg = ?,
                            last_5_games_velo_avg = ?,
                            last_3_games_swinging_strike_pct_avg = ?,
                            last_5_games_swinging_strike_pct_avg = ?,
                            days_rest = ?,
                            team_changed = ?
                        WHERE id = ?
                    ''', (
                        row['last_3_games_strikeouts_avg'],
                        row['last_5_games_strikeouts_avg'],
                        row['last_3_games_k9_avg'],
                        row['last_5_games_k9_avg'],
                        row['last_3_games_era_avg'],
                        row['last_5_games_era_avg'],
                        row['last_3_games_fip_avg'],
                        row['last_5_games_fip_avg'],
                        row['last_3_games_velo_avg'],
                        row['last_5_games_velo_avg'],
                        row['last_3_games_swinging_strike_pct_avg'],
                        row['last_5_games_swinging_strike_pct_avg'],
                        row['days_rest'],
                        row['team_changed'],
                        existing[0]
                    ))
                else:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO prediction_features (
                            pitcher_id, game_id, game_date, season,
                            last_3_games_strikeouts_avg, last_5_games_strikeouts_avg,
                            last_3_games_k9_avg, last_5_games_k9_avg,
                            last_3_games_era_avg, last_5_games_era_avg,
                            last_3_games_fip_avg, last_5_games_fip_avg,
                            last_3_games_velo_avg, last_5_games_velo_avg,
                            last_3_games_swinging_strike_pct_avg, last_5_games_swinging_strike_pct_avg,
                            days_rest, team_changed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['pitcher_id'],
                        row['game_id'],
                        row['game_date'].strftime('%Y-%m-%d'),
                        row['season'],
                        row['last_3_games_strikeouts_avg'],
                        row['last_5_games_strikeouts_avg'],
                        row['last_3_games_k9_avg'],
                        row['last_5_games_k9_avg'],
                        row['last_3_games_era_avg'],
                        row['last_5_games_era_avg'],
                        row['last_3_games_fip_avg'],
                        row['last_5_games_fip_avg'],
                        row['last_3_games_velo_avg'],
                        row['last_5_games_velo_avg'],
                        row['last_3_games_swinging_strike_pct_avg'],
                        row['last_5_games_swinging_strike_pct_avg'],
                        row['days_rest'],
                        row['team_changed']
                    ))
                features_inserted += 1
            except Exception as e:
                logger.error(f"Error inserting features for game {row['game_id']}: {e}")
                continue
        
        conn.commit()
        
        logger.info(f"Stored prediction features for {features_inserted} game records.")
    else:
        logger.warning("No features created.")
    
    conn.close()