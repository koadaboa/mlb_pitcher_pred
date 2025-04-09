# src/features/team_features.py
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection

# Setup logger
logger = setup_logger('team_features')

def load_team_batting_data():
    """Load team batting data from database"""
    logger.info("Loading team batting data...")
    with DBConnection() as conn:
        query = "SELECT * FROM team_batting"
        df = pd.read_sql_query(query, conn)
    
    logger.info(f"Loaded {len(df)} rows of team batting data")
    return df

def create_team_features():
    """Create features from team_batting table"""
    try:
        # Load team batting data
        df = load_team_batting_data()
        
        # Create team-level features dataframe
        team_features = []
        
        # Check if Season column exists
        if 'Season' not in df.columns:
            logger.error("Season column missing from team_batting table")
            return pd.DataFrame()
        
        # Get unique seasons and teams
        seasons = df['Season'].unique()
        logger.info(f"Processing {len(seasons)} seasons of team data")
        
        for season in seasons:
            season_data = df[df['Season'] == season]
            
            for _, row in season_data.iterrows():
                team = row['Team'] if 'Team' in row else None
                if not team:
                    continue
                
                # Extract strikeout-related features
                features = {
                    'team': team,
                    'season': season,
                    'team_k_percent': row['K%'] if 'K%' in row else None,
                    'team_bb_k_ratio': row['BB/K'] if 'BB/K' in row else None,
                    'team_zone_percent': row['Zone%'] if 'Zone%' in row else None,
                    'team_o_swing_percent': row['O-Swing%'] if 'O-Swing%' in row else None,
                    'team_z_contact_percent': row['Z-Contact%'] if 'Z-Contact%' in row else None,
                    'team_contact_percent': row['Contact%'] if 'Contact%' in row else None,
                    'team_swstr_percent': row['SwStr%'] if 'SwStr%' in row else None,
                }
                
                # Add pitch type vulnerability metrics if available
                pitch_types = ['wFB/C', 'wSL/C', 'wCT/C', 'wCB/C', 'wCH/C', 'wFS/C', 'wSF/C']
                for pitch in pitch_types:
                    if pitch in row:
                        clean_name = pitch.replace('/', '_').lower()
                        features[f'team_{clean_name}'] = row[pitch]
                
                team_features.append(features)
        
        # Convert to DataFrame
        team_features_df = pd.DataFrame(team_features)
        
        if not team_features_df.empty:
            logger.info(f"Created {len(team_features_df)} team feature records")
            
            # Store to database
            with DBConnection() as conn:
                team_features_df.to_sql('team_season_features', conn, if_exists='replace', index=False)
                logger.info(f"Stored {len(team_features_df)} team features to database")
        else:
            logger.warning("No team features created")
        
        return team_features_df
        
    except Exception as e:
        logger.error(f"Error creating team features: {str(e)}")
        return pd.DataFrame()

def create_combined_features():
    """Combine pitcher, batter, and team features into a single dataset"""
    logger.info("Creating combined features...")
    
    try:
        with DBConnection() as conn:
            # Load pitcher features
            pitcher_query = "SELECT * FROM predictive_pitch_features"
            pitcher_df = pd.read_sql_query(pitcher_query, conn)
            
            # Load team features
            team_query = "SELECT * FROM team_season_features"
            team_df = pd.read_sql_query(team_query, conn)
            
            # Load game-level batters to get expected lineup
            batters_query = """
            SELECT 
                game_pk, game_date, batter_id, player_name, swinging_strike_pct, 
                strikeouts, total_pitches, stand
            FROM game_level_batters
            """
            batters_df = pd.read_sql_query(batters_query, conn)
            
            # Load batter predictive features
            batter_query = """
            SELECT 
                batter_id, game_date, game_pk,
                rolling_5g_k_pct, rolling_5g_swstr_pct, 
                rolling_5g_chase_pct, rolling_5g_zone_contact_pct
            FROM batter_predictive_features
            """
            batter_features_df = pd.read_sql_query(batter_query, conn)
        
        # Convert dates
        pitcher_df['game_date'] = pd.to_datetime(pitcher_df['game_date'])
        batters_df['game_date'] = pd.to_datetime(batters_df['game_date'])
        batter_features_df['game_date'] = pd.to_datetime(batter_features_df['game_date'])
        
        # First create team-level batter metrics for each game
        game_lineup_stats = []
        
        for game_pk in pitcher_df['game_pk'].unique():
            # Get batters for this game
            game_batters = batters_df[batters_df['game_pk'] == game_pk]
            
            if game_batters.empty:
                continue
                
            # Get batter features for these batters
            batter_stats = batter_features_df[
                (batter_features_df['batter_id'].isin(game_batters['batter_id'])) & 
                (batter_features_df['game_pk'] == game_pk)
            ]
            
            if batter_stats.empty:
                continue
                
            # Calculate lineup-level stats
            game_date = game_batters['game_date'].iloc[0]
            home_team = pitcher_df[pitcher_df['game_pk'] == game_pk]['home_team'].iloc[0]
            away_team = pitcher_df[pitcher_df['game_pk'] == game_pk]['away_team'].iloc[0]
            
            # Calculate aggregated stats
            avg_k_pct = batter_stats['rolling_5g_k_pct'].mean()
            avg_swstr_pct = batter_stats['rolling_5g_swstr_pct'].mean()
            avg_chase_pct = batter_stats['rolling_5g_chase_pct'].mean()
            avg_zone_contact = batter_stats['rolling_5g_zone_contact_pct'].mean()
            
            # Count handedness
            right_batters = sum(game_batters['stand'] == 'R')
            left_batters = sum(game_batters['stand'] == 'L')
            
            # Store lineup stats for both teams
            for team in [home_team, away_team]:
                lineup_stats = {
                    'game_pk': game_pk,
                    'game_date': game_date,
                    'team': team,
                    'lineup_avg_k_pct': avg_k_pct,
                    'lineup_avg_swstr_pct': avg_swstr_pct,
                    'lineup_avg_chase_pct': avg_chase_pct,
                    'lineup_avg_zone_contact': avg_zone_contact,
                    'lineup_right_handed': right_batters,
                    'lineup_left_handed': left_batters,
                    'lineup_handedness_ratio': right_batters / (left_batters + 0.001)
                }
                game_lineup_stats.append(lineup_stats)
                
        # Convert to DataFrame
        lineup_df = pd.DataFrame(game_lineup_stats)
        
        # Now join with pitcher data
        # For each game, determine the opposing team and get their lineup stats
        combined_features = []
        
        for _, row in pitcher_df.iterrows():
            game_pk = row['game_pk']
            pitcher_team = row['home_team']  # Simplified assumption
            opposing_team = row['away_team'] if pitcher_team == row['home_team'] else row['home_team']
            season = row['season']
            
            # Get opposing team features
            team_feats = team_df[(team_df['team'] == opposing_team) & 
                                (team_df['season'] == season)]
            
            # Get lineup features
            lineup_feats = lineup_df[(lineup_df['game_pk'] == game_pk) & 
                                    (lineup_df['team'] == opposing_team)]
            
            # Combine all features
            combined_row = row.to_dict()
            
            # Add team features
            if not team_feats.empty:
                for col in team_feats.columns:
                    if col not in ['team', 'season']:
                        combined_row[f'opp_{col}'] = team_feats.iloc[0][col]
            
            # Add lineup features
            if not lineup_feats.empty:
                for col in lineup_feats.columns:
                    if col not in ['game_pk', 'game_date', 'team']:
                        combined_row[f'opp_{col}'] = lineup_feats.iloc[0][col]
            
            combined_features.append(combined_row)
        
        # Convert to DataFrame
        combined_df = pd.DataFrame(combined_features)
        
        # Handle missing values
        for col in combined_df.select_dtypes(include=['float64', 'int64']).columns:
            if combined_df[col].isnull().sum() > 0:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        
        logger.info(f"Created combined dataset with {len(combined_df)} rows and {len(combined_df.columns)} columns")
        
        # Store to database
        with DBConnection() as conn:
            combined_df.to_sql('combined_predictive_features', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(combined_df)} combined feature records to database")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error creating combined features: {str(e)}")
        return pd.DataFrame()