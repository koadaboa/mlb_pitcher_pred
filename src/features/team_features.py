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

def create_team_features(seasons=None):
    """Create features from team_batting table
    
    Args:
        seasons (tuple, optional): Specific seasons to use
        
    Returns:
        pandas.DataFrame: Team features
    """
    try:
        # Load team batting data
        df = load_team_batting_data()
        
        # Filter to specified seasons if provided
        if seasons is not None:
            df = df[df['Season'].isin(seasons)]
            logger.info(f"Filtered team batting data to seasons {seasons}: {len(df)} rows")
        
        # Create team-level features dataframe
        team_features = []
        
        # Check if Season column exists
        if 'Season' not in df.columns:
            logger.error("Season column missing from team_batting table")
            return pd.DataFrame()
        
        # Get unique seasons and teams
        all_seasons = df['Season'].unique()
        logger.info(f"Processing {len(all_seasons)} seasons of team data")
        
        for season in all_seasons:
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

def create_combined_features(pitcher_df=None, team_df=None, dataset_type="all"):
    """Combine pitcher, batter, and team features into a single dataset
    
    Args:
        pitcher_df (pandas.DataFrame, optional): Pitcher features
        team_df (pandas.DataFrame, optional): Team features
        dataset_type (str): Type of dataset ("train", "test", or "all")
        
    Returns:
        pandas.DataFrame: Combined features dataset
    """
    logger.info(f"Creating combined features for {dataset_type} dataset...")
    
    try:
        # Load data if not provided
        if pitcher_df is None or team_df is None:
            with DBConnection() as conn:
                # Load pitcher features - use appropriate table
                pitcher_table = "predictive_pitch_features"
                if dataset_type == "train":
                    pitcher_table = "train_predictive_pitch_features"
                elif dataset_type == "test":
                    pitcher_table = "test_predictive_pitch_features"
                
                pitcher_query = f"SELECT * FROM {pitcher_table}"
                pitcher_df = pd.read_sql_query(pitcher_query, conn)
                
                # Load team features
                team_query = "SELECT * FROM team_season_features"
                team_df = pd.read_sql_query(team_query, conn)
        
        # Convert dates
        pitcher_df['game_date'] = pd.to_datetime(pitcher_df['game_date'])
        
        # Create container for combined features
        combined_features = []
        
        # Process each pitcher game
        for _, row in pitcher_df.iterrows():
            game_pk = row['game_pk']
            game_date = row['game_date']
            pitcher_team = row['home_team']  # Simplified assumption
            opposing_team = row['away_team'] if pitcher_team == row['home_team'] else row['home_team']
            season = row['season']
            
            # *** CRITICAL FIX: Use only team-level seasonal data from PREVIOUS seasons ***
            # Get opposing team features from previous seasons only
            team_feats = team_df[(team_df['team'] == opposing_team) & 
                                (team_df['season'] < season)]
                                
            # Combine all features
            combined_row = row.to_dict()
            
            # Add team features from previous seasons only
            if not team_feats.empty:
                # Use most recent previous season
                latest_season = team_feats['season'].max()
                latest_data = team_feats[team_feats['season'] == latest_season].iloc[0]
                
                for col in team_feats.columns:
                    if col not in ['team', 'season']:
                        combined_row[f'opp_{col}'] = latest_data[col]
            else:
                # No previous season data, use league averages
                logger.info(f"No previous season data for team {opposing_team}, using defaults")
                combined_row['opp_team_k_percent'] = 0.22  # Default K rate
                combined_row['opp_team_swstr_percent'] = 0.10  # Default SwStr rate
                combined_row['opp_team_contact_percent'] = 0.77  # Default contact rate
            
            combined_features.append(combined_row)
        
        # Convert to DataFrame
        combined_df = pd.DataFrame(combined_features)
        
        # Handle missing values
        for col in combined_df.select_dtypes(include=['float64', 'int64']).columns:
            if combined_df[col].isnull().sum() > 0:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        
        logger.info(f"Created combined dataset with {len(combined_df)} rows and {len(combined_df.columns)} columns")
        
        # Store to database with appropriate table name
        table_name = "combined_predictive_features"
        if dataset_type == "train":
            table_name = "train_combined_features"
        elif dataset_type == "test":
            table_name = "test_combined_features"
        
        with DBConnection() as conn:
            combined_df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(combined_df)} combined feature records to {table_name}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error creating combined features: {str(e)}")
        return pd.DataFrame()