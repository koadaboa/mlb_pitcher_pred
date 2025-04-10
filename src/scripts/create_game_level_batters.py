#   src/scripts/create_game_level_batters.py
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import time

#   Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from config import DataConfig

#   Setup logger
logger = setup_logger('create_game_level_batters')

def aggregate_batters_to_game_level():
    """
    Aggregate raw statcast_batters data to game level statistics
    similar to game_level_pitchers using vectorized operations
    """
    start_time = time.time()
    logger.info("Loading statcast batter data...")

    with DBConnection() as conn:
        #   Load data in chunks to avoid memory issues
        chunk_size = 500000
        chunks = []

        #   Get total count for progress tracking
        count_query = "SELECT COUNT(*) FROM statcast_batters"
        total_rows = pd.read_sql_query(count_query, conn).iloc[0, 0]
        logger.info(f"Processing {total_rows} rows in chunks of {chunk_size}")

        #   Process in chunks
        offset = 0
        while True:
            query = f"""
            SELECT 
                batter, player_name, game_date, game_pk, 
                home_team, away_team, stand, p_throws,
                events, description, pitch_type, zone,
                release_speed, season
            FROM statcast_batters
            LIMIT {chunk_size} OFFSET {offset}
            """
            chunk = pd.read_sql_query(query, conn)

            if chunk.empty:
                break

            chunks.append(chunk)
            offset += chunk_size
            logger.info(f"Loaded chunk {len(chunks)}: {offset}/{total_rows} rows")

        #   Combine chunks
        df = pd.concat(chunks, ignore_index=True)

    logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    logger.info("Aggregating to game level using vectorized operations...")

    #   Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])

    #   Add computed columns
    df['is_strikeout'] = (df['events'] == 'strikeout').astype(int)
    df['is_swinging_strike'] = (df['description'] == 'swinging_strike').astype(int)
    df['is_called_strike'] = (df['description'] == 'called_strike').astype(int)
    df['is_in_zone'] = (df['zone'].between(1, 9)).astype(int)
    df['is_contact'] = df['description'].str.contains('hit|foul', case=False, na=False).astype(int)
    df['is_swing'] = df['description'].str.contains('swing', case=False, na=False).astype(int)
    df['is_chase'] = ((~df['zone'].between(1, 9)) & 
                      df['description'].str.contains('swing', case=False, na=False)).astype(int)

    #   Add pitch type indicators
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'FT', 'FC', 'SI']).astype(int)
    df['is_breaking'] = df['pitch_type'].isin(['SL', 'CU', 'KC']).astype(int)
    df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS', 'KN']).astype(int)

    #   Add handedness indicators
    df['vs_rhp'] = (df['p_throws'] == 'R').astype(int)
    df['vs_lhp'] = (df['p_throws'] == 'L').astype(int)

    #   Define aggregation functions
    agg_funcs = {
        'player_name': 'first',
        'stand': 'first',
        'home_team': 'first',
        'away_team': 'first',
        'season': 'first',
        'is_strikeout': 'sum',
        'is_swinging_strike': 'sum',
        'is_called_strike': 'sum',
        'is_in_zone': 'sum',
        'is_contact': 'sum',
        'is_swing': 'sum',
        'is_chase': 'sum',
        'is_fastball': 'sum',
        'is_breaking': 'sum',
        'is_offspeed': 'sum',
        'vs_rhp': 'sum',
        'vs_lhp': 'sum'
    }

    #   Group by batter and game
    logger.info("Performing groupby aggregation...")
    grouped = df.groupby(['batter', 'game_pk', 'game_date']).agg(agg_funcs).reset_index()

    #   Rename columns to match our schema
    grouped.rename(columns={
        'batter': 'batter_id',
        'is_strikeout': 'strikeouts',
        'is_swinging_strike': 'swinging_strikes',
        'is_called_strike': 'called_strikes',
    }, inplace=True)

    #   Calculate total pitches
    grouped['total_pitches'] = df.groupby(['batter', 'game_pk']).size().reset_index(name='count')['count']

    #   Calculate derived metrics
    grouped['swinging_strike_pct'] = grouped['swinging_strikes'] / grouped['total_pitches']
    grouped['called_strike_pct'] = grouped['called_strikes'] / grouped['total_pitches']

    #   Safe division for zone_contact_pct
    safe_zone = grouped['is_in_zone'].copy()
    safe_zone[safe_zone == 0] = float('inf')  #   Replace zeros with infinity
    grouped['zone_contact_pct'] = grouped['is_contact'] / safe_zone
    grouped['zone_contact_pct'] = grouped['zone_contact_pct'].replace(float('inf'), 0)  #   Replace infinity with 0

    #   Safe division for chase_pct
    safe_denominator = (grouped['total_pitches'] - grouped['is_in_zone']).copy()
    safe_denominator[safe_denominator <= 0] = float('inf')
    grouped['chase_pct'] = grouped['is_chase'] / safe_denominator
    grouped['chase_pct'] = grouped['chase_pct'].replace(float('inf'), 0)

    #   Calculate pitch type whiff rates
    grouped['fastball_whiff_pct'] = grouped.apply(
        lambda x: x['swinging_strikes'] * (x['is_fastball'] > 0) / x['is_fastball'] 
        if x['is_fastball'] > 0 else 0, axis=1
    )

    grouped['breaking_whiff_pct'] = grouped.apply(
        lambda x: x['swinging_strikes'] * (x['is_breaking'] > 0) / x['is_breaking'] 
        if x['is_breaking'] > 0 else 0, axis=1
    )

    grouped['offspeed_whiff_pct'] = grouped.apply(
        lambda x: x['swinging_strikes'] * (x['is_offspeed'] > 0) / x['is_offspeed'] 
        if x['is_offspeed'] > 0 else 0, axis=1
    )

    #   Calculate handedness stats
    grouped['strikeouts_vs_rhp'] = grouped.apply(
        lambda x: x['strikeouts'] * (x['vs_rhp'] > 0) / x['vs_rhp'] 
        if x['vs_rhp'] > 0 else 0, axis=1
    )

    grouped['strikeouts_vs_lhp'] = grouped.apply(
        lambda x: x['strikeouts'] * (x['vs_lhp'] > 0) / x['vs_lhp'] 
        if x['vs_lhp'] > 0 else 0, axis=1
    )

    #   Clean up intermediate columns
    columns_to_drop = ['is_in_zone', 'is_contact', 'is_swing', 'is_chase', 
                       'is_fastball', 'is_breaking', 'is_offspeed',
                       'vs_rhp', 'vs_lhp']
    result_df = grouped.drop(columns=columns_to_drop)

    #   Store to database
    with DBConnection() as conn:
        result_df.to_sql('game_level_batters', conn, if_exists='replace', index=False)
        logger.info(f"Stored {len(result_df)} game-level batter records to database")

    total_time = time.time() - start_time
    logger.info(f"Aggregation completed in {total_time:.2f} seconds")

    return result_df

if __name__ == "__main__":
    logger.info("Starting batter data aggregation...")
    aggregate_batters_to_game_level()
    logger.info("Batter data aggregation completed")