# src/data/aggregate_batters.py (Updated for Count Aggregation)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import time

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.config import DataConfig

logger = setup_logger('aggregate_batters')

def classify_count(balls, strikes):
    """Classifies the pitch count into categories."""
    # Ensure inputs are numeric, handle potential errors
    try:
        balls, strikes = int(balls), int(strikes)
    except (ValueError, TypeError):
        return 'unknown'

    if balls > strikes:
        return 'ahead'
    elif strikes > balls:
        return 'behind'
    elif balls == 0 and strikes == 0:
        return '0-0'
    elif balls == 3 and strikes == 2:
         return '3-2'
    else:
        return 'even'

def aggregate_batters_to_game_level():
    """
    Aggregate raw statcast_batters data to game level statistics,
    including performance splits vs pitcher handedness (LHP/RHP)
    and performance in different pitch counts.
    """
    start_time = time.time()
    logger.info("Loading statcast batter data for game-level aggregation (including count)...")

    # --- Load Data (same as before, ensure 'balls', 'strikes' are included) ---
    with DBConnection() as conn:
        chunk_size = 500000
        chunks = []
        try:
            count_query = "SELECT COUNT(*) FROM statcast_batters"
            total_rows = pd.read_sql_query(count_query, conn).iloc[0, 0]
            logger.info(f"Total rows to process from statcast_batters: {total_rows}")
        except Exception as e:
            logger.warning(f"Could not get total row count: {e}.")
            total_rows = 'Unknown'
        logger.info(f"Processing in chunks of {chunk_size}")

        columns_to_select = [
            'batter', 'player_name', 'game_date', 'game_pk',
            'home_team', 'away_team', 'stand', 'p_throws',
            'events', 'description', 'pitch_type', 'zone',
            'release_speed', 'season', 'woba_value',
            'bb_type', 'balls', 'strikes', # <<< Ensure balls/strikes are selected
            'at_bat_number', 'pitch_number'
        ]
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(statcast_batters)")
        available_columns = {info[1] for info in cursor.fetchall()}
        valid_columns_to_select = [col for col in columns_to_select if col in available_columns]
        if 'balls' not in valid_columns_to_select or 'strikes' not in valid_columns_to_select:
             logger.error("Missing 'balls' or 'strikes' column in statcast_batters. Cannot perform count aggregation.")
             return pd.DataFrame()
        logger.info(f"Selecting columns: {valid_columns_to_select}")

        offset = 0
        processed_rows = 0
        while True:
            query = f"SELECT {', '.join(valid_columns_to_select)} FROM statcast_batters LIMIT {chunk_size} OFFSET {offset}"
            try:
                chunk = pd.read_sql_query(query, conn)
            except Exception as e:
                 logger.error(f"Error querying chunk at offset {offset}: {e}")
                 break
            if chunk.empty: break
            chunks.append(chunk)
            processed_rows += len(chunk)
            offset += chunk_size
            logger.info(f"Loaded chunk {len(chunks)} ({processed_rows}/{total_rows} rows)")

        if not chunks:
             logger.error("No data loaded from statcast_batters.")
             return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        del chunks

    logger.info(f"Data loading complete. Shape: {df.shape}. Preprocessing...")

    # --- Preprocessing & Intermediate Features (Including Count Classification) ---
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df.dropna(subset=['batter', 'game_pk', 'game_date', 'p_throws', 'balls', 'strikes'], inplace=True)

    # Convert balls/strikes to numeric, coercing errors
    df['balls'] = pd.to_numeric(df['balls'], errors='coerce')
    df['strikes'] = pd.to_numeric(df['strikes'], errors='coerce')
    df.dropna(subset=['balls', 'strikes'], inplace=True) # Drop rows where conversion failed

    # Classify count *before* the current pitch
    df['count_category'] = df.apply(lambda row: classify_count(row['balls'], row['strikes']), axis=1)

    df['is_strikeout'] = df['events'].isin(['strikeout', 'strikeout_double_play']).astype(int)
    df['is_walk'] = df['events'].isin(['walk', 'hit_by_pitch']).astype(int)
    df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df['is_single'] = (df['events'] == 'single').astype(int)
    df['is_double'] = (df['events'] == 'double').astype(int)
    df['is_triple'] = (df['events'] == 'triple').astype(int)
    df['is_home_run'] = (df['events'] == 'home_run').astype(int)
    df['is_ab'] = df['events'].notna() & ~df['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf', 'intent_walk'])
    df['is_pa_end'] = df['pitch_number'] == df.groupby(['batter', 'game_pk', 'at_bat_number'])['pitch_number'].transform('max')
    df['is_sac_fly'] = (df['events'] == 'sac_fly').astype(int)

    # --- Plate Discipline & Pitch Type features (same as before) ---
    df['is_swinging_strike'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)
    df['is_called_strike'] = (df['description'] == 'called_strike').astype(int)
    df['is_in_zone'] = df['zone'].between(1, 9, inclusive='both').astype(int)
    df['is_contact'] = df['description'].str.contains('hit_into_play|foul', case=False, na=False).astype(int)
    df['is_swing'] = df['description'].str.contains('swing|hit_into_play', case=False, na=False).astype(int)
    df['is_chase'] = ((~df['zone'].between(1, 9, inclusive='both')) & df['is_swing']).astype(int)
    df['is_zone_swing'] = (df['is_in_zone'] & df['is_swing']).astype(int)
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'FT', 'FC', 'SI', 'FS']).astype(int)
    df['is_breaking'] = df['pitch_type'].isin(['SL', 'CU', 'KC', 'CS', 'KN']).astype(int)
    df['is_offspeed'] = df['pitch_type'].isin(['CH', 'SC', 'FO']).astype(int)
    df['vs_rhp'] = (df['p_throws'] == 'R').astype(int)
    df['vs_lhp'] = (df['p_throws'] == 'L').astype(int)

    # wOBA Contribution (same as before)
    woba_weights = {'walk': 0.69, 'hbp': 0.72, 'single': 0.89, 'double': 1.27, 'triple': 1.62, 'home_run': 2.10}
    if 'woba_value' in df.columns and df['woba_value'].notna().any():
         df['woba_contribution'] = df['woba_value'].fillna(0)
    else:
         logger.info("Calculating 'woba_contribution' based on events.")
         df['woba_contribution'] = 0.0
         for event, weight in woba_weights.items():
              df.loc[df['events'] == event, 'woba_contribution'] = weight
         df.loc[df['events'] == 'hit_by_pitch', 'woba_contribution'] = woba_weights['hbp'] # Explicitly add HBP


    # --- PA Outcome Aggregation (Including Count and Handedness) ---
    logger.info("Aggregating PA outcomes (overall, by count, by handedness)...")
    pa_group_cols = ['batter', 'game_pk', 'game_date', 'at_bat_number']
    # We need the count category at the *end* of the PA, or maybe the most frequent one?
    # Let's take the count category of the *last pitch* of the PA for simplicity.
    df['pa_last_pitch_count_category'] = df.loc[df['is_pa_end'], 'count_category']
    df['pa_last_pitch_count_category'] = df.groupby(pa_group_cols)['pa_last_pitch_count_category'].transform('first') # Propagate last pitch count to all pitches in PA

    pa_outcomes = df[df['is_pa_end']].groupby(pa_group_cols).agg(
        strikeouts_pa=('is_strikeout', 'max'),
        walks_pa=('is_walk', 'max'),
        hits_pa=('is_hit', 'max'),
        # ... (singles, doubles, etc. same as before)
        ab_pa=('is_ab', 'max'),
        sac_fly_pa=('is_sac_fly', 'max'),
        woba_value_pa=('woba_contribution', 'sum'),
        p_throws=('p_throws', 'first'),
        count_category_pa=('pa_last_pitch_count_category', 'first'), # Category for this PA
    ).reset_index()
    pa_outcomes['pa_count'] = 1

    # --- Game Level Aggregation (Overall) ---
    # (Same as before, aggregates totals like strikeouts, walks, woba_numerator, etc.)
    logger.info("Aggregating outcomes to game level (overall)...")
    game_group_cols = ['batter', 'game_pk', 'game_date']
    game_level_stats = pa_outcomes.groupby(game_group_cols).agg(
        total_pa=('pa_count', 'sum'),
        total_ab=('ab_pa', 'sum'),
        strikeouts=('strikeouts_pa', 'sum'),
        walks=('walks_pa', 'sum'),
        # ... hits, doubles, etc.
        sac_flies=('sac_fly_pa', 'sum'),
        woba_numerator=('woba_value_pa', 'sum'),
    ).reset_index()


    # --- Game Level Pitch Detail Aggregation ---
    # (Same as before, aggregates pitch details like swinging_strikes, chases etc.)
    logger.info("Aggregating pitch-level details to game level...")
    pitch_level_stats = df.groupby(game_group_cols).agg(
        total_pitches=('pitch_number', 'count'),
        swinging_strikes=('is_swinging_strike', 'sum'),
        called_strikes=('is_called_strike', 'sum'),
        total_swings=('is_swing', 'sum'),
        chases=('is_chase', 'sum'),
        zone_swings=('is_zone_swing', 'sum'),
        zone_pitches=('is_in_zone', 'sum'),
        contact_on_swing=('is_contact', 'sum'),
        # Pitch Type Counts & Whiffs (same as before)
        fastball_count=('is_fastball', 'sum'),
        breaking_count=('is_breaking', 'sum'),
        offspeed_count=('is_offspeed', 'sum'),
        fastball_whiffs = ('is_swinging_strike', lambda x: x[df.loc[x.index, 'is_fastball'] == 1].sum()),
        breaking_whiffs = ('is_swinging_strike', lambda x: x[df.loc[x.index, 'is_breaking'] == 1].sum()),
        offspeed_whiffs = ('is_swinging_strike', lambda x: x[df.loc[x.index, 'is_offspeed'] == 1].sum()),
        fastball_swings = ('is_swing', lambda x: x[df.loc[x.index, 'is_fastball'] == 1].sum()),
        breaking_swings = ('is_swing', lambda x: x[df.loc[x.index, 'is_breaking'] == 1].sum()),
        offspeed_swings = ('is_swing', lambda x: x[df.loc[x.index, 'is_offspeed'] == 1].sum()),
        zone_contact_swings = ('is_contact', lambda x: x[df.loc[x.index, 'is_zone_swing'] == 1].sum()),
    ).reset_index()

    # --- Merge Aggregations & Calculate Overall Rates ---
    logger.info("Merging aggregations and calculating overall rates...")
    first_vals = df.groupby(game_group_cols).agg(
        player_name=('player_name', 'first'),
        stand=('stand', 'first'),
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
        season=('season', 'first'),
    ).reset_index()
    game_data = pd.merge(game_level_stats, first_vals, on=game_group_cols, how='left')
    game_data = pd.merge(game_data, pitch_level_stats, on=game_group_cols, how='left')

    # Calculate overall rates (k_percent, woba, chase_percent, etc. - same as before)
    game_data['k_percent'] = (game_data['strikeouts'] / game_data['total_pa'].replace(0, np.nan)).fillna(0)
    game_data['walk_percent'] = (game_data['walks'] / game_data['total_pa'].replace(0, np.nan)).fillna(0)
    woba_denom = game_data['total_ab'] + game_data['walks'] + game_data['sac_flies']
    game_data['woba'] = (game_data['woba_numerator'] / woba_denom.replace(0, np.nan)).fillna(0)
    game_data['swing_percent'] = (game_data['total_swings'] / game_data['total_pitches'].replace(0, np.nan)).fillna(0)
    game_data['chase_percent'] = (game_data['chases'] / (game_data['total_pitches'] - game_data['zone_pitches']).replace(0, np.nan)).fillna(0)
    game_data['zone_swing_percent'] = (game_data['zone_swings'] / game_data['zone_pitches'].replace(0, np.nan)).fillna(0)
    game_data['contact_percent'] = (game_data['contact_on_swing'] / game_data['total_swings'].replace(0, np.nan)).fillna(0)
    game_data['zone_contact_percent'] = (game_data['zone_contact_swings'] / game_data['zone_swings'].replace(0, np.nan)).fillna(0)
    game_data['swinging_strike_percent'] = (game_data['swinging_strikes'] / game_data['total_pitches'].replace(0, np.nan)).fillna(0)
    game_data['fastball_whiff_pct'] = (game_data['fastball_whiffs'] / game_data['fastball_swings'].replace(0, np.nan)).fillna(0)
    game_data['breaking_whiff_pct'] = (game_data['breaking_whiffs'] / game_data['breaking_swings'].replace(0, np.nan)).fillna(0)
    game_data['offspeed_whiff_pct'] = (game_data['offspeed_whiffs'] / game_data['offspeed_swings'].replace(0, np.nan)).fillna(0)


    # --- Handedness Splits (Pivot) ---
    # (Same as before, pivot pa_outcomes based on p_throws)
    logger.info("Calculating performance splits vs LHP/RHP...")
    split_group_cols_hand = ['batter', 'game_pk', 'game_date', 'p_throws']
    pa_outcomes_split_hand = pa_outcomes.groupby(split_group_cols_hand).agg(
        split_pa=('pa_count', 'sum'),
        split_k=('strikeouts_pa', 'sum'),
        split_woba_num=('woba_value_pa', 'sum'),
        split_ab=('ab_pa', 'sum'),
        split_walks=('walks_pa','sum'),
        split_sf=('sac_fly_pa','sum'),
    ).reset_index()
    split_woba_denom_hand = pa_outcomes_split_hand['split_ab'] + pa_outcomes_split_hand['split_walks'] + pa_outcomes_split_hand['split_sf']
    pa_outcomes_split_hand['split_k_pct'] = (pa_outcomes_split_hand['split_k'] / pa_outcomes_split_hand['split_pa'].replace(0, np.nan)).fillna(0)
    pa_outcomes_split_hand['split_woba'] = (pa_outcomes_split_hand['split_woba_num'] / split_woba_denom_hand.replace(0, np.nan)).fillna(0)
    split_pivot_hand = pa_outcomes_split_hand.pivot_table(
        index=game_group_cols, columns='p_throws', values=['split_k_pct', 'split_woba', 'split_pa']
    )
    split_pivot_hand.columns = [f'{metric}_vs_{p_throw}' for metric, p_throw in split_pivot_hand.columns]
    split_pivot_hand.reset_index(inplace=True)
    final_game_data = pd.merge(game_data, split_pivot_hand, on=game_group_cols, how='left')
    # Fill NaNs for split columns (same as before)
    split_cols_hand = [col for col in final_game_data.columns if '_vs_L' in col or '_vs_R' in col]
    for col in split_cols_hand: final_game_data[col].fillna(0, inplace=True)


    # --- Count Splits (Pivot) ---
    logger.info("Calculating performance splits by count category...")
    split_group_cols_count = ['batter', 'game_pk', 'game_date', 'count_category_pa']
    # Aggregate PA outcomes by count category
    pa_outcomes_split_count = pa_outcomes[pa_outcomes['count_category_pa'] != 'unknown'].groupby(split_group_cols_count).agg(
        count_pa=('pa_count', 'sum'),
        count_k=('strikeouts_pa', 'sum'),
        count_woba_num=('woba_value_pa', 'sum'),
        count_ab=('ab_pa', 'sum'),
        count_walks=('walks_pa','sum'),
        count_sf=('sac_fly_pa','sum'),
    ).reset_index()
    # Calculate rates per count
    split_woba_denom_count = pa_outcomes_split_count['count_ab'] + pa_outcomes_split_count['count_walks'] + pa_outcomes_split_count['count_sf']
    pa_outcomes_split_count['count_k_pct'] = (pa_outcomes_split_count['count_k'] / pa_outcomes_split_count['count_pa'].replace(0, np.nan)).fillna(0)
    pa_outcomes_split_count['count_woba'] = (pa_outcomes_split_count['count_woba_num'] / split_woba_denom_count.replace(0, np.nan)).fillna(0)
    # Pivot
    split_pivot_count = pa_outcomes_split_count.pivot_table(
        index=game_group_cols, columns='count_category_pa', values=['count_k_pct', 'count_woba', 'count_pa']
    )
    # Flatten columns (e.g., ('count_k_pct', 'ahead') -> 'k_pct_ahead')
    split_pivot_count.columns = [f'{metric.replace("count_", "")}_{category}' for metric, category in split_pivot_count.columns]
    split_pivot_count.reset_index(inplace=True)
    # Merge count splits
    final_game_data = pd.merge(final_game_data, split_pivot_count, on=game_group_cols, how='left')
    # Fill NaNs for count split columns
    split_cols_count = [col for col in final_game_data.columns if any(cat in col for cat in ['_ahead', '_behind', '_even', '_0-0', '_3-2'])]
    for col in split_cols_count: final_game_data[col].fillna(0, inplace=True)


    # --- Final Cleanup and Save ---
    final_game_data.rename(columns={'batter': 'batter_id'}, inplace=True)
    logger.info(f"Aggregation complete. Final shape: {final_game_data.shape}")
    # logger.info(f"Sample columns: {final_game_data.columns.tolist()}") # Log all columns if needed

    with DBConnection() as conn:
        final_game_data.to_sql('game_level_batters', conn, if_exists='replace', index=False)
        logger.info(f"Stored {len(final_game_data)} aggregated game-level batter records (incl. splits/counts) to 'game_level_batters'.")

    total_time = time.time() - start_time
    logger.info(f"Batter aggregation completed in {total_time:.2f} seconds")
    return final_game_data

if __name__ == "__main__":
    logger.info("Starting batter data aggregation process (incl. counts)...")
    aggregate_batters_to_game_level()
    logger.info("Batter data aggregation process finished.")