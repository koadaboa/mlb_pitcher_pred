# src/data/aggregate_batters.py (Rewritten - Reset Index Merge Fix)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import time
import traceback

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.data.utils import setup_logger, DBConnection
    from src.config import DataConfig, DBConfig
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed import in aggregate_batters: {e}")
    MODULE_IMPORTS_OK = False
    # Dummy definitions... (same as before)
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, p=None): self.p=p or "dummy.db"; self.conn = None
        def __enter__(self): import sqlite3; print("WARN: Dummy DB"); self.conn = sqlite3.connect(self.p); return self.conn
        def __exit__(self,t,v,tb):
             if self.conn: self.conn.close()
    class DataConfig: CHUNK_SIZE=500000
    class DBConfig: PATH="data/pitcher_stats.db"

logger = setup_logger('aggregate_batters') if MODULE_IMPORTS_OK else logging.getLogger('aggregate_batters_fallback')

def classify_count(balls, strikes):
    # (No changes needed)
    try: balls, strikes = int(balls), int(strikes)
    except (ValueError, TypeError): return 'unknown'
    if balls > strikes: return 'ahead'
    elif strikes > balls: return 'behind'
    elif balls == 0 and strikes == 0: return '0-0'
    elif balls == 3 and strikes == 2: return '3-2'
    else: return 'even'

def aggregate_batters_to_game_level():
    """
    Aggregate raw statcast_batters data to game level statistics.
    Returns:
        bool: True if aggregation and saving were successful, False otherwise.
    """
    start_time = time.time()
    if not MODULE_IMPORTS_OK: logger.error("Exiting: Module imports failed."); return False
    db_path = project_root / DBConfig.PATH
    logger.info("Loading statcast batter data for game-level aggregation (including count)...")

    # --- Load Data ---
    try:
        with DBConnection(db_path) as conn:
            # (Data loading logic - same as previous version)
            if conn is None: raise ConnectionError("DB Connection failed.")
            chunk_size = DataConfig.CHUNK_SIZE or 500000
            chunks = []
            try:
                count_query = "SELECT COUNT(*) FROM statcast_batters"
                total_rows = pd.read_sql_query(count_query, conn).iloc[0, 0]
                logger.info(f"Total rows to process from statcast_batters: {total_rows}")
            except Exception as e: logger.warning(f"Could not get total row count: {e}."); total_rows = 'Unknown'
            logger.info(f"Processing in chunks of {chunk_size}")
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(statcast_batters)")
            available_columns = {info[1] for info in cursor.fetchall()}
            columns_to_select = [
                'batter', 'player_name', 'game_date', 'game_pk', 'home_team', 'away_team',
                'stand', 'p_throws', 'events', 'description', 'pitch_type', 'zone',
                'release_speed', 'season', 'woba_value', 'bb_type', 'balls', 'strikes',
                'at_bat_number', 'pitch_number']
            valid_columns_to_select = [col for col in columns_to_select if col in available_columns]
            if not all(c in valid_columns_to_select for c in ['balls', 'strikes', 'batter', 'game_pk', 'game_date', 'p_throws', 'at_bat_number']):
                 logger.error(f"Missing essential columns... Need: 'balls', 'strikes', 'batter', 'game_pk', 'game_date', 'p_throws', 'at_bat_number'. Found: {valid_columns_to_select}"); return False
            logger.info(f"Selecting columns: {valid_columns_to_select}")
            query_base = f"SELECT {', '.join(valid_columns_to_select)} FROM statcast_batters"
            sql_iterator = pd.read_sql_query(query_base, conn, chunksize=chunk_size)
            for i, chunk in enumerate(sql_iterator):
                if chunk.empty: break
                chunks.append(chunk)
                if i % 10 == 0 or chunk_size < 10000: # Log more often for small chunks
                    processed_rows = sum(len(c) for c in chunks)
                    logger.info(f"Loaded chunk {i+1} ({processed_rows}/{total_rows} rows)")
            if not chunks: logger.error("No data loaded."); return False
            processed_rows = sum(len(c) for c in chunks)
            logger.info(f"Finished loading {i+1} chunks ({processed_rows}/{total_rows} rows)")
            df = pd.concat(chunks, ignore_index=True); del chunks
    except Exception as e: logger.error(f"Error loading data: {e}", exc_info=True); logger.error(traceback.format_exc()); return False

    logger.info(f"Data loading complete. Shape: {df.shape}. Preprocessing...")

    # --- Preprocessing & Intermediate Features ---
    try:
        # (Preprocessing logic - same as previous version)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        # Ensure key columns are appropriate types before groupby
        df['batter'] = pd.to_numeric(df['batter'], errors='coerce').astype('Int64')
        df['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce').astype('Int64')
        df.dropna(subset=['batter', 'game_pk', 'game_date', 'p_throws', 'balls', 'strikes'], inplace=True)
        df['balls'] = pd.to_numeric(df['balls'], errors='coerce')
        df['strikes'] = pd.to_numeric(df['strikes'], errors='coerce')
        df.dropna(subset=['balls', 'strikes'], inplace=True)
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
        df['is_swinging_strike'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)
        df['is_called_strike'] = (df['description'] == 'called_strike').astype(int)
        df['is_in_zone'] = df['zone'].between(1, 9, inclusive='both').astype(int) if 'zone' in df.columns else 0
        df['is_contact'] = df['description'].str.contains('hit_into_play|foul', case=False, na=False).astype(int)
        df['is_swing'] = df['description'].str.contains('swing|hit_into_play', case=False, na=False).astype(int)
        df['is_chase'] = ((~df['zone'].between(1, 9, inclusive='both')) & df['is_swing']).astype(int) if 'zone' in df.columns else 0
        df['is_zone_swing'] = (df['is_in_zone'] & df['is_swing']).astype(int)
        df['is_fastball'] = df['pitch_type'].isin(['FF', 'FT', 'FC', 'SI', 'FS']).astype(int) if 'pitch_type' in df.columns else 0
        df['is_breaking'] = df['pitch_type'].isin(['SL', 'CU', 'KC', 'CS', 'KN']).astype(int) if 'pitch_type' in df.columns else 0
        df['is_offspeed'] = df['pitch_type'].isin(['CH', 'SC', 'FO']).astype(int) if 'pitch_type' in df.columns else 0
        df['vs_rhp'] = (df['p_throws'] == 'R').astype(int)
        df['vs_lhp'] = (df['p_throws'] == 'L').astype(int)
        woba_weights = {'walk': 0.69, 'hbp': 0.72, 'single': 0.89, 'double': 1.27, 'triple': 1.62, 'home_run': 2.10}
        if 'woba_value' in df.columns and df['woba_value'].notna().any(): df['woba_contribution'] = df['woba_value'].fillna(0)
        else:
             logger.info("Calculating 'woba_contribution' based on events.")
             df['woba_contribution'] = 0.0
             for event, weight in woba_weights.items(): df.loc[df['events'].str.contains(event, case=False, na=False), 'woba_contribution'] = weight
             df.loc[df['events'] == 'hit_by_pitch', 'woba_contribution'] = woba_weights['hbp']

        # --- PA Outcome Aggregation ---
        logger.info("Aggregating PA outcomes (overall, by count, by handedness)...")
        # (PA aggregation logic - same as previous version)
        pa_group_cols = ['batter', 'game_pk', 'game_date', 'at_bat_number']
        df['pa_last_pitch_count_category'] = df.loc[df['is_pa_end'], 'count_category']
        df['pa_last_pitch_count_category'] = df.groupby(pa_group_cols)['pa_last_pitch_count_category'].transform('first')
        pa_outcomes = df[df['is_pa_end']].groupby(pa_group_cols).agg(
            strikeouts_pa=('is_strikeout', 'max'), walks_pa=('is_walk', 'max'), hits_pa=('is_hit', 'max'),
            singles_pa=('is_single', 'max'), doubles_pa=('is_double', 'max'), triples_pa=('is_triple', 'max'),
            homeruns_pa=('is_home_run', 'max'), ab_pa=('is_ab', 'max'), sac_fly_pa=('is_sac_fly', 'max'),
            woba_value_pa=('woba_contribution', 'sum'), p_throws=('p_throws', 'first'),
            count_category_pa=('pa_last_pitch_count_category', 'first'),
        ).reset_index(); pa_outcomes['pa_count'] = 1

        # --- Game Level Aggregation (Overall) ---
        logger.info("Aggregating outcomes to game level (overall)...")
        # (Game level stats aggregation - same as previous version)
        game_group_cols = ['batter', 'game_pk', 'game_date']
        # Ensure types before grouping
        pa_outcomes['batter'] = pd.to_numeric(pa_outcomes['batter'], errors='coerce').astype('Int64')
        pa_outcomes['game_pk'] = pd.to_numeric(pa_outcomes['game_pk'], errors='coerce').astype('Int64')
        pa_outcomes['game_date'] = pd.to_datetime(pa_outcomes['game_date']).dt.date # Use date object for consistency
        game_level_stats = pa_outcomes.groupby(game_group_cols).agg(
            total_pa=('pa_count', 'sum'), total_ab=('ab_pa', 'sum'), strikeouts=('strikeouts_pa', 'sum'),
            walks=('walks_pa', 'sum'), hits=('hits_pa', 'sum'), singles=('singles_pa', 'sum'),
            doubles=('doubles_pa', 'sum'), triples=('triples_pa', 'sum'), homeruns=('homeruns_pa', 'sum'),
            sac_flies=('sac_fly_pa', 'sum'), woba_numerator=('woba_value_pa', 'sum'),
        ).reset_index()

        # --- Game Level Pitch Detail Aggregation ---
        logger.info("Aggregating pitch-level details to game level...")
        # Ensure types before grouping
        df['game_date'] = pd.to_datetime(df['game_date']).dt.date # Match type
        pitch_level_agg_dict = {
            'total_pitches': ('pitch_number', 'count'), 'swinging_strikes': ('is_swinging_strike', 'sum'),
            'called_strikes': ('is_called_strike', 'sum'), 'total_swings': ('is_swing', 'sum'),
            'chases': ('is_chase', 'sum'), 'zone_swings': ('is_zone_swing', 'sum'),
            'zone_pitches': ('is_in_zone', 'sum'), 'contact_on_swing': ('is_contact', 'sum'),
            'fastball_count': ('is_fastball', 'sum'), 'breaking_count': ('is_breaking', 'sum'),
            'offspeed_count': ('is_offspeed', 'sum'),
        }
        pitch_level_stats = df.groupby(game_group_cols).agg(**pitch_level_agg_dict).reset_index()


        # *** FIX START: Calculate pitch type swings/whiffs separately, reset index, then MERGE ***
        logger.info("Calculating pitch type swing/whiff details...")
        pitch_type_swings_grouped = df[df['is_swing'] == 1].groupby(game_group_cols)['pitch_type'].value_counts().unstack(fill_value=0)
        pitch_type_whiffs_grouped = df[df['is_swinging_strike'] == 1].groupby(game_group_cols)['pitch_type'].value_counts().unstack(fill_value=0)
        zone_contact_swings_grouped = df[(df['is_zone_swing'] == 1) & (df['is_contact'] == 1)].groupby(game_group_cols).size()

        # --- Calculate sums into a new DataFrame ---
        swing_whiff_calcs = pd.DataFrame(index=pitch_type_swings_grouped.index) # Use existing index initially

        fb_types = ['FF', 'FT', 'FC', 'SI', 'FS']
        fb_swing_cols = [col for col in fb_types if col in pitch_type_swings_grouped.columns]
        fb_whiff_cols = [col for col in fb_types if col in pitch_type_whiffs_grouped.columns]
        swing_whiff_calcs['fastball_swings'] = pitch_type_swings_grouped[fb_swing_cols].sum(axis=1) if fb_swing_cols else 0
        swing_whiff_calcs['fastball_whiffs'] = pitch_type_whiffs_grouped[fb_whiff_cols].sum(axis=1) if fb_whiff_cols else 0

        br_types = ['SL', 'CU', 'KC', 'CS', 'KN']
        br_swing_cols = [col for col in br_types if col in pitch_type_swings_grouped.columns]
        br_whiff_cols = [col for col in br_types if col in pitch_type_whiffs_grouped.columns]
        swing_whiff_calcs['breaking_swings'] = pitch_type_swings_grouped[br_swing_cols].sum(axis=1) if br_swing_cols else 0
        swing_whiff_calcs['breaking_whiffs'] = pitch_type_whiffs_grouped[br_whiff_cols].sum(axis=1) if br_whiff_cols else 0

        os_types = ['CH', 'SC', 'FO']
        os_swing_cols = [col for col in os_types if col in pitch_type_swings_grouped.columns]
        os_whiff_cols = [col for col in os_types if col in pitch_type_whiffs_grouped.columns]
        swing_whiff_calcs['offspeed_swings'] = pitch_type_swings_grouped[os_swing_cols].sum(axis=1) if os_swing_cols else 0
        swing_whiff_calcs['offspeed_whiffs'] = pitch_type_whiffs_grouped[os_whiff_cols].sum(axis=1) if os_whiff_cols else 0

        # Add zone contact swings
        swing_whiff_calcs['zone_contact_swings'] = zone_contact_swings_grouped

        # Reset index to merge on columns
        swing_whiff_calcs.reset_index(inplace=True)
        # pitch_level_stats is already reset from the agg step

        # Merge calculated DataFrame into pitch_level_stats
        # Ensure key columns have same type before merge
        for col in game_group_cols:
             if col in pitch_level_stats.columns and col in swing_whiff_calcs.columns:
                 common_type = pd.api.types.infer_dtype(pitch_level_stats[col])
                 swing_whiff_calcs[col] = swing_whiff_calcs[col].astype(pitch_level_stats[col].dtype)

        pitch_level_stats = pd.merge(pitch_level_stats, swing_whiff_calcs, on=game_group_cols, how='left')

        # Fill NaNs introduced by merging
        fill_cols = ['fastball_swings', 'fastball_whiffs', 'breaking_swings', 'breaking_whiffs', 'offspeed_swings', 'offspeed_whiffs', 'zone_contact_swings']
        for col in fill_cols:
            if col in pitch_level_stats.columns:
                pitch_level_stats[col].fillna(0, inplace=True)
            else:
                pitch_level_stats[col] = 0 # Add column if merge didn't create it (no swings/whiffs at all)
        # *** FIX END ***

        # --- Merge Aggregations & Calculate Overall Rates ---
        logger.info("Merging aggregations and calculating overall rates...")
        # (Merging and rate calculation logic - same as previous version)
        first_vals = df.groupby(game_group_cols).agg(
            player_name=('player_name', 'first'), stand=('stand', 'first'),
            home_team=('home_team', 'first'), away_team=('away_team', 'first'), season=('season', 'first'),
        ).reset_index()
        # Ensure merge keys are consistent type
        first_vals['game_date'] = pd.to_datetime(first_vals['game_date']).dt.date
        game_level_stats['game_date'] = pd.to_datetime(game_level_stats['game_date']).dt.date
        pitch_level_stats['game_date'] = pd.to_datetime(pitch_level_stats['game_date']).dt.date

        game_data = pd.merge(game_level_stats, first_vals, on=game_group_cols, how='left')
        game_data = pd.merge(game_data, pitch_level_stats, on=game_group_cols, how='left', suffixes=('', '_dropme'))
        game_data.drop(columns=[c for c in game_data.columns if '_dropme' in c], inplace=True)

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

        # --- Handedness Splits ---
        logger.info("Calculating performance splits vs LHP/RHP...")
        # (Handedness split logic - same as previous version)
        pa_outcomes['game_date'] = pd.to_datetime(pa_outcomes['game_date']).dt.date # Ensure date type
        split_group_cols_hand = ['batter', 'game_pk', 'game_date', 'p_throws']
        pa_outcomes_split_hand = pa_outcomes.groupby(split_group_cols_hand).agg(
            split_pa=('pa_count', 'sum'), split_k=('strikeouts_pa', 'sum'), split_woba_num=('woba_value_pa', 'sum'),
            split_ab=('ab_pa', 'sum'), split_walks=('walks_pa','sum'), split_sf=('sac_fly_pa','sum'),
        ).reset_index()
        split_woba_denom_hand = pa_outcomes_split_hand['split_ab'] + pa_outcomes_split_hand['split_walks'] + pa_outcomes_split_hand['split_sf']
        pa_outcomes_split_hand['split_k_pct'] = (pa_outcomes_split_hand['split_k'] / pa_outcomes_split_hand['split_pa'].replace(0, np.nan)).fillna(0)
        pa_outcomes_split_hand['split_woba'] = (pa_outcomes_split_hand['split_woba_num'] / split_woba_denom_hand.replace(0, np.nan)).fillna(0)
        split_pivot_hand = pa_outcomes_split_hand.pivot_table(index=game_group_cols, columns='p_throws', values=['split_k_pct', 'split_woba', 'split_pa'])
        split_pivot_hand.columns = [f'{metric}_vs_{p_throw}' for metric, p_throw in split_pivot_hand.columns]
        split_pivot_hand.reset_index(inplace=True)
        # Ensure merge keys are consistent type
        split_pivot_hand['game_date'] = pd.to_datetime(split_pivot_hand['game_date']).dt.date
        final_game_data = pd.merge(game_data, split_pivot_hand, on=game_group_cols, how='left')
        split_cols_hand = [col for col in final_game_data.columns if '_vs_L' in col or '_vs_R' in col]
        for col in split_cols_hand: final_game_data[col].fillna(0, inplace=True)

        # --- Count Splits ---
        logger.info("Calculating performance splits by count category...")
        # (Count split logic - same as previous version)
        split_group_cols_count = ['batter', 'game_pk', 'game_date', 'count_category_pa']
        pa_outcomes_split_count = pa_outcomes[pa_outcomes['count_category_pa'] != 'unknown'].groupby(split_group_cols_count).agg(
            count_pa=('pa_count', 'sum'), count_k=('strikeouts_pa', 'sum'), count_woba_num=('woba_value_pa', 'sum'),
            count_ab=('ab_pa', 'sum'), count_walks=('walks_pa','sum'), count_sf=('sac_fly_pa','sum'),
        ).reset_index()
        split_woba_denom_count = pa_outcomes_split_count['count_ab'] + pa_outcomes_split_count['count_walks'] + pa_outcomes_split_count['count_sf']
        pa_outcomes_split_count['count_k_pct'] = (pa_outcomes_split_count['count_k'] / pa_outcomes_split_count['count_pa'].replace(0, np.nan)).fillna(0)
        pa_outcomes_split_count['count_woba'] = (pa_outcomes_split_count['count_woba_num'] / split_woba_denom_count.replace(0, np.nan)).fillna(0)
        split_pivot_count = pa_outcomes_split_count.pivot_table(index=game_group_cols, columns='count_category_pa', values=['count_k_pct', 'count_woba', 'count_pa'])
        split_pivot_count.columns = [f'{metric.replace("count_", "")}_{category}' for metric, category in split_pivot_count.columns]
        split_pivot_count.reset_index(inplace=True)
        # Ensure merge keys are consistent type
        split_pivot_count['game_date'] = pd.to_datetime(split_pivot_count['game_date']).dt.date
        final_game_data = pd.merge(final_game_data, split_pivot_count, on=game_group_cols, how='left')
        split_cols_count = [col for col in final_game_data.columns if any(cat in col for cat in ['_ahead', '_behind', '_even', '_0-0', '_3-2'])]
        for col in split_cols_count: final_game_data[col].fillna(0, inplace=True)

        # --- Final Cleanup and Save ---
        final_game_data.rename(columns={'batter': 'batter_id'}, inplace=True)
        logger.info(f"Aggregation complete. Final shape: {final_game_data.shape}")

    except Exception as e: # Catch errors during processing
        logger.error(f"Error during data processing/aggregation: {e}", exc_info=True); logger.error(traceback.format_exc()); return False

    # --- Save to Database ---
    try:
        # (Database saving logic - same as previous version)
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed before saving.")
            try: cursor = conn.cursor(); cursor.execute("DROP TABLE IF EXISTS game_level_batters"); conn.commit(); logger.info("Dropped existing game_level_batters table.")
            except Exception as drop_e: logger.warning(f"Could not drop game_level_batters: {drop_e}")
            final_game_data.to_sql('game_level_batters', conn, if_exists='replace', index=False, chunksize=5000)
            logger.info(f"Stored {len(final_game_data)} aggregated records to 'game_level_batters'.")
        total_time = time.time() - start_time
        logger.info(f"Batter aggregation completed in {total_time:.2f} seconds")
        return True
    except Exception as e: logger.error(f"Error saving aggregated data: {e}", exc_info=True); logger.error(traceback.format_exc()); return False

if __name__ == "__main__":
    # (Main execution block - same as previous version)
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Module imports failed.")
    logger.info("Starting batter data aggregation process (incl. counts)...")
    success = aggregate_batters_to_game_level()
    if success: logger.info("Batter data aggregation process finished successfully.")
    else: logger.error("Batter data aggregation process finished with errors.")