import pandas as pd
import numpy as np
import logging
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import DBConnection
from src import config # Assuming config.py is in the src directory
# Import the new function for boxscore feature engineering
from src.features.boxscore_features import engineer_boxscore_features # Ensure this file exists

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

def _aggregate_starter_pitcher_stats(conn) -> pd.DataFrame:
    """
    Aggregates Statcast pitch data to create game-level statistics for starting pitchers.
    This is a focused helper function.

    Args:
        conn: Active database connection.

    Returns:
        DataFrame with aggregated stats for starting pitchers.
    """
    logger.info("Starting aggregation of Statcast data for starting pitchers.")

    # 1. Load the definitive list of starting pitchers
    logger.info(f"Reading definitive starters list from '{config.STATCAST_STARTING_PITCHERS_TABLE}' table.")
    starters_list_df = pd.read_sql_query(f"SELECT * FROM {config.STATCAST_STARTING_PITCHERS_TABLE}", conn)

    if starters_list_df.empty:
        logger.warning(f"'{config.STATCAST_STARTING_PITCHERS_TABLE}' is empty. No starter pitcher stats to generate.")
        # Return empty DataFrame with expected schema if possible, or just empty
        # For simplicity, returning empty. The orchestrator can decide schema for empty final table.
        return pd.DataFrame()

    starters_for_merge = starters_list_df[['game_pk', 'pitcher']].drop_duplicates()
    logger.info(f"Loaded {len(starters_for_merge)} unique starter (game_pk, pitcher) entries for merging.")

    # 2. Load raw pitch-by-pitch data
    logger.info(f"Reading pitch-by-pitch data from '{config.STATCAST_PITCHERS_TABLE}'.")
    pitch_data_query = f"SELECT * FROM {config.STATCAST_PITCHERS_TABLE} ORDER BY game_date, game_pk, pitcher, inning, at_bat_number, pitch_number"
    pitch_data = pd.read_sql_query(pitch_data_query, conn)

    if pitch_data.empty:
        logger.warning(f"'{config.STATCAST_PITCHERS_TABLE}' is empty. Cannot aggregate pitcher stats.")
        return pd.DataFrame()

    logger.info(f"Pitch data loaded: {len(pitch_data)} rows.")

    # 3. Filter pitch_data to include only pitches from the identified starting pitchers
    sp_pitch_data = pd.merge(pitch_data, starters_for_merge, on=['game_pk', 'pitcher'], how='inner')

    if sp_pitch_data.empty:
        logger.warning("No pitch data found for the identified starters after merge. Pitcher game stats will be empty.")
        return pd.DataFrame()

    logger.info(f"Successfully merged pitch data with starters: {len(sp_pitch_data)} rows of starter pitches.")

    # 4. Feature engineering for aggregation
    sp_pitch_data['is_strikeout'] = np.where(sp_pitch_data['events'] == 'strikeout', 1, 0)
    numeric_sum_cols = ['release_speed', 'release_pos_x', 'release_pos_z', 'release_pos_y',
                        'balls', 'strikes', 'release_spin_rate', 'outs_when_up', 'is_strikeout']
    for col in numeric_sum_cols:
        if col in sp_pitch_data.columns:
            if pd.api.types.is_numeric_dtype(sp_pitch_data[col]):
                sp_pitch_data[col] = sp_pitch_data[col].fillna(0)
            else:
                sp_pitch_data[col] = pd.to_numeric(sp_pitch_data[col], errors='coerce').fillna(0)
                logger.warning(f"Column '{col}' for pitcher stats was non-numeric and coerced; NaNs set to 0.")
        elif col != 'is_strikeout': # is_strikeout is created above
             logger.warning(f"Numeric column '{col}' for pitcher aggregation not found. It will be effectively zero.")
             sp_pitch_data[col] = 0 # Add column as zero if missing to prevent agg error

    aggregation_dict_corrected = {
        'strikeouts_for_sum': ('is_strikeout', 'sum'),
        'total_pitches_count': ('pitcher', 'size'), # Using 'pitcher' (any non-null col) for 'size'
        'release_speed_sum': ('release_speed', 'sum'),
        'release_pos_x_sum': ('release_pos_x', 'sum'),
        'release_pos_z_sum': ('release_pos_z', 'sum'),
        'release_pos_y_sum': ('release_pos_y', 'sum'),
        'balls_sum': ('balls', 'sum'),
        'strikes_sum': ('strikes', 'sum'),
        'release_spin_rate_sum': ('release_spin_rate', 'sum'),
        'outs_when_up_sum': ('outs_when_up', 'sum')
    }
    # Ensure all columns in agg dict exist in sp_pitch_data, if not, they were handled above
    # by being added as a column of zeros.

    logger.info("Grouping by game, pitcher, game_date, and p_throws to aggregate pitcher stats.")
    pitcher_game_stats_sum = sp_pitch_data.groupby(
        ['game_pk', 'pitcher', 'game_date', 'p_throws'], # Ensure game_pk is in group keys
        as_index=False
    ).agg(**aggregation_dict_corrected)

    pitcher_game_stats_agg = pitcher_game_stats_sum.rename(columns={
        'strikeouts_for_sum': 'strikeouts_recorded',
        'total_pitches_count': 'total_pitches_thrown'
    })

    # 5. Calculate averages
    denominator = pitcher_game_stats_agg['total_pitches_thrown']
    pitcher_game_stats_agg['avg_release_speed'] = np.where(denominator > 0, pitcher_game_stats_agg['release_speed_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_release_pos_x'] = np.where(denominator > 0, pitcher_game_stats_agg['release_pos_x_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_release_pos_z'] = np.where(denominator > 0, pitcher_game_stats_agg['release_pos_z_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_release_pos_y'] = np.where(denominator > 0, pitcher_game_stats_agg['release_pos_y_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_balls'] = np.where(denominator > 0, pitcher_game_stats_agg['balls_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_strikes'] = np.where(denominator > 0, pitcher_game_stats_agg['strikes_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_spin_rate'] = np.where(denominator > 0, pitcher_game_stats_agg['release_spin_rate_sum'] / denominator, 0)
    pitcher_game_stats_agg['avg_outs_when_up'] = np.where(denominator > 0, pitcher_game_stats_agg['outs_when_up_sum'] / denominator, 0)

    cols_to_drop_agg = [
        'release_speed_sum', 'release_pos_x_sum', 'release_pos_z_sum', 'release_pos_y_sum',
        'balls_sum', 'strikes_sum', 'release_spin_rate_sum', 'outs_when_up_sum'
    ]
    pitcher_game_stats_agg = pitcher_game_stats_agg.drop(columns=[col for col in cols_to_drop_agg if col in pitcher_game_stats_agg.columns], errors='ignore')
    logger.info(f"Finished aggregating starter pitcher stats. {len(pitcher_game_stats_agg)} games/starters processed.")
    return pitcher_game_stats_agg

def _fetch_and_engineer_boxscore_data(conn) -> pd.DataFrame:
    """
    Fetches raw boxscore data and engineers features using engineer_boxscore_features.

    Args:
        conn: Active database connection.

    Returns:
        DataFrame with engineered boxscore features.
    """
    logger.info(f"Fetching raw boxscore data from {config.MLB_BOXSCORES_TABLE}...")
    try:
        # Ensure 'game_pk' is selected, and all columns needed by engineer_boxscore_features
        raw_boxscores_df = pd.read_sql_query(f"SELECT * FROM {config.MLB_BOXSCORES_TABLE}", conn)
        if raw_boxscores_df.empty:
            logger.warning(f"No data found in {config.MLB_BOXSCORES_TABLE}. Boxscore features will be sparse or default.")
            # engineer_boxscore_features should handle empty df and produce empty df with schema or raise error
            # For safety, call it to get schema or handle its missing checks.
            return engineer_boxscore_features(pd.DataFrame(columns=['game_pk', 'game_date', 'wind', 'weather', 'temp', 'dayNight', 'elevation', 'hp_umpire']))


    except Exception as e:
        logger.error(f"Error fetching data from {config.MLB_BOXSCORES_TABLE}: {e}. Proceeding with empty boxscore features.")
        return pd.DataFrame() # Return empty with no schema, or define one

    if not raw_boxscores_df.empty:
        logger.info("Engineering features from raw boxscore data...")
        try:
            boxscore_features_df = engineer_boxscore_features(raw_boxscores_df)
            # Ensure game_pk is present for merging
            if 'game_pk' not in boxscore_features_df.columns and not boxscore_features_df.empty:
                logger.error("'game_pk' is missing from the output of engineer_boxscore_features. This will cause merge issues.")
                # Potentially add game_pk back if raw_boxscores_df is available and index aligns
                # but this indicates an issue in engineer_boxscore_features
                # For now, return potentially problematic df for caller to handle or fail merge
            return boxscore_features_df
        except ValueError as ve: # Catch the >10% missing error from engineer_boxscore_features
            logger.error(f"ValueError during boxscore feature engineering: {ve}. Boxscore features will be empty.")
            raise # Re-raise to halt pipeline as this is a critical data quality issue
        except Exception as e:
            logger.error(f"Unexpected error during boxscore feature engineering. ErrorType: {type(e).__name__}, Message: {str(e)}", exc_info=True)
            return pd.DataFrame() # Return empty
    else:
        logger.warning("Raw boxscore data was empty, returning empty DataFrame for boxscore features.")
        return pd.DataFrame()


def create_game_level_aggregates():
    """
    Orchestrates the creation of comprehensive game-level aggregated data.
    It combines starter pitcher stats, (eventually team stats), and boxscore features.
    The final enriched data is saved to the database.
    """
    logger.info("Starting creation of comprehensive game-level aggregates.")

    # Ensure necessary table names are available in config
    required_configs = [
        'STATCAST_STARTING_PITCHERS_TABLE', 'STATCAST_PITCHERS_TABLE',
        'PITCHER_GAME_STATS_TABLE', 'MLB_BOXSCORES_TABLE' # For boxscore data
        # Add STATCAST_BATTERS_TABLE, TEAM_GAME_STATS_TABLE when team stats are integrated
    ]
    for req_config in required_configs:
        if not hasattr(config, req_config):
            logger.error(f"Configuration variable '{req_config}' not found in config.py.")
            raise AttributeError(f"Configuration variable '{req_config}' not found in config.py.")

    final_game_data = pd.DataFrame()

    with DBConnection() as conn:
        try:
            # 1. Aggregate Starter Pitcher Stats
            pitcher_stats_df = _aggregate_starter_pitcher_stats(conn)

            # 2. Fetch and Engineer Boxscore Features
            # This might raise ValueError if >10% missing in raw boxscore columns, halting the process.
            boxscore_features_df = _fetch_and_engineer_boxscore_data(conn)

            # 3. (Placeholder) Aggregate Team Batting Stats
            # team_batting_stats_df = _aggregate_team_batting_stats(conn) # Example future function
            # logger.info("Team batting stats aggregation would occur here.")
            team_batting_stats_df = pd.DataFrame() # For now

            # 4. Merge DataFrames
            # Start with pitcher_stats_df as the base, or boxscore_features_df if it's more complete game list
            # If PITCHER_GAME_STATS_TABLE is primarily for games where a starter pitched, left merge from pitcher_stats is appropriate.
            # If it's a general game log, a full outer merge on game_pk with boxscore_features might be better.
            # Given current naming, let's assume we enrich pitcher_stats.

            if not pitcher_stats_df.empty:
                final_game_data = pitcher_stats_df.copy()
                if 'game_pk' not in final_game_data.columns:
                     logger.error("Critical error: 'game_pk' not found in pitcher_stats_df. Cannot merge.")
                     raise KeyError("'game_pk' missing from pitcher_stats_df after aggregation.")


                if not boxscore_features_df.empty and 'game_pk' in boxscore_features_df.columns:
                    logger.info(f"Merging pitcher stats ({len(final_game_data)} rows) with boxscore features ({len(boxscore_features_df)} rows)...")
                    # Ensure game_pk types are compatible for merging
                    final_game_data['game_pk'] = final_game_data['game_pk'].astype(str)
                    boxscore_features_df['game_pk'] = boxscore_features_df['game_pk'].astype(str)
                    final_game_data = pd.merge(final_game_data, boxscore_features_df, on='game_pk', how='left')
                    logger.info(f"Shape after merging boxscore features: {final_game_data.shape}")
                elif not boxscore_features_df.empty:
                    logger.warning("Boxscore features were generated but 'game_pk' was missing. Cannot merge.")
                else:
                    logger.info("Boxscore features DataFrame was empty. No boxscore features merged.")

                # Placeholder for merging team stats
                if not team_batting_stats_df.empty and 'game_pk' in team_batting_stats_df.columns:
                    logger.info("Merging team batting stats...") # (Future implementation)
                    # final_game_data = pd.merge(final_game_data, team_batting_stats_df, on='game_pk', how='left')
                    # logger.info(f"Shape after merging team batting stats: {final_game_data.shape}")
                else:
                    logger.info("Team batting stats DataFrame was empty. No team stats merged.")

            elif not boxscore_features_df.empty: # No pitcher stats, but have boxscore features
                 logger.warning("Pitcher stats are empty, but boxscore features exist. Final data will primarily be boxscore features.")
                 # This implies that the output table might not just be "PITCHER_GAME_STATS" anymore
                 # Or, we decide if pitcher_stats is empty, the output is empty for PITCHER_GAME_STATS_TABLE
                 # For now, if pitcher_stats is empty, let final_game_data remain empty or reflect this.
                 # If the goal is a comprehensive game log even without pitcher stats, then:
                 # final_game_data = boxscore_features_df.copy()
                 # if not team_batting_stats_df.empty and 'game_pk' in team_batting_stats_df.columns:
                 # final_game_data = pd.merge(final_game_data, team_batting_stats_df, on='game_pk', how='left')
                 # This decision depends on the desired content of PITCHER_GAME_STATS_TABLE.
                 # Sticking to the idea that PITCHER_GAME_STATS_TABLE is an enriched version of pitcher stats:
                 final_game_data = pd.DataFrame() # Keep it empty if pitcher_stats_df was empty
                 logger.info("Pitcher stats were empty, so final_game_data for PITCHER_GAME_STATS_TABLE remains empty.")

            else: # All components are empty
                logger.info("All data components (pitcher stats, boxscore features) are empty.")
                final_game_data = pd.DataFrame()


            # 5. Save the final aggregated and enriched game data
            output_table_name = config.PITCHER_GAME_STATS_TABLE # Or a new config variable e.g., GAME_LEVEL_FEATURES_TABLE
            if not final_game_data.empty:
                logger.info(f"Writing final aggregated game data to '{output_table_name}'. Rows: {len(final_game_data)}")
                # Ensure no duplicate columns before writing (e.g., if game_date came from multiple sources)
                # final_game_data = final_game_data.loc[:,~final_game_data.columns.duplicated()]
                final_game_data.to_sql(output_table_name, conn, if_exists='replace', index=False, chunksize=10000)
                logger.info(f"Successfully wrote to '{output_table_name}'.")
            else:
                logger.warning(f"Final game data is empty. Writing an empty table or doing nothing for '{output_table_name}'.")
                # To ensure the table is created with schema even if empty:
                # pd.DataFrame(columns=[...expected schema...]).to_sql(output_table_name, conn, if_exists='replace', index=False)
                # For now, if final_game_data is pd.DataFrame() with no columns, to_sql might do nothing.
                # If it has columns from one of the empty DFs, it will create those.
                # Best practice would be to define the full schema and create an empty table with that schema.
                # However, if no data, saving an empty DataFrame (if it has columns) is acceptable.
                if hasattr(final_game_data, 'columns') and not final_game_data.columns.empty:
                    final_game_data.to_sql(output_table_name, conn, if_exists='replace', index=False)
                    logger.info(f"Wrote empty table (with columns) to '{output_table_name}'.")
                else: # Truly empty df with no columns
                     logger.info(f"Final data is a completely empty DataFrame with no columns. Table '{output_table_name}' might not be created or might be empty without schema by to_sql.")
                     # To create an empty table with a defined schema if final_game_data is completely empty (no columns):
                     # conn.execute(f"DROP TABLE IF EXISTS {output_table_name}") # Or handle existence
                     # conn.execute(f"CREATE TABLE {output_table_name} (game_pk TEXT, column2 TEXT, ...)") # Define your schema
                     # Or more simply, if a schema is critical:
                     # if final_game_data.empty:
                     #     # Create a df with expected columns to establish schema
                     #     # This schema should ideally be defined in config or derived consistently
                     #     logger.info(f"Forcing schema for empty table {output_table_name}")
                     #     # Example schema columns (needs to be comprehensive)
                     #     example_schema_cols = ['game_pk', 'pitcher', 'strikeouts_recorded', 'temperature', 'wind_speed_mph'] 
                     #     pd.DataFrame(columns=example_schema_cols).to_sql(output_table_name, conn, if_exists='replace', index=False)


        except pd.io.sql.DatabaseError as db_err:
            logger.error(f"Database error during game-level aggregation: {db_err}")
            raise
        except AttributeError as attr_err:
            logger.error(f"Configuration error: {attr_err}")
            raise
        except ValueError as val_err: # Catch critical ValueErrors (e.g., from boxscore missing data check)
            logger.error(f"Halting due to ValueError during game-level aggregation: {val_err}")
            # Depending on desired behavior, you might not re-raise if partial processing is acceptable
            # but the >10% missing rule implies it's critical.
            raise
        except KeyError as key_err:
            logger.error(f"KeyError during game-level aggregation (often due to missing columns for merge/processing): {key_err}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during game-level aggregation: {e}")
            raise

    logger.info("Finished creating comprehensive game-level aggregates.")

if __name__ == '__main__':
    # This script now orchestrates multiple aggregation steps.
    # Ensure prerequisite data (starter lists, raw statcast, raw boxscores) is available.
    create_game_level_aggregates()