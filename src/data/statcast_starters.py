# src/data/statcast_starters.py
import pandas as pd
import numpy as np
import gc # Garbage Collector interface
import sqlite3 # For explicit type checking if needed

# Assuming your project structure allows these imports
from src.data.utils import DBConnection

# Define the columns we want to keep from the Statcast data.
# These are chosen for their potential value in predicting pitcher strikeouts.
STATCAST_COLUMNS_TO_KEEP = [
    # Core Identifiers & Game Context
    'game_pk', 'game_date', 'pitcher', 'player_name', 'batter', 'home_team', 'away_team',
    'game_year', 'game_type',
    # Pitch Sequence & In-Game State
    'inning', 'inning_topbot', 'at_bat_number', 'pitch_number', 'balls', 'strikes',
    'outs_when_up', 'stand', 'p_throws', 'on_1b', 'on_2b', 'on_3b', 'home_score', 'away_score',
    # Pitch Attributes
    'pitch_type', 'pitch_name', 'release_speed', 'effective_speed', 'release_spin_rate',
    'spin_axis', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone',
    'release_pos_x', 'release_pos_y', 'release_pos_z', 'release_extension',
    'sz_top', 'sz_bot',
    # Pitch Outcome
    'description', 'type', 'events',
    # Advanced/Contextual (useful for feature engineering)
    'woba_value', 'babip_value', 'iso_value', 'launch_speed', 'launch_angle',
    'hit_distance_sc', 'bb_type', 'fielder_2', # fielder_2 is catcher ID
    # Required for internal logic if not already listed (at_bat_number, pitch_number are duplicated but set will handle)
    'at_bat_number', 'pitch_number'
]
# Remove duplicates from the list and ensure they are sorted for consistent query generation
STATCAST_COLUMNS_TO_KEEP = sorted(list(set(STATCAST_COLUMNS_TO_KEEP)))


def optimize_df_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes memory usage of a DataFrame by downcasting numeric types
    and converting object columns with low cardinality to 'category'.
    Handles 'game_date' specifically for datetime conversion.
    Addresses FutureWarning for pd.to_numeric by using try-except.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with optimized dtypes.
    """
    print(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Convert game_date first and ensure it's datetime
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    for col in df.columns:
        if col == 'game_date': # Already handled, skip further processing for this column here
            continue

        col_type = df[col].dtype

        if pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
            try:
                # Attempt conversion to numeric first
                converted_numeric = pd.to_numeric(df[col])
                # If successful, downcast
                if pd.api.types.is_integer_dtype(converted_numeric.dtype):
                    df[col] = pd.to_numeric(converted_numeric, downcast='integer')
                elif pd.api.types.is_float_dtype(converted_numeric.dtype):
                    df[col] = pd.to_numeric(converted_numeric, downcast='float')
                else: # Other numeric types
                    df[col] = converted_numeric
            except (ValueError, TypeError): # If conversion to numeric fails
                # Try to convert to category if cardinality is suitable
                num_unique_values = df[col].nunique()
                if num_unique_values < len(df) * 0.5 and num_unique_values < 5000:
                    try:
                        df[col] = df[col].astype('category')
                    except Exception:
                        pass # Keep as original type if category conversion fails
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        # Note: datetime64 columns (like game_date) are not further processed here

    print(f"Optimized memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    # print(df.info()) # Optional: for debugging dtypes after optimization
    return df


def _identify_and_filter_starting_pitchers(
    statcast_data: pd.DataFrame, 
    min_innings_pitched: int = 4,
    min_pitches_thrown: int = 50
) -> pd.DataFrame:
    if statcast_data.empty:
        print("DEBUG: _identify_and_filter_starting_pitchers received empty statcast_data.")
        return pd.DataFrame(columns=statcast_data.columns)

    internal_required_cols = ['game_pk', 'pitcher', 'inning', 'at_bat_number', 'pitch_number']
    missing_cols = [col for col in internal_required_cols if col not in statcast_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for starter identification: {', '.join(missing_cols)}")

    print(f"DEBUG: Shape of statcast_data at start of _identify: {statcast_data.shape}")

    cols_for_first_pitch = ['game_pk', 'at_bat_number', 'pitch_number', 'pitcher']
    df_first_pitch_subset = statcast_data[cols_for_first_pitch].copy()
    df_first_pitch_details = df_first_pitch_subset.sort_values(
        ['game_pk', 'at_bat_number', 'pitch_number']
    ).groupby('game_pk', as_index=False).first()
    df_first_pitch_throwers = df_first_pitch_details[['game_pk', 'pitcher']]
    print(f"DEBUG: Shape of df_first_pitch_throwers: {df_first_pitch_throwers.shape}")
    del df_first_pitch_subset, df_first_pitch_details
    gc.collect()

    if df_first_pitch_throwers.empty:
        print("DEBUG: No first pitch throwers found.")
        return pd.DataFrame(columns=statcast_data.columns)

    df_pitcher_game_summary = statcast_data.groupby(['game_pk', 'pitcher'], as_index=False, observed=True).agg(
        innings_pitched=('inning', 'nunique'),
        pitches_thrown=('pitch_number', 'count')
    )
    df_pitcher_game_summary['innings_pitched'] = pd.to_numeric(df_pitcher_game_summary['innings_pitched'], downcast='integer')
    df_pitcher_game_summary['pitches_thrown'] = pd.to_numeric(df_pitcher_game_summary['pitches_thrown'], downcast='integer')
    print(f"DEBUG: Shape of df_pitcher_game_summary: {df_pitcher_game_summary.shape}")

    df_candidate_starters_stats = pd.merge(
        df_first_pitch_throwers,
        df_pitcher_game_summary,
        on=['game_pk', 'pitcher'],
        how='inner'
    )
    print(f"DEBUG: Shape of df_candidate_starters_stats: {df_candidate_starters_stats.shape}")
    del df_first_pitch_throwers, df_pitcher_game_summary
    gc.collect()

    df_qualified_starters = df_candidate_starters_stats[
        (df_candidate_starters_stats['innings_pitched'] >= min_innings_pitched) &
        (df_candidate_starters_stats['pitches_thrown'] >= min_pitches_thrown)
    ]
    print(f"DEBUG: Shape of df_qualified_starters: {df_qualified_starters.shape}")
    del df_candidate_starters_stats
    gc.collect()

    df_qualified_starter_keys = df_qualified_starters[['game_pk', 'pitcher']].drop_duplicates()
    print(f"DEBUG: Shape of df_qualified_starter_keys: {df_qualified_starter_keys.shape}")
    del df_qualified_starters
    gc.collect()

    if df_qualified_starter_keys.empty:
        print("DEBUG: No qualified starter keys found.")
        return pd.DataFrame(columns=statcast_data.columns)

    merge_key_columns = ['game_pk', 'pitcher']
    for col in merge_key_columns:
        if col not in statcast_data.columns:
            raise ValueError(f"Merge key column '{col}' not found in main statcast_data.")
        if col not in df_qualified_starter_keys.columns:
            raise ValueError(f"Merge key column '{col}' not found in df_qualified_starter_keys.")
        try:
            statcast_data[col] = statcast_data[col].astype(np.int64)
            df_qualified_starter_keys[col] = df_qualified_starter_keys[col].astype(np.int64)
        except Exception as e:
            print(f"Error casting column {col} to np.int64: {e}")
            raise

    df_statcast_starting_pitchers = pd.merge(
        statcast_data,
        df_qualified_starter_keys,
        on=merge_key_columns,
        how='inner'
    )
    print(f"DEBUG: Shape of df_statcast_starting_pitchers after final merge: {df_statcast_starting_pitchers.shape}")
    return df_statcast_starting_pitchers


def create_statcast_starting_pitchers_table(
    min_innings_pitched: int = 4,
    min_pitches_thrown: int = 50,
    year_filter: int = None,
    limit: int = None
) -> pd.DataFrame:
    print("Starting to create Statcast starting pitchers table...")

    logic_cols = ['game_pk', 'pitcher', 'inning', 'at_bat_number', 'pitch_number', 'game_year'] # ensure game_year is fetched if used
    cols_to_select_final = sorted(list(set(STATCAST_COLUMNS_TO_KEEP + logic_cols)))
    query_cols_str = ", ".join([f'"{c}"' for c in cols_to_select_final])
    query = f"SELECT {query_cols_str} FROM statcast_pitchers"

    conditions = []
    if year_filter:
        # Ensure 'game_year' is a column that exists in the table
        conditions.append(f"game_year = {year_filter}")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if limit:
        query += f" LIMIT {limit}"
    query += ";"

    print(f"Executing query (first 200 chars): {query[:200]}...")

    try:
        with DBConnection() as conn:
            check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='statcast_pitchers';"
            table_exists = pd.read_sql_query(check_query, conn)
            if table_exists.empty:
                print("Error: 'statcast_pitchers' table does not exist in the database.")
                return pd.DataFrame()
            statcast_data_full = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return pd.DataFrame()

    if statcast_data_full.empty:
        print("No data fetched from 'statcast_pitchers' table matching criteria.")
        return pd.DataFrame()

    print(f"Fetched {len(statcast_data_full)} rows from the database.")

    statcast_data_optimized = optimize_df_memory(statcast_data_full.copy())
    print(f"DEBUG: Shape of statcast_data_optimized: {statcast_data_optimized.shape}")
    del statcast_data_full
    gc.collect()

    df_starters = _identify_and_filter_starting_pitchers(
        statcast_data_optimized,
        min_innings_pitched,
        min_pitches_thrown
    )
    print(f"DEBUG: Shape of df_starters after _identify_and_filter: {df_starters.shape}")

    if df_starters.empty:
        print("No qualifying starting pitcher data was generated after filtering.")
        return df_starters # Return the empty DataFrame

    # Ensure final columns are as expected
    final_output_columns = [col for col in STATCAST_COLUMNS_TO_KEEP if col in df_starters.columns]
    if not final_output_columns: # Should not happen if STATCAST_COLUMNS_TO_KEEP is reasonable
        print("Warning: No columns from STATCAST_COLUMNS_TO_KEEP found in the processed data. Returning all columns.")
    else:
        df_starters = df_starters[final_output_columns]
    
    print(f"Identified {df_starters[['game_pk', 'pitcher']].drop_duplicates().shape[0]} unique starting pitcher games.")
    print(f"Final table shape: {df_starters.shape}")
    if not df_starters.empty:
        print(f"Final memory usage: {df_starters.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("Finished creating Statcast starting pitchers table (in memory).")
    return df_starters


if __name__ == '__main__':
    print("Running statcast_starters.py directly for testing...")

    test_year = None
    test_limit = None

    df_statcast_starters = create_statcast_starting_pitchers_table(
        year_filter=test_year,
        limit=test_limit
    )

    if not df_statcast_starters.empty:
        print("\n--- Resulting DataFrame Info (before save attempt) ---")
        df_statcast_starters.info(memory_usage='deep')

        print("\n--- Sample Data (First 5 rows, before save attempt) ---")
        print(df_statcast_starters.head())

        # --- Store the DataFrame ---
        df_to_save = df_statcast_starters.copy() # Work on a copy for saving modifications

        if 'game_date' in df_to_save.columns:
            if pd.api.types.is_datetime64_any_dtype(df_to_save['game_date']):
                print("DEBUG: Converting 'game_date' from datetime to string for SQLite.")
                df_to_save['game_date'] = df_to_save['game_date'].dt.strftime('%Y-%m-%d')
            elif pd.api.types.is_categorical_dtype(df_to_save['game_date']):
                 print("DEBUG: Converting 'game_date' from category to string for SQLite.")
                 df_to_save['game_date'] = df_to_save['game_date'].astype(str)
            else:
                print(f"DEBUG: 'game_date' is type {df_to_save['game_date'].dtype}, converting to string for SQLite.")
                df_to_save['game_date'] = df_to_save['game_date'].astype(str)
        
        table_name = "statcast_starting_pitchers"
        try:
            with DBConnection() as conn:
                df_to_save.to_sql(
                    table_name,
                    conn,
                    if_exists='replace',
                    index=False,
                    chunksize=10000
                )
                conn.commit() # Explicitly commit
                print(f"Successfully called to_sql for table '{table_name}'.")

                # Verification step: Read back some data
                print("Verifying data in the database...")
                try:
                    verification_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
                    print(f"Verification: Found {len(verification_df)} rows in '{table_name}'.")
                    if not verification_df.empty:
                        print("Sample data from database:")
                        print(verification_df.head())
                    else:
                        print(f"WARNING: Verification query returned no data from '{table_name}'. The table might be empty.")
                except Exception as e_verify:
                    print(f"Error during verification read: {e_verify}")

        except Exception as e_save:
            print(f"Error saving data to SQLite table '{table_name}': {e_save}")
            print("Please check database path, permissions, and table structure if it exists.")

    else:
        print("DataFrame 'df_statcast_starters' is empty. Nothing to save.")





