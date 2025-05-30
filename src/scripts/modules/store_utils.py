import sqlite3
import logging
from src.utils import DBConnection
from .fetch_utils import dedup_pitch_df
logger = logging.getLogger(__name__)


def store_data_to_sql(df, table_name, db_path, if_exists='append'):
    """Stores DataFrame to SQLite table with dynamic chunksize, robust logging, and context manager."""
    if df is None or df.empty:
        logger.debug(f"Empty DataFrame provided for '{table_name}'. Skipping database save.")
        return True # Nothing to save is considered success

    if table_name == 'statcast_pitchers':
        df = dedup_pitch_df(df)

    db_path_str = str(db_path)
    num_columns = len(df.columns)
    if num_columns == 0:
        logger.warning(f"DataFrame for '{table_name}' has 0 columns. Cannot store.")
        return False # Cannot store a DF with no columns

    # Calculate dynamic chunksize based on SQLite variable limit
    SQLITE_MAX_VARS = 30000 # Max variables often around 32766, use a safer limit
    # Ensure num_columns > 0 (checked above)
    pandas_chunksize = max(1, SQLITE_MAX_VARS // num_columns)
    # Cap chunksize to avoid excessive memory usage per chunk (e.g., 1000 rows max)
    pandas_chunksize = min(pandas_chunksize, 1000)
    variables_per_chunk = num_columns * pandas_chunksize

    logger.info(f"Storing {len(df)} records to table '{table_name}' in database '{db_path_str}' (mode: {if_exists}, chunksize: {pandas_chunksize}, vars/chunk: ~{variables_per_chunk})...")

    try:
        # Use the DBConnection context manager
        with DBConnection(db_path_str) as conn:
            if conn is None:
                # The context manager should ideally raise an error if connection fails
                # but we add a check just in case it returns None silently.
                raise ConnectionError(f"DBConnection failed to establish connection to {db_path_str}")

            # Handle 'replace' logic: Drop table before writing
            if if_exists == 'replace':
                logger.info(f"Attempting to drop table '{table_name}' before replacing...")
                try:
                    cursor = conn.cursor()
                    # Use standard SQL syntax for dropping table if exists
                    cursor.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")
                    conn.commit()
                    logger.info(f"Dropped existing table '{table_name}' (if it existed).")
                except sqlite3.Error as drop_e:
                    # Log warning but proceed; to_sql might still work if table didn't exist
                    logger.warning(f"Could not explicitly drop table '{table_name}': {drop_e}. Continuing with to_sql...")


            # Use pandas to_sql for writing data
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists, # Let pandas handle append/replace logic after potential drop
                index=False,
                chunksize=pandas_chunksize,
                method='multi' # Generally recommended for performance with chunking
            )
            logger.info(f"Finished storing data to '{table_name}'.")
            # Context manager handles commit/close on exit
            return True

    # Specific error handling for SQLite operational errors like "too many SQL variables"
    except sqlite3.OperationalError as oe:
         logger.error(f"SQLite OperationalError storing data to '{table_name}': {oe}", exc_info=True)
         if 'too many SQL variables' in str(oe).lower():
             logger.error(f"DYNAMIC CHUNKSIZE FAILED. Calculated chunksize ({pandas_chunksize}) for {num_columns} columns exceeded SQLite variable limit.")
         elif 'has no column named' in str(oe).lower():
             logger.error(f"Schema mismatch? Error indicates table '{table_name}' is missing an expected column from the DataFrame.")
             logger.error(f"DataFrame columns: {df.columns.tolist()}")
         else:
              logger.error(f"Unhandled SQLite OperationalError: {oe}")
         # Log traceback for detailed debugging
         # logger.error(traceback.format_exc()) # Redundant with exc_info=True
         return False # Indicate failure

    # Catch potential connection errors if DBConnection fails
    except ConnectionError as ce:
        logger.error(f"Database connection error for '{db_path_str}': {ce}", exc_info=True)
        return False

    # Catch other general exceptions during the process
    except Exception as e:
         logger.error(f"General Error storing data to table '{table_name}': {e}", exc_info=True)
         # logger.error(traceback.format_exc()) # Redundant with exc_info=True
         return False # Indicate failure

