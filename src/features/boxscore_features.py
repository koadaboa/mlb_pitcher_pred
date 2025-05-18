import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define typical values or methods to calculate them (can be refined)
# These might be calculated on the fly from the non-missing part of the current dataset
# or could be broader defaults if the dataset is too small to get stable typicals.

DEFAULT_HP_UMPIRE = "Unknown Umpire" # Or consider mode if truly necessary
DEFAULT_IS_NIGHT_GAME_MODE = 1 # Assuming more night games, or calculate from data
DEFAULT_ELEVATION_MEAN = 1000 # Example, ideally from data or better source
DEFAULT_TEMPERATURE_MEAN = 70 # Example, ideally from data
DEFAULT_WEATHER_SIMPLIFIED_MODE = "Clear"
DEFAULT_IS_PRECIPITATION_MODE = False
DEFAULT_IS_DOME_WEATHER_MODE = False # Most games are not domes
DEFAULT_WIND_SPEED_MPH_MEAN_OUTDOOR = 7 # Example for non-dome
DEFAULT_WIND_DIRECTION_STR_MODE_OUTDOOR = "Varies" # Or most common actual direction


# --- Private Helper Functions ---

def _calculate_missing_percentage(series: pd.Series) -> float:
    """Calculates the percentage of missing values in a series."""
    return series.isnull().sum() / len(series) * 100

def _parse_temperature(temp_series: pd.Series) -> pd.Series:
    """
    Parses temperature string (e.g., "72 degrees", "72 F", "72") to numeric.
    Handles various formats and potential non-numeric characters.
    """
    if temp_series.empty:
        return pd.Series(dtype=np.float64)
    # Extract numbers, handling cases like "72 F", "72 degrees", or just "72"
    parsed_temps = temp_series.astype(str).str.extract(r'(\d+)').iloc[:, 0]
    return pd.to_numeric(parsed_temps, errors='coerce')


def _parse_wind_data(wind_series: pd.Series) -> pd.DataFrame:
    """
    Parses raw wind string (e.g., "10 mph, Out to CF") into speed and direction.
    Returns a DataFrame with 'parsed_wind_speed_mph' and 'parsed_wind_direction_str'.
    """
    if wind_series.empty:
        return pd.DataFrame(columns=['parsed_wind_speed_mph', 'parsed_wind_direction_str'])

    # Speed: extract first number
    # Ensure we are working with string representation, handle potential float NaNs before astype(str)
    wind_series_str = wind_series.fillna("").astype(str) # fillna("") before str to handle actual NaNs

    speed = wind_series_str.str.extract(r'(\d+)\s*mph', expand=False)
    speed = pd.to_numeric(speed, errors='coerce')

    # Direction:
    # expand=True ensures a DataFrame, n=1 means at most one split
    direction_df = wind_series_str.str.split(r'mph,\s*', n=1, expand=True)

    direction_str = pd.Series(index=wind_series.index, dtype=object)

    # Case 1: Split occurred, and there's text after "mph, " (i.e., column 1 exists and is not empty)
    if 1 in direction_df.columns: # Check if column 1 exists
        # For rows where column 1 is not None and not empty string after strip
        valid_direction_mask = direction_df[1].notna() & (direction_df[1].str.strip() != '')
        direction_str[valid_direction_mask] = direction_df.loc[valid_direction_mask, 1].str.strip()

    # Case 2: No "mph, " delimiter, or no text after it.
    # The original string might be the direction if speed wasn't parsed from it,
    # or it might be something like "0 mph" or "Calm".
    # This part needs to handle "None" for domes when speed is 0.

    # If speed is 0 and the original string contained "None" (case-insensitive)
    # and direction_str is still not set for that row, mark as "Dome Controlled"
    dome_like_cond = (speed == 0) & \
                     (wind_series_str.str.contains("None", case=False, na=False)) & \
                     (direction_str.isnull()) # Only update if not already parsed
    direction_str[dome_like_cond] = "Dome Controlled"

    # Case 3: If no "mph" was found (speed is NaN), the original string might be the direction.
    # Only update if direction_str is still NaN for that row.
    no_mph_speed_is_nan_cond = speed.isnull() & \
                               (wind_series_str != "") & \
                               (direction_str.isnull())
    direction_str[no_mph_speed_is_nan_cond] = wind_series_str[no_mph_speed_is_nan_cond].str.strip()
    
    # If direction_str is still NaN, it means it's truly unparsed or was an empty string initially
    # These will be handled by imputation later if they remain NaN after this function.

    return pd.DataFrame({
        'parsed_wind_speed_mph': speed,
        'parsed_wind_direction_str': direction_str
    })


# --- Main Engineering Function ---

def engineer_boxscore_features(boxscore_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from MLB boxscore data, imputes missing values,
    and checks for high missing percentages.

    Args:
        boxscore_df: DataFrame containing raw boxscore data from 'mlb_boxscores' table.
                     Expected columns: 'game_pk', 'game_date', 'hp_umpire', 'dayNight',
                                     'elevation', 'temp', 'weather', 'wind'.

    Returns:
        DataFrame with new engineered features.

    Raises:
        ValueError: If any key raw column has more than 10% missing values.
    """
    processed_df = boxscore_df.copy()
    cols_to_check_missing = ['hp_umpire', 'dayNight', 'elevation', 'temp', 'weather', 'wind']

    logger.info("Starting boxscore feature engineering.")

    for col in cols_to_check_missing:
        if col not in processed_df.columns:
            logger.warning(f"Raw column '{col}' not found in input DataFrame. Skipping checks and processing for it.")
            # Add placeholder columns for downstream safety if they are critical and not present
            if col == 'hp_umpire': processed_df[col] = DEFAULT_HP_UMPIRE
            elif col == 'dayNight': processed_df[col] = "night" # A common default before processing
            elif col == 'elevation': processed_df[col] = np.nan # Will be imputed
            elif col == 'temp': processed_df[col] = np.nan # Will be imputed
            elif col == 'weather': processed_df[col] = np.nan # Will be imputed
            elif col == 'wind': processed_df[col] = np.nan # Will be imputed
            continue


        missing_pct = _calculate_missing_percentage(processed_df[col])
        logger.info(f"Raw column '{col}': {missing_pct:.2f}% missing values.")
        if missing_pct > 10.0:
            raise ValueError(f"Raw column '{col}' has {missing_pct:.2f}% missing values, which exceeds the 10% threshold.")

    # 1. Home Plate Umpire
    logger.info("Processing 'hp_umpire'...")
    # Impute with a placeholder if missing. Mode imputation for umpires is usually not ideal.
    processed_df['home_plate_umpire'] = processed_df['hp_umpire'].fillna(DEFAULT_HP_UMPIRE)
    if processed_df['hp_umpire'].isnull().any():
        logger.warning(f"Imputed {processed_df['hp_umpire'].isnull().sum()} missing 'hp_umpire' values with '{DEFAULT_HP_UMPIRE}'.")


    # 2. Day/Night
    logger.info("Processing 'dayNight'...")
    processed_df['is_night_game'] = np.where(processed_df['dayNight'].str.lower() == 'night', 1, 0)
    # Impute 'is_night_game' if original 'dayNight' was NaN
    # Calculate mode from non-missing 'dayNight' if available, else use default
    if not processed_df['dayNight'].dropna().empty:
        mode_is_night_game = np.where(processed_df['dayNight'].dropna().str.lower().mode()[0] == 'night', 1, 0)
    else:
        mode_is_night_game = DEFAULT_IS_NIGHT_GAME_MODE
    
    # Identify where original 'dayNight' was null to apply imputation
    # This assumes 'is_night_game' for NaNs in 'dayNight' defaults to 0 due to the np.where condition not being met.
    # We need to specifically target NaNs from original.
    original_dayNight_na_mask = processed_df['dayNight'].isnull()
    processed_df.loc[original_dayNight_na_mask, 'is_night_game'] = mode_is_night_game
    if original_dayNight_na_mask.any():
        logger.warning(f"Imputed {original_dayNight_na_mask.sum()} 'is_night_game' entries (due to missing 'dayNight') with mode value '{mode_is_night_game}'.")


    # 3. Elevation
    logger.info("Processing 'elevation'...")
    processed_df['park_elevation'] = pd.to_numeric(processed_df['elevation'], errors='coerce')
    if not processed_df['park_elevation'].dropna().empty:
        mean_elevation = processed_df['park_elevation'].dropna().mean()
    else:
        mean_elevation = DEFAULT_ELEVATION_MEAN
    num_missing_elevation = processed_df['park_elevation'].isnull().sum()
    if num_missing_elevation > 0:
        processed_df['park_elevation'] = processed_df['park_elevation'].fillna(mean_elevation)
        logger.warning(f"Imputed {num_missing_elevation} missing 'park_elevation' values with mean '{mean_elevation:.2f}'.")

    # 4. Temperature
    logger.info("Processing 'temp'...")
    processed_df['temperature'] = _parse_temperature(processed_df['temp'])
    if not processed_df['temperature'].dropna().empty:
        mean_temp = processed_df['temperature'].dropna().mean()
    else:
        mean_temp = DEFAULT_TEMPERATURE_MEAN
    num_missing_temp = processed_df['temperature'].isnull().sum()
    if num_missing_temp > 0:
        processed_df['temperature'] = processed_df['temperature'].fillna(mean_temp)
        logger.warning(f"Imputed {num_missing_temp} missing 'temperature' values with mean '{mean_temp:.2f}'.")

    # 5. Weather
    # This is more complex as it derives multiple features.
    # Imputation will happen on the derived features if original 'weather' was NaN.
    logger.info("Processing 'weather'...")
    raw_weather_na_mask = processed_df['weather'].isnull()

    # Derive initial features (will be NaN if raw_weather_na_mask is True for a row)
    processed_df['is_dome_weather_raw'] = processed_df['weather'].astype(str).str.contains("Dome|Indoors", case=False, na=False)
    processed_df['is_precipitation_raw'] = processed_df['weather'].astype(str).str.contains("Rain|Snow|Drizzle|Shower", case=False, na=False)

    conditions = [
        processed_df['weather'].astype(str).str.contains("Clear|Sunny", case=False, na=False),
        processed_df['weather'].astype(str).str.contains("Cloudy|Overcast|Partly Cloudy|Hazy", case=False, na=False),
        processed_df['is_precipitation_raw'] & processed_df['weather'].astype(str).str.contains("Snow", case=False, na=False), # Snow takes precedence
        processed_df['is_precipitation_raw'], # Rain/Drizzle if not Snow
        processed_df['is_dome_weather_raw']
    ]
    choices = ["Clear", "Cloudy", "Snow", "Rain", "Dome"]
    processed_df['weather_condition_simplified_raw'] = np.select(conditions, choices, default=np.nan if processed_df['weather'].isnull().all() else "Other") # if weather had values but none matched
    # If raw weather was NaN, simplified should be NaN
    processed_df.loc[raw_weather_na_mask, 'weather_condition_simplified_raw'] = np.nan


    # Impute derived weather features
    # is_dome_weather
    if not processed_df['is_dome_weather_raw'].dropna().empty:
        mode_is_dome = processed_df['is_dome_weather_raw'].dropna().mode()[0]
    else:
        mode_is_dome = DEFAULT_IS_DOME_WEATHER_MODE
    processed_df['is_dome_weather'] = processed_df['is_dome_weather_raw']
    processed_df.loc[raw_weather_na_mask, 'is_dome_weather'] = mode_is_dome # Impute where original weather was NaN
    if raw_weather_na_mask.any(): # Assuming some missing led to imputation for this derived
         logger.warning(f"Imputed 'is_dome_weather' for {raw_weather_na_mask.sum()} entries (due to missing raw 'weather') with mode '{mode_is_dome}'.")

    # is_precipitation
    if not processed_df['is_precipitation_raw'].dropna().empty:
        mode_is_precip = processed_df['is_precipitation_raw'].dropna().mode()[0]
    else:
        mode_is_precip = DEFAULT_IS_PRECIPITATION_MODE
    processed_df['is_precipitation'] = processed_df['is_precipitation_raw']
    processed_df.loc[raw_weather_na_mask, 'is_precipitation'] = mode_is_precip
    # Override precipitation if it's a dome
    processed_df.loc[processed_df['is_dome_weather'] == True, 'is_precipitation'] = False
    if raw_weather_na_mask.any():
        logger.warning(f"Imputed 'is_precipitation' for {raw_weather_na_mask.sum()} entries (due to missing raw 'weather') with mode '{mode_is_precip}' (and set to False for Domes).")


    # weather_condition_simplified
    if not processed_df['weather_condition_simplified_raw'].dropna().empty:
         mode_weather_cond = processed_df['weather_condition_simplified_raw'].dropna().mode()[0]
         # Ensure mode is one of the expected values
         if mode_weather_cond not in choices and mode_weather_cond != "Other": mode_weather_cond = DEFAULT_WEATHER_SIMPLIFIED_MODE
    else:
        mode_weather_cond = DEFAULT_WEATHER_SIMPLIFIED_MODE
    processed_df['weather_condition_simplified'] = processed_df['weather_condition_simplified_raw']
    processed_df.loc[raw_weather_na_mask, 'weather_condition_simplified'] = mode_weather_cond
    # If it's a dome, simplify to "Dome"
    processed_df.loc[processed_df['is_dome_weather'] == True, 'weather_condition_simplified'] = "Dome"
    if raw_weather_na_mask.any():
        logger.warning(f"Imputed 'weather_condition_simplified' for {raw_weather_na_mask.sum()} entries (due to missing raw 'weather') with mode '{mode_weather_cond}' (and set to 'Dome' for Domes).")


    # 6. Wind (depends on imputed is_dome_weather)
    logger.info("Processing 'wind'...")
    raw_wind_na_mask = processed_df['wind'].isnull()
    parsed_wind_df = _parse_wind_data(processed_df['wind'])
    processed_df['parsed_wind_speed_mph'] = parsed_wind_df['parsed_wind_speed_mph']
    processed_df['parsed_wind_direction_str'] = parsed_wind_df['parsed_wind_direction_str']

    # Impute parsed_wind_speed_mph
    # First, set to 0 for domes
    processed_df.loc[processed_df['is_dome_weather'] == True, 'parsed_wind_speed_mph'] = 0
    # Then, impute remaining NaNs (non-dome missing wind speed)
    if not processed_df.loc[processed_df['is_dome_weather'] == False, 'parsed_wind_speed_mph'].dropna().empty:
        mean_wind_speed_outdoor = processed_df.loc[processed_df['is_dome_weather'] == False, 'parsed_wind_speed_mph'].dropna().mean()
    else: # No outdoor wind data to calculate mean
        mean_wind_speed_outdoor = DEFAULT_WIND_SPEED_MPH_MEAN_OUTDOOR

    # Impute NaNs in parsed_wind_speed_mph where it's NOT a dome but wind was missing
    speed_na_outdoor_mask = processed_df['parsed_wind_speed_mph'].isnull() & (processed_df['is_dome_weather'] == False)
    processed_df.loc[speed_na_outdoor_mask, 'parsed_wind_speed_mph'] = mean_wind_speed_outdoor
    num_missing_wind_speed_imputed = speed_na_outdoor_mask.sum()
    if num_missing_wind_speed_imputed > 0:
        logger.warning(f"Imputed {num_missing_wind_speed_imputed} 'parsed_wind_speed_mph' values for outdoor games with mean '{mean_wind_speed_outdoor:.2f}'.")
    processed_df['wind_speed_mph'] = processed_df['parsed_wind_speed_mph']


    # Impute parsed_wind_direction_str
    processed_df.loc[processed_df['is_dome_weather'] == True, 'parsed_wind_direction_str'] = "Dome Controlled"
    if not processed_df.loc[processed_df['is_dome_weather'] == False, 'parsed_wind_direction_str'].dropna().empty:
        mode_wind_dir_outdoor = processed_df.loc[processed_df['is_dome_weather'] == False, 'parsed_wind_direction_str'].dropna().mode()
        mode_wind_dir_outdoor = mode_wind_dir_outdoor[0] if not mode_wind_dir_outdoor.empty else DEFAULT_WIND_DIRECTION_STR_MODE_OUTDOOR
    else: # No outdoor wind direction data
        mode_wind_dir_outdoor = DEFAULT_WIND_DIRECTION_STR_MODE_OUTDOOR

    dir_na_outdoor_mask = processed_df['parsed_wind_direction_str'].isnull() & (processed_df['is_dome_weather'] == False)
    processed_df.loc[dir_na_outdoor_mask, 'parsed_wind_direction_str'] = mode_wind_dir_outdoor
    num_missing_wind_dir_imputed = dir_na_outdoor_mask.sum()
    if num_missing_wind_dir_imputed > 0:
        logger.warning(f"Imputed {num_missing_wind_dir_imputed} 'parsed_wind_direction_str' values for outdoor games with mode '{mode_wind_dir_outdoor}'.")
    # Ensure no NaNs remain in parsed_wind_direction_str before deriving flags
    processed_df['parsed_wind_direction_str'] = processed_df['parsed_wind_direction_str'].fillna("Unknown")


    # Derive final wind features from (now imputed) parsed_wind_speed_mph and parsed_wind_direction_str
    # Wind Speed Category
    speed_bins = [-1, 0, 7, 12, 18, np.inf]
    speed_labels = ["Calm/Dome", "Light Breeze", "Moderate Breeze", "Strong Breeze", "Very Strong Wind"]
    processed_df['wind_speed_category'] = pd.cut(processed_df['wind_speed_mph'], bins=speed_bins, labels=speed_labels, right=True)
    # If wind_speed_mph was 0 and not a dome (e.g. calm outdoor), it'll be "Calm/Dome". This is fine.
    # If is_dome_weather is true, ensure category is "Calm/Dome"
    processed_df.loc[processed_df['is_dome_weather'] == True, 'wind_speed_category'] = "Calm/Dome"


    # is_dome_wind (can be true if weather says dome OR wind implies dome)
    processed_df['is_dome_wind'] = (processed_df['is_dome_weather'] == True) | \
                                  ((processed_df['wind_speed_mph'] == 0) & \
                                   (processed_df['parsed_wind_direction_str'].isin(["Dome Controlled", "None"])))


    # Directional Flags
    # Ensure these are False if it's a dome wind situation
    wind_dir_lower = processed_df['parsed_wind_direction_str'].str.lower()
    processed_df['wind_blowing_out'] = np.where(processed_df['is_dome_wind'], False, wind_dir_lower.str.contains(r'out to|out lf|out rf|out cf|to lf|to rf|to cf', na=False))
    processed_df['wind_blowing_in'] = np.where(processed_df['is_dome_wind'], False, wind_dir_lower.str.contains(r'in from|from cf|from lf|from rf', na=False))
    processed_df['wind_blowing_across_L_to_R'] = np.where(processed_df['is_dome_wind'], False, wind_dir_lower.str.contains(r'l to r|left to right|from left', na=False))
    processed_df['wind_blowing_across_R_to_L'] = np.where(processed_df['is_dome_wind'], False, wind_dir_lower.str.contains(r'r to l|right to left|from right', na=False))

    # wind_direction_varies_or_unknown
    # True if not dome, and not any of the specific directions, or explicitly "varies"/"unknown"
    known_direction_mask = (
        processed_df['wind_blowing_out'] |
        processed_df['wind_blowing_in'] |
        processed_df['wind_blowing_across_L_to_R'] |
        processed_df['wind_blowing_across_R_to_L']
    )
    explicit_varies_unknown = wind_dir_lower.isin(["varies", "unknown"])
    processed_df['wind_direction_varies_or_unknown'] = np.where(processed_df['is_dome_wind'], False, (~known_direction_mask) | explicit_varies_unknown)


    # Clean up intermediate columns
    cols_to_drop = [
        'hp_umpire', 'dayNight', 'elevation', 'temp', 'weather', 'wind', # Raw columns
        'is_dome_weather_raw', 'is_precipitation_raw', 'weather_condition_simplified_raw',
        'parsed_wind_speed_mph', 'parsed_wind_direction_str'
    ]
    processed_df = processed_df.drop(columns=[col for col in cols_to_drop if col in processed_df.columns], errors='ignore')

    logger.info("Finished boxscore feature engineering.")
    return processed_df