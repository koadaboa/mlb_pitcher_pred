# Import necessary libraries
import pandas as pd
import argparse
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import re # Keep re in case needed elsewhere, though not for parsing now

# --- Imports/Fallbacks for DBConfig/DBConnection ---
try:
    from src.config import DBConfig
    from src.data.utils import DBConnection
except ImportError:
    print("Error: Could not import DBConfig or DBConnection from src.", file=sys.stderr)
    print("Using fallback configurations.", file=sys.stderr)
    class DBConfig: PATH = "data/pitcher_stats.db" # Default path
    class DBConnection:
        def __init__(self, db_name): self.db_name = db_name
        def __enter__(self):
            Path(self.db_name).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_name); return self.conn
        def __exit__(self,t,v,tb):
            if self.conn: self.conn.close()

# --- Updated Crew List (Pre-formatted) ---
# Use the clean list you generated previously
CREW_LIST_TEXT = """
Bill Miller, Chad Fairchild, Ben May, Roberto Ortiz
Mark Wegner, Bruce Dreckman, Shane Livensparger, Nate Tomlinson
Alfonso Marquez, Lance Barrett, Carlos Torres, TBD
Dan Iassogna, CB Bucknor, Scott Barry, Adam Beck
Mark Carlson, Jordan Baker, Stu Scheurwater, Dan Merzel
Laz Diaz, Brian O'Nora, Mike Estabrook, Erich Bacchus
Ron Kulpa, Cory Blaser, Manny Gonzalez, Alex Tosi
Marvin Hudson, Tripp Gibson, Ryan Blakney, Junior Valentine
Lance Barksdale, Will Little, Ryan Additon, Ryan Wills
James Hoye, DJ Reyburn, John Libka, Sean Barber
Adrian Johnson, Quinn Wolcott, Ramon De Jesus, Paul Clemons
Dan Bellino, Phil Cuzzi, Tony Randazzo, Clint Vondrak
Todd Tichenor, Hunter Wendelstedt, Adam Hamari, Nestor Ceja
Alan Porter, Jim Wolf, Chris Segal, Alex MacKay
Chris Conroy, John Tumpane, Jeremie Rekah, Brennan Miller
Chris Guccione, David Rackley, Chad Whitson, Edwin Moscoso
Andy Fletcher, Rob Drake, Jansen Visconti, Malachi Moore
Doug Eddings, Mike Muchlinski, Gabe Morales, Emil Jimenez
Vic Carapazza, Mark Ripperger, Nic Lentz, Nick Mahrley
"""

# --- SIMPLIFIED Parsing Function for Pre-formatted List ---
def parse_umpire_crews(crew_text):
    """
    Parses the pre-formatted crew list text (First Last, comma-separated)
    into a dictionary mapping Crew Chief name to the full crew list.
    """
    crews_by_chief = {}
    crew_id_counter = 0 # Use a counter for unique keys if needed

    for line in crew_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Split the pre-formatted line by comma and strip whitespace
        members = [name.strip() for name in line.split(',')]

        # Ensure we have roughly 4 members (allow for TBD)
        if 3 <= len(members) <= 4:
            # Use the first name (Chief) as the key, or generate an ID
            chief_name = members[0]
            # Handle potential TBDs by replacing with None
            crew_list = [(member if member.upper() != 'TBD' else None) for member in members]
            # Pad with None if less than 4 members were listed (e.g., if ending TBD was omitted)
            while len(crew_list) < 4:
                crew_list.append(None)

            if chief_name and chief_name.upper() != 'TBD':
                 crews_by_chief[chief_name] = crew_list
                 # print(f"DEBUG: Parsed Crew {chief_name}: {crew_list}") # Optional
            else:
                 # Handle cases where chief might be TBD or line format is unexpected
                 generic_key = f"Crew_{crew_id_counter}"
                 crews_by_chief[generic_key] = crew_list
                 crew_id_counter += 1
                 print(f"Warning: Using generic key '{generic_key}' for line: {line}", file=sys.stderr)

        else:
            print(f"Warning: Skipping line due to unexpected number of members ({len(members)}): '{line}'", file=sys.stderr)

    print(f"Parsed {len(crews_by_chief)} umpire crews.")
    if not crews_by_chief:
        print("ERROR: Failed to parse any crews. Check CREW_LIST_TEXT format.", file=sys.stderr)
    return crews_by_chief

# Define UMPIRE_CREWS using the simplified parser and pre-formatted text
UMPIRE_CREWS = parse_umpire_crews(CREW_LIST_TEXT)

# --- Functions to find crew info ---
def find_crew_and_chief_from_member(umpire_name, crews_dict_by_chief):
    """ Finds the crew list and chief name given any member's name """
    normalized_ump_name = umpire_name.strip().lower()
    for chief, members in crews_dict_by_chief.items():
        # Check if umpire name is in the crew members list
        if any(member and normalized_ump_name in member.lower() for member in members if member):
            return members, chief # Return the full crew list and the chief name
    return None, None # Return None, None if not found

# --- Helper Functions (get_scheduled_games, get_recent_assignments - unchanged) ---
def get_scheduled_games(target_date_str, db_path):
    """Fetches scheduled games for the target date."""
    print(f"Fetching scheduled games for {target_date_str}...")
    query = "SELECT gamePk, home_team_abbr, away_team_abbr FROM mlb_api WHERE game_date = ?"
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            df = pd.read_sql_query(query, conn, params=(target_date_str,))
            if df.empty: print(f"No scheduled games found for {target_date_str}."); return pd.DataFrame()
            print(f"Found {len(df)} scheduled games.")
            return df
    except Exception as e: print(f"Error fetching scheduled games: {e}", file=sys.stderr); return pd.DataFrame()

def get_recent_assignments(lookback_days, target_date, db_path):
    """Fetches umpire assignments from the last few days."""
    print(f"Fetching umpire assignments from the last {lookback_days} days...")
    start_date = target_date - timedelta(days=lookback_days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    target_date_str = target_date.strftime('%Y-%m-%d')
    query = "SELECT game_date, umpire, home_team, away_team FROM umpire_data WHERE game_date >= ? AND game_date < ? ORDER BY game_date DESC"
    try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            cursor = conn.cursor(); cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='umpire_data'")
            if not cursor.fetchone(): print(f"Error: umpire_data table not found", file=sys.stderr); return pd.DataFrame()
            df = pd.read_sql_query(query, conn, params=(start_date_str, target_date_str))
            if df.empty: print(f"Warning: No recent umpire assignments found.")
            else: print(f"Found {len(df)} recent umpire assignments.")
            return df
    except Exception as e: print(f"Error fetching recent umpire assignments: {e}", file=sys.stderr); return pd.DataFrame()


# --- Updated Prediction Function (WITH DEBUGGING) ---
def predict_umpire_rotation(target_date_str, crews_dict_by_chief):
    """
    Predicts today's home plate umpire based on crew list and recent assignments.
    """
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {target_date_str}", file=sys.stderr)
        return pd.DataFrame()

    db_path = DBConfig.PATH
    lookback_days = 5

    # 1. Get today's schedule
    scheduled_games = get_scheduled_games(target_date_str, db_path)
    if scheduled_games.empty: return pd.DataFrame()

    # 2. Get recent historical assignments (up to yesterday)
    recent_assignments = get_recent_assignments(lookback_days, target_date, db_path)
    # Ensure umpire column is string
    if not recent_assignments.empty and 'umpire' in recent_assignments.columns:
         recent_assignments['umpire'] = recent_assignments['umpire'].astype(str)


    predictions = []
    print("\n--- Debugging Predictions ---") # DEBUG START

    # 3. Iterate through today's games
    for _, game in scheduled_games.iterrows():
        home_team = game['home_team_abbr']
        away_team = game['away_team_abbr']
        game_pk = game['gamePk'] # Get gamePk for debugging
        print(f"\nProcessing Game: {away_team} @ {home_team} (gamePk: {game_pk})") # DEBUG

        predicted_ump = "Unknown - New Series or Crew Unclear"
        predicted_chief = "Unknown"
        confidence = "Low"

        # Find the most recent game in this series from historical data
        series_games = pd.DataFrame() # Initialize
        if not recent_assignments.empty:
            series_games = recent_assignments[
                (recent_assignments['home_team'] == home_team) &
                (recent_assignments['away_team'] == away_team)
            ].sort_values(by='game_date', ascending=False)

        if not series_games.empty:
            last_game_date_str = series_games['game_date'].iloc[0]
            print(f"  Found previous game in series on: {last_game_date_str}") # DEBUG
            yesterdays_assignment = series_games[series_games['game_date'] == last_game_date_str].iloc[0]
            # Ensure umpire name is a string
            yesterdays_hp_umpire = str(yesterdays_assignment['umpire']) if pd.notna(yesterdays_assignment['umpire']) else None
            print(f"  Yesterday's HP Umpire from DB: '{yesterdays_hp_umpire}'") # DEBUG

            if yesterdays_hp_umpire:
                # Try to identify the crew based on yesterday's HP umpire
                full_crew, chief_name = find_crew_and_chief_from_member(yesterdays_hp_umpire, crews_dict_by_chief)
                print(f"  Crew Search Result: Chief='{chief_name}', Crew Found={bool(full_crew)}") # DEBUG

                if full_crew and chief_name:
                    predicted_chief = chief_name
                    print(f"  Identified Crew (Chief: {chief_name}): {full_crew}") # DEBUG

                    # Check if crew has TBDs (represented as None)
                    if None not in full_crew:
                        try:
                            # Find the index of yesterday's HP umpire within the crew list
                            hp_index = -1
                            for idx, member in enumerate(full_crew):
                                 if member and yesterdays_hp_umpire.lower() in member.lower():
                                      hp_index = idx
                                      print(f"    Found Yest HP '{yesterdays_hp_umpire}' at index {idx} in crew list.") # DEBUG
                                      break # Found the match

                            if hp_index != -1:
                                # Apply rotation rule: Assume next person in the list gets HP
                                predicted_index = (hp_index + 1) % 4 # Simple wrap-around
                                predicted_ump = full_crew[predicted_index]
                                confidence = "Medium (Assumed Rotation)"
                                print(f"    Predicted HP index: {predicted_index} -> Ump: '{predicted_ump}'") # DEBUG
                            else:
                                predicted_ump = f"Crew '{chief_name}' ID'd, HP Rot Unclear (Yest HP '{yesterdays_hp_umpire}' not in list)"
                                confidence = "Low"
                                print(f"    Could not find index for '{yesterdays_hp_umpire}' in the crew list.") # DEBUG
                        except Exception as e:
                            print(f"    Error applying rotation for crew {chief_name}: {e}") # DEBUG
                            predicted_ump = f"Crew '{chief_name}' ID'd, Rotation Error"
                            confidence = "Low"
                    else:
                        predicted_ump = f"Crew '{chief_name}' ID'd, but Incomplete (TBD)"
                        confidence = "Low"
                        print(f"    Crew '{chief_name}' has TBD members.") # DEBUG
                else:
                    predicted_ump = f"Crew Unknown (Yest HP: {yesterdays_hp_umpire})"
                    confidence = "Low"
                    print(f"    Could not match Yest HP '{yesterdays_hp_umpire}' to any known crew chief.") # DEBUG
            else:
                 predicted_ump = "Crew Unknown (Yesterday's HP missing?)"
                 confidence = "Low"
                 print("    Yesterday's HP Umpire name is missing in DB.") # DEBUG
        else:
             print("  No previous game found for this matchup in recent history (New Series?).") # DEBUG
             predicted_ump = "Unknown - New Series?"
             confidence = "Low"


        # Append prediction details for this game
        predictions.append({
            'gamePk': game_pk,
            'home_team_abbr': home_team,
            'away_team_abbr': away_team,
            'predicted_hp_umpire': predicted_ump,
            'crew_chief': predicted_chief,
            'confidence': confidence
        })

    print("--- End Debugging ---") # DEBUG END
    results_df = pd.DataFrame(predictions)
    return results_df

def find_crew_and_chief_from_member(umpire_name, crews_dict_by_chief):
    """ Finds the crew list and chief name given any member's name """
    normalized_ump_name = umpire_name.strip().lower()
    for chief, members in crews_dict_by_chief.items():
        # Check if umpire name is in the crew members list
        if any(member and normalized_ump_name in member.lower() for member in members if member):
            return members, chief # Return the full crew list and the chief name
    return None, None # Return None, None if not found

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MLB Home Plate Umpires based on crew list and series rotation.")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD) to predict umpires for.")
    args = parser.parse_args()

    # Make sure UMPIRE_CREWS is populated before predicting
    if not UMPIRE_CREWS:
         print("Exiting: Umpire crew list failed to parse.", file=sys.stderr)
         sys.exit(1)

    prediction_results = predict_umpire_rotation(args.date, UMPIRE_CREWS)

    if not prediction_results.empty:
        print(f"\n--- Predicted Umpire Information for {args.date} ---")
        if 'gamePk' in prediction_results.columns:
             prediction_results['gamePk'] = pd.to_numeric(prediction_results['gamePk'], errors='coerce').astype('Int64')
        print(prediction_results.to_string(index=False))
        print("\nNOTE: Predictions are based on the provided 2025 crew list and an *assumed* rotation pattern.")
        print("Actual assignments can vary. Confidence level indicates reliability.")
    else:
        print(f"\nCould not generate umpire predictions for {args.date}.")