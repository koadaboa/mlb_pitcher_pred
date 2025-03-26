# src/data/player_mapping.py
import logging
import pandas as pd
from pybaseball import playerid_lookup, playerid_reverse_lookup
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def get_player_id_map(statcast_ids: List[int] = None, 
                      fangraphs_ids: List[int] = None) -> pd.DataFrame:
    """
    Get mapping between Statcast (MLBAM) IDs and FanGraphs IDs using Chadwick Bureau data.
    
    Args:
        statcast_ids: List of Statcast (MLBAM) IDs to map
        fangraphs_ids: List of FanGraphs IDs to map
        
    Returns:
        DataFrame with player IDs mapped across different systems
    """
    mapping_df = pd.DataFrame()
    
    try:
        # Case 1: Map from Statcast IDs to FanGraphs IDs
        if statcast_ids and len(statcast_ids) > 0:
            logger.info(f"Mapping {len(statcast_ids)} Statcast IDs to FanGraphs IDs")
            
            # Process in batches to avoid overwhelming the API
            batch_size = 50
            all_results = []
            
            for i in range(0, len(statcast_ids), batch_size):
                batch = statcast_ids[i:i+batch_size]
                try:
                    # Use reverse lookup to get all IDs for these players
                    batch_results = playerid_reverse_lookup(batch, key_type='mlbam')
                    
                    if not batch_results.empty:
                        all_results.append(batch_results)
                        
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size}: {e}")
            
            if all_results:
                mapping_df = pd.concat(all_results, ignore_index=True)
        
        # Case 2: Map from FanGraphs IDs to Statcast IDs
        elif fangraphs_ids and len(fangraphs_ids) > 0:
            logger.info(f"Mapping {len(fangraphs_ids)} FanGraphs IDs to Statcast IDs")
            
            # Process in batches
            batch_size = 50
            all_results = []
            
            for i in range(0, len(fangraphs_ids), batch_size):
                batch = fangraphs_ids[i:i+batch_size]
                try:
                    # Use reverse lookup to get all IDs for these players
                    batch_results = playerid_reverse_lookup(batch, key_type='fangraphs')
                    
                    if not batch_results.empty:
                        all_results.append(batch_results)
                        
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size}: {e}")
            
            if all_results:
                mapping_df = pd.concat(all_results, ignore_index=True)
        
        if not mapping_df.empty:
            # Rename columns to match our database schema
            mapping_df = mapping_df.rename(columns={
                'key_mlbam': 'statcast_id', 
                'key_fangraphs': 'traditional_id',
                'name_first': 'first_name',
                'name_last': 'last_name'
            })
            
            # Add full name column
            mapping_df['player_name'] = mapping_df['first_name'] + ' ' + mapping_df['last_name']
            
            logger.info(f"Successfully retrieved {len(mapping_df)} player ID mappings")
            return mapping_df
        else:
            logger.warning("No player ID mappings found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error retrieving player ID mappings: {e}")
        return pd.DataFrame()

def lookup_player_by_name(last_name: str, first_name: Optional[str] = None) -> pd.DataFrame:
    """
    Look up a player's IDs by name
    
    Args:
        last_name: Player's last name
        first_name: Player's first name (optional)
        
    Returns:
        DataFrame with player's IDs across different systems
    """
    try:
        player_info = playerid_lookup(last_name, first_name)
        
        if not player_info.empty:
            # Rename columns to match our database schema
            player_info = player_info.rename(columns={
                'key_mlbam': 'statcast_id', 
                'key_fangraphs': 'traditional_id',
                'name_first': 'first_name',
                'name_last': 'last_name'
            })
            
            # Add full name column
            player_info['player_name'] = player_info['first_name'] + ' ' + player_info['last_name']
            
            return player_info
        else:
            logger.warning(f"No player found with name: {first_name} {last_name}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error looking up player by name: {e}")
        return pd.DataFrame()