# src/data/utils.py
import re
import logging

logger = logging.getLogger(__name__)

def normalize_name(name):
    """
    Normalize player names for better matching
    
    Args:
        name (str): Player name to normalize
        
    Returns:
        str: Normalized player name
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove all accented characters
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace('ñ', 'n').replace('ç', 'c').replace('ü', 'u')
    
    # Remove suffixes like Jr., Sr., III
    name = re.sub(r'\b(jr|jr\.|sr|sr\.|iii|ii|iv)\b', '', name)
    
    # Remove all punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Handle lastname, firstname format
    if ", " in name:
        parts = name.split(", ", 1)
        if len(parts) == 2:
            last, first = parts
            name = f"{first} {last}"
    
    return name