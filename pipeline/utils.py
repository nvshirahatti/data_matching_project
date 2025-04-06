from shapely.geometry import Point, Polygon, shape
import json
import pandas as pd
import hashlib
from typing import List
import Geohash  # python-geohash package

# Try to import geohash, provide a fallback if not available
try:
    import geohash
    GEOHASH_AVAILABLE = True
except ImportError:
    print("Warning: geohash package not available. Geohash functionality will be disabled.")
    GEOHASH_AVAILABLE = False

def extract_zip(postcode):
    """Extract zip code from postcode, handling both single values and Series."""
    if isinstance(postcode, pd.Series):
        return postcode.apply(lambda x: str(x)[:5] if pd.notna(x) else None)
    if pd.isna(postcode):
        return None
    # Convert to string and take first 5 characters
    return str(postcode)[:5]

def parse_geometry(geom_str):
    if pd.isna(geom_str):
        return None
    try:
        geom_dict = json.loads(geom_str)
        return shape(geom_dict)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Failed to parse geometry: {geom_str}")
        print(f"Error: {str(e)}")
        return None

def process_geometry_columns(df):
    """Process geometry columns in a dataframe"""
    if 'geometry' in df.columns:
        df['geometry'] = df['geometry'].apply(parse_geometry)
    return df

def add_unique_key(df):
    """
    Add a unique key column to the dataframe by combining id and provider columns.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'id' and 'provider' columns
        
    Returns:
        pd.DataFrame: DataFrame with added 'key' column
    """
    df = df.copy()
    df["key"] = df['id'].astype(str) + '_' + df['provider'].astype(str)
    return df

def create_ngram_hashes(text: str, postcode: str, n: int = 3) -> List[str]:
    """
    Create n-gram hashes from a text string, appending postcode to each hash.
    
    Args:
        text (str): Input text to create n-grams from
        postcode (str): Postcode to append to each hash
        n (int): Size of n-grams (default: 3)
        
    Returns:
        List[str]: List of n-gram hashes with postcode
    """
    if pd.isna(text) or pd.isna(postcode):
        return []
        
    # Clean and normalize text
    text = str(text).lower().strip()
    postcode = str(postcode).strip()
    
    # Create n-grams
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    # Create hashes for each n-gram with postcode
    hashes = [hashlib.md5((ng + '_' + postcode).encode()).hexdigest() for ng in ngrams]
    
    return hashes

def add_hash_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hash arrays to the dataframe based on n-grams of the name field with postcode,
    and include geohash in the array.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'name', 'zip', and 'geohash' columns
        
    Returns:
        pd.DataFrame: DataFrame with added 'hashes' column
    """
    df = df.copy()
    
    # Create hash arrays from name field with postcode
    df['hashes'] = df.apply(lambda row: create_ngram_hashes(row['name'], row['zip']), axis=1)
    
    # Add geohash to the array of hashes if available
    df['hashes'] = df.apply(
        lambda row: row['hashes'] + [row['geohash']] if pd.notna(row['geohash']) else row['hashes'],
        axis=1
    )
    
    return df

def create_hash_matches(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Create matches between df1 and df2 based on individual hash matches.
    Expands the hashes arrays and joins on individual hashes.
    
    Args:
        df1 (pd.DataFrame): First dataframe with 'hashes' column
        df2 (pd.DataFrame): Second dataframe with 'hashes' column
        
    Returns:
        pd.DataFrame: DataFrame with matches from df1 to df2
    """
    # Create copies to avoid modifying originals
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Expand hashes into separate rows
    df1_expanded = df1.explode('hashes')
    df2_expanded = df2.explode('hashes')
    
    # Create matches based on hashes
    matches = pd.merge(
        df1_expanded,
        df2_expanded,
        on='hashes',
        suffixes=('_df1', '_df2')
    )
    
    # Group by df1 key to get all matching df2 rows
    matches_grouped = matches.groupby('key_df1').agg({
        'key_df2': lambda x: list(x.unique()),
        'name_df1': 'first',
        'name_df2': lambda x: list(x.unique()),
        'address_df1': 'first',
        'address_df2': lambda x: list(x.unique()),
        'postcode_df1': 'first',
        'postcode_df2': lambda x: list(x.unique())
    }).reset_index()
    
    # Add count of matches
    matches_grouped['match_count'] = matches_grouped['key_df2'].apply(len)
    
    return matches_grouped

def extract_coordinates(geom):
    """
    Extract latitude and longitude from a geometry object.
    
    Args:
        geom: Shapely geometry object
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if invalid
    """
    if geom is None:
        return None, None
    try:
        if isinstance(geom, Point):
            return geom.y, geom.x  # latitude, longitude
        elif isinstance(geom, Polygon):
            # Use centroid for polygons
            centroid = geom.centroid
            return centroid.y, centroid.x
        else:
            return None, None
    except:
        return None, None

def add_geohash(df: pd.DataFrame, precision: int = 7) -> pd.DataFrame:
    """
    Add geohash column to dataframe based on geometry coordinates.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'geometry' column
        precision (int): Length of geohash (default: 7)
        
    Returns:
        pd.DataFrame: DataFrame with added 'geohash' column
    """
    df = df.copy()
    
    # Extract coordinates from geometry
    df[['lat', 'lon']] = pd.DataFrame(
        df['geometry'].apply(extract_coordinates).tolist(),
        index=df.index
    )
    
    # Create geohash
    df['geohash'] = None
    mask = df['lat'].notna() & df['lon'].notna()
    df.loc[mask, 'geohash'] = df.loc[mask].apply(
        lambda row: Geohash.encode(row['lat'], row['lon'], precision),
        axis=1
    )
    
    # Drop temporary columns
    df = df.drop(['lat', 'lon'], axis=1)
    
    return df