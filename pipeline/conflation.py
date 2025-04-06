import argparse
import pandas as pd
import pickle
import numpy as np
from typing import Optional, Union
from utils import add_geohash, add_hash_arrays, create_hash_matches, process_geometry_columns, extract_zip, add_unique_key
from feature_engineering import add_embedding_columns, extract_features
import xgboost as xgb

# Or specify which columns to embed
def extract(input_file: str) -> pd.DataFrame:
    """Extract data from a CSV file."""
    print(f"Extracting data from {input_file}...")
    df = pd.read_csv(input_file)
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Perform simple transformation on the data."""
    print("Transforming data...")
    df = add_unique_key(df)
    df = process_geometry_columns(df)
    # Convert postcode to string and extract zip using map
    df['postcode'] = df['postcode'].fillna('').astype(str)
    df['zip'] = df['postcode'].map(extract_zip)
    df = add_embedding_columns(df)
    df = add_geohash(df)
    df = add_hash_arrays(df)
    return df

def load(df: pd.DataFrame, output_file: str):
    """Load data to a CSV file."""
    print(f"Loading data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Data loaded successfully.")

def get_pairs(df1, df2, matches):
    """
    Get pairs of records from df1 and df2 based on matches.
    
    Args:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        matches (pd.DataFrame): DataFrame with matches
        
    Returns:
        pd.DataFrame: DataFrame with pairs of records
    """
    # Create a copy of matches to avoid modifying the original
    pairs = matches.copy()
    
    # Explode the key_df2 list into separate rows
    pairs = pairs.explode('key_df2')
    
    # Drop columns from matches that will be replaced by the merge
    cols_to_drop = ['name_df1', 'name_df2', 'address_df1', 'address_df2', 
                    'postcode_df1', 'postcode_df2']
    pairs = pairs.drop(columns=[col for col in cols_to_drop if col in pairs.columns])
    
    # Rename columns in df1 and df2 before merge
    df1_renamed = df1.copy()
    df2_renamed = df2.copy()
    
    # Add _df1 suffix to all columns in df1 except 'key'
    df1_cols = {col: f"{col}_df1" for col in df1.columns if col != 'key'}
    df1_renamed = df1_renamed.rename(columns=df1_cols)
    
    # Add _df2 suffix to all columns in df2 except 'key'
    df2_cols = {col: f"{col}_df2" for col in df2.columns if col != 'key'}
    df2_renamed = df2_renamed.rename(columns=df2_cols)
    
    # Join with df1
    pairs = pd.merge(
        pairs,
        df1_renamed,
        left_on='key_df1',
        right_on='key',
        how='left'
    )
    
    # Join with df2
    pairs = pd.merge(
        pairs,
        df2_renamed,
        left_on='key_df2',
        right_on='key',
        how='left'
    )
    
    # Drop redundant key columns
    pairs = pairs.drop(columns=['key_x', 'key_y'], errors='ignore')
    
    return pairs

def load_model_and_scaler(model_path: str, scaler_path: str):
    """
    Load the trained model and scaler.
    
    Args:
        model_path (str): Path to the trained model file
        scaler_path (str): Path to the scaler file
        
    Returns:
        tuple: (model, scaler)
    """
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def match_pipeline(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    model_path: Optional[str] = None, 
    scaler_path: Optional[str] = None, 
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Match businesses from two dataframes based on name similarity and location.
    For each record in df1, select the highest scoring match from df2.
    
    Args:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        model_path (Optional[str]): Path to the trained model file
        scaler_path (Optional[str]): Path to the scaler file
        threshold (float): Threshold for match probability. Defaults to 0.5.
        
    Returns:
        pd.DataFrame: DataFrame with matched pairs, including highest scoring match for each df1 record
    """
    # Create matches based on hash arrays
    matches = create_hash_matches(df1, df2)
    
    # Get pairs and extract features
    pairs = get_pairs(df1, df2, matches)
    pairs = extract_features(pairs)
    
    # If model paths are provided, use model to score pairs
    if model_path is not None and scaler_path is not None:
        print("Loading model and scoring pairs...")
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        
        # Prepare features for scoring
        feature_columns = ['name_similarity', 'address_similarity', 'categories_similarity', 'distance']
        X = pairs[feature_columns].fillna(0)
        
        # Scale features if using non-XGBoost model
        if not isinstance(model, xgb.XGBClassifier):
            X = scaler.transform(X)
        
        # Get match probabilities
        probs = model.predict_proba(X)[:, 1]
        pairs['match_probability'] = probs
        
        # For each df1 record, get the highest scoring match
        best_matches = pairs.loc[pairs.groupby('key_df1')['match_probability'].idxmax()]
        
        # Add pred_label column based on threshold
        best_matches['pred_label'] = (best_matches['match_probability'] >= threshold).astype(int)
        
        # Sort by match probability
        best_matches = best_matches.sort_values('match_probability', ascending=False)
        
        # Remove embedding, derived, hashes, geohash, and zip columns
        columns_to_keep = [
            'key_df1', 'key_df2', 'match_count',
            'id_df1', 'provider_df1', 'name_df1', 'address_df1', 'geometry_df1', 
            'categories_df1', 'city_df1', 'country_df1', 'postcode_df1', 
            'mapbox_id_df1',
            'id_df2', 'provider_df2', 'name_df2', 'address_df2', 'geometry_df2',
            'categories_df2', 'city_df2', 'country_df2', 'postcode_df2',
            'mapbox_id_df2',
            'match_probability', 'pred_label'
        ]
        
        # Filter columns
        best_matches = best_matches[columns_to_keep]
        
        return best_matches
    
    # If no model is provided, still remove embedding, hashes, geohash, and zip columns
    columns_to_keep = [
        'key_df1', 'key_df2', 'match_count',
        'id_df1', 'provider_df1', 'name_df1', 'address_df1', 'geometry_df1', 
        'categories_df1', 'city_df1', 'country_df1', 'postcode_df1', 
        'mapbox_id_df1',
        'id_df2', 'provider_df2', 'name_df2', 'address_df2', 'geometry_df2',
        'categories_df2', 'city_df2', 'country_df2', 'postcode_df2',
        'mapbox_id_df2'
    ]
    
    # Filter columns
    pairs = pairs[columns_to_keep]
    
    return pairs

def etl_pipeline(input_file1: str, input_file2: str, output_file: str):
    """Chain the extract, transform, map and load functions."""
    df1 = extract(input_file1)
    df2 = extract(input_file2)
    df1 = transform(df1)
    df2 = transform(df2)
    df = match_pipeline(df1, df2)
    load(df, output_file)

def main():
    parser = argparse.ArgumentParser(description='Match businesses from two datasets.')
    parser.add_argument('--input1', required=True, help='Path to first input CSV file')
    parser.add_argument('--input2', required=True, help='Path to second input CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--model', help='Path to trained model file')
    parser.add_argument('--scaler', help='Path to scaler file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for match probability')
    
    args = parser.parse_args()
    
    # Extract
    df1 = extract(args.input1)
    df2 = extract(args.input2)
    
    # Transform
    df1 = transform(df1)
    df2 = transform(df2)
    
    # Match
    matches = match_pipeline(df1, df2, args.model, args.scaler, args.threshold)
    
    # Load
    load(matches, args.output)

if __name__ == "__main__":
    main()