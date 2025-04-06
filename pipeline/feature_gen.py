import pandas as pd
import numpy as np
import argparse
import ast
import json
import re
import math
from shapely.geometry import Point
from feature_engineering import extract_features

def convert_string_to_array(string_repr):
    """Convert string representation of array to numpy array"""
    try:
        # Handle empty or NaN values
        if pd.isna(string_repr) or string_repr == '[]':
            return np.array([])
        
        # Clean up the string representation
        # Remove line breaks and extra spaces
        cleaned = re.sub(r'\s+', ' ', string_repr)
        # Remove any trailing commas
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Try to parse as a list
        try:
            # First try with ast.literal_eval
            return np.array(ast.literal_eval(cleaned))
        except (ValueError, SyntaxError):
            # If that fails, try to parse as a space-separated list of numbers
            try:
                # Extract numbers using regex
                numbers = re.findall(r'-?\d+\.?\d*e?-?\d*', cleaned)
                return np.array([float(num) for num in numbers])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {string_repr} to array")
                return np.array([])
    except Exception as e:
        print(f"Warning: Error converting {string_repr} to array: {str(e)}")
        return np.array([])

def convert_string_to_point(string_repr):
    """Convert string representation of geometry to Shapely Point"""
    try:
        # Handle empty or NaN values
        if pd.isna(string_repr) or string_repr == '{}':
            return None
        
        # Check if it's a WKT format (POINT format)
        if string_repr.startswith('POINT'):
            # Extract coordinates from WKT format
            coords_str = string_repr.replace('POINT (', '').replace(')', '')
            try:
                lon, lat = map(float, coords_str.split())
                return Point(lon, lat)
            except (ValueError, TypeError):
                print(f"Warning: Could not parse coordinates from WKT: {string_repr}")
                return None
        
        # Try to parse as JSON
        try:
            geom_dict = json.loads(string_repr)
            
            # Extract coordinates
            if 'coordinates' in geom_dict and len(geom_dict['coordinates']) >= 2:
                lon, lat = geom_dict['coordinates'][0], geom_dict['coordinates'][1]
                return Point(lon, lat)
            else:
                print(f"Warning: Invalid geometry format: {string_repr}")
                return None
        except json.JSONDecodeError:
            # If not JSON, try to extract coordinates using regex
            coords_match = re.search(r'\(([-\d.]+)\s+([-\d.]+)\)', string_repr)
            if coords_match:
                try:
                    lon, lat = float(coords_match.group(1)), float(coords_match.group(2))
                    return Point(lon, lat)
                except (ValueError, TypeError):
                    pass
            
            print(f"Warning: Could not parse geometry: {string_repr}")
            return None
    except Exception as e:
        print(f"Warning: Error converting geometry {string_repr} to Point: {str(e)}")
        return None

def calculate_distance(point1, point2):
    """Calculate distance between two points using Haversine formula (in meters)"""
    if point1 is None or point2 is None:
        return None
    
    try:
        # Extract coordinates
        lon1, lat1 = point1.x, point1.y
        lon2, lat2 = point2.x, point2.y
        
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Calculate distance in meters
        distance = R * c
        return distance
    except Exception as e:
        print(f"Warning: Error calculating distance: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess embeddings and geometry data"""
    # Process embeddings
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    for col in embedding_cols:
        print(f"Processing embedding column: {col}")
        df[col] = df[col].apply(convert_string_to_array)
        # Check if conversion was successful
        sample = df[col].iloc[0] if not df[col].empty else None
        if sample is not None and hasattr(sample, 'shape'):
            print(f"Sample from {col}: {type(sample)}, shape: {sample.shape}")
        else:
            print(f"Sample from {col}: {type(sample)}")
    
    # Process geometry columns
    geometry_cols = [col for col in df.columns if 'geometry' in col.lower()]
    for col in geometry_cols:
        print(f"Processing geometry column: {col}")
        df[col] = df[col].apply(convert_string_to_point)
    
    # Calculate distances between points if both geometry columns exist
    if 'geometry_df1' in df.columns and 'geometry_df2' in df.columns:
        print("Calculating distances between points...")
        df['distance'] = df.apply(lambda row: calculate_distance(row['geometry_df1'], row['geometry_df2']), axis=1)
        print(f"Distance statistics: min={df['distance'].min()}, max={df['distance'].max()}, mean={df['distance'].mean()}")
    
    return df

def generate_features(input_file: str, output_file: str) -> None:
    """
    Generate features from the input pairs data.
    
    Args:
        input_file (str): Path to the input CSV file containing pairs
        output_file (str): Path to save the output CSV file with features
    """
    print(f"Reading pairs from {input_file}...")
    pairs_df = pd.read_csv(input_file)
    
    print("Preprocessing data...")
    pairs_df = preprocess_data(pairs_df)
    
    print("Generating features...")
    # Extract features using the extract_features function
    features_df = extract_features(pairs_df)
    
    # Ensure important columns are included in the final features
    columns_to_preserve = ['distance', 'label']
    for col in columns_to_preserve:
        if col in pairs_df.columns and col not in features_df.columns:
            print(f"Adding {col} column to features...")
            features_df[col] = pairs_df[col]
    
    print(f"Saving features to {output_file}...")
    features_df.to_csv(output_file, index=False)
    print("Feature generation complete!")
    
    # Print feature statistics
    print("\nFeature Statistics:")
    print(features_df.describe())
    
    # Print sample of similarity features
    print("\nSample of similarity features:")
    similarity_cols = [col for col in features_df.columns if 'similarity' in col.lower()]
    if similarity_cols:
        print(features_df[similarity_cols].head())
    
    # Print distance statistics if available
    if 'distance' in features_df.columns:
        print("\nDistance Statistics:")
        print(features_df['distance'].describe())
    
    # Print label distribution
    if 'label' in features_df.columns:
        print("\nLabel Distribution:")
        print(features_df['label'].value_counts())
        print(f"Positive pairs: {features_df['label'].sum()}")
        print(f"Negative pairs: {len(features_df) - features_df['label'].sum()}")

def main():
    parser = argparse.ArgumentParser(description="Generate features from pairs data")
    parser.add_argument("--input", required=True, help="Path to the input CSV file containing pairs")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file with features")
    
    args = parser.parse_args()
    generate_features(args.input, args.output)

if __name__ == "__main__":
    main() 