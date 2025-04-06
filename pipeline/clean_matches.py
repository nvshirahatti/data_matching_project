import pandas as pd
import argparse

def clean_matches(input_path: str, output_path: str) -> None:
    """
    Clean the matches file by removing embedding and derived columns.
    
    Args:
        input_path: Path to the input matches CSV file
        output_path: Path to save the cleaned matches CSV file
    """
    # Read the matches file
    df = pd.read_csv(input_path)
    
    # Columns to keep (excluding embeddings and derived columns)
    columns_to_keep = [
        'key_df1', 'key_df2', 'match_count',
        'id_df1', 'provider_df1', 'name_df1', 'address_df1', 'geometry_df1', 
        'categories_df1', 'city_df1', 'country_df1', 'postcode_df1', 
        'mapbox_id_df1', 'zip_df1', 'geohash_df1', 'hashes_df1',
        'id_df2', 'provider_df2', 'name_df2', 'address_df2', 'geometry_df2',
        'categories_df2', 'city_df2', 'country_df2', 'postcode_df2',
        'mapbox_id_df2', 'zip_df2', 'geohash_df2', 'hashes_df2',
        'match_probability', 'pred_label'
    ]
    
    # Filter columns
    df_cleaned = df[columns_to_keep]
    
    # Save the cleaned matches
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned matches saved to {output_path}")
    print(f"Removed {len(df.columns) - len(columns_to_keep)} columns")

def main():
    parser = argparse.ArgumentParser(description='Clean matches file by removing embedding and derived columns')
    parser.add_argument('--input', type=str, required=True, help='Path to input matches CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save cleaned matches CSV file')
    
    args = parser.parse_args()
    clean_matches(args.input, args.output)

if __name__ == '__main__':
    main() 