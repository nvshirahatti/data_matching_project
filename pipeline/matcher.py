import pandas as pd
import numpy as np
import argparse
import os
import tempfile
from model_gen import EntityMatchingModel
from feature_gen import generate_features

def match_entities(pairs_df, model_path, scaler_path=None, threshold=0.5, batch_size=1000):
    """
    Match entities using a trained model on pre-generated pairs.
    
    Args:
        pairs_df (pd.DataFrame): DataFrame with entity pairs to match
        model_path (str): Path to the saved model
        scaler_path (str, optional): Path to the saved scaler
        threshold (float): Probability threshold for matching
        batch_size (int): Number of pairs to process at once
        
    Returns:
        pd.DataFrame: DataFrame with matched pairs and their probabilities
    """
    # Load the model
    model = EntityMatchingModel.load(model_path, scaler_path)
    
    # Initialize result dataframe
    results = []
    
    # Process in batches to avoid memory issues
    total_pairs = len(pairs_df)
    print(f"Total pairs to process: {total_pairs}")
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(0, total_pairs, batch_size):
            batch_df = pairs_df.iloc[i:i+batch_size]
            
            if len(batch_df) == 0:
                continue
                
            # Save to temporary file
            temp_input = os.path.join(temp_dir, f'pairs_{i}.csv')
            temp_output = os.path.join(temp_dir, f'features_{i}.csv')
            batch_df.to_csv(temp_input, index=False)
            
            # Generate features
            generate_features(temp_input, temp_output)
            features_df = pd.read_csv(temp_output)
            
            # Make predictions
            probabilities = model.predict_proba(features_df)
            match_probabilities = probabilities[:, 1]
            
            # Filter by threshold
            matches = features_df[match_probabilities >= threshold].copy()
            matches['match_probability'] = match_probabilities[match_probabilities >= threshold]
            
            # Add to results
            results.append(matches)
            
            print(f"Processed {len(batch_df)} pairs, found {len(matches)} matches")
    
    # Combine all results
    if results:
        final_results = pd.concat(results, ignore_index=True)
        return final_results
    else:
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Match entities using a trained model")
    parser.add_argument("--pairs", required=True, help="Path to the CSV file with entity pairs")
    parser.add_argument("--model", required=True, help="Path to the saved model")
    parser.add_argument("--scaler", help="Path to the saved scaler")
    parser.add_argument("--output", required=True, help="Path to save the matches")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for matching")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of pairs to process at once")
    
    args = parser.parse_args()
    
    # Load the pairs data
    print(f"Loading pairs data from {args.pairs}...")
    pairs_df = pd.read_csv(args.pairs)
    
    # Match entities
    print("Matching entities...")
    matches = match_entities(
        pairs_df=pairs_df,
        model_path=args.model,
        scaler_path=args.scaler,
        threshold=args.threshold,
        batch_size=args.batch_size
    )
    
    # Save the matches
    print(f"Saving {len(matches)} matches to {args.output}...")
    matches.to_csv(args.output, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
