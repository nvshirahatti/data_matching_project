import pandas as pd
import numpy as np
from typing import Tuple, List
from utils import process_geometry_columns, extract_zip, add_unique_key, add_geohash, add_hash_arrays, extract_coordinates
from feature_engineering import add_embedding_columns, extract_features
import os
import argparse
import Geohash
from shapely.geometry import Point

class DataGenerator:
    def __init__(self, ds1_path: str, ds2_path: str):
        """
        Initialize the data generator with paths to both datasets.
        
        Args:
            ds1_path (str): Path to first dataset
            ds2_path (str): Path to second dataset
        """
        self.ds1_path = ds1_path
        self.ds2_path = ds2_path
        self.ds1 = None
        self.ds2 = None
        
    def load_data(self) -> None:
        """Load and preprocess both datasets."""
        # Load datasets
        self.ds1 = pd.read_csv(self.ds1_path)
        self.ds2 = pd.read_csv(self.ds2_path)
        
        # Check if mapbox_id column exists
        if 'mapbox_id' not in self.ds1.columns or 'mapbox_id' not in self.ds2.columns:
            raise ValueError("Both datasets must contain a 'mapbox_id' column")
        
        print(f"Initial ds1 columns: {self.ds1.columns.tolist()}")
        print(f"Initial ds2 columns: {self.ds2.columns.tolist()}")
        
        # Process both datasets
        for i, df in enumerate([self.ds1, self.ds2]):
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Process geometry columns
            df = process_geometry_columns(df)
            
            # Extract zip code
            df['zip'] = df['postcode'].apply(lambda x: str(x)[:5] if pd.notna(x) else None)
            
            # Add unique key
            df = add_unique_key(df)
            
            # Add geohash
            df = add_geohash(df)
            
            # Add hash arrays
            df = add_hash_arrays(df)
            
            # Add embedding columns
            df = add_embedding_columns(df)
            
            # Verify mapbox_id is still present
            if 'mapbox_id' not in df.columns:
                print(f"Warning: mapbox_id column was lost during processing for dataset {i+1}")
                # Restore mapbox_id from original dataset
                if i == 0:
                    df['mapbox_id'] = self.ds1['mapbox_id']
                else:
                    df['mapbox_id'] = self.ds2['mapbox_id']
            
            # Update the corresponding dataset
            if i == 0:
                self.ds1 = df
            else:
                self.ds2 = df
        
        print(f"Final ds1 columns: {self.ds1.columns.tolist()}")
        print(f"Final ds2 columns: {self.ds2.columns.tolist()}")
    
    def get_positive_pairs(self) -> pd.DataFrame:
        """
        Get positive pairs by joining on mapbox_id.
        These are guaranteed matches between datasets.
        
        Returns:
            pd.DataFrame: DataFrame containing positive pairs
        """
        if self.ds1 is None or self.ds2 is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Join on mapbox_id
        positive_pairs = pd.merge(
            self.ds1,
            self.ds2,
            on='mapbox_id',
            suffixes=('_df1', '_df2')
        )
        
        # Add label column
        positive_pairs['label'] = 1
        
        # Drop mapbox_id column
        positive_pairs = positive_pairs.drop(columns=['mapbox_id'])
        
        return positive_pairs
    
    def get_negative_pairs(self, max_pairs=None, include_low_distance=True) -> pd.DataFrame:
        """
        Get negative pairs by joining on hash keys and filtering for different mapbox_ids.
        Returns pairs where businesses have similar names (hash overlap) but different mapbox_ids.
        
        Args:
            max_pairs (int, optional): Maximum number of negative pairs to return
            include_low_distance (bool): Whether to include pairs with low distance but different names
            
        Returns:
            pd.DataFrame: DataFrame containing negative pairs
        """
        if self.ds1 is None or self.ds2 is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Method 1: Join on hashes and postcode (original method)
        # Expand the hashes into separate rows
        ds1_expanded = self.ds1.explode('hashes')
        ds2_expanded = self.ds2.explode('hashes')
        
        # Join on hashes and postcode
        merged = pd.merge(
            ds1_expanded,
            ds2_expanded,
            on=['hashes', 'postcode'],
            suffixes=('_df1', '_df2')
        )
        
        # Filter for different mapbox_ids
        negative_pairs = merged[merged['mapbox_id_df1'] != merged['mapbox_id_df2']].copy()
        
        # Add label column (0 for negative pairs)
        negative_pairs['label'] = 0
        
        # Drop duplicates based on mapbox_id pairs
        result = negative_pairs.drop_duplicates(subset=['mapbox_id_df1', 'mapbox_id_df2'])
        
        # Method 2: Join on postcode only (for low distance but different names)
        if include_low_distance:
            # Join on postcode only
            postcode_merged = pd.merge(
                self.ds1,
                self.ds2,
                on='postcode',
                suffixes=('_df1', '_df2')
            )
            
            # Filter for different mapbox_ids
            postcode_pairs = postcode_merged[postcode_merged['mapbox_id_df1'] != postcode_merged['mapbox_id_df2']].copy()
            
            # Add label column (0 for negative pairs)
            postcode_pairs['label'] = 0
            
            # Drop duplicates based on mapbox_id pairs
            postcode_pairs = postcode_pairs.drop_duplicates(subset=['mapbox_id_df1', 'mapbox_id_df2'])
            
            # Combine with the original negative pairs
            result = pd.concat([result, postcode_pairs], ignore_index=True)
            
            # Drop duplicates again
            result = result.drop_duplicates(subset=['mapbox_id_df1', 'mapbox_id_df2'])
        
        # Method 3: Join on geohash (for nearby locations with different names)
        if include_low_distance and 'geohash' in self.ds1.columns and 'geohash' in self.ds2.columns:
            # Create a copy of datasets with truncated geohash (less precise)
            ds1_geo = self.ds1.copy()
            ds2_geo = self.ds2.copy()
            
            # Truncate geohash to 5 characters for less precise matching
            ds1_geo['geohash_truncated'] = ds1_geo['geohash'].str[:5]
            ds2_geo['geohash_truncated'] = ds2_geo['geohash'].str[:5]
            
            # Join on truncated geohash
            geohash_merged = pd.merge(
                ds1_geo,
                ds2_geo,
                on='geohash_truncated',
                suffixes=('_df1', '_df2')
            )
            
            # Filter for different mapbox_ids
            geohash_pairs = geohash_merged[geohash_merged['mapbox_id_df1'] != geohash_merged['mapbox_id_df2']].copy()
            
            # Add label column (0 for negative pairs)
            geohash_pairs['label'] = 0
            
            # Drop duplicates based on mapbox_id pairs
            geohash_pairs = geohash_pairs.drop_duplicates(subset=['mapbox_id_df1', 'mapbox_id_df2'])
            
            # Drop the truncated geohash column
            geohash_pairs = geohash_pairs.drop(columns=['geohash_truncated'])
            
            # Combine with the previous negative pairs
            result = pd.concat([result, geohash_pairs], ignore_index=True)
            
            # Drop duplicates again
            result = result.drop_duplicates(subset=['mapbox_id_df1', 'mapbox_id_df2'])
        
        # Limit the number of pairs if specified
        if max_pairs is not None and len(result) > max_pairs:
            result = result.sample(n=max_pairs, random_state=42)
        
        # Drop mapbox_id columns
        result = result.drop(columns=['mapbox_id_df1', 'mapbox_id_df2'])
        
        return result
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two points in degrees."""
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
    
    def create_synthetic_positive_pairs(self, positive_pairs, max_shift_km=1.0):
        """
        Create synthetic positive pairs by slightly shifting coordinates of existing positive pairs.
        This helps reduce the heavy importance on distance in the model.
        
        Args:
            positive_pairs (pd.DataFrame): Original positive pairs
            max_shift_km (float): Maximum shift in kilometers (approximately 1 km = 0.01 degrees)
            
        Returns:
            pd.DataFrame: DataFrame containing synthetic positive pairs
        """
        if len(positive_pairs) == 0:
            return pd.DataFrame()
        
        # Convert max_shift_km to degrees (approximate)
        max_shift_deg = max_shift_km * 0.01
        
        # Create a copy of positive pairs
        synthetic_pairs = positive_pairs.copy()
        
        # Number of synthetic pairs to create (same as original)
        num_synthetic = len(positive_pairs)
        
        # Create synthetic pairs by shifting coordinates
        synthetic_pairs_list = []
        
        for _ in range(num_synthetic):
            # Sample a random positive pair
            pair = positive_pairs.sample(n=1).iloc[0]
            
            # Create a new pair with slightly shifted coordinates
            new_pair = pair.copy()
            
            # Extract coordinates from geometry
            from utils import extract_coordinates
            
            # Extract coordinates for df2
            coords_df2 = extract_coordinates(pair['geometry_df2'])
            if coords_df2[0] is None or coords_df2[1] is None:
                continue
                
            lat_df2, lon_df2 = coords_df2
            
            # Shift latitude and longitude by a random amount
            lat_shift = np.random.uniform(-max_shift_deg, max_shift_deg)
            lon_shift = np.random.uniform(-max_shift_deg, max_shift_deg)
            
            # Apply shifts to coordinates
            new_lat_df2 = lat_df2 + lat_shift
            new_lon_df2 = lon_df2 + lon_shift
            
            # Create a new geometry string with shifted coordinates
            new_geometry = Point(new_lon_df2, new_lat_df2)
            new_pair['geometry_df2'] = new_geometry
            
            # Update geohash using Geohash directly
            new_pair['geohash_df2'] = Geohash.encode(new_lat_df2, new_lon_df2, 7)
            
            # Add to list
            synthetic_pairs_list.append(new_pair)
        
        # Combine all synthetic pairs
        synthetic_df = pd.DataFrame(synthetic_pairs_list)
        
        # Ensure label is 1 (positive)
        synthetic_df['label'] = 1
        
        return synthetic_df
    
    def cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        if isinstance(v1, str):
            v1 = eval(v1)  # Convert string representation to list
        if isinstance(v2, str):
            v2 = eval(v2)  # Convert string representation to list
        
        # Convert to numpy arrays
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def calculate_name_similarity(self, name1, name2):
        """Calculate similarity between two names using simple string matching."""
        name1 = str(name1).lower()
        name2 = str(name2).lower()
        
        # If names are exactly the same
        if name1 == name2:
            return 1.0
        
        # If one name is contained in the other
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # Calculate word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def calculate_category_similarity(self, categories1, categories2):
        """Calculate similarity between two category strings using Jaccard similarity."""
        if pd.isna(categories1) or pd.isna(categories2):
            return 0.0
        
        # Split categories into sets
        set1 = set(str(categories1).split(';'))
        set2 = set(str(categories2).split(';'))
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def load_and_preprocess(self, file_path):
        """Load and preprocess a dataset."""
        df = pd.read_csv(file_path)
        
        # Process geometry columns
        df = process_geometry_columns(df)
        
        # Extract coordinates from geometry
        df[['latitude', 'longitude']] = pd.DataFrame(
            df['geometry'].apply(extract_coordinates).tolist(),
            index=df.index
        )
        
        # Extract zip code from postcode column
        df['zip'] = df['postcode'].apply(lambda x: str(x)[:5] if pd.notna(x) else None)
        
        # Add unique key
        df = add_unique_key(df)
        
        # Add geohash
        df = add_geohash(df)
        
        # Add hash arrays
        df = add_hash_arrays(df)
        
        # Add embedding columns
        df = add_embedding_columns(df)
        
        return df

    def get_special_negative_pairs(self, min_name_sim=0.4, min_cat_sim=0.8, max_distance=0.0001):
        """
        Get special negative pairs where:
        - Distance is virtually 0 (max_distance)
        - Categories have high embedding similarity (min_cat_sim)
        - Names have moderate embedding similarity (min_name_sim)
        - But mapbox_ids don't match
        
        Args:
            min_name_sim (float): Minimum name embedding similarity threshold
            min_cat_sim (float): Minimum category embedding similarity threshold
            max_distance (float): Maximum distance threshold
            
        Returns:
            pd.DataFrame: DataFrame containing special negative pairs
        """
        if self.ds1 is None or self.ds2 is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Extract coordinates from geometry
        ds1_coords = self.ds1['geometry'].apply(lambda x: (x.y, x.x) if x is not None else (None, None))
        ds2_coords = self.ds2['geometry'].apply(lambda x: (x.y, x.x) if x is not None else (None, None))
        
        self.ds1['latitude'] = ds1_coords.apply(lambda x: x[0])
        self.ds1['longitude'] = ds1_coords.apply(lambda x: x[1])
        self.ds2['latitude'] = ds2_coords.apply(lambda x: x[0])
        self.ds2['longitude'] = ds2_coords.apply(lambda x: x[1])
        
        # Join on postcode to get nearby locations
        merged = pd.merge(
            self.ds1,
            self.ds2,
            on='postcode',  # Use original postcode for joining
            suffixes=('_df1', '_df2')
        )
        
        print("Total pairs after merge:", len(merged))
        
        # Filter for different mapbox_ids
        merged = merged[merged['mapbox_id_df1'] != merged['mapbox_id_df2']]
        print("Pairs after filtering different mapbox_ids:", len(merged))
        
        # Calculate distances
        merged['distance'] = merged.apply(
            lambda row: self.calculate_distance(
                row['latitude_df1'], row['longitude_df1'],
                row['latitude_df2'], row['longitude_df2']
            ) if pd.notna(row['latitude_df1']) and pd.notna(row['latitude_df2']) else float('inf'),
            axis=1
        )
        
        # Calculate name similarity using embeddings
        merged['name_similarity'] = merged.apply(
            lambda row: self.cosine_similarity(
                row['name_embedding_df1'],
                row['name_embedding_df2']
            ),
            axis=1
        )
        
        # Calculate category similarity using embeddings
        merged['categories_similarity'] = merged.apply(
            lambda row: self.cosine_similarity(
                row['categories_embedding_df1'],
                row['categories_embedding_df2']
            ),
            axis=1
        )
        
        # Print distribution of similarities and distances
        print("\nDistribution of metrics:")
        print("Distance percentiles:", merged['distance'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
        print("Name similarity percentiles:", merged['name_similarity'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
        print("Category similarity percentiles:", merged['categories_similarity'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
        
        # Adjust thresholds based on data distribution
        max_distance = 0.002  # Increased to catch more nearby pairs
        min_name_sim = 0.25   # Decreased to catch more name variations
        min_cat_sim = 0.5    # Decreased to allow more category variations
        
        # Filter for our special criteria
        special_pairs = merged[
            (merged['distance'] <= max_distance) &
            (merged['name_similarity'] >= min_name_sim) &
            (merged['categories_similarity'] >= min_cat_sim)
        ]
        
        print("\nPairs meeting each criterion:")
        print("Distance ≤", max_distance, ":", len(merged[merged['distance'] <= max_distance]))
        print("Name similarity ≥", min_name_sim, ":", len(merged[merged['name_similarity'] >= min_name_sim]))
        print("Category similarity ≥", min_cat_sim, ":", len(merged[merged['categories_similarity'] >= min_cat_sim]))
        print("All criteria combined:", len(special_pairs))
        
        if len(special_pairs) > 0:
            print("\nExample special pairs:")
            sample = special_pairs.head(5)  # Show more examples
            for _, row in sample.iterrows():
                print(f"\nPair with distance {row['distance']:.6f}:")
                print(f"Name 1: {row['name_df1']} | Name 2: {row['name_df2']} (sim: {row['name_similarity']:.3f})")
                print(f"Categories 1: {row['categories_df1']} | Categories 2: {row['categories_df2']} (sim: {row['categories_similarity']:.3f})")
        
        # Add label column (0 for negative pairs)
        special_pairs['label'] = 0
        
        # Select and rename columns to match our standard format
        result = special_pairs[[
            'name_df1', 'name_df2', 
            'address_df1', 'address_df2',
            'postcode',  # This is the common postcode column
            'categories_df1', 'categories_df2',
            'latitude_df1', 'longitude_df1', 
            'latitude_df2', 'longitude_df2',
            'distance', 'name_similarity', 
            'categories_similarity', 'label'
        ]]
        
        # Duplicate the postcode column to match expected format
        result['postcode_df1'] = result['postcode']
        result['postcode_df2'] = result['postcode']
        result = result.drop(columns=['postcode'])
        
        return result

    def generate_training_data(self, max_negative_pairs=None, include_low_distance=True, max_shift_km=1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate training data with both positive and negative examples.
        
        Args:
            max_negative_pairs (int, optional): Maximum number of negative pairs to include
            include_low_distance (bool): Whether to include pairs with low distance but different names
            max_shift_km (float): Maximum shift in kilometers for synthetic positive pairs
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (training_data, validation_data)
        """
        # Load and preprocess data
        self.load_data()
        
        # Get positive and negative pairs
        positive_pairs = self.get_positive_pairs()
        negative_pairs = self.get_negative_pairs(max_pairs=max_negative_pairs, include_low_distance=include_low_distance)
        
        # Get special negative pairs
        special_negative_pairs = self.get_special_negative_pairs(
            min_name_sim=0.4,
            min_cat_sim=0.8,
            max_distance=0.0001
        )
        
        # Create synthetic positive pairs by slightly shifting coordinates
        synthetic_positive_pairs = self.create_synthetic_positive_pairs(positive_pairs, max_shift_km=max_shift_km)
        
        # Combine original and synthetic positive pairs
        all_positive_pairs = pd.concat([positive_pairs, synthetic_positive_pairs], ignore_index=True)
        
        print("Count of original positive pairs:", len(positive_pairs))
        print("Count of synthetic positive pairs:", len(synthetic_positive_pairs))
        print("Count of regular negative pairs:", len(negative_pairs))
        print("Count of special negative pairs:", len(special_negative_pairs))

        # Combine all pairs
        all_pairs = pd.concat([all_positive_pairs, negative_pairs, special_negative_pairs], ignore_index=True)
        
        # Extract features
        all_pairs = extract_features(all_pairs)
        
        # Shuffle the data
        all_pairs = all_pairs.sample(frac=1, random_state=42)
        
        # Split into train and validation (80-20)
        train_size = int(0.8 * len(all_pairs))
        train_data = all_pairs[:train_size]
        val_data = all_pairs[train_size:]
        
        return train_data, val_data

def main():
    """
    Main function to generate labeled data from command line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate training data for entity matching')
    parser.add_argument('--ds1', required=True, help='Path to first dataset')
    parser.add_argument('--ds2', required=True, help='Path to second dataset')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max-negative', type=int, default=None, help='Maximum number of negative pairs to generate')
    parser.add_argument('--include-low-distance', action='store_true', help='Include low distance negative pairs')
    parser.add_argument('--min-distance', type=float, default=0.0001, help='Minimum distance for positive pairs')
    parser.add_argument('--max-shift-km', type=float, default=1.0, help='Maximum shift in kilometers for synthetic positive pairs')
    
    args = parser.parse_args()
    
    # Initialize data generator
    generator = DataGenerator(args.ds1, args.ds2)
    
    # Load and preprocess datasets
    df1 = generator.load_and_preprocess(args.ds1)
    df2 = generator.load_and_preprocess(args.ds2)
    
    # Generate training data
    train_data, val_data = generator.generate_training_data(
        max_negative_pairs=args.max_negative,
        include_low_distance=args.include_low_distance,
        max_shift_km=args.max_shift_km
    )
    
    # Save the data
    os.makedirs(args.output, exist_ok=True)
    train_data.to_csv(os.path.join(args.output, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(args.output, 'val_data.csv'), index=False)
    
    print(f"Generated {len(train_data)} training examples and {len(val_data)} validation examples")
    print(f"Positive pairs: {len(train_data[train_data['label'] == 1]) + len(val_data[val_data['label'] == 1])}")
    print(f"Negative pairs: {len(train_data[train_data['label'] == 0]) + len(val_data[val_data['label'] == 0])}")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main() 