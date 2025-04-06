import unittest
import pandas as pd
import os
import json
from data_gen import DataGenerator

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        # Create test data directory
        os.makedirs('data', exist_ok=True)
        
        # Create sample datasets with known mapbox_ids
        self.ds1 = pd.DataFrame({
            'id': range(1, 6),
            'provider': ['Provider 1'] * 5,
            'name': ['Business A', 'Business B', 'Business C', 'Business D', 'Business E'],
            'address': ['Address 1', 'Address 2', 'Address 3', 'Address 4', 'Address 5'],
            'geometry': [json.dumps({"coordinates": [-96.7966, 32.7767], "type": "Point"})] * 5,
            'categories': ['restaurant;food', 'retail;shopping', 'restaurant;food', 'retail;shopping', 'restaurant;food'],
            'city': ['Dallas'] * 5,
            'country': ['US'] * 5,
            'postcode': ['75201', '75201', '75202', '75202', '75203'],
            'mapbox_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
            'hashes': [
                ['hash1', 'hash2'],
                ['hash3', 'hash4'],
                ['hash1', 'hash5'],
                ['hash3', 'hash6'],
                ['hash7', 'hash8']
            ]
        })
        
        self.ds2 = pd.DataFrame({
            'id': range(1, 6),
            'provider': ['Provider 2'] * 5,
            'name': ['Business A', 'Business B', 'Business C', 'Business D', 'Business E'],
            'address': ['Address 1', 'Address 2', 'Address 3', 'Address 4', 'Address 5'],
            'geometry': [json.dumps({"coordinates": [-96.7966, 32.7767], "type": "Point"})] * 5,
            'categories': ['restaurant;food', 'retail;shopping', 'restaurant;food', 'retail;shopping', 'restaurant;food'],
            'city': ['Dallas'] * 5,
            'country': ['US'] * 5,
            'postcode': ['75201', '75201', '75202', '75202', '75203'],
            'mapbox_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
            'hashes': [
                ['hash1', 'hash2'],
                ['hash3', 'hash4'],
                ['hash1', 'hash5'],
                ['hash3', 'hash6'],
                ['hash7', 'hash8']
            ]
        })
        
        # Save test data
        self.ds1.to_csv('data/data_source_1.csv', index=False)
        self.ds2.to_csv('data/data_source_2.csv', index=False)
        
        # Initialize DataGenerator
        self.generator = DataGenerator('data/data_source_1.csv', 'data/data_source_2.csv')
        self.generator.load_data()
        
    def test_get_negative_pairs(self):
        # Get negative pairs
        negative_pairs = self.generator.get_negative_pairs()
        
        # Basic checks
        self.assertFalse(negative_pairs.empty, "Should generate negative pairs")
        self.assertTrue('label' in negative_pairs.columns, "Should have label column")
        self.assertTrue(all(negative_pairs['label'] == 0), "All pairs should have label 0")
        
        # Print all columns in negative pairs for debugging
        print("\nNegative pairs (all columns):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(negative_pairs.to_string())
        
    def test_get_positive_pairs(self):
        # Get positive pairs
        positive_pairs = self.generator.get_positive_pairs()
        
        # Basic checks
        self.assertFalse(positive_pairs.empty, "Should generate positive pairs")
        self.assertTrue('label' in positive_pairs.columns, "Should have label column")
        self.assertTrue(all(positive_pairs['label'] == 1), "All pairs should have label 1")
        
        # Print all columns in positive pairs for debugging
        print("\nPositive pairs (all columns):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(positive_pairs.to_string())
        

if __name__ == '__main__':
    unittest.main() 