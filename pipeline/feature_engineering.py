from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import pandas as pd
import math

def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None) -> SentenceTransformer:
    """
    Initialize and return a sentence transformer model.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        device (str, optional): Device to run the model on ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        SentenceTransformer: Initialized sentence transformer model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the model using the exact pattern from the notebook
    model =  SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Move to specified device
    model.to(device)
    return model

model = get_embedding_model()

def get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Get embeddings for a list of texts using the sentence transformer model.
    
    Args:
        model (SentenceTransformer): Initialized sentence transformer model
        texts (list): List of text strings to embed
        batch_size (int): Batch size for processing
    
    Returns:
        np.ndarray: Array of embeddings
    """
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def add_embedding_columns(df: pd.DataFrame, columns: List[str] = ['name', 'address', 'categories']) -> pd.DataFrame:
    """
    Add embedding columns for specified text columns in the dataframe.
    Each embedding is stored as a single column containing the full vector.
    
    Args:
        df (pd.DataFrame): Input dataframe
        model (SentenceTransformer): Initialized sentence transformer model
        dataset_name (str): Name of the dataset ('ds1' or 'ds2')
        columns (List[str]): List of column names to create embeddings for
    
    Returns:
        pd.DataFrame: DataFrame with new embedding columns
    """
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe")
            continue
            
        # Handle missing values
        df[col] = df[col].fillna('')
        
        # Create embeddings
        embeddings = get_embeddings(df[col].tolist())
        
        # Add embedding as a single column containing the full vector with dataset suffix
        df[f'{col}_embedding'] = [emb for emb in embeddings]
    
    return df

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1 (np.ndarray): First vector
        v2 (np.ndarray): Second vector
        
    Returns:
        float: Cosine similarity between the vectors
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if norm_v1 * norm_v2 != 0 else 0

def extract_features(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from pairs of records and append them to the existing DataFrame.
    New features include:
    - Cosine similarity between name, address, and category embeddings
    - Distance between points
    
    Args:
        pairs_df (pd.DataFrame): DataFrame containing pairs of records with embeddings
        
    Returns:
        pd.DataFrame: DataFrame with additional feature columns
    """
    # Calculate cosine similarities for embeddings
    embedding_fields = ['name', 'address', 'categories']
    for field in embedding_fields:
        emb1 = pairs_df[f'{field}_embedding_df1']
        emb2 = pairs_df[f'{field}_embedding_df2']
        pairs_df[f'{field}_similarity'] = [
            cosine_similarity(e1, e2) 
            for e1, e2 in zip(emb1, emb2)
        ]
    
    # Calculate distance between points if geometry columns exist
    if 'geometry_df1' in pairs_df.columns and 'geometry_df2' in pairs_df.columns:
        from shapely.geometry import Point
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the great circle distance between two points on the earth (specified in decimal degrees)"""
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        
        pairs_df['distance'] = [
            haversine_distance(g1.y, g1.x, g2.y, g2.x) if pd.notna(g1) and pd.notna(g2) else None
            for g1, g2 in zip(pairs_df['geometry_df1'], pairs_df['geometry_df2'])
        ]
    
    return pairs_df
