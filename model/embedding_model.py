from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import pandas as pd

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
    model = SentenceTransformer(model_name)
    # Move to specified device
    model.to(device)
    return model

def get_embeddings(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
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

def get_top_weight_vectors(embedding: np.ndarray, top_k: int = 5) -> Dict[int, float]:
    """
    Get the top k highest weight vectors from an embedding.
    
    Args:
        embedding (np.ndarray): The embedding vector
        top_k (int): Number of top vectors to return
    
    Returns:
        Dict[int, float]: Dictionary mapping indices to their corresponding weights
    """
    # Get absolute values of the embedding
    abs_weights = np.abs(embedding)
    
    # Get indices of top k highest weights
    top_indices = np.argsort(abs_weights)[-top_k:][::-1]
    
    # Get the actual weights (not absolute values)
    top_weights = embedding[top_indices]
    
    # Create a dictionary mapping indices to weights
    weight_map = {int(idx): float(weight) for idx, weight in zip(top_indices, top_weights)}
    
    return weight_map

def add_embedding_columns(df: pd.DataFrame, model: SentenceTransformer, dataset_name: str, columns: List[str] = ['name', 'address', 'categories']) -> pd.DataFrame:
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
    if dataset_name not in ['ds1', 'ds2']:
        raise ValueError("dataset_name must be either 'ds1' or 'ds2'")
        
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe")
            continue
            
        # Handle missing values
        df[col] = df[col].fillna('')
        
        # Create embeddings
        embeddings = get_embeddings(model, df[col].tolist())
        
        # Add embedding as a single column containing the full vector with dataset suffix
        df[f'{col}_{dataset_name}_embedding'] = [emb for emb in embeddings]
    
    return df 