import openai
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Any, Callable, Optional

class EmbeddingGenerator:
    """
    Generic class to generate embeddings for any text data using OpenAI's embedding models.
    """
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-large"):
        """
        Initialize the embedder with model and optional API key.
        If no API key is provided, it will use the OPENAI_API_KEY environment variable.
        
        Args:
            api_key: OpenAI API key (optional)
            model: Embedding model to use (default is text-embedding-3-large)
        """
        # Use provided API key or fall back to environment variable
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def get_embedding(self, text: str, dimensions: Optional[int] = None) -> List[float]:
        """
        Get embedding vector for a single text string.
        
        Args:
            text: Text to embed
            dimensions: Optional parameter to reduce dimensions of the embedding
            
        Returns:
            Embedding vector as list of floats
        """
        text = text.replace("\n", " ")  # Normalize text
        params = {
            "input": text,
            "model": self.model
        }
        
        if dimensions is not None:
            params["dimensions"] = dimensions
            
        response = self.client.embeddings.create(**params)
        return response.data[0].embedding
    
    def get_embeddings(self, texts: List[str], dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            dimensions: Optional parameter to reduce dimensions of the embedding
            
        Returns:
            List of embedding vectors
        """
        # Filter out None values and normalize all texts
        filtered_texts = [text.replace("\n", " ") for text in texts if text and not pd.isna(text)]
        
        # Handle empty list case
        if not filtered_texts:
            return []
        
        # Check if any texts are empty strings after filtering
        filtered_texts = [text if text.strip() else "Empty description" for text in filtered_texts]
        
        # Create mapping from original to filtered index
        index_map = {}
        for i, text in enumerate(texts):
            if text and not pd.isna(text):
                index_map[i] = len(index_map)
        
        params = {
            "input": filtered_texts,
            "model": self.model
        }
        
        if dimensions is not None:
            params["dimensions"] = dimensions
        
        response = self.client.embeddings.create(**params)
        
        # Create result list with same length as input
        result = [None] * len(texts)
        
        # Fill in embeddings for non-empty texts
        for orig_idx, filtered_idx in index_map.items():
            result[orig_idx] = response.data[filtered_idx].embedding
        
        # Fill None values with zeros
        default_dim = 1536  # Default for text-embedding-3-large
        for i, emb in enumerate(result):
            if emb is None:
                result[i] = [0.0] * default_dim
        
        return result
    
    def embed_data(self, 
                  data: List[Any], 
                  text_extractor: Callable[[Any], str],
                  id_extractor: Optional[Callable[[Any], str]] = None) -> Dict[str, List[float]]:
        """
        Process a list of data items and generate embeddings.
        
        Args:
            data: List of data items to embed
            text_extractor: Function that extracts text to embed from each data item
            id_extractor: Optional function that extracts ID from each data item
            
        Returns:
            Dictionary mapping data IDs to embedding vectors
        """
        texts = [text_extractor(item) for item in data]
        embeddings = self.get_embeddings(texts)
        
        # Create mapping of ID to embedding
        result = {}
        for i, item in enumerate(data):
            if id_extractor:
                item_id = id_extractor(item)
            else:
                item_id = f"item_{i}"
                
            result[item_id] = embeddings[i]
            
        return result