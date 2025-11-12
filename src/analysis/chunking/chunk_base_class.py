# -*- coding: utf-8 -*-
"""
chunk_base_class.py

Abstract base class for all chunking/pooling strategies.
All chunking methods must inherit from this class and implement the embed() method.
"""

import abc
import numpy as np
from typing import Optional
import sys
from pathlib import Path

# Add src to path for imports
# chunk_base_class.py is in src/analysis/chunking/
# So we go up 3 levels to get to src/
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding.embedding import EmbeddingService


class ChunkBase(abc.ABC):
    """
    Abstract base class for document embedding aggregation methods.
    
    All chunking strategies must implement the embed() method that takes
    a text string and returns a normalized embedding vector.
    """
    
    def __init__(self, embedding_service: EmbeddingService = None, 
                 model_name: str = "BAAI/bge-m3"):
        """
        Initialize the chunking method with an EmbeddingService.
        
        Args:
            embedding_service (EmbeddingService, optional): Pre-initialized EmbeddingService.
                                                           If None, one will be created.
            model_name (str): Name of the model to use (default: BAAI/bge-m3)
        """
        if embedding_service is None:
            # Create EmbeddingService if not provided (let it handle device selection)
            self.embedding_service = EmbeddingService(model_name, None)
        else:
            self.embedding_service = embedding_service
        
        self.model_name = model_name
    
    @abc.abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a text document using the chunking strategy.
        
        Args:
            text (str): The text document to embed
            
        Returns:
            np.ndarray: Normalized embedding vector (L2-normalized)
        """
        pass
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of texts using the chunking strategy.
        Default implementation processes texts individually, but subclasses can override for efficiency.
        
        Args:
            texts (list[str]): List of text documents to embed
            batch_size (int): Batch size for processing (used by some methods)
            
        Returns:
            np.ndarray: Array of normalized embedding vectors [n_texts, embedding_dim]
        """
        # Default: process individually (subclasses can override for batch processing)
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return np.array(embeddings)
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        L2-normalize a vector.
        
        Args:
            vec (np.ndarray): Vector to normalize
            
        Returns:
            np.ndarray: L2-normalized vector
        """
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def cleanup(self):
        """Clean up EmbeddingService resources."""
        if hasattr(self, 'embedding_service') and self.embedding_service is not None:
            self.embedding_service.cleanup()
