# -*- coding: utf-8 -*-
"""
base_strategy.py

This module defines the abstract base class for embedding strategies.
All concrete embedding strategies must implement this interface to ensure
consistency and flexibility across different embedding libraries.
"""

import abc
from typing import Optional
import numpy as np


class AbstractEmbeddingStrategy(abc.ABC):
    """
    Abstract base class defining the interface for all embedding strategies.
    
    This interface allows the EmbeddingService to work with different
    embedding libraries (e.g., sentence-transformers, FlagEmbedding)
    in a uniform way.
    """
    
    def __init__(self, model_name: str, target_devices: list[str]):
        """
        Initialize the embedding strategy.
        
        Args:
            model_name (str): The name of the model to load from
                              Hugging Face or similar repository.
            target_devices (list[str]): List of device identifiers
                                       (e.g., ['cuda:0', 'cuda:1']).
        """
        self.model_name = model_name
        self.target_devices = target_devices
        self.model: Optional[object] = None
    
    @abc.abstractmethod
    def load_model(self) -> None:
        """
        Load the embedding model and prepare it for parallel execution.
        
        This method contains library-specific logic for loading the model
        from self.model_name and preparing it for execution across
        self.target_devices.
        
        Raises:
            Exception: If model loading fails.
        """
        pass
    
    @abc.abstractmethod
    def encode(self, corpus: list[str], batch_size: int) -> dict[str, np.ndarray]:
        """
        Encode a corpus of documents into embeddings.
        
        Args:
            corpus (list[str]): List of documents to encode.
            batch_size (int): Batch size for encoding.
        
        Returns:
            dict[str, np.ndarray]: Dictionary mapping embedding type names
                                  to numpy arrays. For example:
                                  {'dense': np.ndarray} or
                                  {'dense': np.ndarray, 'sparse': np.ndarray}
        
        Raises:
            Exception: If encoding fails.
        """
        pass

