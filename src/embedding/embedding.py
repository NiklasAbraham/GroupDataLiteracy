# -*- coding: utf-8 -*-
"""
embedding_service.py

This module defines the EmbeddingService class, which provides a unified
interface for high-throughput parallel encoding across multiple GPU devices
using various embedding model libraries.
"""

import logging
import numpy as np

# Handle imports - try relative import first, then absolute
try:
    from .models import AbstractEmbeddingStrategy, get_embedding_strategy
except ImportError:
    try:
        from models import AbstractEmbeddingStrategy, get_embedding_strategy
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from embedding.models import AbstractEmbeddingStrategy, get_embedding_strategy

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    A service class for loading embedding models and performing parallel
    text encoding on multiple GPUs.
    
    This service uses the Strategy pattern to support different embedding
    libraries (sentence-transformers, FlagEmbedding, etc.) in a uniform way.
    """

    def __init__(self, model_name: str, target_devices: list[str] = None, strategy: AbstractEmbeddingStrategy = None):
        """
        Initializes the EmbeddingService with a specific strategy.
        
        Args:
            model_name (str): The name of the model to load from
                              Hugging Face (e.g., 'BAAI/bge-m3').
            target_devices (list[str], optional): List of CUDA device identifiers.
                                                  Defaults to ['cuda:0'].
            strategy (AbstractEmbeddingStrategy, optional): A pre-configured strategy.
                                                            If None, one will be
                                                            selected automatically.
        """
        if target_devices is None:
            target_devices = ['cuda:0']
        
        logger.info(f"Initializing EmbeddingService with model: {model_name}")
        
        # Select strategy if not provided
        if strategy is None:
            logger.info("Auto-selecting embedding strategy based on model name")
            self.strategy = get_embedding_strategy(model_name, target_devices)
        else:
            logger.info("Using provided embedding strategy")
            self.strategy = strategy
        
        # Load the model using the strategy
        try:
            self.strategy.load_model()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        logger.info(f"EmbeddingService initialized with {self.strategy.__class__.__name__}")

    def cleanup(self):
        """
        Clean up any multiprocessing resources (pools, semaphores, etc.).
        This should be called when done with the service to prevent resource leaks.
        """
        if hasattr(self.strategy, 'pool') and self.strategy.pool is not None:
            try:
                if hasattr(self.strategy.model, 'stop_multi_process_pool'):
                    logger.info("Cleaning up multiprocessing pool...")
                    self.strategy.model.stop_multi_process_pool(self.strategy.pool)
                    self.strategy.pool = None
                    logger.info("Multiprocessing pool cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up multiprocessing pool: {e}")
        
        # Clean up model resources
        if hasattr(self.strategy, 'model') and self.strategy.model is not None:
            try:
                # Clear CUDA cache if using GPU
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup

    def encode_parallel(self,
                        corpus: list[str],
                        target_devices: list[str],
                        batch_size: int = 128) -> np.ndarray:
        """
        Encodes a large corpus of text in parallel across specified GPUs.
        
        This method is maintained for backwards compatibility with existing code.
        New code should use encode_corpus() for better flexibility.

        Args:
            corpus (list[str]): A list of documents (e.g., movie plots) to encode.
            target_devices (list[str]): A list of CUDA device identifiers
                                         (e.g., ['cuda:0', 'cuda:1']).
            batch_size (int): The batch size to use for encoding on each process.

        Returns:
            np.ndarray: A 2D NumPy array containing the dense embeddings.
        """
        # For backwards compatibility, extract dense embeddings
        results = self.encode_corpus(corpus, batch_size)
        
        # Log what keys are available for debugging
        if not results:
            logger.error(f"encode_corpus returned empty dictionary. Expected keys: 'dense', 'sparse', 'colbert_vecs'")
            return np.array([])
        
        logger.info(f"encode_corpus returned keys: {list(results.keys())}")
        
        # Try to get dense embeddings - check multiple possible key names
        # FlagEmbedding returns 'dense_vecs', other models may return 'dense'
        dense_embeddings = None
        for key in ['dense_vecs', 'dense', 'dense_embedding']:
            if key in results:
                dense_embeddings = results[key]
                logger.info(f"Found dense embeddings under key '{key}' with shape: {dense_embeddings.shape if hasattr(dense_embeddings, 'shape') else type(dense_embeddings)}")
                break
        
        if dense_embeddings is None:
            logger.error(f"Dense embeddings not found. Available keys: {list(results.keys())}")
            # If there's only one key, use it as a fallback
            if len(results) == 1:
                key = list(results.keys())[0]
                logger.warning(f"Using '{key}' as fallback for dense embeddings")
                dense_embeddings = results[key]
            else:
                return np.array([])
        
        # Ensure it's a numpy array
        if not isinstance(dense_embeddings, np.ndarray):
            logger.error(f"Dense embeddings is not a numpy array: {type(dense_embeddings)}")
            return np.array([])
        
        return dense_embeddings

    def encode_corpus(self, corpus: list[str], batch_size: int = 128) -> dict[str, np.ndarray]:
        """
        Encodes a corpus of documents using the configured strategy.
        
        This method returns a dictionary that may contain multiple embedding
        types (dense, sparse, etc.) depending on the strategy used.

        Args:
            corpus (list[str]): A list of documents to encode.
            batch_size (int): The batch size to use for encoding.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping embedding type names to numpy arrays.
        """
        logger.info(f"Delegating encoding to {self.strategy.__class__.__name__}...")
        logger.info(f"Processing {len(corpus)} documents with batch size {batch_size}")
        
        return self.strategy.encode(corpus, batch_size)
