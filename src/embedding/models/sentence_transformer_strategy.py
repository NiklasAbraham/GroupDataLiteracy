# -*- coding: utf-8 -*-
"""
sentence_transformer_strategy.py

This module implements the embedding strategy for sentence-transformers models.
"""

import time
import logging
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from .base_strategy import AbstractEmbeddingStrategy

logger = logging.getLogger(__name__)


class SentenceTransformerStrategy(AbstractEmbeddingStrategy):
    """
    Embedding strategy implementation for sentence-transformers models.
    
    This strategy uses the sentence-transformers library's built-in
    multi-process pool for parallel encoding across multiple GPUs.
    """
    
    def __init__(self, model_name: str, target_devices: list[str]):
        """
        Initialize the SentenceTransformer strategy.
        
        Args:
            model_name (str): The name of the model to load.
            target_devices (list[str]): List of CUDA device identifiers.
        """
        super().__init__(model_name, target_devices)
        self.model: Optional[SentenceTransformer] = None
        self.pool = None
    
    def load_model(self) -> None:
        """
        Load the SentenceTransformer model onto CPU.
        
        The model is loaded onto CPU because this strategy's encode method
        uses start_multi_process_pool, which manages its own GPU processes.
        """
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device='cpu')
            
            if hasattr(self.model, 'get_max_seq_length'):
                model_max_len = self.model.get_max_seq_length()
                logger.info(f"Model max sequence length: {model_max_len} tokens")
            
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise
    
    def encode(self, corpus: list[str], batch_size: int) -> dict[str, np.ndarray]:
        """
        Encode a corpus of documents using sentence-transformers.
        
        Args:
            corpus (list[str]): List of documents to encode.
            batch_size (int): Batch size for encoding.
        
        Returns:
            dict[str, np.ndarray]: Dictionary with 'dense' key containing
                                  the embedding matrix.
        """
        if not corpus:
            logger.warning("Input corpus is empty. Returning empty array.")
            return {'dense': np.array([])}
        
        if not self.target_devices:
            logger.error("No target devices specified for parallel encoding.")
            raise ValueError("target_devices list cannot be empty.")
        
        logger.info(f"Starting SentenceTransformer parallel encoding on {len(self.target_devices)} devices")
        logger.info(f"Processing {len(corpus)} documents with batch size {batch_size}")
        
        try:
            # Start the multi-process pool
            logger.info("Starting multi-process pool...")
            self.pool = self.model.start_multi_process_pool(target_devices=self.target_devices)
            
            start_time = time.time()
            
            # Encode the corpus in parallel
            embeddings = self.model.encode_multi_process(
                sentences=corpus,
                pool=self.pool,
                batch_size=batch_size,
                chunk_size=None  # Let the library auto-determine chunk size
            )
            
            end_time = time.time()
            duration = end_time - start_time
            docs_per_sec = len(corpus) / duration if duration > 0 else 0
            
            logger.info(f"Encoding complete. Time taken: {duration:.2f} seconds")
            logger.info(f"Throughput: {docs_per_sec:.2f} docs/sec")
            
            return {'dense': embeddings}
        
        except Exception as e:
            logger.error(f"An error occurred during SentenceTransformer encoding: {e}")
            raise
        finally:
            # Always ensure the pool is stopped
            if self.pool:
                try:
                    logger.info("Stopping multi-process pool...")
                    self.model.stop_multi_process_pool(self.pool)
                    self.pool = None
                    logger.info("Pool stopped.")
                except Exception as e:
                    logger.error(f"Error stopping multiprocessing pool: {e}")
                    # Force cleanup by setting to None even if stop failed
                    self.pool = None

