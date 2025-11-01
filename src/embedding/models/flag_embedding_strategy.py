# -*- coding: utf-8 -*-
"""
flag_embedding_strategy.py

This module implements the embedding strategy for FlagEmbedding models (e.g., BGE-M3).
"""

import time
import logging
import numpy as np
import torch
from typing import Optional

from .base_strategy import AbstractEmbeddingStrategy

logger = logging.getLogger(__name__)


class FlagEmbeddingStrategy(AbstractEmbeddingStrategy):
    """
    Embedding strategy implementation for FlagEmbedding models.
    
    This strategy uses the FlagEmbedding library for models like BGE-M3,
    which provides sparse and multi-vector embedding capabilities.
    """
    
    def __init__(self, model_name: str, target_devices: list[str]):
        """
        Initialize the FlagEmbedding strategy.
        
        Args:
            model_name (str): The name of the model to load.
            target_devices (list[str]): List of CUDA device identifiers.
        """
        super().__init__(model_name, target_devices)
        self.model: Optional[object] = None
    
    def load_model(self) -> None:
        """
        Load the FlagEmbedding model and prepare it for multi-GPU execution.
        
        The model is loaded directly onto the primary device and wrapped
        in DataParallel for multi-GPU support.
        """
        logger.info(f"Loading FlagEmbedding model: {self.model_name}")
        try:
            # Import FlagEmbedding
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError:
                logger.error("FlagEmbedding library not found. Please install it with: pip install FlagEmbedding")
                raise
            
            # Determine primary device
            primary_device = self.target_devices[0] if self.target_devices else 'cuda:0'
            logger.info(f"Loading model on primary device: {primary_device}")
            
            # Load the model
            self.model = BGEM3FlagModel(self.model_name, device=primary_device, use_fp16=True)
            
            # Handle multi-GPU with DataParallel
            if len(self.target_devices) > 1:
                logger.info(f"Wrapping model in DataParallel for {len(self.target_devices)} devices")
                # Extract device indices for DataParallel
                device_ids = [int(device.split(':')[1]) for device in self.target_devices]
                self.model.model = torch.nn.DataParallel(self.model.model, device_ids=device_ids)
            
            logger.info("FlagEmbedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FlagEmbedding model: {e}")
            raise
    
    def encode(self, corpus: list[str], batch_size: int) -> dict[str, np.ndarray]:
        """
        Encode a corpus of documents using FlagEmbedding.
        
        Args:
            corpus (list[str]): List of documents to encode.
            batch_size (int): Batch size for encoding.
        
        Returns:
            dict[str, np.ndarray]: Dictionary with embedding types as keys.
                                  May include 'dense', 'sparse', 'colbert_vecs'.
        """
        if not corpus:
            logger.warning("Input corpus is empty. Returning empty dictionary.")
            return {}
        
        logger.info(f"Starting FlagEmbedding encoding on {len(self.target_devices)} devices")
        logger.info(f"Processing {len(corpus)} documents with batch size {batch_size}")
        
        try:
            start_time = time.time()
            
            # Encode using FlagEmbedding's optimized method
            # This already handles batching and (if wrapped) multi-GPU processing
            output = self.model.encode(
                corpus,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            docs_per_sec = len(corpus) / duration if duration > 0 else 0
            
            logger.info(f"Encoding complete. Time taken: {duration:.2f} seconds")
            logger.info(f"Throughput: {docs_per_sec:.2f} docs/sec")
            
            # Convert tensors to numpy arrays and return dictionary
            result = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    result[key] = value
                elif isinstance(value, list):
                    # Handle sparse embeddings which might be lists of dicts
                    result[key] = value
                else:
                    logger.warning(f"Unexpected output type for key '{key}': {type(value)}")
                    result[key] = value
            
            return result
        
        except Exception as e:
            logger.error(f"An error occurred during FlagEmbedding encoding: {e}")
            raise

