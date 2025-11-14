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
    
    def __init__(self, model_name: str, target_devices: list[str], max_length: int = 8192):
        """
        Initialize the FlagEmbedding strategy.
        
        Args:
            model_name (str): The name of the model to load.
            target_devices (list[str]): List of CUDA device identifiers.
            max_length (int): Maximum sequence length for encoding. Defaults to 8192
                            (BGE-M3's maximum context window). Set to None to use
                            FlagEmbedding's default (512).
        """
        super().__init__(model_name, target_devices)
        self.model: Optional[object] = None
        self.max_length = max_length
    
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
            # FlagEmbedding handles multi-GPU internally via its own multi-process pool
            # We don't need to wrap it in DataParallel - let it handle it natively
            self.model = BGEM3FlagModel(self.model_name, device=primary_device, use_fp16=True)
            
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
            # FlagEmbedding handles multi-GPU internally via its own multi-process pool
            # We just pass the parameters and let it handle everything
            encode_kwargs = {
                'sentences': corpus,  # FlagEmbedding uses 'sentences', not 'corpus'
                'batch_size': batch_size,
                'return_dense': True,
                'return_sparse': True,
                'return_colbert_vecs': True,
            }
            if self.max_length is not None:
                encode_kwargs['max_length'] = self.max_length
            
            # Let FlagEmbedding handle multi-GPU automatically via its internal mechanisms
            output = self.model.encode(**encode_kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            docs_per_sec = len(corpus) / duration if duration > 0 else 0
            
            logger.info(f"Encoding complete. Time taken: {duration:.2f} seconds")
            logger.info(f"Throughput: {docs_per_sec:.2f} docs/sec")
            
            # Log the raw output structure for debugging
            if output is None:
                logger.error("FlagEmbedding model.encode() returned None")
                return {}
            
            # Check if output is a dictionary
            if not isinstance(output, dict):
                logger.error(f"FlagEmbedding model.encode() returned non-dict type: {type(output)}. Expected dict.")
                # Try to convert to dict if it's a tuple or has attributes
                if isinstance(output, tuple) and len(output) > 0:
                    logger.warning(f"Output is a tuple with {len(output)} elements. Attempting to extract first element as dense embeddings.")
                    # If it's a tuple, the first element is usually the dense embeddings
                    dense_val = output[0]
                    if isinstance(dense_val, torch.Tensor):
                        return {'dense': dense_val.cpu().numpy()}
                    elif isinstance(dense_val, np.ndarray):
                        return {'dense': dense_val}
                    else:
                        logger.error(f"First element of tuple is unexpected type: {type(dense_val)}")
                        return {}
                else:
                    return {}
            
            logger.info(f"Raw output type: {type(output)}, keys: {list(output.keys())}")
            
            # Convert tensors to numpy arrays and return dictionary
            result = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    numpy_value = value.cpu().numpy()
                    result[key] = numpy_value
                    logger.debug(f"Converted tensor '{key}' to numpy array with shape: {numpy_value.shape}")
                elif isinstance(value, np.ndarray):
                    result[key] = value
                    logger.debug(f"Key '{key}' is already numpy array with shape: {value.shape}")
                elif isinstance(value, list):
                    # Handle sparse embeddings which might be lists of dicts
                    result[key] = value
                    logger.debug(f"Key '{key}' is a list with length: {len(value)}")
                else:
                    logger.warning(f"Unexpected output type for key '{key}': {type(value)}")
                    result[key] = value
            
            logger.info(f"Returning result dictionary with keys: {list(result.keys())}")
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        logger.warning(f"  {key}: numpy array is EMPTY (shape: {value.shape})")
                    else:
                        logger.info(f"  {key}: numpy array with shape {value.shape}, dtype {value.dtype}")
                elif value is None:
                    logger.warning(f"  {key}: value is None")
                else:
                    logger.info(f"  {key}: {type(value)}")
            
            # Validate that we have at least one non-empty numpy array
            has_valid_embedding = False
            for key, value in result.items():
                if isinstance(value, np.ndarray) and value.size > 0:
                    has_valid_embedding = True
                    break
            
            if not has_valid_embedding:
                logger.error("Result dictionary contains no valid (non-empty) embeddings arrays!")
            
            return result
        
        except Exception as e:
            logger.error(f"An error occurred during FlagEmbedding encoding: {e}")
            raise

