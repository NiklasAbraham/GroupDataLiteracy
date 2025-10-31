# -*- coding: utf-8 -*-
"""
embedding_service.py

This module defines the EmbeddingService class, which encapsulates the
SentenceTransformer model and provides methods for high-throughput
parallel encoding across multiple GPU devices.
"""

import time
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    A service class for loading a SentenceTransformer model and
    performing parallel text encoding on multiple GPUs.
    """

    def __init__(self, model_name: str):
        """
        Initializes the EmbeddingService by loading the specified model.
        The model is intentionally loaded onto the CPU first; the
        multi_process_pool will handle distributing copies to target GPUs.

        Args:
            model_name (str): The name of the model to load from
                              Hugging Face (e.g., 'BAAI/bge-m3').
        """
        logger.info(f"Initializing EmbeddingService with model: {model_name}")
        try:
            # Load model onto CPU. The parallel processes will manage GPU memory.
            self.model = SentenceTransformer(model_name, device='cpu')
            
            # Set max_seq_length if the model supports it (like bge-m3)
            if hasattr(self.model, 'max_seq_length'):
                # bge-m3 supports 8192 tokens
                model_max_len = self.model.get_max_seq_length()
                logger.info(f"Model max sequence length: {model_max_len} tokens")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        logger.info("EmbeddingService initialized successfully.")

    def encode_parallel(self,
                        corpus: list[str],
                        target_devices: list[str],
                        batch_size: int = 128) -> np.ndarray:
        """
        Encodes a large corpus of text in parallel across specified GPUs.

        Args:
            corpus (list[str]): A list of documents (e.g., movie plots) to encode.
            target_devices (list[str]): A list of CUDA device identifiers
                                         (e.g., ['cuda:0', 'cuda:1']).
            batch_size (int): The batch size to use for encoding on each process.
                              Tune this based on VRAM (24GB 3090s can handle
                              a large batch_size, e.g., 128 or 256).

        Returns:
            np.ndarray: A 2D NumPy array containing the embeddings.
        """
        if not corpus:
            logger.warning("Input corpus is empty. Returning empty array.")
            return np.array([])

        if not target_devices:
            logger.error("No target devices specified for parallel encoding.")
            raise ValueError("target_devices list cannot be empty.")

        logger.info(f"Starting parallel encoding on {len(target_devices)} devices: {target_devices}")
        logger.info(f"Processing {len(corpus)} documents with batch size {batch_size}.")

        pool = None
        try:
            # Start the multi-process pool
            # This creates a separate process for each target_device
            logger.info("Starting multi-process pool...")
            pool = self.model.start_multi_process_pool(target_devices=target_devices)

            start_time = time.time()
            
            # Encode the corpus in parallel
            # The library handles chunking the data, sending it to the
            # processes, encoding it, and gathering the results.
            embeddings = self.model.encode_multi_process(
                sentences=corpus,
                pool=pool,
                batch_size=batch_size,
                chunk_size=None  # Let the library auto-determine chunk size
            )
            
            end_time = time.time()
            duration = end_time - start_time
            docs_per_sec = len(corpus) / duration
            
            logger.info(f"Encoding complete. Time taken: {duration:.2f} seconds.")
            logger.info(f"Throughput: {docs_per_sec:.2f} docs/sec.")

            return embeddings

        except Exception as e:
            logger.error(f"An error occurred during parallel encoding: {e}")
            raise
        finally:
            # Always ensure the pool is stopped
            if pool:
                logger.info("Stopping multi-process pool...")
                self.model.stop_multi_process_pool(pool)
                logger.info("Pool stopped.")
