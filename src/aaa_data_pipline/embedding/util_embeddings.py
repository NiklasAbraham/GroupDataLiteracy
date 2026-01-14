# -*- coding: utf-8 -*-
"""
util_embeddings.py

Utility functions for embedding operations, including GPU setup verification
and embedding validation.
"""

import logging
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s] %(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def verify_gpu_setup(target_devices: list[str] = None) -> list[str]:
    """
    Verifies that CUDA is available and adjusts target devices if needed.
    
    Args:
        target_devices (list[str], optional): List of CUDA device identifiers.
                                             If None, auto-detects available devices.
    
    Returns:
        list[str]: Adjusted list of target devices. Returns ['cpu'] if CUDA is not available.
    
    Raises:
        SystemExit: If CUDA is required but not available (when target_devices is provided).
    """
    logger.info("Verifying GPU setup...")
    
    if not torch.cuda.is_available():
        if target_devices is None:
            logger.warning("CUDA is not available. Returning ['cpu'] as default.")
            return ['cpu']
        else:
            logger.error("CUDA is not available. This script requires a GPU setup.")
            raise SystemExit(1)
    
    available_gpus = torch.cuda.device_count()
    logger.info(f"Found {available_gpus} available CUDA devices.")
    
    # Auto-detect if no devices specified
    if target_devices is None:
        target_devices = [f'cuda:{i}' for i in range(available_gpus)]
        logger.info(f"Auto-detected {len(target_devices)} devices: {target_devices}")
        return target_devices
    
    # Verify specified devices
    if available_gpus < len(target_devices):
        logger.warning(f"Warning: Configured for {len(target_devices)} devices, "
                       f"but only {available_gpus} are available.")
        # Adjust to available devices
        adjusted_devices = target_devices[:available_gpus]
        if not adjusted_devices:
            logger.error("No target devices are available to run on.")
            raise SystemExit(1)
        logger.info(f"Adjusting to use {len(adjusted_devices)} devices: {adjusted_devices}")
        return adjusted_devices
    else:
        logger.info("GPU configuration matches available devices.")
        return target_devices


def verify_embeddings(embeddings: np.ndarray, corpus: list[str]) -> bool:
    """
    Verifies that embeddings were created correctly.
    
    Args:
        embeddings (np.ndarray): The embedding matrix.
        corpus (list[str]): The original corpus.
    
    Returns:
        bool: True if embeddings are valid, False otherwise.
    """
    logger.info("--- Verifying Embeddings ---")
    
    if embeddings.size == 0:
        logger.error("Embeddings array is empty.")
        return False
    
    if len(corpus) != embeddings.shape[0]:
        logger.error(f"Mismatch: {len(corpus)} documents but {embeddings.shape[0]} embeddings.")
        return False
    
    if embeddings.shape[1] == 0:
        logger.error("Embedding dimension is 0.")
        return False
    
    # Check for NaN or Inf values
    if np.isnan(embeddings).any():
        logger.error("Embeddings contain NaN values.")
        return False
    
    if np.isinf(embeddings).any():
        logger.error("Embeddings contain Inf values.")
        return False
    
    logger.info(f"✓ Embeddings verified: shape {embeddings.shape}, dtype {embeddings.dtype}")
    logger.info(f"✓ Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    logger.info(f"✓ Embedding mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    
    return True

