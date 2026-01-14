# -*- coding: utf-8 -*-
"""
strategy_factory.py

This module provides a factory function to select the appropriate embedding
strategy based on the model name or configuration.
"""

import logging
from typing import Optional

from .sentence_transformer_strategy import SentenceTransformerStrategy
from .flag_embedding_strategy import FlagEmbeddingStrategy
from .qwen3_strategy import Qwen3EmbeddingStrategy
from .base_strategy import AbstractEmbeddingStrategy

logger = logging.getLogger(__name__)


def get_embedding_strategy(model_name: str, target_devices: list[str]) -> AbstractEmbeddingStrategy:
    """
    Factory function to select the appropriate embedding strategy.
    
    Args:
        model_name (str): The name of the model to use.
        target_devices (list[str]): List of device identifiers.
    
    Returns:
        AbstractEmbeddingStrategy: An instance of the appropriate strategy class.
    
    Raises:
        ValueError: If no strategy is defined for the given model.
    """
    logger.info(f"Selecting strategy for model: {model_name}")
    
    # Normalize model name for comparison
    model_name_lower = model_name.lower()
    
    # Qwen3-Embedding models
    if 'qwen3-embedding' in model_name_lower or 'qwen3_embedding' in model_name_lower:
        logger.info("Selected Qwen3EmbeddingStrategy")
        return Qwen3EmbeddingStrategy(model_name, target_devices)
    
    # FlagEmbedding models (BGE-M3 and similar)
    if 'bge-m3' in model_name_lower or 'flag' in model_name_lower:
        logger.info("Selected FlagEmbeddingStrategy")
        return FlagEmbeddingStrategy(model_name, target_devices)
    
    # SentenceTransformer models
    # These are the most common, so we default to this strategy
    # for most models unless explicitly overridden above
    logger.info("Selected SentenceTransformerStrategy")
    return SentenceTransformerStrategy(model_name, target_devices)


def get_embedding_strategy_by_library(library: str, model_name: str, target_devices: list[str]) -> Optional[AbstractEmbeddingStrategy]:
    """
    Factory function to select a strategy by explicitly specifying the library.
    
    Args:
        library (str): The library name ('sentence-transformers', 'FlagEmbedding', or 'qwen3').
        model_name (str): The name of the model to use.
        target_devices (list[str]): List of device identifiers.
    
    Returns:
        Optional[AbstractEmbeddingStrategy]: An instance of the appropriate strategy,
                                             or None if library is not supported.
    """
    logger.info(f"Selecting strategy by library: {library}")
    
    library_lower = library.lower()
    
    if library_lower in ['qwen3', 'qwen3-embedding', 'qwen3_embedding']:
        logger.info("Selected Qwen3EmbeddingStrategy")
        return Qwen3EmbeddingStrategy(model_name, target_devices)
    
    elif library_lower in ['sentence-transformers', 'sentence_transformer', 'sentence transformer']:
        logger.info("Selected SentenceTransformerStrategy")
        return SentenceTransformerStrategy(model_name, target_devices)
    
    elif library_lower in ['flagembedding', 'flag_embedding', 'flag-embedding']:
        logger.info("Selected FlagEmbeddingStrategy")
        return FlagEmbeddingStrategy(model_name, target_devices)
    
    else:
        logger.warning(f"Unsupported library: {library}")
        return None

