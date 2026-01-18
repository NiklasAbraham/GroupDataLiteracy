# -*- coding: utf-8 -*-
"""
models package

This package contains embedding strategy implementations for different
embedding libraries (sentence-transformers, FlagEmbedding, etc.).
"""

from .base_strategy import AbstractEmbeddingStrategy
from .sentence_transformer_strategy import SentenceTransformerStrategy
from .flag_embedding_strategy import FlagEmbeddingStrategy
from .qwen3_strategy import Qwen3EmbeddingStrategy
from .strategy_factory import get_embedding_strategy, get_embedding_strategy_by_library

__all__ = [
    'AbstractEmbeddingStrategy',
    'SentenceTransformerStrategy',
    'FlagEmbeddingStrategy',
    'Qwen3EmbeddingStrategy',
    'get_embedding_strategy',
    'get_embedding_strategy_by_library',
]

