# -*- coding: utf-8 -*-
"""
chunking module

Provides different chunking/pooling strategies for document embeddings.
"""

from .chunk_base_class import ChunkBase
from .chunk_mean_pooling import MeanPooling
from .chunk_no_chunking_cls_token import CLSToken
from .chunk_first_then_embed import ChunkFirstEmbed
from .chunk_late_chunking import LateChunking

__all__ = [
    'ChunkBase',
    'MeanPooling',
    'CLSToken',
    'ChunkFirstEmbed',
    'LateChunking'
]

