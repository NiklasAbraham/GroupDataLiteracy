# -*- coding: utf-8 -*-
"""
chunk_first_then_embed.py

Chunk-first strategy: split text into chunks, embed each chunk independently,
then pool the chunk embeddings. The final vector is L2-normalized.
Chunks are overlapping with adjustable overlap and window size.
"""

import numpy as np
from .chunk_base_class import ChunkBase
from transformers import AutoTokenizer


class ChunkFirstEmbed(ChunkBase):
    """
    Chunk-first embedding: chunk text first, then embed each chunk.
    """
    
    def __init__(self, embedding_service=None, model_name: str = "BAAI/bge-m3", 
                 chunk_size: int = 512, stride: int = 256):
        """
        Initialize chunk-first embedding method.
        
        Args:
            embedding_service (EmbeddingService, optional): Pre-initialized EmbeddingService
            model_name (str): Model name
            chunk_size (int): Size of each chunk in tokens (default: 512)
            stride (int): Stride for overlapping chunks (default: 256)
        """
        super().__init__(embedding_service, model_name)
        self.chunk_size = chunk_size
        self.stride = stride
        # Load tokenizer for text chunking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            list[str]: List of text chunks
        """
        # Tokenize to get token positions
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        # Create overlapping chunks
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            start += self.stride
        
        return chunks
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text by chunking first, then embedding each chunk.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: L2-normalized pooled chunk embeddings
        """
        if not text or not isinstance(text, str):
            # Get embedding dimension from a dummy encoding
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            # Check keys in same order as embedding.py: FlagEmbedding returns 'dense_vecs', others return 'dense'
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in dummy_result:
                    emb_dim = dummy_result[key].shape[-1]
                    return np.zeros(emb_dim)
            # Fallback: use first available array
            first_key = list(dummy_result.keys())[0]
            emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros(emb_dim)
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        if not chunks:
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            if 'dense' in dummy_result:
                emb_dim = dummy_result['dense'].shape[-1]
            elif 'dense_vecs' in dummy_result:
                emb_dim = dummy_result['dense_vecs'].shape[-1]
            else:
                first_key = list(dummy_result.keys())[0]
                emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros(emb_dim)
        
        # Embed each chunk using EmbeddingService and get dense (CLS) embedding
        chunk_embeddings = []
        
        for chunk in chunks:
            results = self.embedding_service.encode_corpus([chunk], batch_size=1)
            
            # Get dense embedding (CLS token) for each chunk
            # Check keys in same order as embedding.py: FlagEmbedding returns 'dense_vecs', others return 'dense'
            cls_embedding = None
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in results:
                    cls_embedding = results[key][0]  # Remove batch dimension
                    break
            
            if cls_embedding is None:
                # Fallback: use first available array
                first_key = list(results.keys())[0]
                cls_embedding = results[first_key][0]
            
            chunk_embeddings.append(cls_embedding)
        
        # Mean pool chunk embeddings
        chunk_embeddings = np.array(chunk_embeddings)
        mean_embedding = chunk_embeddings.mean(axis=0)
        
        # L2 normalize
        return self._normalize(mean_embedding)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently by batching all chunks together."""
        if not texts:
            return np.array([])
        
        # Step 1: Chunk all texts and keep track of which chunks belong to which text
        all_chunks = []
        
        for text_idx, text in enumerate(texts):
            if not text or not isinstance(text, str):
                # Empty text - will handle later
                all_chunks.append([])
                continue
            
            chunks = self._chunk_text(text)
            all_chunks.append(chunks)
        
        # Step 2: Collect all chunks from all texts into a flat list
        flat_chunks = []
        chunk_offsets = [0]  # Track where each text's chunks start in flat_chunks
        for chunks in all_chunks:
            flat_chunks.extend(chunks)
            chunk_offsets.append(len(flat_chunks))
        
        if not flat_chunks:
            # All texts are empty - return zero vectors
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in dummy_result:
                    emb_dim = dummy_result[key].shape[-1]
                    return np.zeros((len(texts), emb_dim))
            first_key = list(dummy_result.keys())[0]
            emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros((len(texts), emb_dim))
        
        # Step 3: Embed all chunks in batches
        results = self.embedding_service.encode_corpus(flat_chunks, batch_size=batch_size)
        
        # Step 4: Extract dense embeddings for all chunks
        dense_key = None
        for key in ['dense_vecs', 'dense', 'dense_embedding']:
            if key in results:
                dense_key = key
                break
        
        if dense_key:
            chunk_embeddings = results[dense_key]
            # Normalize each chunk embedding
            chunk_embeddings = np.array([self._normalize(emb) for emb in chunk_embeddings])
        else:
            # Fallback
            first_key = list(results.keys())[0]
            chunk_embeddings = np.array([self._normalize(emb) for emb in results[first_key]])
        
        # Step 5: Reassemble - pool chunks for each text
        final_embeddings = []
        for text_idx in range(len(texts)):
            chunks = all_chunks[text_idx]
            if not chunks:
                # Empty text - use zero vector
                final_embeddings.append(np.zeros_like(chunk_embeddings[0]) if len(chunk_embeddings) > 0 else np.zeros(1024))
            else:
                # Get embeddings for this text's chunks
                start_idx = chunk_offsets[text_idx]
                end_idx = chunk_offsets[text_idx + 1]
                text_chunk_embeddings = chunk_embeddings[start_idx:end_idx]
                
                # Mean pool chunk embeddings
                mean_embedding = text_chunk_embeddings.mean(axis=0)
                final_embeddings.append(self._normalize(mean_embedding))
        
        return np.array(final_embeddings)
