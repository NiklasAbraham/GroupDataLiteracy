# -*- coding: utf-8 -*-
"""
chunk_no_chunking_cls_token.py

CLS token strategy: use the dense embedding (CLS token) from EmbeddingService.
The final vector is L2-normalized.
"""

import numpy as np
from .chunk_base_class import ChunkBase


class CLSToken(ChunkBase):
    """
    CLS token: use the dense embedding (CLS token) as document embedding.
    """
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text using CLS token (dense embedding).
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: L2-normalized CLS token embedding
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
        
        # Encode using EmbeddingService
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        
        # Get dense embedding (CLS token)
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
        
        # L2 normalize
        return self._normalize(cls_embedding)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            # Return zero vectors for empty texts
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in dummy_result:
                    emb_dim = dummy_result[key].shape[-1]
                    return np.zeros((len(texts), emb_dim))
            first_key = list(dummy_result.keys())[0]
            emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros((len(texts), emb_dim))
        
        # Encode batch using EmbeddingService
        # CLSToken only needs dense embeddings, not token-level (saves memory and time)
        results = self.embedding_service.encode_corpus(valid_texts, batch_size=batch_size, 
                                                      require_token_embeddings=False)
        
        # Get dense embeddings (CLS token)
        dense_key = None
        for key in ['dense_vecs', 'dense', 'dense_embedding']:
            if key in results:
                dense_key = key
                break
        
        if dense_key:
            dense_embeddings = results[dense_key]
            batch_embeddings = [self._normalize(emb) for emb in dense_embeddings]
        else:
            # Fallback
            first_key = list(results.keys())[0]
            batch_embeddings = [self._normalize(emb) for emb in results[first_key]]
        
        # Handle empty texts by inserting zero vectors
        final_embeddings = []
        valid_idx = 0
        for text in texts:
            if text and isinstance(text, str):
                final_embeddings.append(batch_embeddings[valid_idx])
                valid_idx += 1
            else:
                # Zero vector for empty text
                final_embeddings.append(np.zeros_like(batch_embeddings[0]) if batch_embeddings else np.zeros(1024))
        
        return np.array(final_embeddings)
