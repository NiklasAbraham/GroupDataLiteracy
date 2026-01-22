# -*- coding: utf-8 -*-

import numpy as np
from .chunk_base_class import ChunkBase


def _get_embedding_dim(embedding_service) -> int:
    """Get embedding dimension from a dummy encoding."""
    dummy_result = embedding_service.encode_corpus(["test"], batch_size=1)
    for key in ['dense_vecs', 'dense', 'dense_embedding']:
        if key in dummy_result:
            return dummy_result[key].shape[-1]
    first_key = list(dummy_result.keys())[0]
    return dummy_result[first_key].shape[-1]


def _get_dense_embedding(results):
    """Extract dense embedding from results."""
    for key in ['dense_vecs', 'dense', 'dense_embedding']:
        if key in results:
            return results[key]
    first_key = list(results.keys())[0]
    return results[first_key]


class CLSToken(ChunkBase):
    """CLS token: use the dense embedding (CLS token) as document embedding."""
    
    def embed(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros(emb_dim)
        
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        dense_embeddings = _get_dense_embedding(results)
        return self._normalize(dense_embeddings[0])
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros((len(texts), emb_dim))
        
        results = self.embedding_service.encode_corpus(
            valid_texts, batch_size=batch_size, require_token_embeddings=False
        )
        dense_embeddings = _get_dense_embedding(results)
        batch_embeddings = [self._normalize(emb) for emb in dense_embeddings]
        
        final_embeddings = []
        valid_idx = 0
        for text in texts:
            if text and isinstance(text, str):
                final_embeddings.append(batch_embeddings[valid_idx])
                valid_idx += 1
            else:
                final_embeddings.append(
                    np.zeros_like(batch_embeddings[0]) if batch_embeddings else np.zeros(1024)
                )
        
        return np.array(final_embeddings)
