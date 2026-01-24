# -*- coding: utf-8 -*-

import numpy as np
from .chunk_base_class import ChunkBase


def _get_embedding_dim(embedding_service):
    """Get embedding dimension from a dummy encoding."""
    dummy_result = embedding_service.encode_corpus(["test"], batch_size=1)
    if 'colbert_vecs' in dummy_result:
        return dummy_result['colbert_vecs'].shape[-1]
    for key in ['dense_vecs', 'dense', 'dense_embedding']:
        if key in dummy_result:
            return dummy_result[key].shape[-1]
    first_key = list(dummy_result.keys())[0]
    return dummy_result[first_key].shape[-1]


def _get_token_embeddings(results):
    """Extract token embeddings from results."""
    if 'colbert_vecs' in results:
        return results['colbert_vecs']
    for key in ['dense_vecs', 'dense', 'dense_embedding']:
        if key in results:
            return results[key]
    first_key = list(results.keys())[0]
    return results[first_key]


def _mean_pool_token_embeddings(token_emb):
    """Mean pool token embeddings, handling different formats."""
    if isinstance(token_emb, np.ndarray):
        if token_emb.shape[0] == 0:
            emb_dim = token_emb.shape[1] if len(token_emb.shape) > 1 else 1024
            return np.zeros(emb_dim)
        return token_emb.mean(axis=0)
    
    if isinstance(token_emb, list):
        if len(token_emb) == 0:
            return np.zeros(1024)
        token_emb = np.array(token_emb)
    
    if not isinstance(token_emb, np.ndarray):
        try:
            token_emb = np.array(token_emb)
        except Exception:
            return np.zeros(1024)
    
    if token_emb.shape[0] == 0:
        emb_dim = token_emb.shape[1] if len(token_emb.shape) > 1 else 1024
        return np.zeros(emb_dim)
    
    return token_emb.mean(axis=0)


class MeanPooling(ChunkBase):
    """Mean pooling: average all token embeddings to get document representation."""
    
    def embed(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros(emb_dim)
        
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        token_embeddings = _get_token_embeddings(results)
        mean_emb = _mean_pool_token_embeddings(token_embeddings[0])
        return self._normalize(mean_emb)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros((len(texts), emb_dim))
        
        results = self.embedding_service.encode_corpus(valid_texts, batch_size=batch_size)
        
        if 'colbert_vecs' in results:
            batch_embeddings = [
                self._normalize(_mean_pool_token_embeddings(colbert_vec))
                for colbert_vec in results['colbert_vecs']
            ]
        else:
            dense_embeddings = _get_token_embeddings(results)
            batch_embeddings = [self._normalize(emb) for emb in dense_embeddings]
        
        final_embeddings = []
        valid_idx = 0
        for text in texts:
            if text and isinstance(text, str):
                if self._collect_preL2:
                    if 'colbert_vecs' in results:
                        colbert_vec = results['colbert_vecs'][valid_idx]
                        pre_norm = np.linalg.norm(_mean_pool_token_embeddings(colbert_vec))
                    else:
                        pre_norm = np.linalg.norm(dense_embeddings[valid_idx])
                    self._last_preL2_norms.append(float(pre_norm))
                final_embeddings.append(batch_embeddings[valid_idx])
                valid_idx += 1
            else:
                if self._collect_preL2:
                    self._last_preL2_norms.append(0.0)
                final_embeddings.append(
                    np.zeros_like(batch_embeddings[0]) if batch_embeddings else np.zeros(1024)
                )
        
        return np.array(final_embeddings)
