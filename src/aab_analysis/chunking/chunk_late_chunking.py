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


class LateChunking(ChunkBase):
    """Late chunking: embed full text, then chunk hidden states (colbert_vecs)."""
    
    def __init__(self, embedding_service=None, model_name: str = "BAAI/bge-m3", 
                 window_size: int = 512, stride: int = 256):
        super().__init__(embedding_service, model_name)
        self.window_size = window_size
        self.stride = window_size if stride <= 0 else stride
    
    def _process_hidden_states(self, hidden_states: np.ndarray) -> np.ndarray:
        """Process hidden states into final embedding using late chunking."""
        seq_len = hidden_states.shape[0]
        
        if seq_len <= self.window_size:
            mean_embedding = hidden_states.mean(axis=0)
            self._last_preL2_embedding = mean_embedding
            return self._normalize(mean_embedding)
        
        window_embeddings = []
        start = 0
        
        while start < seq_len:
            end = min(start + self.window_size, seq_len)
            window_hidden = hidden_states[start:end]
            window_embedding = self._normalize(window_hidden.mean(axis=0))
            window_embeddings.append(window_embedding)
            
            if end >= seq_len:
                break
            start += self.stride
        
        window_embeddings = np.array(window_embeddings)
        final_embedding = window_embeddings.mean(axis=0)
        self._last_preL2_embedding = final_embedding
        
        return self._normalize(final_embedding)
    
    def embed(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros(emb_dim)
        
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        
        if 'colbert_vecs' not in results:
            raise ValueError(
                "LateChunking requires token-level hidden states (colbert_vecs). "
                "The embedding service did not provide colbert_vecs."
            )
        
        hidden_states = results['colbert_vecs'][0]
        return self._process_hidden_states(hidden_states)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros((len(texts), emb_dim))
        
        results = self.embedding_service.encode_corpus(valid_texts, batch_size=batch_size)
        
        if 'colbert_vecs' not in results:
            raise ValueError(
                "LateChunking requires token-level hidden states (colbert_vecs). "
                "The embedding service did not provide colbert_vecs."
            )
        
        colbert_vecs = results['colbert_vecs']
        final_embeddings = []
        valid_idx = 0
        
        for text in texts:
            if not text or not isinstance(text, str):
                if len(final_embeddings) > 0:
                    final_embeddings.append(np.zeros_like(final_embeddings[0]))
                else:
                    if isinstance(colbert_vecs, (list, np.ndarray)) and len(colbert_vecs) > 0:
                        first_colbert = colbert_vecs[0] if isinstance(colbert_vecs, list) else colbert_vecs[0]
                        if isinstance(first_colbert, np.ndarray) and len(first_colbert.shape) > 0:
                            emb_dim = first_colbert.shape[-1]
                        else:
                            emb_dim = _get_embedding_dim(self.embedding_service)
                    else:
                        emb_dim = _get_embedding_dim(self.embedding_service)
                    final_embeddings.append(np.zeros(emb_dim))
                continue
            
            if isinstance(colbert_vecs, list):
                hidden_states = colbert_vecs[valid_idx]
                if not isinstance(hidden_states, np.ndarray):
                    hidden_states = np.array(hidden_states)
            else:
                hidden_states = colbert_vecs[valid_idx]
            
            final_embedding = self._process_hidden_states(hidden_states)
            
            if self._collect_preL2:
                pre_norm = np.linalg.norm(self._last_preL2_embedding)
                self._last_preL2_norms.append(float(pre_norm))
            
            final_embeddings.append(final_embedding)
            valid_idx += 1
        
        return np.array(final_embeddings)
