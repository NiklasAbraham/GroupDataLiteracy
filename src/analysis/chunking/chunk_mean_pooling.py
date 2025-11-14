# -*- coding: utf-8 -*-
"""
chunk_mean_pooling.py

Mean pooling strategy: use colbert_vecs (token-level hidden states) and return the global mean.
The final vector is L2-normalized.
"""

import numpy as np
from .chunk_base_class import ChunkBase


class MeanPooling(ChunkBase):
    """
    Mean pooling: average all token embeddings (from colbert_vecs) to get document representation.
    """
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text by averaging all token embeddings from colbert_vecs.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: L2-normalized mean-pooled embedding
        """
        if not text or not isinstance(text, str):
            # Get embedding dimension from a dummy encoding
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            if 'colbert_vecs' in dummy_result:
                emb_dim = dummy_result['colbert_vecs'].shape[-1]
            else:
                # Fallback: check for dense embeddings (same order as embedding.py)
                for key in ['dense_vecs', 'dense', 'dense_embedding']:
                    if key in dummy_result:
                        emb_dim = dummy_result[key].shape[-1]
                        return np.zeros(emb_dim)
                # Last resort: use first available array
                first_key = list(dummy_result.keys())[0]
                emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros(emb_dim)
        
        # Encode using EmbeddingService
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        
        # Get colbert_vecs (token-level hidden states)
        if 'colbert_vecs' in results:
            # colbert_vecs shape: [batch_size, seq_len, hidden_dim]
            token_embeddings = results['colbert_vecs'][0]  # Remove batch dimension
        else:
            # Fallback to dense if colbert_vecs not available (same order as embedding.py)
            token_embeddings = None
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in results:
                    token_embeddings = results[key][0]
                    break
            if token_embeddings is None:
                # Last resort: use first available array
                first_key = list(results.keys())[0]
                token_embeddings = results[first_key][0]
        
        # Mean pool (average over sequence length)
        mean_embedding = token_embeddings.mean(axis=0)
        
        # L2 normalize
        return self._normalize(mean_embedding)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            # Return zero vectors for empty texts
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            if 'colbert_vecs' in dummy_result:
                emb_dim = dummy_result['colbert_vecs'].shape[-1]
            else:
                for key in ['dense_vecs', 'dense', 'dense_embedding']:
                    if key in dummy_result:
                        emb_dim = dummy_result[key].shape[-1]
                        break
                else:
                    first_key = list(dummy_result.keys())[0]
                    emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros((len(texts), emb_dim))
        
        # Encode batch using EmbeddingService
        results = self.embedding_service.encode_corpus(valid_texts, batch_size=batch_size)
        
        # Get colbert_vecs (token-level hidden states)
        if 'colbert_vecs' in results:
            # colbert_vecs is a list of arrays, one per document
            # Each element shape: [seq_len, hidden_dim]
            batch_embeddings = []
            colbert_vecs = results['colbert_vecs']
            for colbert_vec in colbert_vecs:
                if isinstance(colbert_vec, np.ndarray):
                    # Check if array is empty (shape[0] == 0)
                    if colbert_vec.shape[0] == 0:
                        # Empty array - use zero vector with correct dimension
                        emb_dim = colbert_vec.shape[1] if len(colbert_vec.shape) > 1 else 1024
                        mean_emb = np.zeros(emb_dim)
                    else:
                        # Mean pool over sequence length
                        mean_emb = colbert_vec.mean(axis=0)
                    batch_embeddings.append(self._normalize(mean_emb))
                elif isinstance(colbert_vec, list):
                    # Handle list format - convert to array first
                    if len(colbert_vec) == 0:
                        # Empty list - use zero vector
                        mean_emb = np.zeros(1024)  # Default dimension
                    else:
                        colbert_array = np.array(colbert_vec)
                        if colbert_array.shape[0] == 0:
                            emb_dim = colbert_array.shape[1] if len(colbert_array.shape) > 1 else 1024
                            mean_emb = np.zeros(emb_dim)
                        else:
                            mean_emb = colbert_array.mean(axis=0)
                    batch_embeddings.append(self._normalize(mean_emb))
                else:
                    # Try to convert to array
                    try:
                        colbert_array = np.array(colbert_vec)
                        if colbert_array.shape[0] == 0:
                            emb_dim = colbert_array.shape[1] if len(colbert_array.shape) > 1 else 1024
                            mean_emb = np.zeros(emb_dim)
                        else:
                            mean_emb = colbert_array.mean(axis=0)
                        batch_embeddings.append(self._normalize(mean_emb))
                    except Exception:
                        # Skip if can't process - use zero vector as fallback
                        batch_embeddings.append(self._normalize(np.zeros(1024)))
        else:
            # Fallback to dense embeddings
            dense_key = None
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in results:
                    dense_key = key
                    break
            
            if dense_key:
                dense_embeddings = results[dense_key]
                batch_embeddings = [self._normalize(emb) for emb in dense_embeddings]
            else:
                # Last resort
                first_key = list(results.keys())[0]
                batch_embeddings = [self._normalize(emb) for emb in results[first_key]]
        
        # Handle empty texts by inserting zero vectors
        final_embeddings = []
        valid_idx = 0
        for text in texts:
            if text and isinstance(text, str):
                # Record pre-L2 norm if collection enabled (from colbert or dense)
                if self._collect_preL2:
                    if 'colbert_vecs' in results:
                        colbert_vec = results['colbert_vecs'][valid_idx]
                        if isinstance(colbert_vec, np.ndarray):
                            if colbert_vec.shape[0] > 0:
                                pre_norm = np.linalg.norm(colbert_vec.mean(axis=0))
                            else:
                                pre_norm = 0.0  # Empty array
                        else:
                            colbert_array = np.array(colbert_vec)
                            if colbert_array.shape[0] > 0:
                                pre_norm = np.linalg.norm(colbert_array.mean(axis=0))
                            else:
                                pre_norm = 0.0  # Empty array
                    else:
                        pre_norm = np.linalg.norm(dense_embeddings[valid_idx]) if 'dense_embeddings' in locals() else np.linalg.norm(results[list(results.keys())[0]][valid_idx])
                    self._last_preL2_norms.append(float(pre_norm))
                final_embeddings.append(batch_embeddings[valid_idx])
                valid_idx += 1
            else:
                if self._collect_preL2:
                    self._last_preL2_norms.append(0.0)  # Zero vector
                # Zero vector for empty text
                final_embeddings.append(np.zeros_like(batch_embeddings[0]) if batch_embeddings else np.zeros(1024))
        
        return np.array(final_embeddings)
