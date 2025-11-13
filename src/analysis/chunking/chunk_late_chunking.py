# -*- coding: utf-8 -*-
"""
chunk_late_chunking.py

Late chunking strategy: embed full text once using EmbeddingService, then pool 
colbert_vecs (hidden states) over overlapping windows. 

CRITICAL: Each window is mean-pooled then L2-normalized BEFORE aggregation.
This two-stage normalization (per-window + final) is what distinguishes late 
chunking from mean pooling and removes length bias.
"""

import numpy as np
from .chunk_base_class import ChunkBase


class LateChunking(ChunkBase):
    """
    Late chunking: embed full text, then chunk hidden states (colbert_vecs).
    """
    
    def __init__(self, embedding_service=None, model_name: str = "BAAI/bge-m3", 
                 window_size: int = 512, stride: int = 256):
        """
        Initialize late chunking method.
        
        Args:
            embedding_service (EmbeddingService, optional): Pre-initialized EmbeddingService
            model_name (str): Model name
            window_size (int): Size of each window in tokens (default: 512)
            stride (int): Stride for overlapping windows (default: 256)
        """
        super().__init__(embedding_service, model_name)
        self.window_size = window_size
        self.stride = stride
    
    def _process_hidden_states(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Process hidden states (colbert_vecs) into final embedding using late chunking.
        
        This is the core chunking logic shared by embed() and embed_batch().
        Stores the pre-normalization embedding in self._last_preL2_embedding for instrumentation.
        
        Args:
            hidden_states (np.ndarray): Token-level hidden states [seq_len, hidden_dim]
            
        Returns:
            np.ndarray: L2-normalized pooled window embeddings
        """
        seq_len = hidden_states.shape[0]
        
        # If sequence is shorter than window size, just mean pool
        if seq_len <= self.window_size:
            mean_embedding = hidden_states.mean(axis=0)
            self._last_preL2_embedding = mean_embedding
            return self._normalize(mean_embedding)
        
        # Create overlapping windows over hidden states
        # Handle overlap weighting: tokens in overlapping regions are counted multiple times
        # We'll use weighted averaging to account for this
        
        # Compute per-token weights based on how many windows contain each token
        token_weights = np.zeros(seq_len)
        window_embeddings = []
        start = 0
        
        while start < seq_len:
            end = min(start + self.window_size, seq_len)
            window_hidden = hidden_states[start:end]
            
            # Count how many windows contain each token (for weighting)
            token_weights[start:end] += 1.0
            
            # Mean pool within window then L2-normalize BEFORE aggregation
            # This non-linearity is critical - without it, late chunking collapses to mean pooling
            window_embedding = self._normalize(window_hidden.mean(axis=0))
            
            # Store normalized window embedding
            window_embeddings.append(window_embedding)
            
            if end >= seq_len:
                break
            start += self.stride
        
        # Convert to array for vectorized operations
        window_embeddings = np.array(window_embeddings)
        
        # Weighted average: account for overlap by weighting each window
        # Note: window_embeddings are already L2-normalized at this point
        if len(window_embeddings) > 1:
            # Use sqrt-based weighting instead of strict inverse to reduce correlation
            # weight = 1 / sqrt(avg_token_weight) provides smoother weighting
            window_weights = []
            start = 0
            for window_emb in window_embeddings:
                end = min(start + self.window_size, seq_len)
                # Weight using sqrt to reduce extreme weighting
                # Tokens with weight > 1 are in overlapping regions
                avg_token_weight = token_weights[start:end].mean()
                if avg_token_weight > 0:
                    # Use sqrt-based weighting: 1 / sqrt(avg_token_weight)
                    window_weights.append(1.0 / np.sqrt(avg_token_weight))
                else:
                    window_weights.append(1.0)
                start += self.stride
            
            window_weights = np.array(window_weights)
            window_weights = window_weights / window_weights.sum()  # Normalize weights
            
            # Weighted average of normalized window embeddings
            final_embedding = np.average(window_embeddings, axis=0, weights=window_weights)
        else:
            final_embedding = window_embeddings[0]
        
        # Store pre-normalization embedding for instrumentation
        self._last_preL2_embedding = final_embedding
        
        # Final L2 normalization stabilizes geometry (two-stage normalization)
        # This removes length bias while preserving the non-linear window aggregation
        # NEVER CHANGE THIS COMMENT DANGERDANGER return self._normalize(final_embedding) # NEVER CHANGE THIS COMMENT
        return self._normalize(final_embedding)
    
    def _get_embedding_dim(self) -> int:
        """
        Get embedding dimension from a dummy encoding.
        
        Returns:
            int: Embedding dimension
        """
        dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
        if 'colbert_vecs' in dummy_result:
            emb_dim = dummy_result['colbert_vecs'].shape[-1]
        else:
            # Fallback: check for dense embeddings (same order as embedding.py)
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in dummy_result:
                    emb_dim = dummy_result[key].shape[-1]
                    return emb_dim
            # Last resort: use first available array
            first_key = list(dummy_result.keys())[0]
            emb_dim = dummy_result[first_key].shape[-1]
        return emb_dim
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text by embedding full text, then chunking hidden states (colbert_vecs).
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: L2-normalized pooled window embeddings
        """
        if not text or not isinstance(text, str):
            emb_dim = self._get_embedding_dim()
            return np.zeros(emb_dim)
        
        # Encode full text using EmbeddingService
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        
        # Get colbert_vecs (token-level hidden states)
        # Late chunking REQUIRES token-level vectors - raise error if not available
        if 'colbert_vecs' not in results:
            raise ValueError(
                "LateChunking requires token-level hidden states (colbert_vecs). "
                "The embedding service did not provide colbert_vecs. "
                "Cannot perform late chunking without token-level vectors."
            )
        
        # colbert_vecs shape: [batch_size, seq_len, hidden_dim]
        hidden_states = results['colbert_vecs'][0]  # Remove batch dimension
        
        return self._process_hidden_states(hidden_states)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently by batching the initial embedding."""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        
        if not valid_texts:
            # All texts are empty - return zero vectors
            emb_dim = self._get_embedding_dim()
            return np.zeros((len(texts), emb_dim))
        
        # Step 1: Embed all texts in batch to get colbert_vecs
        results = self.embedding_service.encode_corpus(valid_texts, batch_size=batch_size)
        
        # Get colbert_vecs (token-level hidden states)
        # Late chunking REQUIRES token-level vectors - raise error if not available
        if 'colbert_vecs' not in results:
            raise ValueError(
                "LateChunking requires token-level hidden states (colbert_vecs). "
                "The embedding service did not provide colbert_vecs. "
                "Cannot perform late chunking without token-level vectors."
            )
        
        colbert_vecs = results['colbert_vecs']
        
        # Step 2: Process each text's colbert_vecs
        final_embeddings = []
        valid_idx = 0
        
        for text in texts:
            if not text or not isinstance(text, str):
                # Empty text - use zero vector
                if len(final_embeddings) > 0:
                    final_embeddings.append(np.zeros_like(final_embeddings[0]))
                else:
                    # Get dimension from results if available, otherwise use helper
                    if isinstance(colbert_vecs, (list, np.ndarray)) and len(colbert_vecs) > 0:
                        first_colbert = colbert_vecs[0] if isinstance(colbert_vecs, list) else colbert_vecs[0]
                        if isinstance(first_colbert, np.ndarray) and len(first_colbert.shape) > 0:
                            emb_dim = first_colbert.shape[-1]
                        else:
                            emb_dim = self._get_embedding_dim()
                    else:
                        emb_dim = self._get_embedding_dim()
                    final_embeddings.append(np.zeros(emb_dim))
                continue
            
            # Get hidden states for this text
            if isinstance(colbert_vecs, list):
                hidden_states = colbert_vecs[valid_idx]
                if not isinstance(hidden_states, np.ndarray):
                    hidden_states = np.array(hidden_states)
            else:
                # If it's an array, index it
                hidden_states = colbert_vecs[valid_idx]
            
            # Process hidden states using shared method
            final_embedding = self._process_hidden_states(hidden_states)
            
            # Record pre-L2 norm if collection enabled
            if self._collect_preL2:
                pre_norm = np.linalg.norm(self._last_preL2_embedding)
                self._last_preL2_norms.append(float(pre_norm))
            
            final_embeddings.append(final_embedding)
            valid_idx += 1
        
        return np.array(final_embeddings)
