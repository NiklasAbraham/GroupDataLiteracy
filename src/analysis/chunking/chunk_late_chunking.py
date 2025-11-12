# -*- coding: utf-8 -*-
"""
chunk_late_chunking.py

Late chunking strategy: embed full text once using EmbeddingService, then pool 
colbert_vecs (hidden states) over overlapping windows. Each window embedding is 
L2-normalized, then all windows are averaged together.
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
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text by embedding full text, then chunking hidden states (colbert_vecs).
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: L2-normalized pooled window embeddings
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
        
        # Encode full text using EmbeddingService
        results = self.embedding_service.encode_corpus([text], batch_size=1)
        
        # Get colbert_vecs (token-level hidden states)
        if 'colbert_vecs' in results:
            # colbert_vecs shape: [batch_size, seq_len, hidden_dim]
            hidden_states = results['colbert_vecs'][0]  # Remove batch dimension
        else:
            # Fallback: if no colbert_vecs, use dense (but this won't work for chunking)
            # In this case, just return the dense embedding (same order as embedding.py)
            for key in ['dense_vecs', 'dense', 'dense_embedding']:
                if key in results:
                    return self._normalize(results[key][0])
            # Last resort: use first available array
            first_key = list(results.keys())[0]
            hidden_states = results[first_key][0]
        
        seq_len = hidden_states.shape[0]
        
        # If sequence is shorter than window size, just mean pool
        if seq_len <= self.window_size:
            mean_embedding = hidden_states.mean(axis=0)
            return self._normalize(mean_embedding)
        
        # Create overlapping windows over hidden states
        window_embeddings = []
        start = 0
        
        while start < seq_len:
            end = min(start + self.window_size, seq_len)
            window_hidden = hidden_states[start:end]
            
            # Mean pool within window
            window_embedding = window_hidden.mean(axis=0)
            
            # L2 normalize each window embedding
            window_embedding = self._normalize(window_embedding)
            
            window_embeddings.append(window_embedding)
            
            if end >= seq_len:
                break
            start += self.stride
        
        # Average all window embeddings
        window_embeddings = np.array(window_embeddings)
        final_embedding = window_embeddings.mean(axis=0)
        
        # Final L2 normalization
        return self._normalize(final_embedding)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently by batching the initial embedding."""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        
        if not valid_texts:
            # All texts are empty - return zero vectors
            dummy_result = self.embedding_service.encode_corpus(["test"], batch_size=1)
            if 'colbert_vecs' in dummy_result:
                emb_dim = dummy_result['colbert_vecs'].shape[-1] if hasattr(dummy_result['colbert_vecs'], 'shape') else 1024
            else:
                for key in ['dense_vecs', 'dense', 'dense_embedding']:
                    if key in dummy_result:
                        emb_dim = dummy_result[key].shape[-1]
                        break
                else:
                    first_key = list(dummy_result.keys())[0]
                    emb_dim = dummy_result[first_key].shape[-1]
            return np.zeros((len(texts), emb_dim))
        
        # Step 1: Embed all texts in batch to get colbert_vecs
        results = self.embedding_service.encode_corpus(valid_texts, batch_size=batch_size)
        
        # Step 2: Process each text's colbert_vecs
        final_embeddings = []
        valid_idx = 0
        
        for text_idx, text in enumerate(texts):
            if not text or not isinstance(text, str):
                # Empty text - use zero vector
                if len(final_embeddings) > 0:
                    final_embeddings.append(np.zeros_like(final_embeddings[0]))
                else:
                    # Get dimension from results
                    if 'colbert_vecs' in results and len(results['colbert_vecs']) > 0:
                        first_colbert = results['colbert_vecs'][0]
                        if isinstance(first_colbert, np.ndarray):
                            emb_dim = first_colbert.shape[-1]
                        else:
                            emb_dim = 1024
                    else:
                        emb_dim = 1024
                    final_embeddings.append(np.zeros(emb_dim))
                continue
            
            # Get colbert_vecs for this text
            if 'colbert_vecs' in results:
                colbert_vecs = results['colbert_vecs']
                if isinstance(colbert_vecs, list):
                    hidden_states = colbert_vecs[valid_idx]
                    if not isinstance(hidden_states, np.ndarray):
                        hidden_states = np.array(hidden_states)
                else:
                    # If it's an array, index it
                    hidden_states = colbert_vecs[valid_idx]
            else:
                # Fallback to dense (but this won't work for chunking)
                for key in ['dense_vecs', 'dense', 'dense_embedding']:
                    if key in results:
                        final_embeddings.append(self._normalize(results[key][valid_idx]))
                        valid_idx += 1
                        break
                else:
                    # Last resort
                    first_key = list(results.keys())[0]
                    final_embeddings.append(self._normalize(results[first_key][valid_idx]))
                    valid_idx += 1
                continue
            
            seq_len = hidden_states.shape[0]
            
            # If sequence is shorter than window size, just mean pool
            if seq_len <= self.window_size:
                mean_embedding = hidden_states.mean(axis=0)
                final_embeddings.append(self._normalize(mean_embedding))
                valid_idx += 1
                continue
            
            # Create overlapping windows over hidden states
            window_embeddings = []
            start = 0
            
            while start < seq_len:
                end = min(start + self.window_size, seq_len)
                window_hidden = hidden_states[start:end]
                
                # Mean pool within window
                window_embedding = window_hidden.mean(axis=0)
                
                # L2 normalize each window embedding
                window_embedding = self._normalize(window_embedding)
                
                window_embeddings.append(window_embedding)
                
                if end >= seq_len:
                    break
                start += self.stride
            
            # Average all window embeddings
            window_embeddings = np.array(window_embeddings)
            final_embedding = window_embeddings.mean(axis=0)
            
            # Final L2 normalization
            final_embeddings.append(self._normalize(final_embedding))
            valid_idx += 1
        
        return np.array(final_embeddings)
