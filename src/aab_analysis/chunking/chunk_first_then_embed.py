# -*- coding: utf-8 -*-

import numpy as np
from .chunk_base_class import ChunkBase
from transformers import AutoTokenizer


def _get_embedding_dim(embedding_service):
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


class ChunkFirstEmbed(ChunkBase):
    """Chunk-first embedding: chunk text first, then embed each chunk."""
    
    def __init__(self, embedding_service=None, model_name: str = "BAAI/bge-m3", 
                 chunk_size: int = 512, stride: int = 256):
        super().__init__(embedding_service, model_name)
        self.chunk_size = chunk_size
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
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
        if not text or not isinstance(text, str):
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros(emb_dim)
        
        chunks = self._chunk_text(text)
        if not chunks:
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros(emb_dim)
        
        chunk_embeddings = []
        for chunk in chunks:
            results = self.embedding_service.encode_corpus([chunk], batch_size=1)
            dense_embeddings = _get_dense_embedding(results)
            chunk_embeddings.append(dense_embeddings[0])
        
        chunk_embeddings = np.array(chunk_embeddings)
        mean_embedding = chunk_embeddings.mean(axis=0)
        return self._normalize(mean_embedding)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        
        all_chunks = []
        for text in texts:
            if not text or not isinstance(text, str):
                all_chunks.append([])
            else:
                all_chunks.append(self._chunk_text(text))
        
        flat_chunks = []
        chunk_offsets = [0]
        for chunks in all_chunks:
            flat_chunks.extend(chunks)
            chunk_offsets.append(len(flat_chunks))
        
        if not flat_chunks:
            emb_dim = _get_embedding_dim(self.embedding_service)
            return np.zeros((len(texts), emb_dim))
        
        chunk_batch_size = self._compute_chunk_batch_size(batch_size)
        results = self.embedding_service.encode_corpus(
            flat_chunks, batch_size=chunk_batch_size, require_token_embeddings=False
        )
        
        dense_embeddings = _get_dense_embedding(results)
        chunk_embeddings = np.array([self._normalize(emb) for emb in dense_embeddings])
        
        final_embeddings = []
        for text_idx in range(len(texts)):
            chunks = all_chunks[text_idx]
            if not chunks:
                final_embeddings.append(
                    np.zeros_like(chunk_embeddings[0]) if len(chunk_embeddings) > 0 else np.zeros(1024)
                )
            else:
                start_idx = chunk_offsets[text_idx]
                end_idx = chunk_offsets[text_idx + 1]
                text_chunk_embeddings = chunk_embeddings[start_idx:end_idx]
                mean_embedding = text_chunk_embeddings.mean(axis=0)
                final_embeddings.append(self._normalize(mean_embedding))
        
        return np.array(final_embeddings)
    
    def _compute_chunk_batch_size(self, base_batch_size: int) -> int:
        """Compute appropriate batch size for chunks based on chunk size and GPU memory."""
        import torch
        
        if self.chunk_size <= 512:
            base = min(max(base_batch_size * 2, 32), 64)
        elif self.chunk_size <= 1024:
            base = min(max(base_batch_size * 2, 16), 32)
        elif self.chunk_size <= 2048:
            base = min(max(base_batch_size, 8), 16)
        else:
            base = min(max(base_batch_size // 2, 4), 8)
        
        if not torch.cuda.is_available():
            return base
        
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory_gb = total - allocated
        except Exception:
            return base
        
        if available_memory_gb < 8.0:
            return max(1, base // 4)
        elif available_memory_gb < 12.0:
            return max(1, base // 2)
        return base
