# -*- coding: utf-8 -*-

import abc
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.aaa_data_pipline.embedding.embedding import EmbeddingService  # noqa: E402


class ChunkBase(abc.ABC):
    """Abstract base class for document embedding aggregation methods."""
    
    def __init__(self, embedding_service: EmbeddingService = None, 
                 model_name: str = "BAAI/bge-m3"):
        if embedding_service is None:
            self.embedding_service = EmbeddingService(model_name, None)
        else:
            self.embedding_service = embedding_service
        
        self.model_name = model_name
        self._collect_preL2 = False
        self._last_preL2_norms = []
    
    @abc.abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a text document using the chunking strategy."""
        pass
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts. Subclasses can override for efficiency."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return np.array(embeddings)
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def enable_preL2_collection(self, flag: bool = True):
        """Enable or disable pre-L2 norm collection during embedding."""
        self._collect_preL2 = flag
        self._last_preL2_norms = []
    
    def fetch_preL2_norms(self) -> np.ndarray:
        """Fetch collected pre-L2 norms and clear the buffer."""
        vals = np.array(self._last_preL2_norms) if self._last_preL2_norms else np.array([])
        self._last_preL2_norms = []
        return vals
    
    def cleanup(self):
        """Clean up EmbeddingService resources."""
        if hasattr(self, 'embedding_service') and self.embedding_service is not None:
            self.embedding_service.cleanup()
