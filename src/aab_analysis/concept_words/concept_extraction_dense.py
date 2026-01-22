"""
Concept extraction from text using dense embeddings and concept space similarity.

Extracts semantic concepts by:
1. Getting dense embeddings of the text
2. Comparing to embeddings in the concept space using cosine similarity
3. Ranking concept words by similarity score and returning top K

Uses the same concept space built with spaCy as the sparse extraction method.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .concept_space import (
    ConceptSpace,
    DEFAULT_CONCEPT_DIR,
    DEFAULT_CONCEPT_MODEL,
    build_wordnet_concept_vocab,
    embed_and_save_concept_vocab,
    get_concept_space_filenames,
)



def extract_dense_embedding_from_results(embedding_results: dict) -> np.ndarray:
    """
    Extract dense embedding vector from embedding service results dictionary.
    Handles various formats that embedding services might return.
    """
    dense_embedding = None
    
    for key in ['dense_vecs', 'dense', 'dense_embedding']:
        if key in embedding_results:
            dense_data = embedding_results[key]
            if isinstance(dense_data, (list, tuple)) and len(dense_data) > 0:
                dense_embedding = np.asarray(dense_data[0], dtype=np.float32)
            elif hasattr(dense_data, 'shape'):
                dense_embedding = np.asarray(dense_data, dtype=np.float32)
                if dense_embedding.ndim == 2:
                    dense_embedding = dense_embedding[0]
                elif dense_embedding.ndim > 2:
                    raise ValueError(f"Unexpected embedding shape: {dense_embedding.shape}")
            break
    
    if dense_embedding is None:
        raise ValueError("No dense embedding found in embedding_results. Expected 'dense_vecs' or 'dense' key.")
    
    if dense_embedding.ndim != 1:
        raise ValueError(f"Dense embedding must be 1D, got shape: {dense_embedding.shape}")
    
    return dense_embedding


def extract_concepts_from_dense_embedding(
    dense_embedding: np.ndarray,
    concept_space: Optional[ConceptSpace] = None,
    concept_dir: Optional[Path] = None,
    concept_model: str = DEFAULT_CONCEPT_MODEL,
    top_k: int = 30,
    build_concept_space: bool = True,
    min_zipf_vocab: float = 4.0,
    max_vocab: int = 20000
) -> List[Tuple[str, float]]:
    """
    Extract top concepts from dense embedding by comparing to concept space.
    Computes cosine similarity between the dense embedding and all concept embeddings.
    """
    dense_embedding = np.asarray(dense_embedding, dtype=np.float32)
    if dense_embedding.ndim != 1:
        raise ValueError(f"dense_embedding must be 1D, got shape {dense_embedding.shape}")
    
    dense_norm = np.linalg.norm(dense_embedding)
    if dense_norm == 0:
        return []
    dense_embedding = dense_embedding / dense_norm
    
    if concept_space is None:
        if concept_dir is None:
            concept_dir = DEFAULT_CONCEPT_DIR
        
        concept_dir = Path(concept_dir)
        
        words_filename, vecs_filename = get_concept_space_filenames(
            min_zipf_vocab, max_vocab, concept_model
        )
        concept_words_path = concept_dir / words_filename
        concept_vecs_path = concept_dir / vecs_filename
        
        if not concept_words_path.exists() or not concept_vecs_path.exists():
            if build_concept_space:
                vocab = build_wordnet_concept_vocab(
                    min_zipf=min_zipf_vocab, 
                    max_vocab=max_vocab, 
                    filter_verbs=False, 
                    filter_generic=True
                )
                embed_and_save_concept_vocab(
                    vocab, concept_dir, 
                    model_name=concept_model,
                    min_zipf=min_zipf_vocab,
                    max_vocab=max_vocab
                )
            else:
                raise FileNotFoundError(
                    f"Concept space files not found: {concept_words_path}"
                )
        
        concept_space = ConceptSpace(concept_words_path, concept_vecs_path, model_name=concept_model)
    
    if dense_embedding.shape[0] != concept_space.concept_vecs.shape[1]:
        raise ValueError(
            f"Dimension mismatch: embedding {dense_embedding.shape[0]} != concept space {concept_space.concept_vecs.shape[1]}. "
            f"Use concept_model='{concept_model}' to match embedding model."
        )
    
    similarities = dense_embedding @ concept_space.concept_vecs.T
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        concept_word = concept_space.concept_words[idx]
        results.append((concept_word, score))
    
    return results


def extract_concepts_from_embedding_results(
    embedding_results: Dict,
    concept_space: Optional[ConceptSpace] = None,
    concept_dir: Optional[Path] = None,
    concept_model: str = DEFAULT_CONCEPT_MODEL,
    top_k: int = 30,
    build_concept_space: bool = True,
    min_zipf_vocab: float = 4.0,
    max_vocab: int = 20000
) -> List[Tuple[str, float]]:
    """
    Extract concepts from embedding service results.
    Convenience wrapper that extracts dense_embedding from embedding_results.
    """
    dense_embedding = extract_dense_embedding_from_results(embedding_results)
    
    return extract_concepts_from_dense_embedding(
        dense_embedding=dense_embedding,
        concept_space=concept_space,
        concept_dir=concept_dir,
        concept_model=concept_model,
        top_k=top_k,
        build_concept_space=build_concept_space,
        min_zipf_vocab=min_zipf_vocab,
        max_vocab=max_vocab
    )


