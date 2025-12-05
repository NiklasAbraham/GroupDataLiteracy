"""
Concept extraction from text using dense embeddings and concept space similarity.

This module provides functionality to extract semantic concepts from text by:
1. Getting dense embeddings of the text
2. Comparing to embeddings in the concept space using cosine similarity
3. Ranking concept words by similarity score and returning top 30

Uses the same concept space built with spaCy as the sparse extraction method.
"""

from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict

from .concept_space import (
    ConceptSpace,
    DEFAULT_CONCEPT_DIR,
    DEFAULT_CONCEPT_MODEL,
    get_concept_space_filenames,
    build_wordnet_concept_vocab,
    embed_and_save_concept_vocab
)



def extract_dense_embedding_from_results(embedding_results: dict) -> np.ndarray:
    """
    Extract dense embedding vector from embedding service results dictionary.
    
    This helper function handles various formats that embedding services might return.
    
    Args:
        embedding_results: Dictionary returned by EmbeddingService.encode_corpus()
                          Must contain dense embeddings under key 'dense_vecs', 'dense', or 'dense_embedding'
    
    Returns:
        Dense embedding vector as numpy array of shape (d,)
    
    Raises:
        ValueError: If no dense embedding is found in the results
    """
    dense_embedding = None
    
    # Check common keys for dense embeddings
    for key in ['dense_vecs', 'dense', 'dense_embedding']:
        if key in embedding_results:
            dense_data = embedding_results[key]
            if isinstance(dense_data, (list, tuple)) and len(dense_data) > 0:
                dense_embedding = np.asarray(dense_data[0], dtype=np.float32)
            elif hasattr(dense_data, 'shape'):
                dense_embedding = np.asarray(dense_data, dtype=np.float32)
                if dense_embedding.ndim == 2:
                    dense_embedding = dense_embedding[0]  # Take first if batch
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
    
    This function computes cosine similarity between the dense embedding and
    all concept embeddings in the concept space, then returns the top K concepts.
    Uses the same concept space built with spaCy as the sparse extraction method.
    
    Args:
        dense_embedding: Dense embedding vector of shape (d,) where d is embedding dimension
        concept_space: Optional pre-loaded ConceptSpace instance
        concept_dir: Directory containing concept space files (default: data/concept_space)
        concept_model: SentenceTransformer model name for concept space (default: BAAI/bge-small-en-v1.5)
        top_k: Number of top concepts to return (default: 30)
        build_concept_space: If True, build concept space if it doesn't exist (default: True)
        min_zipf_vocab: Minimum Zipf frequency for building concept vocabulary (default: 4.0)
        max_vocab: Maximum vocabulary size for concept space (default: 20000)
    
    Returns:
        List of (concept, score) tuples sorted by score descending, where score is cosine similarity
    """
    # Ensure dense_embedding is a numpy array and 1D
    dense_embedding = np.asarray(dense_embedding, dtype=np.float32)
    if dense_embedding.ndim != 1:
        raise ValueError(f"dense_embedding must be 1D, got shape {dense_embedding.shape}")
    
    # Normalize the dense embedding for cosine similarity
    dense_norm = np.linalg.norm(dense_embedding)
    if dense_norm == 0:
        return []
    dense_embedding = dense_embedding / dense_norm
    
    # Load or build concept space (same logic as sparse extraction)
    if concept_space is None:
        if concept_dir is None:
            concept_dir = DEFAULT_CONCEPT_DIR
        
        concept_dir = Path(concept_dir)
        
        # Generate filenames based on parameters
        words_filename, vecs_filename = get_concept_space_filenames(
            min_zipf_vocab, max_vocab, concept_model
        )
        concept_words_path = concept_dir / words_filename
        concept_vecs_path = concept_dir / vecs_filename
        
        if not concept_words_path.exists() or not concept_vecs_path.exists():
            if build_concept_space:
                vocab = build_wordnet_concept_vocab(min_zipf=min_zipf_vocab, max_vocab=max_vocab, filter_verbs=False, filter_generic=True)
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
    
    # Validate dimension compatibility
    if dense_embedding.shape[0] != concept_space.concept_vecs.shape[1]:
        raise ValueError(
            f"Dimension mismatch: embedding {dense_embedding.shape[0]} != concept space {concept_space.concept_vecs.shape[1]}. "
            f"Use concept_model='{concept_model}' to match embedding model."
        )
    
    # Compute cosine similarities with all concept embeddings
    # concept_space.concept_vecs is shape (N, d) and should already be normalized
    # dense_embedding is shape (d,) and is now normalized
    # Compute: dense_embedding @ concept_vecs.T = (d,) @ (d, N) = (N,)
    similarities = dense_embedding @ concept_space.concept_vecs.T
    
    # Get top K concepts by similarity score
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
    
    Convenience wrapper that extracts dense_embedding from embedding_results
    and calls extract_concepts_from_dense_embedding.
    Uses the same concept space built with spaCy as the sparse extraction method.
    
    Args:
        embedding_results: Dictionary returned by EmbeddingService.encode_corpus()
        concept_space: Optional pre-loaded ConceptSpace instance
        concept_dir: Directory containing concept space files
        concept_model: SentenceTransformer model name for concept space
        top_k: Number of top concepts to return
        build_concept_space: If True, build concept space if it doesn't exist
        min_zipf_vocab: Minimum Zipf frequency for building concept vocabulary
        max_vocab: Maximum vocabulary size for concept space
    
    Returns:
        List of (concept, score) tuples sorted by score descending
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


