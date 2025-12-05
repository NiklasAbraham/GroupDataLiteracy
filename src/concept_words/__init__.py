"""
Concept extraction module.

This module provides two methods for extracting concepts from text:
- Sparse method: Uses lexical weights from sparse embeddings
- Dense method: Uses dense embeddings directly
"""

# Export sparse method
from .concept_extraction_sparse import (
    extract_concepts_from_text,
    extract_concepts_from_embedding_results,
    build_noun_lemma_weights,
    filter_by_zipf,
)

# Export dense method
from .concept_extraction_dense import (
    extract_concepts_from_dense_embedding,
    extract_dense_embedding_from_results,
)

# Export shared concept space functionality
from .concept_space import (
    ConceptSpace,
    DEFAULT_CONCEPT_DIR,
    DEFAULT_CONCEPT_MODEL,
    build_wordnet_concept_vocab,
    embed_and_save_concept_vocab,
)

__all__ = [
    # Sparse method
    'extract_concepts_from_text',
    'extract_concepts_from_embedding_results',
    'build_noun_lemma_weights',
    'filter_by_zipf',
    # Dense method
    'extract_concepts_from_dense_embedding',
    'extract_dense_embedding_from_results',
    # Shared
    'ConceptSpace',
    'DEFAULT_CONCEPT_DIR',
    'DEFAULT_CONCEPT_MODEL',
    'build_wordnet_concept_vocab',
    'embed_and_save_concept_vocab',
]
