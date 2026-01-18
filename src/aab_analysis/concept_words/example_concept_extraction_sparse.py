"""
Example script demonstrating how to use the concept extraction module.

This shows the easy-to-use API for extracting concepts from text.
"""

import sys
from pathlib import Path

import numpy as np
import spacy
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.aab_analysis.concept_words.concept_extraction_dense import (  # noqa: E402
    extract_dense_embedding_from_results,
)
from src.aab_analysis.concept_words.concept_extraction_sparse import (  # noqa: E402
    extract_concepts_from_embedding_results,
)
from src.aab_analysis.concept_words.concept_space import (  # noqa: E402
    DEFAULT_CONCEPT_DIR,
    ConceptSpace,
    get_concept_space_filenames,
)
from src.aaa_data_pipline.embedding.embedding import EmbeddingService  # noqa: E402
from src.utils.data_utils import load_final_dataset  # noqa: E402

# Configuration
DATA_DIR = str(BASE_DIR / "data" / "data_final")
CSV_PATH = str(BASE_DIR / "data" / "data_final" / "final_dataset.csv")
MODEL_NAME = "BAAI/bge-m3"
TARGET_DEVICES = ['cuda:0']


def main():
    """Example usage of concept extraction."""
    # Load data
    df = load_final_dataset(CSV_PATH, verbose=True)
    df = df[df['plot'].notna() & (df['plot'].str.len() > 2000)].copy()
    
    # Select a random movie
    random_movie = df.sample(n=1, random_state=39).iloc[0]
    plot_text = random_movie['plot']
    
    print(f"\nSelected movie ID: {random_movie['movie_id']}")
    print(f"Plot length: {len(plot_text)} characters\n")
    
    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize embedding service
    embedding_service = EmbeddingService(MODEL_NAME, TARGET_DEVICES)
    
    # Get embeddings with lexical weights
    print("Encoding text with embedding service...")
    results = embedding_service.encode_corpus([plot_text], batch_size=1)
    
    # Extract concepts using the high-level API
    print("\nExtracting concepts...")
    top_concepts = extract_concepts_from_embedding_results(
        text=plot_text,
        embedding_results=results,
        tokenizer=tokenizer,
        nlp=nlp,
        concept_model=MODEL_NAME,  # Use same model as embeddings
        top_k=30,
        zipf_threshold=4.0,
        min_zipf_vocab=2.5,
        max_vocab=10000
    )
    
    # Compute cosine similarities for each concept
    print("\nComputing cosine similarities...")
    dense_embedding = extract_dense_embedding_from_results(results)
    
    # Normalize dense embedding for cosine similarity
    dense_norm = np.linalg.norm(dense_embedding)
    if dense_norm > 0:
        dense_embedding = dense_embedding / dense_norm
    
    # Load concept space to get concept vectors
    concept_dir = DEFAULT_CONCEPT_DIR
    min_zipf_vocab = 2.5
    max_vocab = 10000
    words_filename, vecs_filename = get_concept_space_filenames(
        min_zipf_vocab, max_vocab, MODEL_NAME
    )
    concept_words_path = concept_dir / words_filename
    concept_vecs_path = concept_dir / vecs_filename
    
    concept_space = ConceptSpace(concept_words_path, concept_vecs_path, model_name=MODEL_NAME)
    
    # Create a mapping from concept word to index
    concept_word_to_idx = {word: idx for idx, word in enumerate(concept_space.concept_words)}
    
    # Compute cosine similarities for each concept in results
    concepts_with_cosine = []
    for concept, lexical_score in top_concepts:
        if concept in concept_word_to_idx:
            concept_idx = concept_word_to_idx[concept]
            concept_vec = concept_space.concept_vecs[concept_idx]
            # Concept vectors should already be normalized, but ensure it
            concept_vec_norm = np.linalg.norm(concept_vec)
            if concept_vec_norm > 0:
                concept_vec = concept_vec / concept_vec_norm
            cosine_score = float(dense_embedding @ concept_vec)
            concepts_with_cosine.append((concept, lexical_score, cosine_score))
        else:
            # If concept not found, set cosine to 0
            concepts_with_cosine.append((concept, lexical_score, 0.0))
    
    # Display results
    if top_concepts:
        print(f"\n{'='*80}")
        print(f"TOP 30 CONCEPTS:")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Concept':<30} {'Lexical Score':<15} {'Cosine Score':<15}")
        print(f"{'-'*80}")
        for idx, (concept, lexical_score, cosine_score) in enumerate(concepts_with_cosine, 1):
            print(f"{idx:2d}.   {concept:30s} {lexical_score:15.6f} {cosine_score:15.6f}")
        print(f"{'='*80}\n")
    else:
        print("No concepts found.\n")
    
    # Cleanup
    embedding_service.cleanup()


if __name__ == '__main__':
    main()

