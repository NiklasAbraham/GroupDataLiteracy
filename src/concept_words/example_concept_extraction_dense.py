"""
Example script demonstrating how to use the dense embedding concept extraction module.

This shows the easy-to-use API for extracting concepts from text using dense embeddings.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from data_utils import load_movie_data
from embedding.embedding import EmbeddingService
from concept_words.concept_extraction_dense import (
    extract_concepts_from_dense_embedding,
    extract_dense_embedding_from_results
)
from concept_words.concept_space import get_concept_space_filenames

# Configuration
DATA_DIR = str(BASE_DIR / "data" / "data_final")  # Movie data location
CONCEPT_DIR = BASE_DIR / "data" / "concept_space"  # Concept space folder location
MODEL_NAME = "BAAI/bge-m3"
TARGET_DEVICES = ['cuda:0']



def main():
    """Example usage of dense embedding concept extraction."""
    # Load data
    df = load_movie_data(DATA_DIR, verbose=True)
    df = df[df['plot'].notna() & (df['plot'].str.len() > 2000)].copy()
    
    # Select a random movie
    random_movie = df.sample(n=1, random_state=38).iloc[0]
    plot_text = random_movie['plot']
    
    print(f"\nSelected movie ID: {random_movie['movie_id']}")
    print(f"Plot length: {len(plot_text)} characters\n")
    
    # Display the plot text
    print("="*80)
    print("PLOT TEXT:")
    print("="*80)
    # Print plot text with word wrapping for better readability
    import textwrap
    wrapped_text = textwrap.fill(plot_text, width=78, initial_indent="  ", subsequent_indent="  ")
    print(wrapped_text)
    print("="*80)
    print()
    
    # Initialize embedding service
    embedding_service = EmbeddingService(MODEL_NAME, TARGET_DEVICES)
    
    # Get embeddings with dense vectors
    print("Encoding text with embedding service...")
    results = embedding_service.encode_corpus([plot_text], batch_size=1)
    
    # Extract dense embedding from results
    dense_embedding = extract_dense_embedding_from_results(results)
    print(f"Dense embedding extracted: shape {dense_embedding.shape}")
    
    # Extract concepts using dense embedding directly
    print("\nExtracting concepts using dense embeddings...")
    print("Note: Concept space is built from WordNet nouns filtered by frequency.\n")
    
    # Show which concept space file will be used
    min_zipf_vocab = 2.5  # Lower threshold to include more words
    max_vocab = 20000  # Vocabulary size
    words_filename, vecs_filename = get_concept_space_filenames(
        min_zipf_vocab, max_vocab, MODEL_NAME
    )
    concept_words_path = CONCEPT_DIR / words_filename
    print(f"Concept space files:")
    print(f"  Words: {concept_words_path}")
    print(f"  Vectors: {CONCEPT_DIR / vecs_filename}")
    if concept_words_path.exists():
        from concept_words.concept_space import ConceptSpace
        concept_space = ConceptSpace(concept_words_path, CONCEPT_DIR / vecs_filename, MODEL_NAME)
        print(f"  Loaded {len(concept_space.concept_words)} concepts")
        print(f"  Sample concepts: {', '.join(concept_space.concept_words[:10])}")
    else:
        print(f"  Concept space will be built (this may take a while)...")
    print()
    
    top_concepts = extract_concepts_from_dense_embedding(
        dense_embedding=dense_embedding,
        concept_dir=CONCEPT_DIR,
        concept_model=MODEL_NAME,
        top_k=80,
        min_zipf_vocab=min_zipf_vocab,
        max_vocab=max_vocab
    )
    
    # Display results
    if top_concepts:
        print(f"\n{'='*80}")
        print(f"TOP {len(top_concepts)} CONCEPTS (Dense Embedding Method):")
        print(f"{'='*80}")
        for idx, (concept, score) in enumerate(top_concepts, 1):
            print(f"{idx:2d}. {concept:30s}: {score:.6f}")
        print(f"{'='*80}\n")
    else:
        print("No concepts found.\n")
    
    # Cleanup
    embedding_service.cleanup()


if __name__ == '__main__':
    main()
