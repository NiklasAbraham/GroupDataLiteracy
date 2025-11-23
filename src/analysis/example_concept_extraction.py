"""
Example script demonstrating how to use the concept extraction module.

This shows the easy-to-use API for extracting concepts from text.
"""

import sys
from pathlib import Path
import spacy
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from data_utils import load_movie_data
from embedding.embedding import EmbeddingService
from analysis.concept_extraction import extract_concepts_from_embedding_results

# Configuration
DATA_DIR = str(BASE_DIR / "data")
MODEL_NAME = "BAAI/bge-m3"
TARGET_DEVICES = ['cuda:0']


def main():
    """Example usage of concept extraction."""
    # Load data
    df = load_movie_data(DATA_DIR, verbose=True)
    df = df[df['plot'].notna() & (df['plot'].str.len() > 2000)].copy()
    
    # Select a random movie
    random_movie = df.sample(n=1, random_state=41).iloc[0]
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
        top_k=30,
        zipf_threshold=4.0
    )
    
    # Display results
    if top_concepts:
        print(f"\n{'='*80}")
        print(f"TOP 30 CONCEPTS:")
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

