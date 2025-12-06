"""
Example script demonstrating PCA analysis on movie embeddings with concept space.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from data_utils import load_final_data_with_embeddings
from concept_words.pca_analysis import run_pca_analysis


def example_random_movies():
    """PCA on random movies."""
    # Load and filter data
    df = load_final_data_with_embeddings(verbose=True)
    # df = df[df['genre_cluster_names'].str.contains('drama', case=False, na=False)]
    # size of df
    print(f"Size of df: {len(df)}")
    # df = df[(df['year'] >= 2010) & (df['year'] <= 2020)]
    df = df.sample(n=20000, random_state=42)
    
    # Run PCA analysis
    run_pca_analysis(
        df=df,
        concept_model="BAAI/bge-m3",
        top_k_concepts=5000,
        n_components=10,
        max_vocab=10000,
        min_zipf_vocab=2.5,
        verbose=True,
        output_path="pca_analysis_random_movies.png"
    )


if __name__ == "__main__":
    example_random_movies()
