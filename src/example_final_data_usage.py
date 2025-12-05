"""
Example script demonstrating the new final data loading functions.

This script shows common use cases for loading and working with the final dataset.
"""

import sys
sys.path.insert(0, '/home/nab/Niklas/GroupDataLiteracy/src')

from data_utils import (
    load_final_dataset,
    load_final_dense_embeddings,
    load_final_sparse_embeddings,
    load_final_data_with_embeddings,
    load_final_data_limited
)
import numpy as np


def example_1_load_metadata_only():
    """Example 1: Load movie metadata without embeddings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Load Metadata Only")
    print("="*80)
    
    # Load the final dataset
    df = load_final_dataset(verbose=False)

    # count the number of nan entires in the genre column
    print(f"\nNumber of nan entries in genre column: {df['genre'].isna().sum()}")
    # now in the genre_cluster_names column
    print(f"\nNumber of nan entries in genre_cluster_names column: {df['genre_cluster_names'].isna().sum()}")
    
    print(f"\nTotal movies: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample movies:")
    print(df[['movie_id', 'title', 'year', 'genre']].head(10))
    
    # Filter by year
    movies_2020 = df[df['year'] == 2020]
    print(f"\nMovies from 2020: {len(movies_2020)}")


def example_2_load_embeddings_only():
    """Example 2: Load embeddings without metadata."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Load Embeddings Only")
    print("="*80)
    
    # Load dense embeddings
    embeddings, movie_ids = load_final_dense_embeddings(verbose=False)
    
    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"\nFirst 5 movie IDs: {movie_ids[:5]}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Compute some statistics
    mean_embedding = embeddings.mean(axis=0)
    print(f"\nMean embedding norm: {np.linalg.norm(mean_embedding):.4f}")


def example_3_load_data_with_embeddings():
    """Example 3: Load metadata merged with embeddings."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Load Metadata with Embeddings")
    print("="*80)
    
    # Load everything together
    df = load_final_data_with_embeddings(verbose=False)
    
    print(f"\nTotal movies: {len(df)}")
    print(f"Has embedding: {'embedding' in df.columns}")
    print(f"Has genre clusters: {'new_genre' in df.columns}")
    
    # Show sample movie with embedding
    sample = df.iloc[0]
    print(f"\nSample movie:")
    print(f"  ID: {sample['movie_id']}")
    print(f"  Title: {sample['title']}")
    print(f"  Year: {sample['year']}")
    print(f"  Genre: {sample['genre']}")
    print(f"  Genre cluster: {sample['new_genre']}")
    print(f"  Embedding shape: {sample['embedding'].shape}")


def example_4_sample_movies_per_year():
    """Example 4: Sample a limited number of movies per year."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Sample Movies Per Year")
    print("="*80)
    
    # Load 10 movies per year
    df = load_final_data_limited(movies_per_year=10, verbose=False)
    
    print(f"\nTotal sampled movies: {len(df)}")
    
    # Show distribution by year
    year_counts = df['year'].value_counts().sort_index()
    print(f"\nMovies per year (first 10 years):")
    print(year_counts.head(10))
    
    # Show distribution by genre
    if 'new_genre' in df.columns:
        genre_counts = df['new_genre'].str.split('|').explode().value_counts()
        print(f"\nTop genres in sample:")
        print(genre_counts.head(10))


def example_5_load_sparse_embeddings():
    """Example 5: Load and examine sparse embeddings."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Load Sparse Embeddings")
    print("="*80)
    
    # Load sparse embeddings
    result = load_final_sparse_embeddings(verbose=False)
    
    if result is not None:
        token_indices, weights, movie_ids = result
        
        print(f"\nTotal documents: {len(token_indices)}")
        print(f"Sample document (first movie):")
        print(f"  Movie ID: {movie_ids[0]}")
        print(f"  Number of non-zero tokens: {len(token_indices[0])}")
        print(f"  First 10 token indices: {token_indices[0][:10]}")
        print(f"  First 10 weights: {weights[0][:10]}")
        
        # Calculate sparsity
        avg_non_zero = sum(len(ti) for ti in token_indices) / len(token_indices)
        print(f"\nAverage non-zero tokens per document: {avg_non_zero:.1f}")
    else:
        print("\nSparse embeddings not found")


def example_6_compute_similarity():
    """Example 6: Compute similarity between movies."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Compute Movie Similarity")
    print("="*80)
    
    # Load data with embeddings
    df = load_final_data_with_embeddings(verbose=False)
    
    # Get embeddings matrix
    embeddings = np.vstack(df['embedding'].values)
    
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute similarity for first movie with all others
    query_idx = 0
    similarities = embeddings_norm @ embeddings_norm[query_idx]
    
    # Find top 5 most similar movies (excluding itself)
    top_indices = np.argsort(similarities)[-6:-1][::-1]
    
    query_movie = df.iloc[query_idx]
    print(f"\nQuery movie:")
    print(f"  Title: {query_movie['title']}")
    print(f"  Year: {query_movie['year']}")
    print(f"  Genre: {query_movie['genre']}")
    
    print(f"\nTop 5 most similar movies:")
    for i, idx in enumerate(top_indices, 1):
        movie = df.iloc[idx]
        sim = similarities[idx]
        print(f"{i}. {movie['title']} ({movie['year']}) - Similarity: {sim:.4f}")


def example_7_filter_by_genre():
    """Example 7: Filter movies by genre."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Filter by Genre")
    print("="*80)
    
    # Load data
    df = load_final_data_with_embeddings(verbose=False)
    
    # Filter for action movies
    if 'new_genre' in df.columns:
        action_movies = df[df['new_genre'].str.contains('action', case=False, na=False)]
        
        print(f"\nTotal movies: {len(df)}")
        print(f"Action movies: {len(action_movies)}")
        
        # Show sample action movies
        print(f"\nSample action movies:")
        sample = action_movies[['movie_id', 'title', 'year', 'new_genre']].head(10)
        for idx, row in sample.iterrows():
            print(f"  {row['title']} ({row['year']}) - {row['new_genre']}")
    else:
        print("\nGenre cluster column not found")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("FINAL DATA LOADING - USAGE EXAMPLES")
    print("="*80)
    
    try:
        example_1_load_metadata_only()
        example_2_load_embeddings_only()
        example_3_load_data_with_embeddings()
        example_4_sample_movies_per_year()
        example_5_load_sparse_embeddings()
        example_6_compute_similarity()
        example_7_filter_by_genre()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
