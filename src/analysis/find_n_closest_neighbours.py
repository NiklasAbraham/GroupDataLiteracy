"""
Find n closest neighbors in the latent space for a given qid.

This script finds the n closest neighbors to a specified movie (by qid)
in the latent embedding space using cosine similarity.
"""

import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import functions from data_utils
from src.data_utils import (
    load_movie_embeddings, 
    load_movie_data
)
from src.data_cleaning import clean_dataset

DATA_DIR = os.path.join(BASE_DIR, 'data', 'data_final')
START_YEAR = 1930
END_YEAR = 2024
CHUNKING_SUFFIX = None  # Auto-detect


def find_n_closest_neighbours(
    qid: str,
    n: int = 10,
    chunking_suffix: str = None,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR
):
    """
    Find the n closest neighbors to a given qid in the latent space.
    
    Parameters:
    - qid: The movie_id (qid) to find neighbors for
    - n: Number of closest neighbors to find (default: 10)
    - chunking_suffix: Suffix for embedding files (None for auto-detect)
    - start_year: First year to load embeddings from
    - end_year: Last year to load embeddings from
    - data_dir: Directory containing the data files
    
    Returns:
    - List of tuples (qid, title, distance, similarity) for the n closest neighbors
    """
    # Auto-detect chunking suffix if not specified
    if chunking_suffix is None:
        test_year = start_year
        found_suffix = None
        for suffix in ['_cls_token', '_mean_pooling', '']:
            test_path = os.path.join(data_dir, f'movie_embeddings_{test_year}{suffix}.npy')
            if os.path.exists(test_path):
                found_suffix = suffix
                break
        
        if found_suffix is not None:
            chunking_suffix = found_suffix
            logger.info(f"Auto-detected chunking suffix: '{chunking_suffix}'")
        else:
            chunking_suffix = ''
            logger.info("No chunking suffix detected, using default (no suffix)")
    else:
        logger.info(f"Using chunking suffix: '{chunking_suffix}'")
    
    # Load all embeddings and corresponding movie IDs
    logger.info("Loading embeddings...")
    all_embeddings, all_movie_ids = load_movie_embeddings(
        data_dir,
        chunking_suffix=chunking_suffix,
        start_year=start_year,
        end_year=end_year,
        verbose=False
    )
    
    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")
    
    logger.info(f"Total movies with embeddings: {len(all_movie_ids)}")
    logger.info(f"Embedding shape: {all_embeddings.shape}")
    
    # Find the index of the query qid
    query_indices = np.where(all_movie_ids == qid)[0]
    if len(query_indices) == 0:
        raise ValueError(f"QID '{qid}' not found in embeddings")
    
    if len(query_indices) > 1:
        logger.warning(f"Multiple embeddings found for qid '{qid}', using first one")
    
    query_idx = query_indices[0]
    query_embedding = all_embeddings[query_idx:query_idx+1]  # Keep 2D shape for sklearn
    
    logger.info(f"Found query movie at index {query_idx}")
    
    # Load movie metadata to get titles
    logger.info("Loading movie metadata...")
    movie_data = load_movie_data(data_dir, verbose=False)
    
    if movie_data.empty:
        raise ValueError(f"No movie data found in {data_dir}")
    
    logger.info(f"Loaded {len(movie_data)} movies from metadata files")
    
    # Get query movie title
    query_movie = movie_data[movie_data['movie_id'] == qid]
    if not query_movie.empty:
        query_title = query_movie.iloc[0]['title']
        logger.info(f"Query movie: {query_title} (QID: {qid})")
    else:
        query_title = "Unknown"
        logger.warning(f"Could not find title for QID '{qid}'")
    
    # Calculate cosine similarity between query and all other embeddings
    logger.info("Calculating cosine similarities...")
    
    # Normalize embeddings
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    if query_norm[0, 0] == 0:
        raise ValueError(f"Query embedding is zero vector")
    query_normalized = query_embedding / query_norm
    
    all_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1  # Avoid division by zero
    all_normalized = all_embeddings / all_norms
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_normalized, all_normalized)[0]
    
    # Convert to distances (1 - similarity)
    distances = 1 - similarities
    
    # Exclude the query movie itself
    distances[query_idx] = np.inf
    
    # Find n closest neighbors
    n_neighbors = min(n, len(all_movie_ids) - 1)  # -1 to exclude query itself
    closest_indices = np.argsort(distances)[:n_neighbors]
    
    # Get results
    results = []
    for idx in closest_indices:
        neighbor_qid = all_movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]
        
        # Get title
        neighbor_movie = movie_data[movie_data['movie_id'] == neighbor_qid]
        if not neighbor_movie.empty:
            title = neighbor_movie.iloc[0]['title']
        else:
            title = "Unknown"
        
        results.append((neighbor_qid, title, distance, similarity))
    
    return results


def main(
    qid: str = "Q1931001",
    n: int = 10,
    chunking_suffix: str = CHUNKING_SUFFIX,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR
):
    """
    Main function to find and print n closest neighbors.
    
    Parameters:
    - qid: The movie_id (qid) to find neighbors for
    - n: Number of closest neighbors to find
    - chunking_suffix: Suffix for embedding files
    - start_year: First year to load embeddings from
    - end_year: Last year to load embeddings from
    - data_dir: Directory containing the data files
    """
    logger.info(f"{'='*60}")
    logger.info(f"Finding {n} closest neighbors for QID: {qid}")
    logger.info(f"{'='*60}")
    
    try:
        results = find_n_closest_neighbours(
            qid=qid,
            n=n,
            chunking_suffix=chunking_suffix,
            start_year=start_year,
            end_year=end_year,
            data_dir=data_dir
        )
        
        # Print results
        logger.info(f"\n{'='*60}")
        logger.info(f"Top {len(results)} closest neighbors:")
        logger.info(f"{'='*60}\n")
        
        for i, (neighbor_qid, title, distance, similarity) in enumerate(results, 1):
            logger.info(f"{i}. QID: {neighbor_qid}")
            logger.info(f"   Title: {title}")
            logger.info(f"   Cosine Distance: {distance:.6f}")
            logger.info(f"   Cosine Similarity: {similarity:.6f}")
            logger.info("")
        
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error finding neighbors: {e}")
        raise


if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    main(
        qid="Q14786561",  # Change this to your desired qid
        n=30             # Change this to desired number of neighbors
    )

