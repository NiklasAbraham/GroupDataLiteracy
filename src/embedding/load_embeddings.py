"""
Utility functions for loading embeddings with movie_id mapping.

This module provides functions to load embeddings that were saved with
movie_id indexing, ensuring embeddings can be matched back to movies.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def load_embeddings_with_ids(year: int, data_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and corresponding movie_ids for a specific year.
    
    Args:
        year: The year to load embeddings for
        data_dir: Directory containing the data files (defaults to 'data' relative to project root)
    
    Returns:
        Tuple of (embeddings, movie_ids) where:
        - embeddings: numpy array of shape (n_movies, embedding_dim)
        - movie_ids: numpy array of shape (n_movies,) with corresponding movie_ids
        
        embedding[i] corresponds to movie_ids[i]
    
    Raises:
        FileNotFoundError: If embedding or movie_id files don't exist
    """
    if data_dir is None:
        # Try to find data directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, 'data')
    
    embeddings_path = os.path.join(data_dir, f'movie_embeddings_{year}.npy')
    movie_ids_path = os.path.join(data_dir, f'movie_ids_{year}.npy')
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(movie_ids_path):
        raise FileNotFoundError(f"Movie IDs file not found: {movie_ids_path}")
    
    embeddings = np.load(embeddings_path)
    movie_ids = np.load(movie_ids_path)
    
    if len(embeddings) != len(movie_ids):
        raise ValueError(
            f"Mismatch: embeddings has {len(embeddings)} entries but "
            f"movie_ids has {len(movie_ids)} entries"
        )
    
    logger.info(f"Loaded {len(embeddings)} embeddings for year {year}")
    return embeddings, movie_ids


def get_embedding_by_movie_id(
    movie_id: str,
    year: int,
    data_dir: str = None
) -> Optional[np.ndarray]:
    """
    Get embedding for a specific movie_id.
    
    Args:
        movie_id: The Wikidata movie_id (e.g., 'Q191753')
        year: The year to search in
        data_dir: Directory containing the data files
    
    Returns:
        The embedding vector for the movie, or None if not found
    """
    embeddings, movie_ids = load_embeddings_with_ids(year, data_dir)
    
    # Find index of movie_id
    indices = np.where(movie_ids == movie_id)[0]
    if len(indices) == 0:
        logger.warning(f"Movie ID {movie_id} not found in embeddings for year {year}")
        return None
    
    if len(indices) > 1:
        logger.warning(f"Multiple embeddings found for movie_id {movie_id}, returning first")
    
    return embeddings[indices[0]]


def create_embedding_to_movie_mapping(
    year: int,
    data_dir: str = None
) -> Dict[str, np.ndarray]:
    """
    Create a dictionary mapping movie_id to embedding.
    
    Args:
        year: The year to load embeddings for
        data_dir: Directory containing the data files
    
    Returns:
        Dictionary mapping movie_id (str) to embedding vector (np.ndarray)
    """
    embeddings, movie_ids = load_embeddings_with_ids(year, data_dir)
    
    return {movie_id: embedding for movie_id, embedding in zip(movie_ids, embeddings)}


def load_all_years_embeddings(
    start_year: int,
    end_year: int,
    data_dir: str = None
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load embeddings for multiple years.
    
    Args:
        start_year: First year to load
        end_year: Last year to load (inclusive)
        data_dir: Directory containing the data files
    
    Returns:
        Dictionary mapping year to (embeddings, movie_ids) tuple
    """
    results = {}
    
    for year in range(start_year, end_year + 1):
        try:
            embeddings, movie_ids = load_embeddings_with_ids(year, data_dir)
            results[year] = (embeddings, movie_ids)
            logger.info(f"Loaded embeddings for year {year}")
        except FileNotFoundError:
            logger.debug(f"Embeddings not found for year {year}, skipping")
            continue
    
    return results


def verify_embedding_alignment(
    year: int,
    csv_file: str = None,
    data_dir: str = None
) -> bool:
    """
    Verify that embeddings align correctly with CSV file movie_ids.
    
    Args:
        year: The year to verify
        csv_file: Path to CSV file (if None, constructs path from year)
        data_dir: Directory containing the data files
    
    Returns:
        True if alignment is correct, False otherwise
    """
    if csv_file is None:
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            data_dir = os.path.join(project_root, 'data')
        csv_file = os.path.join(data_dir, f'wikidata_movies_{year}.csv')
    
    # Load embeddings and movie_ids
    embeddings, embedding_movie_ids = load_embeddings_with_ids(year, data_dir)
    
    # Load CSV and get movie_ids with plots
    df = pd.read_csv(csv_file, dtype=str, low_memory=False)
    if 'plot' not in df.columns or 'movie_id' not in df.columns:
        logger.error("CSV file missing 'plot' or 'movie_id' column")
        return False
    
    # Get movie_ids that have plots
    has_plot_mask = df['plot'].notna() & (df['plot'].astype(str).str.strip() != '') & (df['plot'].astype(str) != 'nan')
    csv_movie_ids = df[has_plot_mask]['movie_id'].tolist()
    
    # Verify counts match
    if len(embeddings) != len(csv_movie_ids):
        logger.error(
            f"Count mismatch: embeddings has {len(embeddings)} entries, "
            f"CSV has {len(csv_movie_ids)} movies with plots"
        )
        return False
    
    # Verify movie_ids match
    if not np.array_equal(embedding_movie_ids, np.array(csv_movie_ids)):
        logger.error("Movie IDs in embeddings don't match CSV movie IDs")
        mismatched = embedding_movie_ids != np.array(csv_movie_ids)
        logger.error(f"Mismatches at indices: {np.where(mismatched)[0][:10]}")  # Show first 10
        return False
    
    logger.info(f"Embedding alignment verified for year {year}: {len(embeddings)} embeddings match CSV")
    return True

