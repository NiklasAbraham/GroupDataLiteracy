"""
Find most dissimilar items in embedding space using cosine distance.

This module provides functions to find the n most dissimilar items to a
reference embedding in a corpus of embeddings using cosine similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_most_dissimilar_movies(
    reference: str | np.ndarray,
    embeddings: np.ndarray,
    movie_ids: np.ndarray,
    movie_data,
    n: int = 10,
):
    """
    Find the n most dissimilar movies based on distance from a reference embedding.

    The function calculates cosine similarity between the reference embedding and
    all embeddings in the corpus, then returns the n items with the highest
    cosine distance (lowest similarity).

    Mathematical formulation:
    - Cosine similarity: sim(a, b) = (a · b) / (||a|| ||b||)
    - Cosine distance: dist(a, b) = 1 - sim(a, b)
    - Normalized embeddings: a_norm = a / ||a||

    Parameters:
    - reference: Reference for comparison. Can be:
        - str: Movie ID (qid) to use as reference (must exist in movie_ids)
        - np.ndarray: Embedding vector to use as reference (shape: [embedding_dim])
    - embeddings: Array of embeddings to search through
        (shape: [n_movies, embedding_dim])
    - movie_ids: Array of movie IDs corresponding to embeddings
        (shape: [n_movies,])
    - movie_data: DataFrame with movie metadata
        (must contain 'movie_id', 'title', and 'year' columns)
    - n: Number of most dissimilar movies to find (default: 10)

    Returns:
    - List of tuples (qid, title, distance, similarity, year) for the n most
        dissimilar movies, sorted by distance (descending)

    Raises:
    - ValueError: If reference is invalid type or movie ID not found
    """
    if len(movie_ids) == 0:
        raise ValueError("No movie IDs provided")

    if embeddings.shape[0] != len(movie_ids):
        raise ValueError(
            f"Mismatch between embeddings ({embeddings.shape[0]}) "
            f"and movie_ids ({len(movie_ids)})"
        )

    # Handle reference input
    reference_idx = None
    if isinstance(reference, str):
        reference_indices = np.where(movie_ids == reference)[0]
        if len(reference_indices) == 0:
            raise ValueError(f"Movie ID '{reference}' not found in movie_ids")
        reference_idx = reference_indices[0]
        reference_embedding = embeddings[reference_idx]
    elif isinstance(reference, np.ndarray):
        if reference.ndim != 1:
            raise ValueError(
                f"Reference embedding must be 1D array, got shape {reference.shape}"
            )
        if reference.shape[0] != embeddings.shape[1]:
            raise ValueError(
                f"Mismatch between reference embedding dimension ({reference.shape[0]}) "
                f"and corpus embedding dimension ({embeddings.shape[1]})"
            )
        reference_embedding = reference
    else:
        raise ValueError("reference must be a movie ID string or a numpy array")

    # Normalize reference embedding
    reference_norm = np.linalg.norm(reference_embedding)
    if reference_norm == 0:
        raise ValueError("Reference embedding is zero vector")
    reference_normalized = reference_embedding / reference_norm

    # Normalize corpus embeddings
    all_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1  # Avoid division by zero
    all_normalized = embeddings / all_norms

    # Reshape reference to 2D for sklearn compatibility
    reference_reshaped = reference_normalized.reshape(1, -1)

    # Calculate cosine similarities
    # cosine_similarity returns a matrix where entry (0, j) is the cosine
    # similarity between reference_embedding[0] and embeddings[j]
    similarities = cosine_similarity(reference_reshaped, all_normalized)[0]

    # Convert to distances (1 - similarity)
    # Distance ranges from 0 (identical) to 2 (opposite directions)
    distances = 1 - similarities

    # Exclude the reference movie itself if reference is a movie ID
    if reference_idx is not None:
        distances[reference_idx] = np.inf

    # Find n most dissimilar movies (highest distances)
    n_movies = min(n, len(movie_ids) - (1 if reference_idx is not None else 0))
    most_dissimilar_indices = np.argsort(distances)[-n_movies:][::-1]

    # Get results
    results = []
    for idx in most_dissimilar_indices:
        movie_id = movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]

        # Get title and year from movie_data
        movie_row = movie_data[movie_data["movie_id"] == movie_id]
        if not movie_row.empty:
            title = movie_row.iloc[0]["title"]
            year = movie_row.iloc[0]["year"]
        else:
            title = "Unknown"
            year = None

        results.append((movie_id, title, distance, similarity, year))

    return results
