"""
Utility functions for finding neighbors and dissimilar items in embedding space.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_n_closest_neighbours(
    embeddings_corpus: np.ndarray,
    anchor_embedding: np.ndarray,
    movie_ids: np.ndarray,
    movie_data,
    n: int = 10,
    anchor_idx: int = None,
):
    """
    Find the n closest neighbors to an anchor embedding using cosine similarity.

    Parameters:
    - embeddings_corpus: Array of embeddings to search through (shape: [n_movies, embedding_dim])
    - anchor_embedding: The anchor embedding (shape: [1, embedding_dim])
    - movie_ids: Array of movie IDs corresponding to embeddings_corpus
    - movie_data: DataFrame with movie metadata (must contain 'movie_id' and 'title' columns)
    - n: Number of closest neighbors to find (default: 10)
    - anchor_idx: Index of anchor in the corpus (if None, will not exclude). Used to exclude the anchor from results.

    Returns:
    - List of tuples (qid, title, distance, similarity) for the n closest neighbors sorted by distance (ascending)
    """
    if len(movie_ids) == 0:
        raise ValueError("No movie IDs provided")

    if embeddings_corpus.shape[0] != len(movie_ids):
        raise ValueError(
            f"Mismatch between embeddings ({embeddings_corpus.shape[0]}) "
            f"and movie_ids ({len(movie_ids)})"
        )

    if embeddings_corpus.shape[1] != anchor_embedding.shape[1]:
        raise ValueError(
            f"Mismatch between corpus embedding dimension "
            f"({embeddings_corpus.shape[1]}) and anchor embedding dimension "
            f"({anchor_embedding.shape[1]})"
        )

    anchor_norm = np.linalg.norm(anchor_embedding, axis=1, keepdims=True)
    if anchor_norm[0, 0] == 0:
        raise ValueError("Anchor embedding is zero vector")
    anchor_normalized = anchor_embedding / anchor_norm

    corpus_norms = np.linalg.norm(embeddings_corpus, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0] = 1
    corpus_normalized = embeddings_corpus / corpus_norms

    similarities = cosine_similarity(anchor_normalized, corpus_normalized)[0]
    distances = 1 - similarities

    if anchor_idx is not None:
        distances[anchor_idx] = np.inf

    n_neighbors = min(n, len(movie_ids) - (1 if anchor_idx is not None else 0))
    closest_indices = np.argsort(distances)[:n_neighbors]

    results = []
    for idx in closest_indices:
        neighbor_qid = movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]

        neighbor_movie = movie_data[movie_data["movie_id"] == neighbor_qid]
        if not neighbor_movie.empty:
            title = neighbor_movie.iloc[0]["title"]
        else:
            title = "Unknown"

        results.append((neighbor_qid, title, distance, similarity))

    return results


def find_most_dissimilar_movies(
    reference: str | np.ndarray,
    embeddings: np.ndarray,
    movie_ids: np.ndarray,
    movie_data,
    n: int = 10,
):
    """
    Find the n most dissimilar movies based on distance from a reference embedding.

    Parameters:
    - reference: Reference for comparison. Can be:
        - str: Movie ID (qid) to use as reference (must exist in movie_ids)
        - np.ndarray: Embedding vector to use as reference (shape: [embedding_dim])
    - embeddings: Array of embeddings to search through (shape: [n_movies, embedding_dim])
    - movie_ids: Array of movie IDs corresponding to embeddings (shape: [n_movies,])
    - movie_data: DataFrame with movie metadata (must contain 'movie_id', 'title', and 'year' columns)
    - n: Number of most dissimilar movies to find (default: 10)

    Returns:
    - List of tuples (qid, title, distance, similarity, year) for the n most dissimilar movies, sorted by distance (descending)
    """
    if len(movie_ids) == 0:
        raise ValueError("No movie IDs provided")

    if embeddings.shape[0] != len(movie_ids):
        raise ValueError(
            f"Mismatch between embeddings ({embeddings.shape[0]}) "
            f"and movie_ids ({len(movie_ids)})"
        )

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

    reference_norm = np.linalg.norm(reference_embedding)
    if reference_norm == 0:
        raise ValueError("Reference embedding is zero vector")
    reference_normalized = reference_embedding / reference_norm

    all_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1
    all_normalized = embeddings / all_norms

    reference_reshaped = reference_normalized.reshape(1, -1)
    similarities = cosine_similarity(reference_reshaped, all_normalized)[0]
    distances = 1 - similarities

    if reference_idx is not None:
        distances[reference_idx] = np.inf

    n_movies = min(n, len(movie_ids) - (1 if reference_idx is not None else 0))
    most_dissimilar_indices = np.argsort(distances)[-n_movies:][::-1]

    results = []
    for idx in most_dissimilar_indices:
        movie_id = movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]

        movie_row = movie_data[movie_data["movie_id"] == movie_id]
        if not movie_row.empty:
            title = movie_row.iloc[0]["title"]
            year = movie_row.iloc[0]["year"]
        else:
            title = "Unknown"
            year = None

        results.append((movie_id, title, distance, similarity, year))

    return results
