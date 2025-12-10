"""
Find closest neighbors in embedding space using cosine similarity.

This module provides functions to find the n closest neighbors to an anchor
embedding in a corpus of embeddings using cosine similarity.
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
    Find the n closest neighbors to an anchor embedding in the latent space.

    The function calculates cosine similarity between the anchor embedding and
    all embeddings in the corpus, then returns the n most similar items.

    Mathematical formulation:
    - Cosine similarity: sim(a, b) = (a · b) / (||a|| ||b||)
    - Cosine distance: dist(a, b) = 1 - sim(a, b)
    - Normalized embeddings: a_norm = a / ||a||

    Parameters:
    - embeddings_corpus: Array of embeddings to search through
        (shape: [n_movies, embedding_dim])
    - anchor_embedding: The anchor embedding to find neighbors for
        (shape: [1, embedding_dim])
    - movie_ids: Array of movie IDs corresponding to embeddings_corpus
    - movie_data: DataFrame with movie metadata
        (must contain 'movie_id' and 'title' columns)
    - n: Number of closest neighbors to find (default: 10)
    - anchor_idx: Index of anchor in the corpus (if None, will not exclude).
        Used to exclude the anchor from results.

    Returns:
    - List of tuples (qid, title, distance, similarity) for the n closest neighbors
        sorted by distance (ascending)

    Raises:
    - ValueError: If inputs are invalid or mismatched
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

    # Normalize embeddings for cosine similarity calculation
    # Normalize anchor embedding
    anchor_norm = np.linalg.norm(anchor_embedding, axis=1, keepdims=True)
    if anchor_norm[0, 0] == 0:
        raise ValueError("Anchor embedding is zero vector")
    anchor_normalized = anchor_embedding / anchor_norm

    # Normalize corpus embeddings
    corpus_norms = np.linalg.norm(embeddings_corpus, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0] = 1  # Avoid division by zero
    corpus_normalized = embeddings_corpus / corpus_norms

    # Calculate cosine similarities
    # cosine_similarity returns a matrix where entry (0, j) is the cosine
    # similarity between anchor_embedding[0] and embeddings_corpus[j]
    similarities = cosine_similarity(anchor_normalized, corpus_normalized)[0]

    # Convert to distances (1 - similarity)
    # Distance ranges from 0 (identical) to 2 (opposite directions)
    distances = 1 - similarities

    # Exclude the anchor movie itself if anchor_idx is provided
    if anchor_idx is not None:
        distances[anchor_idx] = np.inf

    # Find n closest neighbors
    n_neighbors = min(n, len(movie_ids) - (1 if anchor_idx is not None else 0))
    closest_indices = np.argsort(distances)[:n_neighbors]

    # Get results
    results = []
    for idx in closest_indices:
        neighbor_qid = movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]

        # Get title from movie_data
        neighbor_movie = movie_data[movie_data["movie_id"] == neighbor_qid]
        if not neighbor_movie.empty:
            title = neighbor_movie.iloc[0]["title"]
        else:
            title = "Unknown"

        results.append((neighbor_qid, title, distance, similarity))

    return results
