"""
Epsilon ball analysis functions for finding movies within a distance threshold.

This module provides functions to find all movies within an epsilon distance
(epsilon ball) around anchor movie embeddings using cosine distance.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_movies_in_epsilon_ball(
    embeddings_corpus: np.ndarray,
    anchor_embedding: np.ndarray,
    movie_ids: np.ndarray,
    epsilon: float,
    exclude_anchor_ids: list = None,
):
    """
    Find all movies within an epsilon distance (epsilon ball) around an anchor embedding.

    The function calculates cosine distance between the anchor embedding and
    all embeddings in the corpus, then returns all movies within the epsilon threshold.

    Mathematical formulation:
    - Cosine similarity: sim(a, b) = (a · b) / (||a|| ||b||)
    - Cosine distance: dist(a, b) = 1 - sim(a, b)
    - Epsilon ball: all movies where dist(anchor, movie) <= epsilon

    Parameters:
    - embeddings_corpus: Array of embeddings to search through
        (shape: [n_movies, embedding_dim])
    - anchor_embedding: The anchor embedding (can be average of multiple anchors)
        (shape: [1, embedding_dim])
    - movie_ids: Array of movie IDs corresponding to embeddings_corpus
    - epsilon: Maximum cosine distance threshold (0 <= epsilon <= 2)
    - exclude_anchor_ids: List of movie IDs to exclude from results (e.g., anchor movies)

    Returns:
    - Tuple of (indices, distances, similarities) for movies within epsilon ball
        - indices: Array of indices in embeddings_corpus
        - distances: Array of cosine distances
        - similarities: Array of cosine similarities
        All sorted by distance (ascending)

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

    if epsilon < 0 or epsilon > 2:
        raise ValueError(f"Epsilon must be between 0 and 2, got {epsilon}")

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

    # Exclude anchor movies if specified
    if exclude_anchor_ids is not None:
        exclude_mask = np.array([mid in exclude_anchor_ids for mid in movie_ids])
        distances[exclude_mask] = np.inf

    # Find all movies within epsilon ball
    within_epsilon_mask = distances <= epsilon
    within_epsilon_indices = np.where(within_epsilon_mask)[0]

    if len(within_epsilon_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    # Get distances and similarities for movies within epsilon
    within_epsilon_distances = distances[within_epsilon_indices]
    within_epsilon_similarities = similarities[within_epsilon_indices]

    # Sort by distance (ascending)
    sort_order = np.argsort(within_epsilon_distances)
    sorted_indices = within_epsilon_indices[sort_order]
    sorted_distances = within_epsilon_distances[sort_order]
    sorted_similarities = within_epsilon_similarities[sort_order]

    return sorted_indices, sorted_distances, sorted_similarities


def compute_anchor_embedding(
    anchor_qids: list,
    embeddings_corpus: np.ndarray,
    movie_ids: np.ndarray,
    method: str = "average",
):
    """
    Compute anchor embedding from multiple anchor movie QIDs.

    Parameters:
    - anchor_qids: List of movie QIDs to use as anchors
    - embeddings_corpus: Array of embeddings to search through
        (shape: [n_movies, embedding_dim])
    - movie_ids: Array of movie IDs corresponding to embeddings_corpus
    - method: Method to combine anchor embeddings
        - "average": Average of all anchor embeddings (default)
        - "medoid": Use the medoid (most central) embedding

    Returns:
    - Anchor embedding (shape: [1, embedding_dim])

    Raises:
    - ValueError: If anchor QIDs are not found or method is invalid
    """
    if len(anchor_qids) == 0:
        raise ValueError("No anchor QIDs provided")

    # Find indices of anchor movies
    anchor_indices = []
    for qid in anchor_qids:
        indices = np.where(movie_ids == qid)[0]
        if len(indices) == 0:
            raise ValueError(f"Anchor QID '{qid}' not found in embeddings")
        if len(indices) > 1:
            # Use first occurrence if multiple found
            anchor_indices.append(indices[0])
        else:
            anchor_indices.append(indices[0])

    # Get anchor embeddings
    anchor_embeddings = embeddings_corpus[anchor_indices]

    if method == "average":
        # Compute average embedding
        anchor_embedding = np.mean(anchor_embeddings, axis=0, keepdims=True)
    elif method == "medoid":
        # Find medoid (most central embedding)
        from .cosine_distance_util import find_nearest_and_furthest_medoid

        medoid_idx, _ = find_nearest_and_furthest_medoid(anchor_embeddings)
        anchor_embedding = anchor_embeddings[medoid_idx : medoid_idx + 1]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average' or 'medoid'")

    return anchor_embedding
