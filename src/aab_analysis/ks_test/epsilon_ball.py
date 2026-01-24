"""Epsilon ball analysis functions for finding movies within a distance threshold."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_movies_in_epsilon_ball(
    embeddings_corpus: np.ndarray,
    anchor_embedding: np.ndarray,
    movie_ids: np.ndarray,
    epsilon: float,
    exclude_anchor_ids: list = None,
):
    """Find all movies within an epsilon distance (epsilon ball) around an anchor embedding."""
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

    anchor_norm = np.linalg.norm(anchor_embedding, axis=1, keepdims=True)
    if anchor_norm[0, 0] == 0:
        raise ValueError("Anchor embedding is zero vector")
    anchor_normalized = anchor_embedding / anchor_norm

    corpus_norms = np.linalg.norm(embeddings_corpus, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0] = 1
    corpus_normalized = embeddings_corpus / corpus_norms

    similarities = cosine_similarity(anchor_normalized, corpus_normalized)[0]
    distances = 1 - similarities

    if exclude_anchor_ids is not None:
        exclude_mask = np.array([mid in exclude_anchor_ids for mid in movie_ids])
        distances[exclude_mask] = np.inf

    within_epsilon_mask = distances <= epsilon
    within_epsilon_indices = np.where(within_epsilon_mask)[0]

    if len(within_epsilon_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    within_epsilon_distances = distances[within_epsilon_indices]
    within_epsilon_similarities = similarities[within_epsilon_indices]

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
    """Compute anchor embedding from multiple anchor movie QIDs."""
    if len(anchor_qids) == 0:
        raise ValueError("No anchor QIDs provided")

    anchor_indices = []
    for qid in anchor_qids:
        indices = np.where(movie_ids == qid)[0]
        if len(indices) == 0:
            raise ValueError(f"Anchor QID '{qid}' not found in embeddings")
        anchor_indices.append(indices[0])

    anchor_embeddings = embeddings_corpus[anchor_indices]

    if method == "average":
        anchor_embedding = np.mean(anchor_embeddings, axis=0, keepdims=True)
    elif method == "medoid":
        from src.aab_analysis.math_functions.cosine_distance_util import find_nearest_and_furthest_medoid

        medoid_idx, _ = find_nearest_and_furthest_medoid(anchor_embeddings)
        anchor_embedding = anchor_embeddings[medoid_idx : medoid_idx + 1]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average' or 'medoid'")

    return anchor_embedding
