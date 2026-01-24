"""
Cosine distance calculation functions for embedding analysis.

This module provides functions to calculate average cosine distances
within groups and between groups of embeddings.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def calculate_average_cosine_distance(embeddings):
    """
    Calculate average pairwise cosine distance within a group of embeddings.

    Parameters:
    - embeddings: numpy array of shape (n_samples, embedding_dim)

    Returns:
    - Average cosine distance (float)
    """
    if len(embeddings) < 2:
        return 0.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_embeddings = embeddings / norms

    similarity_matrix = cosine_similarity(normalized_embeddings)
    distance_matrix = 1 - similarity_matrix

    n = len(embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_triangle_indices]

    return np.mean(distances)


def calculate_average_cosine_distance_between_groups(
    group1_embeddings, group2_embeddings
):
    """
    Calculate average cosine distance between two groups of embeddings.

    Parameters:
    - group1_embeddings: numpy array of shape (n1, embedding_dim)
    - group2_embeddings: numpy array of shape (n2, embedding_dim)

    Returns:
    - Average cosine distance (float)

    Important: When computing genre drift for two embedding groups we first calculate the mean embedding for each group and then call
    this function, so the group embeddings will be dx1, so it only outputs one pair
    """
    if group1_embeddings is None or group2_embeddings is None:
        return None

    if pd.isna(group1_embeddings).all() or pd.isna(group2_embeddings).all():
        return None

    if (len(group1_embeddings) == 0 or
            len(group2_embeddings) == 0 or
            np.any(np.isnan(group1_embeddings)) or
            np.any(np.isnan(group2_embeddings))):
        return None

    if group1_embeddings.ndim == 1:
        group1_embeddings = np.expand_dims(group1_embeddings, axis=0)

    if group2_embeddings.ndim == 1:
        group2_embeddings = np.expand_dims(group2_embeddings, axis=0)

    norms1 = np.linalg.norm(group1_embeddings, axis=1, keepdims=True)
    norms1[norms1 == 0] = 1
    normalized_group1 = group1_embeddings / norms1

    norms2 = np.linalg.norm(group2_embeddings, axis=1, keepdims=True)
    norms2[norms2 == 0] = 1
    normalized_group2 = group2_embeddings / norms2

    similarity_matrix = cosine_similarity(normalized_group1, normalized_group2)
    distance_matrix = 1 - similarity_matrix

    return np.mean(distance_matrix)


def find_nearest_and_furthest_medoid(embeddings: np.ndarray) -> tuple[int, int]:
    """
    Returns the nearest & furthest medoid index of a set of embeddings.
    Uses cosine distance as the distance measure.

    :param embeddings: 2D np array containing rows of data
    :type embeddings: np.ndarray
    :rtype: tuple[int, ndarray[Any, Any]]
    """

    pairwise_distance_matrix = cdist(embeddings, embeddings, metric="cosine")
    most_sim_medoid_index = np.argmin(pairwise_distance_matrix.sum(axis=0))
    most_dissim_medoid_index = np.argmax(pairwise_distance_matrix.sum(axis=0))

    return int(most_sim_medoid_index), int(most_dissim_medoid_index)

def get_medoid_embedding(embeddings: np.ndarray) -> np.ndarray:
    """
    Finds and returns the most centered embedding (medoid) vector.

    :param embeddings: 2D np array containing rows of data (the embeddings)
    :type embeddings: np.ndarray
    :rtype: np.ndarray
    """
    medoid_index, _ = find_nearest_and_furthest_medoid(embeddings)
    medoid_embedding = embeddings[medoid_index]

    return medoid_embedding

def get_average_embedding(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the average embedding across all embeddings.

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])

    Returns:
    - Average embedding vector (shape: [embedding_dim])
    """
    return np.mean(embeddings, axis=0)
