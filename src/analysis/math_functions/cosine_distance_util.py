"""
Cosine distance calculation functions for embedding analysis.

This module provides functions to calculate average cosine distances
within groups and between groups of embeddings.
"""

import numpy as np
import pandas as pd
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
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    # Calculate pairwise cosine similarities
    # cosine_similarity returns a matrix where entry (i,j) is the cosine similarity
    # between embeddings[i] and embeddings[j]
    similarity_matrix = cosine_similarity(normalized_embeddings)
    
    # Convert to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Get upper triangle (excluding diagonal) to avoid counting pairs twice
    n = len(embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_triangle_indices]
    
    return np.mean(distances)


def calculate_average_cosine_distance_between_groups(group1_embeddings, group2_embeddings):
    """
    Calculate average cosine distance between two groups of embeddings.
    
    Parameters:
    - group1_embeddings: numpy array of shape (n1, embedding_dim)
    - group2_embeddings: numpy array of shape (n2, embedding_dim)
    
    Returns:
    - Average cosine distance (float)
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

    # 3. Reshape 1D vectors to 2D matrices (1, D) to satisfy axis=1 requirement
    if group1_embeddings.ndim == 1:
        group1_embeddings = np.expand_dims(group1_embeddings, axis=0)

    if group2_embeddings.ndim == 1:
        group2_embeddings = np.expand_dims(group2_embeddings, axis=0)

    # Normalize embeddings
    norms1 = np.linalg.norm(group1_embeddings, axis=1, keepdims=True)
    norms1[norms1 == 0] = 1
    normalized_group1 = group1_embeddings / norms1
    
    norms2 = np.linalg.norm(group2_embeddings, axis=1, keepdims=True)
    norms2[norms2 == 0] = 1
    normalized_group2 = group2_embeddings / norms2
    
    # Calculate pairwise cosine similarities between groups
    similarity_matrix = cosine_similarity(normalized_group1, normalized_group2)
    
    # Convert to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Return mean of all pairwise distances
    return np.mean(distance_matrix)

