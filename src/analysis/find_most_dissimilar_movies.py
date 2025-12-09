"""
Find the most dissimilar movies in the latent embedding space.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from src.data_utils import (  # noqa: E402
    load_final_dataset,
    load_final_dense_embeddings,
)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024


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
        - str: Movie ID (qid) to use as reference
        - np.ndarray: Embedding vector to use as reference
    - embeddings: Filtered embeddings array (n_movies, embedding_dim)
    - movie_ids: Filtered movie IDs array (n_movies,)
    - movie_data: Filtered movie metadata DataFrame
    - n: Number of most dissimilar movies to find (default: 10)

    Returns:
    - List of tuples (qid, title, distance, similarity, year) for the n most dissimilar movies
    """
    if isinstance(reference, str):
        reference_idx = np.where(movie_ids == reference)[0]
        reference_embedding = embeddings[reference_idx[0]]
    elif isinstance(reference, np.ndarray):
        reference_embedding = reference
    else:
        raise ValueError("reference must be a movie ID string or a numpy array")

    reference_norm = np.linalg.norm(reference_embedding)
    reference_normalized = (
        reference_embedding / reference_norm
        if reference_norm > 0
        else reference_embedding
    )

    all_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1
    all_normalized = embeddings / all_norms

    reference_reshaped = reference_normalized.reshape(1, -1)
    similarities = cosine_similarity(reference_reshaped, all_normalized)[0]
    distances = 1 - similarities

    if isinstance(reference, str):
        ref_idx = np.where(movie_ids == reference)[0]
        if len(ref_idx) > 0:
            distances[ref_idx[0]] = np.inf

    n_movies = min(n, len(movie_ids))
    most_dissimilar_indices = np.argsort(distances)[-n_movies:][::-1]

    results = []
    for idx in most_dissimilar_indices:
        movie_id = movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]

        movie_row = movie_data[movie_data["movie_id"] == movie_id]
        title = movie_row.iloc[0]["title"]
        year = movie_row.iloc[0]["year"]

        results.append((movie_id, title, distance, similarity, year))

    return results


if __name__ == "__main__":
    all_embeddings, all_movie_ids = load_final_dense_embeddings(DATA_DIR, verbose=False)
    movie_data = load_final_dataset(CSV_PATH, verbose=False)
    movie_data = movie_data[
        (movie_data["year"] >= START_YEAR) & (movie_data["year"] <= END_YEAR)
    ].copy()
    valid_movie_ids = set(movie_data["movie_id"].values)
    valid_indices = np.array(
        [i for i, mid in enumerate(all_movie_ids) if mid in valid_movie_ids]
    )
    filtered_embeddings = all_embeddings[valid_indices]
    filtered_movie_ids = all_movie_ids[valid_indices]

    # Example 1: Find movies most dissimilar to the mean
    mean_embedding = np.mean(filtered_embeddings, axis=0)
    results = find_most_dissimilar_movies(
        reference=mean_embedding,
        embeddings=filtered_embeddings,
        movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        n=20,
    )
    print("Most dissimilar to mean:")
    for i, (movie_id, title, distance, similarity, year) in enumerate(results, 1):
        print(f"{i}. {title} ({movie_id}, {year})")
        print(f"   Distance: {distance:.6f}, Similarity: {similarity:.6f}")

    # Example 2: Find movies most dissimilar to a specific movie
    print("\n\nMost dissimilar to specific movie:")
    results = find_most_dissimilar_movies(
        reference="Q1931001",
        embeddings=filtered_embeddings,
        movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        n=10,
    )
    for i, (movie_id, title, distance, similarity, year) in enumerate(results, 1):
        print(f"{i}. {title} ({movie_id}, {year})")
        print(f"   Distance: {distance:.6f}, Similarity: {similarity:.6f}")
