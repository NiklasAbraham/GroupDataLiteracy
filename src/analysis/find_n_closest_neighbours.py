"""
Find n closest neighbors in the latent space for a given qid.

This script finds the n closest neighbors to a specified movie (by qid)
in the latent embedding space using cosine similarity.
"""

import logging
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from data_utils
# Path setup must occur before this import
from src.data_utils import (  # type: ignore  # noqa: E402
    load_final_dataset,
    load_final_dense_embeddings,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024


def find_n_closest_neighbours(
    qid: str,
    n: int = 10,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
):
    """
    Find the n closest neighbors to a given qid in the latent space.

    Parameters:
    - qid: The movie_id (qid) to find neighbors for
    - n: Number of closest neighbors to find (default: 10)
    - start_year: First year to filter movies (default: 1930)
    - end_year: Last year to filter movies (default: 2024)
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv

    Returns:
    - List of tuples (qid, title, distance, similarity) for the n closest neighbors
    """
    # Load all embeddings and corresponding movie IDs
    logger.info("Loading embeddings...")
    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")

    logger.info(f"Total movies with embeddings: {len(all_movie_ids)}")
    logger.info(f"Embedding shape: {all_embeddings.shape}")

    # Load movie metadata to get titles and filter by year
    logger.info("Loading movie metadata...")
    movie_data = load_final_dataset(csv_path, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {csv_path}")

    # Filter by year range if year column exists
    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()
        logger.info(
            f"Filtered to {len(movie_data)} movies between {start_year} and {end_year}"
        )

    # Filter embeddings to only include movies in the filtered dataset
    valid_movie_ids = set(movie_data["movie_id"].values)
    valid_indices = np.array(
        [i for i, mid in enumerate(all_movie_ids) if mid in valid_movie_ids]
    )

    if len(valid_indices) == 0:
        raise ValueError(
            f"No movies found in the specified year range ({start_year}-{end_year})"
        )

    # Filter embeddings and movie_ids to only include valid movies
    filtered_embeddings = all_embeddings[valid_indices]
    filtered_movie_ids = all_movie_ids[valid_indices]

    logger.info(
        f"Filtered to {len(filtered_movie_ids)} movies with embeddings in year range"
    )

    # Find the index of the query qid in the filtered data
    query_indices = np.where(filtered_movie_ids == qid)[0]
    if len(query_indices) == 0:
        raise ValueError(
            f"QID '{qid}' not found in embeddings for the specified year range"
        )

    if len(query_indices) > 1:
        logger.warning(f"Multiple embeddings found for qid '{qid}', using first one")

    query_idx = query_indices[0]
    query_embedding = filtered_embeddings[
        query_idx : query_idx + 1
    ]  # Keep 2D shape for sklearn

    logger.info(f"Found query movie at index {query_idx} in filtered data")

    # Get query movie title
    query_movie = movie_data[movie_data["movie_id"] == qid]
    if not query_movie.empty:
        query_title = query_movie.iloc[0]["title"]
        logger.info(f"Query movie: {query_title} (QID: {qid})")
    else:
        query_title = "Unknown"
        logger.warning(f"Could not find title for QID '{qid}'")

    # Calculate cosine similarity between query and all other embeddings
    logger.info("Calculating cosine similarities...")

    # Normalize embeddings
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    if query_norm[0, 0] == 0:
        raise ValueError("Query embedding is zero vector")
    query_normalized = query_embedding / query_norm

    all_norms = np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1  # Avoid division by zero
    all_normalized = filtered_embeddings / all_norms

    # Calculate cosine similarities
    similarities = cosine_similarity(query_normalized, all_normalized)[0]

    # Convert to distances (1 - similarity)
    distances = 1 - similarities

    # Exclude the query movie itself
    distances[query_idx] = np.inf

    # Find n closest neighbors
    n_neighbors = min(n, len(filtered_movie_ids) - 1)  # -1 to exclude query itself
    closest_indices = np.argsort(distances)[:n_neighbors]

    # Get results
    results = []
    for idx in closest_indices:
        neighbor_qid = filtered_movie_ids[idx]
        distance = distances[idx]
        similarity = similarities[idx]

        # Get title
        neighbor_movie = movie_data[movie_data["movie_id"] == neighbor_qid]
        if not neighbor_movie.empty:
            title = neighbor_movie.iloc[0]["title"]
        else:
            title = "Unknown"

        results.append((neighbor_qid, title, distance, similarity))

    return results


def main(
    qid: str = "Q1931001",
    n: int = 10,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
):
    """
    Main function to find and print n closest neighbors.

    Parameters:
    - qid: The movie_id (qid) to find neighbors for
    - n: Number of closest neighbors to find
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Finding {n} closest neighbors for QID: {qid}")
    logger.info(f"{'=' * 60}")

    try:
        results = find_n_closest_neighbours(
            qid=qid,
            n=n,
            start_year=start_year,
            end_year=end_year,
            data_dir=data_dir,
            csv_path=csv_path,
        )

        # Print results
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Top {len(results)} closest neighbors:")
        logger.info(f"{'=' * 60}\n")

        for i, (neighbor_qid, title, distance, similarity) in enumerate(results, 1):
            logger.info(f"{i}. QID: {neighbor_qid}")
            logger.info(f"   Title: {title}")
            logger.info(f"   Cosine Distance: {distance:.6f}")
            logger.info(f"   Cosine Similarity: {similarity:.6f}")
            logger.info("")

        logger.info(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Error finding neighbors: {e}")
        raise


if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    main(
        qid="Q104123",  # Change this to your desired qid
        n=30,  # Change this to desired number of neighbors
    )
