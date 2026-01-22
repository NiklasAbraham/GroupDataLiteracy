"""
Find the most dissimilar movies in the latent embedding space.
"""

import logging
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.aab_analysis.neighbourhood.neighbourhood_002_neighbor_utils import find_most_dissimilar_movies
from src.utils.data_utils import load_final_dataset, load_final_dense_embeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024


def main(
    reference=None,
    n: int = 10,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
):
    """
    Find and print n most dissimilar movies.

    Parameters:
    - reference: Reference for comparison. Can be:
        - str: Movie ID (qid) to use as reference
        - np.ndarray: Embedding vector to use as reference
        - None: Use mean embedding as reference (default)
    - n: Number of most dissimilar movies to find
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    """
    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")

    movie_data = load_final_dataset(csv_path, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {csv_path}")

    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()

    valid_movie_ids = set(movie_data["movie_id"].values)
    valid_indices = np.array(
        [i for i, mid in enumerate(all_movie_ids) if mid in valid_movie_ids]
    )

    if len(valid_indices) == 0:
        raise ValueError(
            f"No movies found in the specified year range ({start_year}-{end_year})"
        )

    filtered_embeddings = all_embeddings[valid_indices]
    filtered_movie_ids = all_movie_ids[valid_indices]

    if reference is None:
        reference = np.mean(filtered_embeddings, axis=0)
        logger.info("Using mean embedding as reference")
    elif isinstance(reference, str):
        logger.info(f"Using movie ID '{reference}' as reference")

    results = find_most_dissimilar_movies(
        reference=reference,
        embeddings=filtered_embeddings,
        movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        n=n,
    )

    logger.info(f"\nTop {len(results)} most dissimilar movies:")
    logger.info("=" * 60)
    for i, (movie_id, title, distance, similarity, year) in enumerate(results, 1):
        logger.info(f"{i}. {title} ({movie_id}, {year})")
        logger.info(f"   Distance: {distance:.6f}, Similarity: {similarity:.6f}")

    return results


if __name__ == "__main__":
    main(reference=None, n=20)
