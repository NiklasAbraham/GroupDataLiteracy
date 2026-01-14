"""
Find the most dissimilar movies in the latent embedding space.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from src.aab_analysis.math_functions import (  # noqa: E402
    find_most_dissimilar_movies,
)
from src.utils.data_utils import (  # noqa: E402
    load_final_dataset,
    load_final_dense_embeddings,
)

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
    Main function to find and print n most dissimilar movies.

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
    # Load all embeddings and corresponding movie IDs
    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")

    # Load movie metadata to get titles and filter by year
    movie_data = load_final_dataset(csv_path, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {csv_path}")

    # Filter by year range if year column exists
    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()

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

    # Determine reference embedding
    if reference is None:
        # Use mean embedding as reference
        reference = np.mean(filtered_embeddings, axis=0)
        print("Using mean embedding as reference")
    elif isinstance(reference, str):
        print(f"Using movie ID '{reference}' as reference")

    # Call the function with loaded data
    results = find_most_dissimilar_movies(
        reference=reference,
        embeddings=filtered_embeddings,
        movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        n=n,
    )

    # Print results
    print(f"\nTop {len(results)} most dissimilar movies:")
    print("=" * 60)
    for i, (movie_id, title, distance, similarity, year) in enumerate(results, 1):
        print(f"{i}. {title} ({movie_id}, {year})")
        print(f"   Distance: {distance:.6f}, Similarity: {similarity:.6f}")

    return results


if __name__ == "__main__":
    # Example 1: Find movies most dissimilar to the mean
    print("Example 1: Most dissimilar to mean embedding")
    print("=" * 60)
    main(reference=None, n=20)

    # Example 2: Find movies most dissimilar to a specific movie
    print("\n\nExample 2: Most dissimilar to specific movie")
    print("=" * 60)
    main(reference="Q1931001", n=10)
