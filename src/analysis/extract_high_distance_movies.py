"""
Extract titles and QIDs for movies with Mahalanobis distance > threshold.

This script loads cached Gaussian analysis results and extracts movie information
for movies with Mahalanobis distances above a specified threshold.
"""

import hashlib
import logging
import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd

from src.data_utils import load_final_dataset, load_final_dense_embeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "gaussian_analysis_cache")
START_YEAR = 1930
END_YEAR = 2024


def _generate_cache_key(
    start_year: int,
    end_year: int,
    robust: bool,
    n_samples: int,
    random_seed: int,
    movie_ids: np.ndarray,
    debias: bool = False,
    debias_k: int = 3,
    debias_normalize: bool = False,
    whiten: bool = False,
    whiten_n_components: int = None,
    whiten_normalize: bool = True,
) -> str:
    """
    Generate a cache key based on parameters (same as in gaussian_fit.py).
    """
    movie_ids_str = "_".join(sorted(map(str, movie_ids)))
    movie_ids_hash = hashlib.md5(movie_ids_str.encode()).hexdigest()[:8]

    n_samples_str = "all" if n_samples is None else str(n_samples)
    robust_str = "robust" if robust else "standard"

    if whiten:
        norm_str = "norm" if whiten_normalize else "nonorm"
        comp_str = (
            f"ncomp{whiten_n_components}"
            if whiten_n_components is not None
            else "allcomp"
        )
        transform_str = f"whiten_{comp_str}_{norm_str}"
    elif debias:
        norm_str = "norm" if debias_normalize else "nonorm"
        transform_str = f"debias_k{debias_k}_{norm_str}"
    else:
        transform_str = "raw"

    cache_key = (
        f"gaussian_base_"
        f"y{start_year}_{end_year}_"
        f"n{n_samples_str}_"
        f"seed{random_seed}_"
        f"{robust_str}_"
        f"{transform_str}_"
        f"ids{movie_ids_hash}"
    )
    return cache_key


def _reconstruct_movie_ids(
    start_year: int,
    end_year: int,
    data_dir: str,
    csv_path: str,
    n_samples: int,
    random_seed: int,
    debias: bool = False,
    whiten: bool = False,
) -> np.ndarray:
    """
    Reconstruct the movie_ids array using the same filtering/sampling logic as main().

    This replicates the exact same process to get the movie_ids that correspond
    to the cached distances.
    """
    # Load all embeddings and corresponding movie IDs
    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    # Load movie metadata to filter by year
    movie_data = load_final_dataset(csv_path, verbose=False)

    # Filter by year range
    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()

    # Filter embeddings to only include movies in the filtered dataset
    valid_movie_ids = set(movie_data["movie_id"].values)
    valid_indices = np.array(
        [i for i, mid in enumerate(all_movie_ids) if mid in valid_movie_ids]
    )

    # Filter movie_ids to only include valid movies
    filtered_movie_ids = all_movie_ids[valid_indices]

    # Sample subset if n_samples is specified
    if n_samples is not None and n_samples < len(filtered_movie_ids):
        np.random.seed(random_seed)
        sample_indices = np.random.choice(
            len(filtered_movie_ids), size=n_samples, replace=False
        )
        filtered_movie_ids = filtered_movie_ids[sample_indices]

    return filtered_movie_ids


def extract_high_distance_movies(
    cache_key: str = None,
    distance_threshold: float = 38.0,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    cache_dir: str = CACHE_DIR,
    n_samples: int = None,
    random_seed: int = 42,
    robust: bool = True,
    debias: bool = False,
    debias_k: int = 3,
    debias_normalize: bool = False,
    whiten: bool = False,
    whiten_n_components: int = None,
    whiten_normalize: bool = True,
) -> pd.DataFrame:
    """
    Extract movies with Mahalanobis distance > threshold from cached results.

    Parameters:
    - cache_key: Specific cache key to load. If None, will generate from parameters.
    - distance_threshold: Threshold for Mahalanobis distance (default: 38.0)
    - start_year, end_year: Year filtering parameters
    - data_dir: Directory containing embeddings
    - csv_path: Path to final_dataset.csv
    - cache_dir: Directory containing cached results
    - n_samples: Number of samples used (must match cache)
    - random_seed: Random seed used (must match cache)
    - robust: Whether robust covariance was used (must match cache)
    - debias, debias_k, debias_normalize: Debias parameters (must match cache)
    - whiten, whiten_n_components, whiten_normalize: Whitening parameters (must match cache)

    Returns:
    - DataFrame with columns: movie_id (QID), title, distance, year
    """
    # Load cached results
    if cache_key is None:
        # Need to reconstruct movie_ids first to generate cache key
        logger.info("Reconstructing movie IDs to generate cache key...")
        temp_movie_ids = _reconstruct_movie_ids(
            start_year,
            end_year,
            data_dir,
            csv_path,
            n_samples,
            random_seed,
            debias,
            whiten,
        )
        cache_key = _generate_cache_key(
            start_year=start_year,
            end_year=end_year,
            robust=robust,
            n_samples=n_samples,
            random_seed=random_seed,
            movie_ids=temp_movie_ids,
            debias=debias,
            debias_k=debias_k,
            debias_normalize=debias_normalize,
            whiten=whiten,
            whiten_n_components=whiten_n_components,
            whiten_normalize=whiten_normalize,
        )

    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Available cache files: {os.listdir(cache_dir) if os.path.exists(cache_dir) else 'Cache directory does not exist'}"
        )

    logger.info(f"Loading cached results from {cache_path}")
    with open(cache_path, "rb") as f:
        cached_data = pickle.load(f)

    distances_squared = cached_data["distances_squared"]
    distances = np.sqrt(distances_squared)

    # Reconstruct movie_ids
    logger.info("Reconstructing movie IDs...")
    movie_ids = _reconstruct_movie_ids(
        start_year, end_year, data_dir, csv_path, n_samples, random_seed, debias, whiten
    )

    if len(movie_ids) != len(distances):
        raise ValueError(
            f"Number of movie_ids ({len(movie_ids)}) does not match "
            f"number of distances ({len(distances)}). "
            f"Parameters may not match the cache."
        )

    # Find movies with distance > threshold
    high_distance_mask = distances > distance_threshold
    high_distance_indices = np.where(high_distance_mask)[0]

    logger.info(
        f"Found {len(high_distance_indices)} movies with distance > {distance_threshold}"
    )

    if len(high_distance_indices) == 0:
        logger.warning("No movies found above threshold")
        return pd.DataFrame(columns=["movie_id", "title", "distance", "year"])

    # Get movie IDs and distances
    high_distance_qids = movie_ids[high_distance_indices]
    high_distances = distances[high_distance_indices]

    # Load movie data to get titles
    logger.info("Loading movie metadata...")
    movie_data = load_final_dataset(csv_path, verbose=False)

    # Filter by year
    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()

    # Create result DataFrame
    results = []
    for qid, distance in zip(high_distance_qids, high_distances):
        movie_row = movie_data[movie_data["movie_id"] == qid]
        if not movie_row.empty:
            title = movie_row.iloc[0].get("title", "Unknown")
            year = movie_row.iloc[0].get("year", None)
            results.append(
                {"movie_id": qid, "title": title, "distance": distance, "year": year}
            )
        else:
            results.append(
                {
                    "movie_id": qid,
                    "title": "Unknown",
                    "distance": distance,
                    "year": None,
                }
            )

    result_df = pd.DataFrame(results)

    # Sort by distance (descending)
    # Sort by distance (descending - highest distance first)
    result_df = result_df.sort_values("distance", ascending=False).reset_index(
        drop=True
    )

    return result_df


def main(
    distance_threshold: float = 38.0,
    cache_key: str = None,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    cache_dir: str = CACHE_DIR,
    n_samples: int = 5000,
    random_seed: int = 42,
    robust: bool = True,
    debias: bool = False,
    debias_k: int = 3,
    debias_normalize: bool = False,
    whiten: bool = True,
    whiten_n_components: int = None,
    whiten_normalize: bool = False,
    output_file: str = None,
):
    """
    Main function to extract and print high-distance movies.

    Parameters:
    - distance_threshold: Threshold for Mahalanobis distance
    - cache_key: Specific cache key to use. If None, will generate from other parameters.
    - output_file: Optional path to save results as CSV
    - Other parameters: Must match the parameters used when creating the cache
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Extracting movies with distance > {distance_threshold}")
    logger.info(f"{'=' * 60}")

    try:
        result_df = extract_high_distance_movies(
            cache_key=cache_key,
            distance_threshold=distance_threshold,
            start_year=start_year,
            end_year=end_year,
            data_dir=data_dir,
            csv_path=csv_path,
            cache_dir=cache_dir,
            n_samples=n_samples,
            random_seed=random_seed,
            robust=robust,
            debias=debias,
            debias_k=debias_k,
            debias_normalize=debias_normalize,
            whiten=whiten,
            whiten_n_components=whiten_n_components,
            whiten_normalize=whiten_normalize,
        )

        # Print results (already sorted by distance descending)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Movies with Mahalanobis distance > {distance_threshold}")
        logger.info("Sorted by distance (highest first)")
        logger.info(f"{'=' * 60}\n")

        if len(result_df) == 0:
            logger.info("No movies found above threshold.")
        else:
            # Verify sorting
            distances = result_df["distance"].values
            is_sorted = all(
                distances[i] >= distances[i + 1] for i in range(len(distances) - 1)
            )
            if not is_sorted:
                logger.warning(
                    "Warning: Results are not properly sorted! Re-sorting..."
                )
                result_df = result_df.sort_values(
                    "distance", ascending=False
                ).reset_index(drop=True)

            # Print all results
            for idx, row in result_df.iterrows():
                year_str = str(int(row["year"])) if pd.notna(row["year"]) else "N/A"
                logger.info(
                    f"{idx + 1:4d}. Distance: {row['distance']:6.2f} | "
                    f"QID: {row['movie_id']:12s} | "
                    f"Year: {year_str:>4s} | "
                    f"Title: {row['title']}"
                )

            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Total: {len(result_df)} movies (sorted by distance, highest first)"
            )
            logger.info(
                f"Distance range: {result_df['distance'].min():.2f} - {result_df['distance'].max():.2f}"
            )

        # Save to file if requested
        if output_file:
            result_df.to_csv(output_file, index=False)
            logger.info(f"\nResults saved to: {output_file}")

        return result_df

    except Exception as e:
        logger.error(f"Error extracting high-distance movies: {e}")
        raise


if __name__ == "__main__":
    # Example usage - adjust parameters to match your cached data
    # You can also specify cache_key directly if you know it

    # Option 1: Use specific cache key (recommended if you know it)
    # main(
    #     cache_key="gaussian_base_y1930_2024_n5000_seed42_robust_whiten_allcomp_nonorm_ids3d1c0339",
    #     distance_threshold=38.0,
    #     output_file="high_distance_movies.csv",
    # )

    # Option 2: Specify parameters to match cache
    main(
        distance_threshold=35.0,
        start_year=1930,
        end_year=2024,
        n_samples=5000,
        random_seed=42,
        robust=True,
        whiten=True,
        whiten_n_components=None,
        whiten_normalize=False,
        output_file="high_distance_movies.csv",
    )
