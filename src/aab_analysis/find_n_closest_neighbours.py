"""
Find n closest neighbors in the latent space for a given qid.

This script finds the n closest neighbors to a specified movie (by qid)
in the latent embedding space using cosine similarity.
"""

import logging
import os
import sys
from typing import Union

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from data_utils
# Path setup must occur before this import
from src.aab_analysis.math_functions import (
    find_n_closest_neighbours,  # type: ignore  # noqa: E402
)
from src.utils.data_utils import (  # type: ignore  # noqa: E402
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


def main(
    qid: Union[str, list[str]] = "Q1931001",
    n: int = 10,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    aggregation: str = "mean",
):
    """
    Main function to find and print n closest neighbors.

    Parameters:
    - qid: The movie_id (qid) or list of qids to find neighbors for.
           If a list is provided, the mean or median embedding will be used.
    - n: Number of closest neighbors to find
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - aggregation: Method to aggregate multiple QID embeddings ("mean" or "median")
    """
    # Normalize qid to always be a list for easier processing
    if isinstance(qid, str):
        qid_list = [qid]
        is_single = True
    else:
        qid_list = qid
        is_single = False

    if aggregation not in ["mean", "median"]:
        raise ValueError(f"aggregation must be 'mean' or 'median', got '{aggregation}'")

    logger.info(f"{'=' * 60}")
    if is_single:
        logger.info(f"Finding {n} closest neighbors for QID: {qid_list[0]}")
    else:
        logger.info(f"Finding {n} closest neighbors for QIDs: {qid_list}")
        logger.info(f"Using {aggregation} aggregation for multiple QIDs")
    logger.info(f"{'=' * 60}")

    try:
        # Load all embeddings and corresponding movie IDs
        logger.info("Loading embeddings...")
        all_embeddings, all_movie_ids = load_final_dense_embeddings(
            data_dir, verbose=False
        )

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

        # Find indices for all query qids in the filtered data
        query_indices_list = []
        query_embeddings_list = []
        found_qids = []

        for current_qid in qid_list:
            query_indices = np.where(filtered_movie_ids == current_qid)[0]
            if len(query_indices) == 0:
                logger.warning(
                    f"QID '{current_qid}' not found in embeddings for the specified year range"
                )
                continue

            if len(query_indices) > 1:
                logger.warning(
                    f"Multiple embeddings found for qid '{current_qid}', using first one"
                )

            query_idx = query_indices[0]
            query_indices_list.append(query_idx)
            query_embeddings_list.append(filtered_embeddings[query_idx])
            found_qids.append(current_qid)

        if len(query_embeddings_list) == 0:
            raise ValueError(
                f"None of the QIDs {qid_list} found in embeddings for the specified year range"
            )

        # Aggregate embeddings if multiple QIDs provided
        if len(query_embeddings_list) == 1:
            query_embedding = query_embeddings_list[0].reshape(1, -1)  # Keep 2D shape
            logger.info(
                f"Found query movie at index {query_indices_list[0]} in filtered data"
            )
        else:
            # Stack embeddings and compute mean or median
            stacked_embeddings = np.stack(query_embeddings_list, axis=0)
            if aggregation == "mean":
                aggregated_embedding = np.mean(stacked_embeddings, axis=0)
            else:  # median
                aggregated_embedding = np.median(stacked_embeddings, axis=0)
            query_embedding = aggregated_embedding.reshape(1, -1)  # Keep 2D shape
            logger.info(
                f"Found {len(query_embeddings_list)} query movies at indices {query_indices_list} in filtered data"
            )
            logger.info(
                f"Computed {aggregation} embedding from {len(query_embeddings_list)} movies"
            )

        # Get query movie title(s)
        if is_single:
            query_movie = movie_data[movie_data["movie_id"] == found_qids[0]]
            if not query_movie.empty:
                query_title = query_movie.iloc[0]["title"]
                logger.info(f"Query movie: {query_title} (QID: {found_qids[0]})")
            else:
                query_title = "Unknown"
                logger.warning(f"Could not find title for QID '{found_qids[0]}'")
        else:
            query_titles = []
            for qid in found_qids:
                query_movie = movie_data[movie_data["movie_id"] == qid]
                if not query_movie.empty:
                    query_titles.append(f"{query_movie.iloc[0]['title']} ({qid})")
                else:
                    query_titles.append(f"Unknown ({qid})")
            logger.info(f"Query movies: {', '.join(query_titles)}")

        # Call the function with loaded data
        # For single QID, exclude it from results. For multiple QIDs, we'll filter after
        anchor_idx_for_call = query_indices_list[0] if is_single else None
        results = find_n_closest_neighbours(
            embeddings_corpus=filtered_embeddings,
            anchor_embedding=query_embedding,
            movie_ids=filtered_movie_ids,
            movie_data=movie_data,
            n=n + len(found_qids),  # Get extra results in case we need to filter
            anchor_idx=anchor_idx_for_call,
        )

        # Filter out any query QIDs from results if we have multiple QIDs
        if not is_single:
            results = [r for r in results if r[0] not in found_qids][
                :n
            ]  # Take top n after filtering

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
    # Skyfall (QID: Q4941)
    main(
        qid=[
            "Q103474",
            "Q184843",
            "Q21500755",
            "Q162255",
            "Q170564",
            "Q16635326",
            "Q788822",
            "Q131191955",
            "Q221113",
            "Q83495",
            "Q189600",
            "Q207536",
            "Q200572",
            "Q504697",
            "Q626483",
            "Q18954",
            "Q244604",
            "Q1066948",
            "Q22575835",
            "Q10384115",
            "Q3549863",
            "Q30611788",
            "Q26751",
        ],  # Change this to your desired qid
        n=200,  # Change this to desired number of neighbors
    )
