"""
Find n closest neighbors in the latent space for a given movie ID.

This script finds the n closest neighbors to a specified movie (by qid)
in the latent embedding space using cosine similarity.
"""

import logging
import os
import sys
from typing import Union

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import importlib.util  # noqa: E402
neighbor_utils_path = os.path.join(os.path.dirname(__file__), "002_neighbor_utils.py")
spec = importlib.util.spec_from_file_location("neighbor_utils", neighbor_utils_path)
neighbor_utils = importlib.util.module_from_spec(spec)
sys.modules["neighbor_utils"] = neighbor_utils
spec.loader.exec_module(neighbor_utils)
find_n_closest_neighbours = neighbor_utils.find_n_closest_neighbours
from src.utils.data_utils import load_final_dataset, load_final_dense_embeddings  # noqa: E402

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
    Find and print n closest neighbors.

    Parameters:
    - qid: The movie_id (qid) or list of qids to find neighbors for. If a list is provided, the mean or median embedding will be used.
    - n: Number of closest neighbors to find
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - aggregation: Method to aggregate multiple QID embeddings ("mean" or "median")
    """
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

    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")

    logger.info(f"Total movies with embeddings: {len(all_movie_ids)}")

    movie_data = load_final_dataset(csv_path, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {csv_path}")

    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()
        logger.info(
            f"Filtered to {len(movie_data)} movies between {start_year} and {end_year}"
        )

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

    logger.info(
        f"Filtered to {len(filtered_movie_ids)} movies with embeddings in year range"
    )

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

    if len(query_embeddings_list) == 1:
        query_embedding = query_embeddings_list[0].reshape(1, -1)
        logger.info(
            f"Found query movie at index {query_indices_list[0]} in filtered data"
        )
    else:
        stacked_embeddings = np.stack(query_embeddings_list, axis=0)
        if aggregation == "mean":
            aggregated_embedding = np.mean(stacked_embeddings, axis=0)
        else:
            aggregated_embedding = np.median(stacked_embeddings, axis=0)
        query_embedding = aggregated_embedding.reshape(1, -1)
        logger.info(
            f"Found {len(query_embeddings_list)} query movies at indices {query_indices_list} in filtered data"
        )
        logger.info(
            f"Computed {aggregation} embedding from {len(query_embeddings_list)} movies"
        )

    if is_single:
        query_movie = movie_data[movie_data["movie_id"] == found_qids[0]]
        if not query_movie.empty:
            query_title = query_movie.iloc[0]["title"]
            logger.info(f"Query movie: {query_title} (QID: {found_qids[0]})")
        else:
            logger.warning(f"Could not find title for QID '{found_qids[0]}'")

    anchor_idx_for_call = query_indices_list[0] if is_single else None
    results = find_n_closest_neighbours(
        embeddings_corpus=filtered_embeddings,
        anchor_embedding=query_embedding,
        movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        n=n + len(found_qids),
        anchor_idx=anchor_idx_for_call,
    )

    if not is_single:
        results = [r for r in results if r[0] not in found_qids][:n]

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

    return results


if __name__ == "__main__":
    main(
        qid="Q1931001",
        n=10,
    )
