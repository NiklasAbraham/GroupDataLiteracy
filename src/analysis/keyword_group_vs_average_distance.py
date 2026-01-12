"""
Calculate average cosine distance for keyword groups vs random movies.

This script computes the average cosine distance within keyword groups
and compares it to the average cosine distance between keyword groups
and random movies in the latent embedding space.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import functions from data_utils
# Import cosine distance functions from math module
from src.analysis.math_functions.cosine_distance_util import (
    calculate_average_cosine_distance,
    calculate_average_cosine_distance_between_groups,
)
from src.data_utils import (
    load_final_dataset,
    load_final_dense_embeddings,
    search_movies_by_keywords,
)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024

# Set random seed for reproducibility
np.random.seed(42)

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Year range: {START_YEAR} to {END_YEAR}")


def main(
    use_qids=False,
    qids_list=None,
    keywords_list=None,
    exclude_qids=None,
    n_random_movies=5000,
):
    """
    Calculate average cosine distance for keyword groups or QID groups vs random movies.

    Parameters:
    - use_qids: If True, use QIDs directly from qids_list. If False, use keywords from keywords_list.
    - qids_list: List of lists of QIDs to analyze (used when use_qids=True).
                 Each inner list represents a group of QIDs.
    - keywords_list: List of lists of keywords to analyze (used when use_qids=False).
                     Each inner list represents a keyword group.
    - exclude_qids: List of lists of QIDs to exclude (one list per group).
                    Only used when use_qids=False.
    - n_random_movies: Number of random movies to compare against. Default is 5000.
    """
    # Load all embeddings and corresponding movie IDs
    logger.info("Loading embeddings...")
    all_embeddings, all_movie_ids = load_final_dense_embeddings(DATA_DIR, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {DATA_DIR}")

    logger.info(f"Total movies with embeddings: {len(all_movie_ids)}")
    logger.info(f"Embedding shape: {all_embeddings.shape}")

    # Load movie metadata from consolidated CSV
    logger.info("Loading movie metadata...")
    movie_data = load_final_dataset(CSV_PATH, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {CSV_PATH}")

    logger.info(f"Loaded {len(movie_data)} movies from metadata file")

    # Filter by year range if year column exists
    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= START_YEAR) & (movie_data["year"] <= END_YEAR)
        ].copy()
        logger.info(
            f"Filtered to {len(movie_data)} movies between {START_YEAR} and {END_YEAR}"
        )

    # Filter embeddings to only include movies in the filtered metadata
    logger.info("Filtering embeddings to match metadata...")
    movie_ids_set = set(movie_data["movie_id"].values)
    mask = np.array([mid in movie_ids_set for mid in all_movie_ids])
    all_embeddings = all_embeddings[mask]
    all_movie_ids = all_movie_ids[mask]

    logger.info(
        f"After filtering: {len(all_movie_ids)} movies with both embeddings and metadata"
    )
    logger.info(f"Embedding shape: {all_embeddings.shape}")

    # Create mapping from movie_id to index in all_movie_ids
    movie_id_to_index = {mid: idx for idx, mid in enumerate(all_movie_ids)}

    # Set default values if not provided
    if use_qids:
        if qids_list is None:
            raise ValueError("qids_list must be provided when use_qids=True")
        groups_list = qids_list
        exclude_qids = (
            exclude_qids if exclude_qids is not None else [[]] * len(qids_list)
        )
    else:
        if keywords_list is None:
            # Default keywords if not provided
            keywords_list = [
                ["Star Wars"],
                ["Furious"],
                ["Harry Potter"],
                ["Jurassic Park"],
                ["Iron Man"],
            ]
        groups_list = keywords_list
        if exclude_qids is None:
            # Default exclude_qids if not provided
            exclude_qids = [
                [
                    "Q1931001",
                    "Q101863729",
                    "Q7428298",
                    "Q7601132",
                    "Q108730566",
                    "Q1538082",
                ],
                ["Q5509553", "Q1891656"],
                [],
                [],
                ["Q1758603", "Qi3820040"],
            ]

    # Ensure exclude_qids has the same length as groups_list
    if len(exclude_qids) < len(groups_list):
        exclude_qids.extend([[]] * (len(groups_list) - len(exclude_qids)))

    # Store results for plotting
    results = {"keywords": [], "within_group_distance": [], "vs_random_distance": []}

    # Process each group
    for idx, group in enumerate(groups_list):
        if use_qids:
            # When using QIDs directly
            group_label = (
                f"QIDs: {', '.join(group[:3])}{'...' if len(group) > 3 else ''}"
            )
            logger.info(f"{'=' * 60}")
            logger.info(f"Processing QID group: {group_label}")
            logger.info(f"{'=' * 60}")

            # Use QIDs directly
            matching_qids = list(group)

            # Apply exclusion filter if any
            keyword_exclude_qids = exclude_qids[idx] if idx < len(exclude_qids) else []
            if keyword_exclude_qids:
                initial_count = len(matching_qids)
                matching_qids = [
                    qid for qid in matching_qids if qid not in keyword_exclude_qids
                ]
                logger.info(
                    f"After excluding {len(keyword_exclude_qids)} QIDs, {len(matching_qids)} movies remain (from {initial_count})"
                )

            logger.info(f"Using {len(matching_qids)} QIDs: {group}")
        else:
            # When using keywords
            keyword_str = " ".join(group)
            logger.info(f"{'=' * 60}")
            logger.info(f"Processing keyword: {keyword_str}")
            logger.info(f"{'=' * 60}")

            # Search for matching movies
            matching_qids = search_movies_by_keywords(
                movie_data,
                keywords=group,
                search_columns=["title"],
                case_sensitive=False,
            )

            # Apply exclusion filter for this specific keyword
            keyword_exclude_qids = exclude_qids[idx] if idx < len(exclude_qids) else []
            if keyword_exclude_qids:
                initial_count = len(matching_qids)
                matching_qids = [
                    qid for qid in matching_qids if qid not in keyword_exclude_qids
                ]
                logger.info(
                    f"After excluding {len(keyword_exclude_qids)} QIDs, {len(matching_qids)} movies remain (from {initial_count})"
                )

            logger.info(f"Found {len(matching_qids)} movies matching keywords: {group}")

        # Print list of movie titles and QIDs
        if len(matching_qids) > 0:
            if use_qids:
                logger.info("\nMovies for QID group:")
            else:
                logger.info(f"\nMovies found for '{keyword_str}':")
            matching_movies_info = movie_data[
                movie_data["movie_id"].isin(matching_qids)
            ][["movie_id", "title"]].copy()
            matching_movies_info = matching_movies_info.sort_values("title")
            for _, row in matching_movies_info.iterrows():
                logger.info(f"  QID: {row['movie_id']}, Title: {row['title']}")
            logger.info("")  # Empty line for readability

        if len(matching_qids) == 0:
            if use_qids:
                logger.warning(f"No movies found for QIDs {group}, skipping...")
            else:
                logger.warning(f"No movies found for keywords {group}, skipping...")
            continue

        # Get embeddings for matching movies
        matching_embeddings = []
        matching_movie_ids_valid = []
        for qid in matching_qids:
            if qid in movie_id_to_index:
                embedding_idx = movie_id_to_index[qid]
                matching_embeddings.append(all_embeddings[embedding_idx])
                matching_movie_ids_valid.append(qid)

        if len(matching_embeddings) < 2:
            logger.warning(
                f"Only {len(matching_embeddings)} movies with embeddings found, need at least 2 for distance calculation"
            )
            continue

        matching_embeddings = np.array(matching_embeddings)
        logger.info(
            f"Using {len(matching_embeddings)} movies with embeddings for analysis"
        )

        # Calculate average distance within keyword group
        within_distance = calculate_average_cosine_distance(matching_embeddings)
        logger.info(
            f"Average cosine distance within keyword group: {within_distance:.4f}"
        )

        # Sample random movies (excluding the keyword group movies)
        all_other_indices = [
            idx
            for idx, mid in enumerate(all_movie_ids)
            if mid not in matching_movie_ids_valid
        ]

        if len(all_other_indices) < n_random_movies:
            n_random = len(all_other_indices)
            logger.warning(f"Only {n_random} other movies available, using all of them")
        else:
            n_random = n_random_movies

        random_indices = np.random.choice(
            all_other_indices, size=n_random, replace=False
        )
        random_embeddings = all_embeddings[random_indices]

        # Calculate average distance between group and random movies
        vs_random_distance = calculate_average_cosine_distance_between_groups(
            matching_embeddings, random_embeddings
        )
        logger.info(
            f"Average cosine distance vs {n_random} random movies: {vs_random_distance:.4f}"
        )

        # Store results
        if use_qids:
            # Create label from QIDs (show first few)
            group_label = (
                f"QIDs: {', '.join(group[:3])}{'...' if len(group) > 3 else ''}"
            )
            results["keywords"].append(group_label)
        else:
            keyword_str = " ".join(group)
            results["keywords"].append(keyword_str)
        results["within_group_distance"].append(within_distance)
        results["vs_random_distance"].append(vs_random_distance)

    # Calculate global average distance
    logger.info(f"{'=' * 60}")
    logger.info("Calculating global average distance...")
    logger.info(f"{'=' * 60}")

    # Sample pairs from all embeddings to calculate global average
    n_global_samples = min(5000, len(all_embeddings) * (len(all_embeddings) - 1) // 2)
    logger.info(
        f"Sampling {n_global_samples} random pairs for global average calculation..."
    )

    # Sample random pairs
    np.random.seed(42)  # For reproducibility
    if len(all_embeddings) <= 100:
        # For small datasets, use all pairs
        from itertools import combinations

        all_pairs = list(combinations(range(len(all_embeddings)), 2))
        if len(all_pairs) > n_global_samples:
            pair_indices = np.random.choice(
                len(all_pairs), size=n_global_samples, replace=False
            )
            selected_pairs = [all_pairs[i] for i in pair_indices]
        else:
            selected_pairs = all_pairs
    else:
        # For large datasets, sample random pairs
        selected_pairs = []
        for _ in range(n_global_samples):
            idx1, idx2 = np.random.choice(len(all_embeddings), size=2, replace=False)
            selected_pairs.append((idx1, idx2))

    # Calculate distances for sampled pairs
    global_distances = []
    for idx1, idx2 in selected_pairs:
        dist = calculate_average_cosine_distance_between_groups(
            all_embeddings[idx1 : idx1 + 1], all_embeddings[idx2 : idx2 + 1]
        )
        global_distances.append(dist)

    global_avg_distance = np.mean(global_distances)
    logger.info(f"Global average cosine distance: {global_avg_distance:.4f}")

    # Create bar plot
    logger.info(f"{'=' * 60}")
    logger.info("Creating bar plot...")
    logger.info(f"{'=' * 60}")

    n_keywords = len(results["keywords"])
    if n_keywords == 0:
        logger.warning("No results to plot!")
        return

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up bar positions
    x = np.arange(n_keywords)
    width = 0.35  # Width of bars

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        results["within_group_distance"],
        width,
        label="Within Group",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        results["vs_random_distance"],
        width,
        label="Vs Random Movies",
        color="coral",
        alpha=0.8,
    )

    # Add horizontal line for global average
    ax.axhline(
        y=global_avg_distance,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Global Average ({global_avg_distance:.3f})",
        alpha=0.8,
    )

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Customize plot
    if use_qids:
        xlabel = "QID Groups"
        title = "Average Cosine Distance: Within QID Groups vs Random Movies\n(Latent Embedding Space)"
        filename = "qid_group_vs_average_distance.png"
    else:
        xlabel = "Keyword Groups"
        title = "Average Cosine Distance: Within Keyword Groups vs Random Movies\n(Latent Embedding Space)"
        filename = "keyword_group_vs_average_distance.png"

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Cosine Distance", fontsize=12, fontweight="bold")
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(results["keywords"], fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(BASE_DIR, "src", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to: {output_path}")

    # Also show plot
    plt.show()

    # Print summary
    logger.info(f"{'=' * 60}")
    logger.info("Summary of Results:")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nGlobal average distance: {global_avg_distance:.4f}")
    for i, keyword in enumerate(results["keywords"]):
        logger.info(f"\n{keyword}:")
        logger.info(
            f"  Within group distance: {results['within_group_distance'][i]:.4f}"
        )
        logger.info(f"  Vs random distance:    {results['vs_random_distance'][i]:.4f}")
        logger.info(f"  Global average:        {global_avg_distance:.4f}")
        diff = results["vs_random_distance"][i] - results["within_group_distance"][i]
        logger.info(
            f"  Difference:            {diff:.4f} ({diff / results['within_group_distance'][i] * 100:.1f}% higher)"
        )


if __name__ == "__main__":
    # Use QIDs directly
    qids_list = [
        [
            "Q18602670",
            "Q212145",
            "Q207916",
            "Q151904",
            "Q591272",
            "Q19089",
            "Q181540",
            "Q107914",
            "Q320423",
            "Q332368",
            "Q21534241",
            "Q30931",
            "Q106440",
        ]
    ]
    main(use_qids=True, qids_list=qids_list)

    # To use keywords instead, comment the above and uncomment below:
    # main()
