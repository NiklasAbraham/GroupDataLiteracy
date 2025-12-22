"""
Number of neighbors distribution analysis.

This script analyzes how the number of movies in an epsilon ball grows as epsilon
increases, starting from an anchor. It compares the distance distributions of
the anchor vs the mean vector of the entire ensemble.
"""

import hashlib
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from data_utils
# Path setup must occur before this import
from src.analysis.math_functions import (  # type: ignore  # noqa: E402
    compute_anchor_embedding,
)
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


def format_movie_name_for_filename(title: str) -> str:
    """
    Format a movie title for use in filenames.

    Parameters:
    - title: Movie title string

    Returns:
    - Formatted string safe for filenames (spaces replaced with underscores,
      special characters removed, etc.)
    """
    # Replace spaces with underscores
    formatted = title.replace(" ", "_")
    # Remove or replace special characters that might cause issues in filenames
    formatted = re.sub(r'[<>:"/\\|?*]', "", formatted)
    # Remove multiple consecutive underscores
    formatted = re.sub(r"_+", "_", formatted)
    # Remove leading/trailing underscores
    formatted = formatted.strip("_")
    return formatted


def get_anchor_names_string(anchor_qids: list, movie_data: pd.DataFrame) -> str:
    """
    Get a formatted string of anchor movie names for use in filenames.

    Parameters:
    - anchor_qids: List of movie QIDs
    - movie_data: DataFrame with movie metadata

    Returns:
    - String with formatted movie names separated by '__'
    """
    anchor_names = []
    for qid in anchor_qids:
        anchor_movie = movie_data[movie_data["movie_id"] == qid]
        if not anchor_movie.empty:
            title = anchor_movie.iloc[0]["title"]
            formatted_name = format_movie_name_for_filename(title)
            anchor_names.append(formatted_name)
        else:
            # Fallback to QID if title not found
            anchor_names.append(qid)

    return "__".join(anchor_names)


def truncate_filename_component(component: str, max_length: int = 120) -> str:
    """
    Truncate a filename component if it's too long, adding a hash suffix for uniqueness.

    Parameters:
    - component: The string component to truncate
    - max_length: Maximum length for the component (default: 120)

    Returns:
    - Truncated string with hash suffix if needed
    """
    if len(component) <= max_length:
        return component

    # Create a hash of the full string for uniqueness
    hash_suffix = hashlib.md5(component.encode()).hexdigest()[:8]
    # Truncate to leave room for the hash separator
    truncated = component[: max_length - 9]  # -9 for "__" + 8 char hash
    return f"{truncated}__{hash_suffix}"


def compute_all_distances(
    anchor_embedding: np.ndarray,
    embeddings_corpus: np.ndarray,
):
    """
    Compute cosine distances from anchor embedding to all embeddings in corpus.

    Parameters:
    - anchor_embedding: The anchor embedding (shape: [1, embedding_dim])
    - embeddings_corpus: Array of embeddings to compute distances to
        (shape: [n_movies, embedding_dim])

    Returns:
    - Array of cosine distances (shape: [n_movies])
    """
    # Normalize embeddings for cosine similarity calculation
    anchor_norm = np.linalg.norm(anchor_embedding, axis=1, keepdims=True)
    if anchor_norm[0, 0] == 0:
        raise ValueError("Anchor embedding is zero vector")
    anchor_normalized = anchor_embedding / anchor_norm

    # Normalize corpus embeddings
    corpus_norms = np.linalg.norm(embeddings_corpus, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0] = 1  # Avoid division by zero
    corpus_normalized = embeddings_corpus / corpus_norms

    # Calculate cosine similarities
    similarities = cosine_similarity(anchor_normalized, corpus_normalized)[0]

    # Convert to distances (1 - similarity)
    distances = 1 - similarities

    return distances


def count_neighbors_at_epsilon(
    distances: np.ndarray,
    epsilon: float,
):
    """
    Count how many movies are within epsilon distance.

    Parameters:
    - distances: Array of cosine distances
    - epsilon: Distance threshold

    Returns:
    - Number of movies within epsilon distance
    """
    return np.sum(distances <= epsilon)


def plot_neighbor_distribution(
    anchor_distances: np.ndarray,
    mean_distances: np.ndarray,
    output_path: str = None,
    title: str = "Distance Distribution: Anchor vs Mean Vector",
    figsize: tuple = (14, 6),
):
    """
    Plot CDF and histogram comparing anchor and mean vector distance distributions.

    Parameters:
    - anchor_distances: Array of cosine distances from anchor to all movies
    - mean_distances: Array of cosine distances from mean vector to all movies
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title
    - figsize: Figure size tuple
    """
    # Sort distances for CDF calculation
    anchor_sorted = np.sort(anchor_distances)
    mean_sorted = np.sort(mean_distances)

    # Calculate CDFs
    anchor_cdf = np.arange(1, len(anchor_sorted) + 1) / len(anchor_sorted)
    mean_cdf = np.arange(1, len(mean_sorted) + 1) / len(mean_sorted)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: CDF comparison
    ax1.plot(
        anchor_sorted,
        anchor_cdf,
        label="Anchor",
        linewidth=2,
        color="blue",
    )
    ax1.plot(
        mean_sorted,
        mean_cdf,
        label="Mean Vector",
        linewidth=2,
        color="red",
        linestyle="--",
    )
    ax1.set_xlabel("Cosine Distance", fontsize=12)
    ax1.set_ylabel("Cumulative Probability", fontsize=12)
    ax1.set_title("Cumulative Distribution Function", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Histogram comparison with dual y-axes
    # Use shared bin edges for proper normalization and comparison
    all_distances_combined = np.concatenate([anchor_distances, mean_distances])
    bins = np.linspace(
        all_distances_combined.min(),
        all_distances_combined.max(),
        50,
    )

    # Normalize both datasets so their highest peak is 1
    anchor_hist, _ = np.histogram(anchor_distances, bins=bins)
    mean_hist, _ = np.histogram(mean_distances, bins=bins)

    anchor_max = anchor_hist.max()
    mean_max = mean_hist.max()

    anchor_normalized = anchor_hist / anchor_max if anchor_max > 0 else anchor_hist
    mean_normalized = mean_hist / mean_max if mean_max > 0 else mean_hist

    # Plot anchor on left y-axis
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax2.bar(
        bin_centers,
        anchor_normalized,
        width=(bins[1] - bins[0]),
        alpha=0.6,
        label="Anchor (normalized)",
        color="blue",
        align="center",
    )
    ax2.set_xlabel("Cosine Distance", fontsize=12)
    ax2.set_ylabel("Normalized Count (Anchor)", fontsize=12, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_title(
        "Distance Distribution Histograms (Normalized)", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Create second y-axis for mean vector
    ax2_twin = ax2.twinx()
    ax2_twin.bar(
        bin_centers,
        mean_normalized,
        width=(bins[1] - bins[0]),
        alpha=0.6,
        label="Mean Vector (normalized)",
        color="red",
        align="center",
    )
    ax2_twin.set_ylabel("Normalized Count (Mean Vector)", fontsize=12, color="red")
    ax2_twin.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_neighbors_vs_epsilon(
    anchor_distances: np.ndarray,
    mean_distances: np.ndarray,
    epsilon_range: np.ndarray,
    output_path: str = None,
    title: str = "Number of Neighbors vs Epsilon",
    figsize: tuple = (10, 6),
):
    """
    Plot how the number of neighbors grows as epsilon increases.

    Parameters:
    - anchor_distances: Array of cosine distances from anchor to all movies
    - mean_distances: Array of cosine distances from mean vector to all movies
    - epsilon_range: Array of epsilon values to test
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title
    - figsize: Figure size tuple
    """
    # Count neighbors for each epsilon
    anchor_counts = [
        count_neighbors_at_epsilon(anchor_distances, eps) for eps in epsilon_range
    ]
    mean_counts = [
        count_neighbors_at_epsilon(mean_distances, eps) for eps in epsilon_range
    ]

    plt.figure(figsize=figsize)
    plt.plot(
        epsilon_range,
        anchor_counts,
        label="Anchor",
        linewidth=2,
        color="blue",
    )
    plt.plot(
        epsilon_range,
        mean_counts,
        label="Mean Vector",
        linewidth=2,
        color="red",
        linestyle="--",
    )
    plt.xlabel("Epsilon (Cosine Distance)", fontsize=12)
    plt.ylabel("Number of Movies in Epsilon Ball", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main(
    anchor_qids: list = None,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    anchor_method: str = "average",
    exclude_anchors: bool = True,
    epsilon_max: float = 2.0,
    epsilon_steps: int = 200,
    output_dir: str = None,
):
    """
    Main function to run neighbor distribution analysis.

    Parameters:
    - anchor_qids: List of movie QIDs to use as anchors
    - start_year: First year to filter movies (default: 1930)
    - end_year: Last year to filter movies (default: 2024)
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - anchor_method: Method to combine anchor embeddings ("average" or "medoid")
    - exclude_anchors: Whether to exclude anchor movies from distance computation
    - epsilon_max: Maximum epsilon value to test (default: 2.0)
    - epsilon_steps: Number of epsilon values to test (default: 200)
    - output_dir: Directory to save plots (if None, uses current directory)
    """
    if anchor_qids is None:
        # Example: James Bond movie
        anchor_qids = ["Q4941"]  # Dr. No (first James Bond film)
        logger.info("Using default anchor: James Bond (Q4941)")

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

    # Compute anchor embedding
    logger.info("Computing anchor embedding...")
    anchor_embedding = compute_anchor_embedding(
        anchor_qids=anchor_qids,
        embeddings_corpus=filtered_embeddings,
        movie_ids=filtered_movie_ids,
        method=anchor_method,
    )

    # Get anchor movie titles for logging
    anchor_titles = []
    for qid in anchor_qids:
        anchor_movie = movie_data[movie_data["movie_id"] == qid]
        if not anchor_movie.empty:
            anchor_titles.append(anchor_movie.iloc[0]["title"])
        else:
            anchor_titles.append("Unknown")

    logger.info(f"Anchor movies: {list(zip(anchor_qids, anchor_titles))}")

    # Compute mean embedding of entire ensemble
    logger.info("Computing mean embedding of entire ensemble...")
    mean_embedding = np.mean(filtered_embeddings, axis=0, keepdims=True)
    logger.info(f"Mean embedding computed from {len(filtered_embeddings)} movies")

    # Compute all distances from anchor
    logger.info("Computing distances from anchor to all movies...")
    anchor_distances = compute_all_distances(anchor_embedding, filtered_embeddings)

    # Exclude anchor movies if specified
    if exclude_anchors:
        anchor_mask = np.array([mid in anchor_qids for mid in filtered_movie_ids])
        anchor_distances[anchor_mask] = np.inf  # Mark as excluded

    # Compute all distances from mean vector
    logger.info("Computing distances from mean vector to all movies...")
    mean_distances = compute_all_distances(mean_embedding, filtered_embeddings)

    logger.info(
        f"Anchor distance range: {np.min(anchor_distances[anchor_distances != np.inf]):.6f} - {np.max(anchor_distances[anchor_distances != np.inf]):.6f}"
    )
    logger.info(
        f"Mean distance range: {np.min(mean_distances):.6f} - {np.max(mean_distances):.6f}"
    )

    # Create epsilon range
    epsilon_range = np.linspace(0, epsilon_max, epsilon_steps)

    # Create plots
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
        anchor_names_str = truncate_filename_component(anchor_names_str)

        # Plot 1: CDF and Histogram comparison
        dist_plot_path = os.path.join(
            output_dir, f"n_neighbours_dist_{anchor_names_str}.png"
        )
        plot_neighbor_distribution(
            anchor_distances[anchor_distances != np.inf],
            mean_distances,
            output_path=dist_plot_path,
            title="Distance Distribution: Anchor vs Mean Vector",
        )

        # Plot 2: Number of neighbors vs epsilon
        neighbors_plot_path = os.path.join(
            output_dir, f"n_neighbours_vs_epsilon_{anchor_names_str}.png"
        )
        plot_neighbors_vs_epsilon(
            anchor_distances[anchor_distances != np.inf],
            mean_distances,
            epsilon_range,
            output_path=neighbors_plot_path,
            title="Number of Neighbors vs Epsilon",
        )
    else:
        plot_neighbor_distribution(
            anchor_distances[anchor_distances != np.inf],
            mean_distances,
            title="Distance Distribution: Anchor vs Mean Vector",
        )
        plot_neighbors_vs_epsilon(
            anchor_distances[anchor_distances != np.inf],
            mean_distances,
            epsilon_range,
            title="Number of Neighbors vs Epsilon",
        )

    return {
        "anchor_distances": anchor_distances[anchor_distances != np.inf],
        "mean_distances": mean_distances,
        "epsilon_range": epsilon_range,
    }


if __name__ == "__main__":
    results = main(
        anchor_qids=["Q4941"],  # Dr. No (first James Bond film)
        start_year=1930,
        end_year=2024,
        anchor_method="average",
        exclude_anchors=True,
        epsilon_max=2.0,
        epsilon_steps=200,
        output_dir=f"{BASE_DIR}/figures/n_neighbours_dist",
    )
