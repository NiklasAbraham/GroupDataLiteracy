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
sys.path.insert(0, BASE_DIR)

from src.aab_analysis.math_functions.epsilon_ball import compute_anchor_embedding  # noqa: E402
from src.utils.data_utils import load_final_dataset, load_final_dense_embeddings  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024


def format_movie_name_for_filename(title: str) -> str:
    """Format a movie title for use in filenames."""
    formatted = title.replace(" ", "_")
    formatted = re.sub(r'[<>:"/\\|?*]', "", formatted)
    formatted = re.sub(r"_+", "_", formatted)
    formatted = formatted.strip("_")
    return formatted


def get_anchor_names_string(anchor_qids: list, movie_data: pd.DataFrame) -> str:
    """Get a formatted string of anchor movie names for use in filenames."""
    anchor_names = []
    for qid in anchor_qids:
        anchor_movie = movie_data[movie_data["movie_id"] == qid]
        if not anchor_movie.empty:
            title = anchor_movie.iloc[0]["title"]
            formatted_name = format_movie_name_for_filename(title)
            anchor_names.append(formatted_name)
        else:
            anchor_names.append(qid)

    return "__".join(anchor_names)


def truncate_filename_component(component: str, max_length: int = 120) -> str:
    """Truncate a filename component if it's too long, adding a hash suffix for uniqueness."""
    if len(component) <= max_length:
        return component

    hash_suffix = hashlib.md5(component.encode()).hexdigest()[:8]
    truncated = component[: max_length - 9]
    return f"{truncated}__{hash_suffix}"


def compute_all_distances(
    anchor_embedding: np.ndarray,
    embeddings_corpus: np.ndarray,
):
    """Compute cosine distances from anchor embedding to all embeddings in corpus."""
    anchor_norm = np.linalg.norm(anchor_embedding, axis=1, keepdims=True)
    if anchor_norm[0, 0] == 0:
        raise ValueError("Anchor embedding is zero vector")
    anchor_normalized = anchor_embedding / anchor_norm

    corpus_norms = np.linalg.norm(embeddings_corpus, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0] = 1
    corpus_normalized = embeddings_corpus / corpus_norms

    similarities = cosine_similarity(anchor_normalized, corpus_normalized)[0]
    distances = 1 - similarities

    return distances


def count_neighbors_at_epsilon(
    distances: np.ndarray,
    epsilon: float,
):
    """Count how many movies are within epsilon distance."""
    return np.sum(distances <= epsilon)


def plot_neighbor_distribution(
    anchor_distances: np.ndarray,
    mean_distances: np.ndarray,
    output_path: str = None,
    title: str = "Distance Distribution: Anchor vs Mean Vector",
    figsize: tuple = (14, 6),
):
    """Plot CDF and histogram comparing anchor and mean vector distance distributions."""
    anchor_sorted = np.sort(anchor_distances)
    mean_sorted = np.sort(mean_distances)

    anchor_cdf = np.arange(1, len(anchor_sorted) + 1) / len(anchor_sorted)
    mean_cdf = np.arange(1, len(mean_sorted) + 1) / len(mean_sorted)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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

    all_distances_combined = np.concatenate([anchor_distances, mean_distances])
    bins = np.linspace(
        all_distances_combined.min(),
        all_distances_combined.max(),
        50,
    )

    anchor_hist, _ = np.histogram(anchor_distances, bins=bins)
    mean_hist, _ = np.histogram(mean_distances, bins=bins)

    anchor_max = anchor_hist.max()
    mean_max = mean_hist.max()

    anchor_normalized = anchor_hist / anchor_max if anchor_max > 0 else anchor_hist
    mean_normalized = mean_hist / mean_max if mean_max > 0 else mean_hist

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
    """Plot how the number of neighbors grows as epsilon increases."""
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


def find_most_average_movies(
    mean_distances: np.ndarray,
    movie_ids: np.ndarray,
    movie_data: pd.DataFrame,
    n: int = 10,
):
    """Find the n movies closest to the mean embedding (most average movies)."""
    top_n_indices = np.argsort(mean_distances)[:n]

    most_average = []
    for idx in top_n_indices:
        movie_id = movie_ids[idx]
        distance = mean_distances[idx]
        movie_info = movie_data[movie_data["movie_id"] == movie_id]

        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            year = movie_info.iloc[0].get("year", None)
        else:
            title = "Unknown"
            year = None

        most_average.append(
            {
                "movie_id": movie_id,
                "title": title,
                "year": year,
                "distance": distance,
                "rank": len(most_average) + 1,
            }
        )

    return most_average


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
    Run neighbor distribution analysis.

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
        anchor_qids = ["Q4941"]
        logger.info("Using default anchor: James Bond (Q4941)")

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

    logger.info("Computing anchor embedding...")
    anchor_embedding = compute_anchor_embedding(
        anchor_qids=anchor_qids,
        embeddings_corpus=filtered_embeddings,
        movie_ids=filtered_movie_ids,
        method=anchor_method,
    )

    anchor_titles = []
    for qid in anchor_qids:
        anchor_movie = movie_data[movie_data["movie_id"] == qid]
        if not anchor_movie.empty:
            anchor_titles.append(anchor_movie.iloc[0]["title"])
        else:
            anchor_titles.append("Unknown")

    logger.info(f"Anchor movies: {list(zip(anchor_qids, anchor_titles))}")

    logger.info("Computing mean embedding of entire ensemble...")
    mean_embedding = np.mean(filtered_embeddings, axis=0, keepdims=True)
    logger.info(f"Mean embedding computed from {len(filtered_embeddings)} movies")

    logger.info("Computing distances from anchor to all movies...")
    anchor_distances = compute_all_distances(anchor_embedding, filtered_embeddings)

    if exclude_anchors:
        anchor_mask = np.array([mid in anchor_qids for mid in filtered_movie_ids])
        anchor_distances[anchor_mask] = np.inf

    logger.info("Computing distances from mean vector to all movies...")
    mean_distances = compute_all_distances(mean_embedding, filtered_embeddings)

    logger.info(
        f"Anchor distance range: {np.min(anchor_distances[anchor_distances != np.inf]):.6f} - {np.max(anchor_distances[anchor_distances != np.inf]):.6f}"
    )
    logger.info(
        f"Mean distance range: {np.min(mean_distances):.6f} - {np.max(mean_distances):.6f}"
    )

    logger.info("Finding most average movies (closest to mean vector)...")
    most_average_movies = find_most_average_movies(
        mean_distances=mean_distances,
        movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        n=10,
    )

    logger.info("\n" + "=" * 60)
    logger.info("10 Most Average Movies (Closest to Mean Vector):")
    logger.info("=" * 60)
    for movie in most_average_movies:
        logger.info(
            f"Rank {movie['rank']}: {movie['title']} ({movie['movie_id']}) - "
            f"Distance: {movie['distance']:.6f}, Year: {movie.get('year', 'N/A')}"
        )
    logger.info("=" * 60 + "\n")

    epsilon_range = np.linspace(0, epsilon_max, epsilon_steps)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
        anchor_names_str = truncate_filename_component(anchor_names_str)

        dist_plot_path = os.path.join(
            output_dir, f"n_neighbours_dist_{anchor_names_str}.png"
        )
        plot_neighbor_distribution(
            anchor_distances[anchor_distances != np.inf],
            mean_distances,
            output_path=dist_plot_path,
            title="Distance Distribution: Anchor vs Mean Vector",
        )

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
        "most_average_movies": most_average_movies,
    }


if __name__ == "__main__":
    main(
        anchor_qids=["Q4941"],
        start_year=1930,
        end_year=2024,
        anchor_method="average",
        exclude_anchors=True,
        epsilon_max=2.0,
        epsilon_steps=200,
        output_dir=f"{BASE_DIR}/figures/n_neighbours_dist",
    )
