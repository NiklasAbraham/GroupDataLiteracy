"""
HDBSCAN Clustering Analysis for Movie Embeddings.

This script performs HDBSCAN clustering on movie embeddings and analyzes:
1. Intra-cluster cosine variance and measurements
2. Inter-cluster cosine variance and measurements
3. Comparison with existing genre classifications
"""

import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from src.utils.data_utils import (
    load_final_dataset,
    load_final_dense_embeddings,
    preprocess_genres,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.

    Parameters:
    - embeddings: Array of shape (n_samples, embedding_dim)

    Returns:
    - Normalized embeddings of same shape
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def mean_center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Mean-center embeddings by subtracting the global mean.

    Parameters:
    - embeddings: Array of shape (n_samples, embedding_dim)

    Returns:
    - Mean-centered embeddings of same shape
    """
    mean = np.mean(embeddings, axis=0, keepdims=True)
    return embeddings - mean


def whiten_embeddings(embeddings: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Whiten embeddings using PCA to restore isotropy.

    Whitening removes correlations and scales variance to 1 in all directions.
    This helps with clustering when data has anisotropic structure.

    Parameters:
    - embeddings: Array of shape (n_samples, embedding_dim)
    - n_components: Number of PCA components to keep. If None, keeps all.

    Returns:
    - Whitened embeddings
    """
    # First mean-center
    centered = mean_center_embeddings(embeddings)

    # Apply PCA
    if n_components is None:
        n_components = min(embeddings.shape[0], embeddings.shape[1])

    pca = PCA(n_components=n_components, whiten=True)
    whitened = pca.fit_transform(centered)

    return whitened


def compute_intra_cluster_cosine_stats(
    embeddings: np.ndarray, cluster_labels: np.ndarray
) -> dict:
    """
    Compute intra-cluster cosine similarity statistics.

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - cluster_labels: Cluster assignments (n_samples,), -1 for noise

    Returns:
    - Dictionary with mean, variance, std of intra-cluster cosines
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

    intra_cosines = []
    cluster_stats = {}

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = normalized_embeddings[cluster_mask]

        if len(cluster_embeddings) < 2:
            continue

        # Compute pairwise cosine similarities within cluster
        cluster_cosine_matrix = cosine_similarity(cluster_embeddings)
        upper_triangle = np.triu_indices(len(cluster_embeddings), k=1)
        cluster_cosines = cluster_cosine_matrix[upper_triangle]

        intra_cosines.extend(cluster_cosines)

        cluster_stats[cluster_id] = {
            "mean": np.mean(cluster_cosines),
            "variance": np.var(cluster_cosines),
            "std": np.std(cluster_cosines),
            "size": len(cluster_embeddings),
        }

    if len(intra_cosines) == 0:
        return {
            "overall_mean": 0.0,
            "overall_variance": 0.0,
            "overall_std": 0.0,
            "cluster_stats": {},
        }

    return {
        "overall_mean": np.mean(intra_cosines),
        "overall_variance": np.var(intra_cosines),
        "overall_std": np.std(intra_cosines),
        "cluster_stats": cluster_stats,
    }


def compute_inter_cluster_cosine_stats(
    embeddings: np.ndarray, cluster_labels: np.ndarray
) -> dict:
    """
    Compute inter-cluster cosine similarity statistics.

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - cluster_labels: Cluster assignments (n_samples,), -1 for noise

    Returns:
    - Dictionary with mean, variance, std of inter-cluster cosines
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

    inter_cosines = []

    # Compute cosine similarities between different clusters
    for i, cluster_i in enumerate(unique_clusters):
        cluster_i_mask = cluster_labels == cluster_i
        cluster_i_embeddings = normalized_embeddings[cluster_i_mask]

        for cluster_j in unique_clusters[i + 1 :]:
            cluster_j_mask = cluster_labels == cluster_j
            cluster_j_embeddings = normalized_embeddings[cluster_j_mask]

            # Sample some pairwise similarities between clusters
            n_samples = min(100, len(cluster_i_embeddings), len(cluster_j_embeddings))
            if n_samples > 0:
                np.random.seed(42)
                indices_i = np.random.choice(
                    len(cluster_i_embeddings), n_samples, replace=False
                )
                indices_j = np.random.choice(
                    len(cluster_j_embeddings), n_samples, replace=False
                )
                pairwise_cosines = np.sum(
                    cluster_i_embeddings[indices_i] * cluster_j_embeddings[indices_j],
                    axis=1,
                )
                inter_cosines.extend(pairwise_cosines)

    if len(inter_cosines) == 0:
        return {
            "overall_mean": 0.0,
            "overall_variance": 0.0,
            "overall_std": 0.0,
        }

    return {
        "overall_mean": np.mean(inter_cosines),
        "overall_variance": np.var(inter_cosines),
        "overall_std": np.std(inter_cosines),
    }


def compute_genre_cluster_stats(
    embeddings: np.ndarray, genre_labels: np.ndarray
) -> dict:
    """
    Compute cosine statistics for genre-based clusters.

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - genre_labels: Genre labels (n_samples,), can be pipe-separated for multi-genre

    Returns:
    - Dictionary with intra-genre and inter-genre statistics
    """
    normalized_embeddings = normalize_embeddings(embeddings)

    # Handle multi-genre labels (pipe-separated)
    # For simplicity, use the first genre for each sample
    single_genre_labels = []
    for label in genre_labels:
        if pd.isna(label) or label is None:
            single_genre_labels.append("Unknown")
        else:
            genres = str(label).split("|")
            single_genre_labels.append(genres[0] if genres else "Unknown")

    single_genre_labels = np.array(single_genre_labels)
    unique_genres = np.unique(single_genre_labels)
    unique_genres = unique_genres[unique_genres != "Unknown"]

    intra_genre_cosines = []
    inter_genre_cosines = []

    # Intra-genre statistics
    for genre in unique_genres:
        genre_mask = single_genre_labels == genre
        genre_embeddings = normalized_embeddings[genre_mask]

        if len(genre_embeddings) < 2:
            continue

        genre_cosine_matrix = cosine_similarity(genre_embeddings)
        upper_triangle = np.triu_indices(len(genre_embeddings), k=1)
        genre_cosines = genre_cosine_matrix[upper_triangle]
        intra_genre_cosines.extend(genre_cosines)

    # Inter-genre statistics
    for i, genre_i in enumerate(unique_genres):
        genre_i_mask = single_genre_labels == genre_i
        genre_i_embeddings = normalized_embeddings[genre_i_mask]

        for genre_j in unique_genres[i + 1 :]:
            genre_j_mask = single_genre_labels == genre_j
            genre_j_embeddings = normalized_embeddings[genre_j_mask]

            n_samples = min(100, len(genre_i_embeddings), len(genre_j_embeddings))
            if n_samples > 0:
                np.random.seed(42)
                indices_i = np.random.choice(
                    len(genre_i_embeddings), n_samples, replace=False
                )
                indices_j = np.random.choice(
                    len(genre_j_embeddings), n_samples, replace=False
                )
                pairwise_cosines = np.sum(
                    genre_i_embeddings[indices_i] * genre_j_embeddings[indices_j],
                    axis=1,
                )
                inter_genre_cosines.extend(pairwise_cosines)

    return {
        "intra_genre": {
            "mean": np.mean(intra_genre_cosines)
            if len(intra_genre_cosines) > 0
            else 0.0,
            "variance": np.var(intra_genre_cosines)
            if len(intra_genre_cosines) > 0
            else 0.0,
            "std": np.std(intra_genre_cosines) if len(intra_genre_cosines) > 0 else 0.0,
        },
        "inter_genre": {
            "mean": np.mean(inter_genre_cosines)
            if len(inter_genre_cosines) > 0
            else 0.0,
            "variance": np.var(inter_genre_cosines)
            if len(inter_genre_cosines) > 0
            else 0.0,
            "std": np.std(inter_genre_cosines) if len(inter_genre_cosines) > 0 else 0.0,
        },
    }


def create_visualizations(
    hdbscan_stats: dict,
    genre_stats: dict,
    cluster_labels: np.ndarray,
    genre_labels: np.ndarray,
    output_dir: str,
    prefix: str = "hdbscan_analysis",
):
    """
    Create visualization plots for HDBSCAN clustering analysis.

    Parameters:
    - hdbscan_stats: Dictionary with HDBSCAN cluster statistics
    - genre_stats: Dictionary with genre-based statistics
    - cluster_labels: HDBSCAN cluster assignments
    - genre_labels: Genre labels
    - output_dir: Directory to save plots
    - prefix: Prefix for output filenames
    """
    saved_files = {}

    # 1. Comparison plot: Intra vs Inter cluster cosine statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Intra-cluster statistics
    ax = axes[0, 0]
    categories = ["HDBSCAN\nIntra", "HDBSCAN\nInter", "Genre\nIntra", "Genre\nInter"]
    means = [
        hdbscan_stats["intra"]["overall_mean"],
        hdbscan_stats["inter"]["overall_mean"],
        genre_stats["intra_genre"]["mean"],
        genre_stats["inter_genre"]["mean"],
    ]
    stds = [
        hdbscan_stats["intra"]["overall_std"],
        hdbscan_stats["inter"]["overall_std"],
        genre_stats["intra_genre"]["std"],
        genre_stats["inter_genre"]["std"],
    ]

    x_pos = np.arange(len(categories))
    ax.bar(
        x_pos,
        means,
        yerr=stds,
        alpha=0.7,
        color=["steelblue", "coral", "green", "orange"],
    )
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Mean Cosine Similarity Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Variance comparison
    ax = axes[0, 1]
    variances = [
        hdbscan_stats["intra"]["overall_variance"],
        hdbscan_stats["inter"]["overall_variance"],
        genre_stats["intra_genre"]["variance"],
        genre_stats["inter_genre"]["variance"],
    ]
    ax.bar(x_pos, variances, alpha=0.7, color=["steelblue", "coral", "green", "orange"])
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title("Cosine Variance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Cluster size distribution
    ax = axes[1, 0]
    unique_clusters, cluster_counts = np.unique(
        cluster_labels[cluster_labels != -1], return_counts=True
    )
    ax.hist(cluster_counts, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Cluster Size", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"HDBSCAN Cluster Size Distribution\n({len(unique_clusters)} clusters)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Genre distribution
    ax = axes[1, 1]
    single_genre_labels = []
    for label in genre_labels:
        if pd.isna(label) or label is None:
            single_genre_labels.append("Unknown")
        else:
            genres = str(label).split("|")
            single_genre_labels.append(genres[0] if genres else "Unknown")
    unique_genres, genre_counts = np.unique(single_genre_labels, return_counts=True)
    top_genres = sorted(zip(genre_counts, unique_genres), reverse=True)[:15]
    top_counts, top_labels = zip(*top_genres) if top_genres else ([], [])

    if len(top_labels) > 0:
        ax.barh(range(len(top_labels)), top_counts, alpha=0.7, color="green")
        ax.set_yticks(range(len(top_labels)))
        ax.set_yticklabels(top_labels, fontsize=9)
        ax.set_xlabel("Number of Movies", fontsize=12)
        ax.set_title("Top 15 Genre Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{prefix}_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    saved_files["comparison"] = comparison_path
    plt.close(fig)

    # 2. Summary statistics text plot
    fig, ax = plt.subplots(figsize=(12, 10))
    stats_text = f"""
HDBSCAN Clustering Analysis Summary

HDBSCAN Clustering:
  Number of clusters: {len(unique_clusters)}
  Noise points: {np.sum(cluster_labels == -1)}
  Total samples: {len(cluster_labels)}
  
  Intra-cluster cosine:
    Mean: {hdbscan_stats["intra"]["overall_mean"]:.6f}
    Variance: {hdbscan_stats["intra"]["overall_variance"]:.6f}
    Std: {hdbscan_stats["intra"]["overall_std"]:.6f}
  
  Inter-cluster cosine:
    Mean: {hdbscan_stats["inter"]["overall_mean"]:.6f}
    Variance: {hdbscan_stats["inter"]["overall_variance"]:.6f}
    Std: {hdbscan_stats["inter"]["overall_std"]:.6f}

Genre Classification:
  Number of unique genres: {len(np.unique(single_genre_labels))}
  
  Intra-genre cosine:
    Mean: {genre_stats["intra_genre"]["mean"]:.6f}
    Variance: {genre_stats["intra_genre"]["variance"]:.6f}
    Std: {genre_stats["intra_genre"]["std"]:.6f}
  
  Inter-genre cosine:
    Mean: {genre_stats["inter_genre"]["mean"]:.6f}
    Variance: {genre_stats["inter_genre"]["variance"]:.6f}
    Std: {genre_stats["inter_genre"]["std"]:.6f}

Comparison:
  Intra-cluster separation (HDBSCAN vs Genre):
    Difference in mean: {hdbscan_stats["intra"]["overall_mean"] - genre_stats["intra_genre"]["mean"]:.6f}
    HDBSCAN is {"better" if hdbscan_stats["intra"]["overall_mean"] > genre_stats["intra_genre"]["mean"] else "worse"} at grouping similar items
  
  Inter-cluster separation (HDBSCAN vs Genre):
    Difference in mean: {hdbscan_stats["inter"]["overall_mean"] - genre_stats["inter_genre"]["mean"]:.6f}
    HDBSCAN clusters are {"more" if hdbscan_stats["inter"]["overall_mean"] < genre_stats["inter_genre"]["mean"] else "less"} separated than genre clusters
"""
    ax.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("HDBSCAN Clustering Analysis Summary", fontsize=16, fontweight="bold")

    summary_path = os.path.join(output_dir, f"{prefix}_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    saved_files["summary"] = summary_path
    plt.close(fig)

    return saved_files


def main(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    output_dir: str = None,
    n_samples: int = 5000,
    random_seed: int = 42,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    apply_whitening: bool = True,
    n_whiten_components: int = None,
):
    """
    Main function to perform HDBSCAN clustering analysis on movie embeddings.

    Parameters:
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - output_dir: Directory to save plots. If None, uses figures/ directory
    - n_samples: Number of samples to use for analysis
    - random_seed: Random seed for sampling
    - min_cluster_size: Minimum number of points required to form a cluster.
                       Smaller groups are labeled as noise. Higher = fewer, larger clusters.
    - min_samples: Minimum number of neighbors required for a point to be in a cluster.
                   Controls clustering conservativeness. Higher = more noise points.
                   Typically set to min_cluster_size / 2 or equal to min_cluster_size.
    - apply_whitening: If True, apply PCA whitening to restore isotropy before clustering.
                      Helps when data has anisotropic structure.
    - n_whiten_components: Number of PCA components for whitening. If None, keeps all.
                           Can reduce dimensionality if set to a smaller value.
    """
    logger.info(f"{'=' * 60}")
    logger.info("HDBSCAN Clustering Analysis for Movie Embeddings")
    logger.info(f"{'=' * 60}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

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

        # Load movie metadata to filter by year
        logger.info("Loading movie metadata...")
        movie_data = load_final_dataset(csv_path, verbose=False)

        if movie_data.empty:
            raise ValueError(f"No movie data found in {csv_path}")

        # Filter by year range
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

        filtered_embeddings = all_embeddings[valid_indices]
        filtered_movie_ids = all_movie_ids[valid_indices]

        logger.info(
            f"Filtered to {len(filtered_movie_ids)} movies with embeddings in year range"
        )

        # Sample subset
        if n_samples is not None and n_samples < len(filtered_embeddings):
            logger.info(
                f"Sampling {n_samples} embeddings from {len(filtered_embeddings)} available (seed={random_seed})"
            )
            np.random.seed(random_seed)
            sample_indices = np.random.choice(
                len(filtered_embeddings), size=n_samples, replace=False
            )
            sampled_embeddings = filtered_embeddings[sample_indices]
            sampled_movie_ids = filtered_movie_ids[sample_indices]
        else:
            sampled_embeddings = filtered_embeddings
            sampled_movie_ids = filtered_movie_ids
            logger.info(f"Using all {len(sampled_embeddings)} embeddings")

        # Get corresponding movie data for sampled movies
        sampled_movie_data = movie_data[
            movie_data["movie_id"].isin(sampled_movie_ids)
        ].copy()

        # Process genres
        logger.info("Processing genre classifications...")
        sampled_movie_data["processed_genre"] = sampled_movie_data["genre"].apply(
            preprocess_genres
        )
        genre_labels = sampled_movie_data["processed_genre"].values

        # Apply preprocessing: mean-centering and/or whitening
        if apply_whitening:
            logger.info("Applying PCA whitening to restore isotropy...")
            processed_embeddings = whiten_embeddings(
                sampled_embeddings, n_components=n_whiten_components
            )
            logger.info(f"Whitened embeddings shape: {processed_embeddings.shape}")
        else:
            logger.info("Mean-centering embeddings...")
            processed_embeddings = mean_center_embeddings(sampled_embeddings)

        # Normalize embeddings after preprocessing
        logger.info("Normalizing embeddings...")
        normalized_embeddings = normalize_embeddings(processed_embeddings)

        # Precompute cosine distance matrix for HDBSCAN
        logger.info("Precomputing cosine distance matrix...")
        cosine_distance_matrix = pairwise_distances(
            normalized_embeddings, metric="cosine"
        )

        # Run HDBSCAN clustering with precomputed cosine distances
        logger.info(
            f"Running HDBSCAN clustering (min_cluster_size={min_cluster_size}, min_samples={min_samples})..."
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        cluster_labels = clusterer.fit_predict(cosine_distance_matrix)

        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        n_noise = np.sum(cluster_labels == -1)
        logger.info(f"HDBSCAN found {n_clusters} clusters with {n_noise} noise points")

        # Compute intra-cluster statistics
        logger.info("Computing intra-cluster cosine statistics...")
        intra_stats = compute_intra_cluster_cosine_stats(
            normalized_embeddings, cluster_labels
        )

        # Compute inter-cluster statistics
        logger.info("Computing inter-cluster cosine statistics...")
        inter_stats = compute_inter_cluster_cosine_stats(
            normalized_embeddings, cluster_labels
        )

        hdbscan_stats = {
            "intra": intra_stats,
            "inter": inter_stats,
        }

        # Compute genre-based statistics
        logger.info("Computing genre-based cosine statistics...")
        genre_stats = compute_genre_cluster_stats(normalized_embeddings, genre_labels)

        # Create visualizations
        logger.info("Creating visualizations...")
        whitening_suffix = "_whitened" if apply_whitening else "_meancentered"
        prefix = f"hdbscan_analysis_n{n_samples}_minclust{min_cluster_size}_minsamp{min_samples}{whitening_suffix}"
        plot_files = create_visualizations(
            hdbscan_stats, genre_stats, cluster_labels, genre_labels, output_dir, prefix
        )

        # Print summary
        logger.info(f"\n{'=' * 60}")
        logger.info("HDBSCAN Clustering Analysis Results")
        logger.info(f"{'=' * 60}\n")

        logger.info(f"Number of samples: {len(sampled_embeddings)}")
        logger.info(f"Number of dimensions: {sampled_embeddings.shape[1]}")
        logger.info("\nHDBSCAN Clustering:")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(
            f"  Noise points: {n_noise} ({100 * n_noise / len(cluster_labels):.2f}%)"
        )

        logger.info("\nIntra-cluster Cosine Statistics:")
        logger.info(f"  Mean: {intra_stats['overall_mean']:.6f}")
        logger.info(f"  Variance: {intra_stats['overall_variance']:.6f}")
        logger.info(f"  Std: {intra_stats['overall_std']:.6f}")

        logger.info("\nInter-cluster Cosine Statistics:")
        logger.info(f"  Mean: {inter_stats['overall_mean']:.6f}")
        logger.info(f"  Variance: {inter_stats['overall_variance']:.6f}")
        logger.info(f"  Std: {inter_stats['overall_std']:.6f}")

        logger.info("\nGenre Classification Statistics:")
        logger.info(f"  Intra-genre Mean: {genre_stats['intra_genre']['mean']:.6f}")
        logger.info(
            f"  Intra-genre Variance: {genre_stats['intra_genre']['variance']:.6f}"
        )
        logger.info(f"  Inter-genre Mean: {genre_stats['inter_genre']['mean']:.6f}")
        logger.info(
            f"  Inter-genre Variance: {genre_stats['inter_genre']['variance']:.6f}"
        )

        logger.info("\nComparison:")
        intra_diff = intra_stats["overall_mean"] - genre_stats["intra_genre"]["mean"]
        inter_diff = inter_stats["overall_mean"] - genre_stats["inter_genre"]["mean"]
        logger.info(
            f"  Intra-cluster mean difference (HDBSCAN - Genre): {intra_diff:.6f}"
        )
        logger.info(
            f"  Inter-cluster mean difference (HDBSCAN - Genre): {inter_diff:.6f}"
        )

        logger.info("\nVisualization plots saved:")
        for plot_name, plot_path in plot_files.items():
            logger.info(f"  {plot_name}: {plot_path}")

        logger.info(f"\n{'=' * 60}")
        logger.info("Analysis complete!")
        logger.info(f"{'=' * 60}")

        return {
            "hdbscan_stats": hdbscan_stats,
            "genre_stats": genre_stats,
            "cluster_labels": cluster_labels,
            "genre_labels": genre_labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    except Exception as e:
        logger.error(f"Error performing HDBSCAN clustering analysis: {e}")
        raise


if __name__ == "__main__":
    # To get more clusters: lower both min_cluster_size and min_samples
    # while keeping the ratio ~0.5 (min_samples / min_cluster_size)
    # Examples:
    #   More clusters: min_cluster_size=10, min_samples=5
    #   Even more: min_cluster_size=5, min_samples=2 or 3
    #   For ~20 clusters with 10k samples: min_cluster_size=3-5, min_samples=2-3
    #   Fewer clusters: min_cluster_size=50, min_samples=25
    main(
        start_year=1930,
        end_year=2024,
        n_samples=20_000,
        random_seed=42,
        min_cluster_size=5,  # Lowered to get more clusters (~20 target)
        min_samples=2,  # Ratio: 3/5 = 0.6 (slightly more conservative)
        apply_whitening=False,  # Apply PCA whitening to restore isotropy
        n_whiten_components=None,  # Keep all components (can reduce for dimensionality reduction)
        output_dir=os.path.join(BASE_DIR, "figures"),
        data_dir=os.path.join(BASE_DIR, "data", "data_final"),
        csv_path=os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv"),
    )
