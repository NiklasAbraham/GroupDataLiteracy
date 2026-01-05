"""
Epsilon ball analysis for finding movies within a distance threshold around anchor movies.

This script finds all movies within an epsilon distance (epsilon ball) around
specified anchor movies. The anchor can be a single movie or the average of
multiple movies. Results include distances, rankings, and temporal analysis.
"""

import hashlib
import json
import logging
import os
import re
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from data_utils
# Path setup must occur before this import
from src.analysis.math_functions import (  # type: ignore  # noqa: E402
    compute_anchor_embedding,
    find_movies_in_epsilon_ball,
    interpret_ks_test,
    kolmogorov_smirnov_test,
    kolmogorov_smirnov_test_temporal,
)
from src.data_utils import (  # type: ignore  # noqa: E402
    load_final_dataset,
    load_final_dense_embeddings,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "data", "final_dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache", "mean_embeddings")
START_YEAR = 1930
END_YEAR = 2024


def analyze_epsilon_ball(
    anchor_qids: list,
    epsilon: float,
    filtered_embeddings: np.ndarray,
    filtered_movie_ids: np.ndarray,
    movie_data: pd.DataFrame,
    anchor_method: str = "average",
    exclude_anchors: bool = True,
):
    """
    Analyze movies within an epsilon ball around anchor movies.

    Parameters:
    - anchor_qids: List of movie QIDs to use as anchors
    - epsilon: Maximum cosine distance threshold (0 <= epsilon <= 2)
    - filtered_embeddings: Array of embeddings already filtered by year range
        (shape: [n_movies, embedding_dim])
    - filtered_movie_ids: Array of movie IDs corresponding to filtered_embeddings
    - movie_data: DataFrame with movie metadata already filtered by year range
    - anchor_method: Method to combine anchor embeddings ("average" or "medoid")
    - exclude_anchors: Whether to exclude anchor movies from results

    Returns:
    - DataFrame with columns: movie_id, title, year, distance, similarity, rank
    """
    logger.info(f"{'=' * 60}")
    logger.info("Epsilon ball analysis")
    logger.info(f"Anchor QIDs: {anchor_qids}")
    logger.info(f"Epsilon: {epsilon}")
    logger.info(f"{'=' * 60}")

    try:
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

        # Find movies within epsilon ball
        logger.info(f"Finding movies within epsilon ball (epsilon={epsilon})...")
        exclude_anchor_ids = anchor_qids if exclude_anchors else None
        indices, distances, similarities = find_movies_in_epsilon_ball(
            embeddings_corpus=filtered_embeddings,
            anchor_embedding=anchor_embedding,
            movie_ids=filtered_movie_ids,
            epsilon=epsilon,
            exclude_anchor_ids=exclude_anchor_ids,
        )

        logger.info(f"Found {len(indices)} movies within epsilon ball")

        if len(indices) == 0:
            logger.warning("No movies found within epsilon ball")
            return pd.DataFrame(
                columns=["movie_id", "title", "year", "distance", "similarity", "rank"]
            )

        # Create results dataframe
        results = []
        for rank, (idx, dist, sim) in enumerate(
            zip(indices, distances, similarities), 1
        ):
            movie_id = filtered_movie_ids[idx]
            movie_info = movie_data[movie_data["movie_id"] == movie_id]

            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                year = movie_info.iloc[0].get("year", None)
            else:
                title = "Unknown"
                year = None

            results.append(
                {
                    "movie_id": movie_id,
                    "title": title,
                    "year": year,
                    "distance": dist,
                    "similarity": sim,
                    "rank": rank,
                }
            )

        results_df = pd.DataFrame(results)

        logger.info(f"\n{'=' * 60}")
        logger.info("Results summary:")
        logger.info(f"Total movies in epsilon ball: {len(results_df)}")
        logger.info(
            f"Distance range: {results_df['distance'].min():.6f} - {results_df['distance'].max():.6f}"
        )
        if "year" in results_df.columns and results_df["year"].notna().any():
            logger.info(
                f"Year range: {int(results_df['year'].min())} - {int(results_df['year'].max())}"
            )
        logger.info(f"{'=' * 60}")

        return results_df

    except Exception as e:
        logger.error(f"Error in epsilon ball analysis: {e}")
        raise


def plot_movies_over_time(
    results_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Movies in Epsilon Ball Over Time",
    figsize: tuple = (12, 6),
    random_results_df: pd.DataFrame = None,
):
    """
    Plot the number of movies in the epsilon ball over time.

    Parameters:
    - results_df: DataFrame from analyze_epsilon_ball
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title (will have total count appended)
    - figsize: Figure size tuple
    - random_results_df: Optional DataFrame from control group (mean embedding) analysis for comparison
    """
    if results_df.empty:
        logger.warning("No data to plot")
        return

    if "year" not in results_df.columns or results_df["year"].isna().all():
        logger.warning("No year data available for plotting")
        return

    # Filter out movies without year
    df_with_year = results_df[results_df["year"].notna()].copy()
    df_with_year["year"] = df_with_year["year"].astype(int)

    if df_with_year.empty:
        logger.warning("No movies with valid year data")
        return

    # Count movies per year
    year_counts = df_with_year["year"].value_counts().sort_index()
    total_movies = len(results_df)

    # Calculate SMA of 3 and SMA of 10
    year_counts_series = pd.Series(year_counts.values, index=year_counts.index)
    sma_3 = year_counts_series.rolling(window=3, center=False, min_periods=1).mean()
    sma_10 = year_counts_series.rolling(window=10, center=False, min_periods=1).mean()

    # Check if random results are provided for comparison
    has_random_comparison = (
        random_results_df is not None
        and not random_results_df.empty
        and "year" in random_results_df.columns
        and random_results_df["year"].notna().any()
    )

    if has_random_comparison:
        # Get random year counts
        if "count" in random_results_df.columns:
            # Use the pre-calculated averaged counts
            random_year_counts_series = pd.Series(
                random_results_df["count"].values,
                index=random_results_df["year"].values,
            ).sort_index()
        else:
            # Fallback to original method
            random_df_with_year = random_results_df[
                random_results_df["year"].notna()
            ].copy()
            random_df_with_year["year"] = random_df_with_year["year"].astype(int)
            random_year_counts = random_df_with_year["year"].value_counts().sort_index()
            random_year_counts_series = pd.Series(
                random_year_counts.values, index=random_year_counts.index
            )

        # Normalize both datasets so their highest peak is 1
        anchor_max = year_counts_series.max()
        random_max = random_year_counts_series.max()

        anchor_normalized = (
            year_counts_series / anchor_max if anchor_max > 0 else year_counts_series
        )
        random_normalized = (
            random_year_counts_series / random_max
            if random_max > 0
            else random_year_counts_series
        )

        # Normalize SMAs as well
        sma_3_normalized = sma_3 / anchor_max if anchor_max > 0 else sma_3
        sma_10_normalized = sma_10 / anchor_max if anchor_max > 0 else sma_10

        random_sma_3 = random_year_counts_series.rolling(
            window=3, center=False, min_periods=1
        ).mean()
        random_sma_10 = random_year_counts_series.rolling(
            window=10, center=False, min_periods=1
        ).mean()
        random_sma_3_normalized = (
            random_sma_3 / random_max if random_max > 0 else random_sma_3
        )
        random_sma_10_normalized = (
            random_sma_10 / random_max if random_max > 0 else random_sma_10
        )

        # Create plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot anchor movies on left y-axis
        ax1.bar(
            anchor_normalized.index,
            anchor_normalized.values,
            alpha=0.7,
            edgecolor="black",
            color="steelblue",
            label="Anchor Movies (normalized)",
        )
        ax1.plot(
            sma_3_normalized.index,
            sma_3_normalized.values,
            color="red",
            linewidth=2,
            label="Anchor SMA (3)",
        )
        ax1.plot(
            sma_10_normalized.index,
            sma_10_normalized.values,
            color="darkred",
            linewidth=2,
            label="Anchor SMA (10)",
        )
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel(
            "Normalized Count (Anchor Movies)", fontsize=12, color="steelblue"
        )
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.grid(True, alpha=0.3, axis="y")

        # Create second y-axis for random movies
        ax2 = ax1.twinx()

        # Align years for random data
        all_years = sorted(set(anchor_normalized.index) | set(random_normalized.index))
        random_aligned = pd.Series(0, index=all_years)
        random_aligned.loc[random_normalized.index] = random_normalized.values

        ax2.bar(
            random_aligned.index,
            random_aligned.values,
            alpha=0.5,
            edgecolor="black",
            color="coral",
            label="Control Group (normalized)",
        )
        ax2.plot(
            random_sma_3_normalized.index,
            random_sma_3_normalized.values,
            color="lightcoral",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label="Control Group SMA (3)",
        )
        ax2.plot(
            random_sma_10_normalized.index,
            random_sma_10_normalized.values,
            color="lightpink",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label="Control Group SMA (10)",
        )
        ax2.set_ylabel("Normalized Count (Control Group)", fontsize=12, color="coral")
        ax2.tick_params(axis="y", labelcolor="coral")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Replace "Random" labels with "Control Group"
        labels2 = [label.replace("Random", "Control Group") for label in labels2]
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(
            f"{title} (Total: {total_movies} movies, Normalized)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
    else:
        # Create plot without dual axes (original behavior)
        plt.figure(figsize=figsize)
        plt.bar(year_counts.index, year_counts.values, alpha=0.7, edgecolor="black")
        plt.plot(sma_3.index, sma_3.values, color="red", linewidth=2, label="SMA (3)")
        plt.plot(
            sma_10.index, sma_10.values, color="darkred", linewidth=2, label="SMA (10)"
        )
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Number of Movies", fontsize=12)
        plt.title(
            f"{title} (Total: {total_movies} movies)", fontsize=14, fontweight="bold"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_distance_distribution(
    results_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Distance Distribution in Epsilon Ball",
    figsize: tuple = (10, 6),
):
    """
    Plot the distribution of distances in the epsilon ball.

    Parameters:
    - results_df: DataFrame from analyze_epsilon_ball
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title (will have total count appended)
    - figsize: Figure size tuple
    """
    if results_df.empty:
        logger.warning("No data to plot")
        return

    total_movies = len(results_df)

    plt.figure(figsize=figsize)
    plt.hist(
        results_df["distance"],
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    plt.xlabel("Cosine Distance", fontsize=12)
    plt.ylabel("Number of Movies", fontsize=12)
    plt.title(f"{title} (Total: {total_movies} movies)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.axvline(
        results_df["distance"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {results_df['distance'].mean():.4f}",
    )
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_ks_test_cdf(
    anchor_distances: np.ndarray,
    random_distances: np.ndarray,
    ks_statistic: float,
    p_value: float,
    output_path: str = None,
    title: str = "Kolmogorov-Smirnov Test: Distance Distributions",
    figsize: tuple = (12, 8),
    interpretation: dict = None,
):
    """
    Plot cumulative distribution functions (CDFs) for K-S test visualization.

    Parameters:
    - anchor_distances: Array of cosine distances from anchor epsilon ball
    - random_distances: Array of cosine distances from control group epsilon ball
    - ks_statistic: K-S test statistic
    - p_value: p-value from K-S test
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title
    - figsize: Figure size tuple
    """
    # Sort distances for CDF calculation
    anchor_sorted = np.sort(anchor_distances)
    random_sorted = np.sort(random_distances)

    # Calculate CDFs
    anchor_cdf = np.arange(1, len(anchor_sorted) + 1) / len(anchor_sorted)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)

    # Find the point of maximum difference
    # Interpolate to find where max difference occurs
    all_distances = np.sort(np.unique(np.concatenate([anchor_sorted, random_sorted])))
    anchor_cdf_interp = np.interp(all_distances, anchor_sorted, anchor_cdf)
    random_cdf_interp = np.interp(all_distances, random_sorted, random_cdf)
    diff = np.abs(anchor_cdf_interp - random_cdf_interp)
    max_diff_idx = np.argmax(diff)
    max_diff_dist = all_distances[max_diff_idx]
    max_diff_anchor_cdf = anchor_cdf_interp[max_diff_idx]
    max_diff_random_cdf = random_cdf_interp[max_diff_idx]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: CDF comparison
    ax1.plot(
        anchor_sorted, anchor_cdf, label="Anchor Movies", linewidth=2, color="blue"
    )
    ax1.plot(
        random_sorted,
        random_cdf,
        label="Control Group",
        linewidth=2,
        color="red",
        linestyle="--",
    )

    # Highlight maximum difference
    ax1.plot(
        [max_diff_dist, max_diff_dist],
        [max_diff_anchor_cdf, max_diff_random_cdf],
        "k-",
        linewidth=2,
        label=f"K-S Statistic = {ks_statistic:.4f}",
    )
    ax1.plot(max_diff_dist, max_diff_anchor_cdf, "ko", markersize=8)
    ax1.plot(max_diff_dist, max_diff_random_cdf, "ko", markersize=8)

    ax1.set_xlabel("Cosine Distance", fontsize=12)
    ax1.set_ylabel("Cumulative Probability", fontsize=12)
    ax1.set_title("Cumulative Distribution Functions", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Histogram comparison
    # Use shared bin edges for proper normalization and comparison
    all_distances_combined = np.concatenate([anchor_distances, random_distances])
    bins = np.linspace(
        all_distances_combined.min(),
        all_distances_combined.max(),
        50,
    )
    ax2.hist(
        anchor_distances,
        bins=bins,
        alpha=0.6,
        label="Anchor Movies",
        color="blue",
        density=True,
    )
    ax2.hist(
        random_distances,
        bins=bins,
        alpha=0.6,
        label="Control Group",
        color="red",
        density=True,
    )
    ax2.axvline(
        max_diff_dist,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Max Diff Point (D={ks_statistic:.4f})",
    )
    ax2.set_xlabel("Cosine Distance", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Distance Distribution Histograms", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add test results as text
    fig.suptitle(
        f"{title}\nK-S Statistic: {ks_statistic:.6f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"K-S test plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_ks_test_temporal_cdf(
    anchor_year_counts: pd.Series,
    random_year_counts: pd.Series,
    ks_statistic: float,
    p_value: float,
    output_path: str = None,
    title: str = "Kolmogorov-Smirnov Test: Temporal Distributions",
    figsize: tuple = (14, 6),
    interpretation: dict = None,
):
    """
    Plot cumulative distribution functions for temporal K-S test.

    Parameters:
    - anchor_year_counts: Series of movie counts per year for anchor
    - random_year_counts: Series of movie counts per year for control group
    - ks_statistic: K-S test statistic
    - p_value: p-value from K-S test
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title
    - figsize: Figure size tuple
    """
    # Create sample arrays from counts
    anchor_samples = []
    for year, count in anchor_year_counts.items():
        anchor_samples.extend([year] * int(count))

    random_samples = []
    for year, count in random_year_counts.items():
        random_samples.extend([year] * int(count))

    # Sort for CDF calculation
    anchor_sorted = np.sort(anchor_samples)
    random_sorted = np.sort(random_samples)

    # Calculate CDFs
    anchor_cdf = np.arange(1, len(anchor_sorted) + 1) / len(anchor_sorted)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)

    # Find maximum difference point
    all_years = np.sort(np.unique(np.concatenate([anchor_sorted, random_sorted])))
    anchor_cdf_interp = np.interp(all_years, anchor_sorted, anchor_cdf)
    random_cdf_interp = np.interp(all_years, random_sorted, random_cdf)
    diff = np.abs(anchor_cdf_interp - random_cdf_interp)
    max_diff_idx = np.argmax(diff)
    max_diff_year = all_years[max_diff_idx]
    max_diff_anchor_cdf = anchor_cdf_interp[max_diff_idx]
    max_diff_random_cdf = random_cdf_interp[max_diff_idx]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: CDF comparison
    ax1.plot(
        anchor_sorted, anchor_cdf, label="Anchor Movies", linewidth=2, color="blue"
    )
    ax1.plot(
        random_sorted,
        random_cdf,
        label="Control Group",
        linewidth=2,
        color="red",
        linestyle="--",
    )

    # Highlight maximum difference
    ax1.plot(
        [max_diff_year, max_diff_year],
        [max_diff_anchor_cdf, max_diff_random_cdf],
        "k-",
        linewidth=2,
        label=f"K-S Statistic = {ks_statistic:.4f}",
    )
    ax1.plot(max_diff_year, max_diff_anchor_cdf, "ko", markersize=8)
    ax1.plot(max_diff_year, max_diff_random_cdf, "ko", markersize=8)

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Cumulative Probability", fontsize=12)
    ax1.set_title(
        "Cumulative Distribution Functions (by Year)", fontsize=12, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Year count comparison with dual y-axes and normalization
    all_years_aligned = sorted(
        set(anchor_year_counts.index) | set(random_year_counts.index)
    )
    anchor_aligned = pd.Series(0, index=all_years_aligned)
    random_aligned = pd.Series(0, index=all_years_aligned)
    anchor_aligned.loc[anchor_year_counts.index] = anchor_year_counts.values
    random_aligned.loc[random_year_counts.index] = random_year_counts.values

    # Normalize both datasets so their highest peak is 1
    anchor_max = anchor_aligned.max()
    random_max = random_aligned.max()

    anchor_normalized = (
        anchor_aligned / anchor_max if anchor_max > 0 else anchor_aligned
    )
    random_normalized = (
        random_aligned / random_max if random_max > 0 else random_aligned
    )

    x = np.arange(len(all_years_aligned))
    width = 0.35

    # Plot anchor movies on left y-axis
    ax2.bar(
        x - width / 2,
        anchor_normalized.values,
        width,
        label="Anchor Movies (normalized)",
        alpha=0.7,
        color="blue",
    )
    ax2.axvline(
        np.where(all_years_aligned == max_diff_year)[0][0],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Max Diff Year ({int(max_diff_year)})",
    )
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Normalized Count (Anchor Movies)", fontsize=12, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_title("Movie Counts per Year (Normalized)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x[:: max(1, len(x) // 10)])
    ax2.set_xticklabels(
        [
            all_years_aligned[i]
            for i in range(0, len(all_years_aligned), max(1, len(x) // 10))
        ],
        rotation=45,
        ha="right",
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Create second y-axis for random movies
    ax2_twin = ax2.twinx()
    ax2_twin.bar(
        x + width / 2,
        random_normalized.values,
        width,
        label="Random Movies (normalized)",
        alpha=0.7,
        color="red",
    )
    ax2_twin.set_ylabel("Normalized Count (Control Group)", fontsize=12, color="red")
    ax2_twin.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Add test results as text
    fig.suptitle(
        f"{title}\nK-S Statistic: {ks_statistic:.6f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"K-S test temporal plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


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


def compute_embeddings_hash(
    filtered_embeddings: np.ndarray,
    start_year: int,
    end_year: int,
) -> str:
    """
    Compute a hash of the filtered embeddings and parameters to verify cache validity.

    Parameters:
    - filtered_embeddings: Array of embeddings (shape: [n_movies, embedding_dim])
    - start_year: Start year used for filtering
    - end_year: End year used for filtering

    Returns:
    - Hash string that uniquely identifies this set of embeddings and parameters
    """
    # Create a hash from:
    # 1. Shape of embeddings
    # 2. Sample of the data (first, middle, last rows and a few random ones)
    # 3. Sum of all values (quick integrity check)
    # 4. Year range
    hash_data = {
        "shape": filtered_embeddings.shape,
        "dtype": str(filtered_embeddings.dtype),
        "start_year": start_year,
        "end_year": end_year,
    }

    # Add sample data for verification
    n_samples = min(100, len(filtered_embeddings))
    if len(filtered_embeddings) > 0:
        # Sample indices: first, last, middle, and some random ones
        sample_indices = [0]
        if len(filtered_embeddings) > 1:
            sample_indices.append(len(filtered_embeddings) - 1)
        if len(filtered_embeddings) > 2:
            sample_indices.append(len(filtered_embeddings) // 2)
        # Add random samples
        if len(filtered_embeddings) > n_samples:
            np.random.seed(42)  # Fixed seed for reproducibility
            random_indices = np.random.choice(
                len(filtered_embeddings),
                size=n_samples - len(sample_indices),
                replace=False,
            )
            sample_indices.extend(random_indices.tolist())

        # Hash the sampled rows
        sample_data = filtered_embeddings[sample_indices].tobytes()
        hash_data["sample_hash"] = hashlib.md5(sample_data).hexdigest()

        # Add sum as a quick integrity check
        hash_data["sum"] = float(np.sum(filtered_embeddings))

    # Create hash from the hash data
    hash_string = json.dumps(hash_data, sort_keys=True)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def load_cached_mean_embedding(
    cache_dir: str,
    embeddings_hash: str,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Load cached mean embedding if it exists and hash matches.

    Parameters:
    - cache_dir: Directory to store/load cache files
    - embeddings_hash: Hash of the embeddings to verify cache validity

    Returns:
    - Tuple of (mean_embedding, found) where:
      - mean_embedding: The cached mean embedding if found, None otherwise
      - found: Boolean indicating if valid cache was found
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.npy")
    metadata_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.json")

    if not os.path.exists(cache_file) or not os.path.exists(metadata_file):
        return None, False

    try:
        # Load metadata to verify hash
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if metadata.get("hash") != embeddings_hash:
            logger.warning("Cache hash mismatch, will recompute mean embedding")
            return None, False

        # Load the cached mean embedding
        mean_embedding = np.load(cache_file)
        logger.info(
            f"Loaded cached mean embedding from {cache_file} "
            f"(computed from {metadata.get('n_movies', 'unknown')} movies)"
        )
        return mean_embedding, True

    except Exception as e:
        logger.warning(f"Error loading cached mean embedding: {e}, will recompute")
        return None, False


def save_cached_mean_embedding(
    mean_embedding: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    n_movies: int,
) -> None:
    """
    Save mean embedding to cache with metadata.

    Parameters:
    - mean_embedding: The computed mean embedding to cache
    - cache_dir: Directory to store cache files
    - embeddings_hash: Hash of the embeddings used for this computation
    - n_movies: Number of movies used to compute the mean
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.npy")
    metadata_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.json")

    try:
        # Save the mean embedding
        np.save(cache_file, mean_embedding)

        # Save metadata
        metadata = {
            "hash": embeddings_hash,
            "n_movies": n_movies,
            "shape": list(mean_embedding.shape),
            "dtype": str(mean_embedding.dtype),
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Cached mean embedding to {cache_file}")

    except Exception as e:
        logger.warning(f"Error saving cached mean embedding: {e}")


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


def main(
    anchor_qids: list = None,
    epsilon: float = 0.3,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    anchor_method: str = "average",
    exclude_anchors: bool = True,
    plot_over_time: bool = True,
    plot_distance_dist: bool = True,
    compare_with_random: bool = False,
    output_dir: str = None,
):
    """
    Main function to run epsilon ball analysis.

    Parameters:
    - anchor_qids: List of movie QIDs to use as anchors (e.g., ["Q4941"] for James Bond)
    - epsilon: Maximum cosine distance threshold (default: 0.3)
    - start_year: First year to filter movies (default: 1930)
    - end_year: Last year to filter movies (default: 2024)
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - anchor_method: Method to combine anchor embeddings ("average" or "medoid")
    - exclude_anchors: Whether to exclude anchor movies from results
    - plot_over_time: Whether to create plot of movies over time
    - plot_distance_dist: Whether to create distance distribution plot
    - compare_with_random: Whether to compare with control group (mean of entire ensemble) (default: False)
    - output_dir: Directory to save plots (if None, uses current directory)
    """
    if anchor_qids is None:
        # Example: James Bond movies
        anchor_qids = ["Q4941"]  # Dr. No (first James Bond film)
        logger.info("Using default anchor: James Bond (Q4941)")

    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)
    movie_data = load_final_dataset(csv_path, verbose=False)

    anchor_qids = [
        "Q182692", "Q190643", "Q243439", "Q201674", "Q585203", "Q623502", "Q471159",
        "Q1126637", "Q180706", "Q478333", "Q1423020", "Q244876", "Q1114683",
        "Q15733016", "Q85842235", "Q1645944", "Q639864", "Q3520085",
        "Q112226601", "Q62277203",
    ]

    N = 750  # max movies per year

    # --- 1) Split anchors from non-anchors ---
    anchors_df = movie_data[movie_data["movie_id"].isin(anchor_qids)]
    non_anchors_df = movie_data[~movie_data["movie_id"].isin(anchor_qids)]

    # --- 2) Apply per-year cap ONLY to non-anchors ---
    non_anchors_df = (
        non_anchors_df
        .sort_values("movie_id")
        .groupby("year", group_keys=False)
        .head(N)
    )

    # --- 3) Recombine (anchors always included) ---
    movie_data = pd.concat([anchors_df, non_anchors_df], ignore_index=True)

    # --- 4) Slice embeddings directly (no mask) ---
    id_to_idx = {mid: i for i, mid in enumerate(all_movie_ids)}
    indices = [id_to_idx[mid] for mid in movie_data["movie_id"] if mid in id_to_idx]

    all_embeddings = all_embeddings[indices]
    all_movie_ids = all_movie_ids[indices]

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

    # Run analysis for anchor movies
    results_df = analyze_epsilon_ball(
        anchor_qids=anchor_qids,
        epsilon=epsilon,
        filtered_embeddings=filtered_embeddings,
        filtered_movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        anchor_method=anchor_method,
        exclude_anchors=exclude_anchors,
    )

    # Run analysis for control group (mean of entire ensemble) if comparison is enabled
    random_results_df = None
    random_distances_list = []
    if compare_with_random:
        logger.info("Computing mean embedding of entire ensemble as control group...")

        # Compute hash of filtered embeddings to check cache
        embeddings_hash = compute_embeddings_hash(
            filtered_embeddings=filtered_embeddings,
            start_year=start_year,
            end_year=end_year,
        )

        # Try to load cached mean embedding
        mean_embedding, cache_found = load_cached_mean_embedding(
            cache_dir=CACHE_DIR,
            embeddings_hash=embeddings_hash,
        )

        if not cache_found:
            # Compute mean embedding of all movies in the filtered dataset
            logger.info("Computing mean embedding (this may take a while)...")
            mean_embedding = np.mean(filtered_embeddings, axis=0, keepdims=True)
            logger.info(
                f"Mean embedding computed from {len(filtered_embeddings)} movies"
            )

            # Save to cache
            save_cached_mean_embedding(
                mean_embedding=mean_embedding,
                cache_dir=CACHE_DIR,
                embeddings_hash=embeddings_hash,
                n_movies=len(filtered_embeddings),
            )
        else:
            logger.info(
                f"Using cached mean embedding (computed from {len(filtered_embeddings)} movies)"
            )

        # Find movies within epsilon ball around the mean embedding
        logger.info(
            f"Finding movies within epsilon ball (epsilon={epsilon}) around mean embedding..."
        )
        indices, distances, similarities = find_movies_in_epsilon_ball(
            embeddings_corpus=filtered_embeddings,
            anchor_embedding=mean_embedding,
            movie_ids=filtered_movie_ids,
            epsilon=epsilon,
            exclude_anchor_ids=None,  # Don't exclude anything for mean embedding
        )

        logger.info(
            f"Found {len(indices)} movies within epsilon ball of mean embedding"
        )

        # Create results dataframe
        results = []
        for rank, (idx, dist, sim) in enumerate(
            zip(indices, distances, similarities), 1
        ):
            movie_id = filtered_movie_ids[idx]
            movie_info = movie_data[movie_data["movie_id"] == movie_id]

            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                year = movie_info.iloc[0].get("year", None)
            else:
                title = "Unknown"
                year = None

            results.append(
                {
                    "movie_id": movie_id,
                    "title": title,
                    "year": year,
                    "distance": dist,
                    "similarity": sim,
                    "rank": rank,
                }
            )

        random_results_df = pd.DataFrame(results)

        # Extract distances for K-S test
        if not random_results_df.empty and "distance" in random_results_df.columns:
            random_distances_list.append(random_results_df["distance"].values)

    # Perform Kolmogorov-Smirnov tests if control group comparison is enabled
    if (
        compare_with_random
        and random_results_df is not None
        and not random_results_df.empty
        and not results_df.empty
    ):
        logger.info(f"\n{'=' * 60}")
        logger.info("Kolmogorov-Smirnov Test Results")
        logger.info(f"{'=' * 60}")

        # Test 1: Distance distribution comparison
        if random_distances_list and "distance" in results_df.columns:
            # Get control group distances
            all_random_distances = np.concatenate(random_distances_list)
            anchor_distances = results_df["distance"].values

            try:
                ks_stat_dist, p_value_dist = kolmogorov_smirnov_test(
                    anchor_distances, all_random_distances
                )
                interpretation_dist = interpret_ks_test(
                    ks_stat_dist,
                    p_value_dist,
                    sample_size_1=len(anchor_distances),
                    sample_size_2=len(all_random_distances),
                )

                logger.info("\n1. Distance Distribution Comparison:")
                logger.info(f"   K-S Statistic: {ks_stat_dist:.6f}")

                # Create K-S test plot for distances
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
                    anchor_names_str = truncate_filename_component(anchor_names_str)
                    ks_plot_path = os.path.join(
                        output_dir,
                        f"ks_test_distances_{anchor_names_str}_eps{epsilon:.2f}.png",
                    )
                    plot_ks_test_cdf(
                        anchor_distances,
                        all_random_distances,
                        ks_stat_dist,
                        p_value_dist,
                        output_path=ks_plot_path,
                        title=f"K-S Test: Distance Distributions (ε={epsilon})",
                        interpretation=interpretation_dist,
                    )
            except Exception as e:
                logger.warning(f"Could not perform distance K-S test: {e}")

        # Test 2: Temporal distribution comparison
        if random_results_df is not None and not random_results_df.empty:
            # Prepare anchor year counts
            anchor_df_with_year = results_df[results_df["year"].notna()].copy()
            anchor_df_with_year["year"] = anchor_df_with_year["year"].astype(int)
            anchor_year_counts = anchor_df_with_year["year"].value_counts().sort_index()

            # Prepare control group year counts
            random_df_with_year = random_results_df[
                random_results_df["year"].notna()
            ].copy()
            random_df_with_year["year"] = random_df_with_year["year"].astype(int)
            random_year_counts = random_df_with_year["year"].value_counts().sort_index()

            # Align years
            all_years = sorted(
                set(anchor_year_counts.index) | set(random_year_counts.index)
            )
            anchor_aligned = pd.Series(0, index=all_years)
            random_aligned = pd.Series(0, index=all_years)
            anchor_aligned.loc[anchor_year_counts.index] = anchor_year_counts.values
            random_aligned.loc[random_year_counts.index] = random_year_counts.values

            try:
                ks_stat_temp, p_value_temp = kolmogorov_smirnov_test_temporal(
                    anchor_aligned.values,
                    random_aligned.values,
                    all_years,
                )
                # Calculate total sample sizes for temporal test
                anchor_temporal_size = int(anchor_aligned.sum())
                random_temporal_size = int(random_aligned.sum())

                interpretation_temp = interpret_ks_test(
                    ks_stat_temp,
                    p_value_temp,
                    sample_size_1=anchor_temporal_size,
                    sample_size_2=random_temporal_size,
                )

                logger.info("\n2. Temporal Distribution Comparison:")
                logger.info(f"   K-S Statistic: {ks_stat_temp:.6f}")

                # Create K-S test plot for temporal distribution
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
                    anchor_names_str = truncate_filename_component(anchor_names_str)
                    ks_plot_path = os.path.join(
                        output_dir,
                        f"ks_test_temporal_{anchor_names_str}_eps{epsilon:.2f}.png",
                    )
                    plot_ks_test_temporal_cdf(
                        anchor_year_counts,
                        random_year_counts,
                        ks_stat_temp,
                        p_value_temp,
                        output_path=ks_plot_path,
                        title=f"K-S Test: Temporal Distributions (ε={epsilon})",
                        interpretation=interpretation_temp,
                    )
            except Exception as e:
                logger.warning(f"Could not perform temporal K-S test: {e}")

        logger.info(f"\n{'=' * 60}")

    # Print top results
    logger.info(f"\n{'=' * 60}")
    logger.info("Top 20 movies in epsilon ball:")
    logger.info(f"{'=' * 60}")
    logger.info("")
    for _, row in results_df.head(20).iterrows():
        logger.info(
            f"Rank {row['rank']}: {row['title']} ({row['movie_id']}) - "
            f"Distance: {row['distance']:.6f}, Year: {row.get('year', 'N/A')}"
        )

    # Create plots
    if plot_over_time and not results_df.empty:
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
            anchor_names_str = truncate_filename_component(anchor_names_str)
            output_path = os.path.join(
                output_dir,
                f"epsilon_ball_over_time_{anchor_names_str}_eps{epsilon:.2f}.png",
            )
        plot_movies_over_time(
            results_df,
            output_path=output_path,
            title=f"Movies in Epsilon Ball (ε={epsilon}) Over Time",
            random_results_df=random_results_df,
        )

    if plot_distance_dist and not results_df.empty:
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
            anchor_names_str = truncate_filename_component(anchor_names_str)
            output_path = os.path.join(
                output_dir,
                f"epsilon_ball_distance_dist_{anchor_names_str}_eps{epsilon:.2f}.png",
            )
        plot_distance_distribution(
            results_df,
            output_path=output_path,
            title=f"Distance Distribution in Epsilon Ball (ε={epsilon})",
        )

    return results_df


if __name__ == "__main__":
    # ["Q18602670", "Q212145", "Q207916", "Q151904", "Q591272", "Q19089", "Q181540", "Q107914", "Q320423", "Q332368", "Q21534241", "Q30931", "Q106440"] # James Bond movies

    # ['Q1187607', 'Q488655', 'Q174992', 'Q785406', 'Q22350712'] # Edge of Tomorrow, Groundhog Day, 12:01, The Time Traveler's Wife, Before I Fall

    # ['Q507994', 'Q1140085', 'Q62730', 'Q244929', 'Q662342', 'Q1320705', 'Q1263897', 'Q1124501', 'Q1469774', 'Q747395', 'Q27888483'] # The Hunt for Red October, Crimson Tide, Das Boot, The Bedford Incident, K-19: The Widowmaker, Ice Station Zebra, The Enemy Below, By Dawn's Early Light, On the Beach, Hunter Killer

    # ['Q1366386', 'Q25136484', 'Q2171744', 'Q2364210', 'Q1490812', 'Q2746506', 'Q7617650', 'Q50650165'] # It, It Chapter Two, Killer Klowns from Outer Space, Clownhouse, Gacy, House of 1000 Corpses, Drive-Thru, Stitches, Terrifier

    # ['Q220735', 'Q261209', 'Q657079', 'Q7763422', 'Q110206', 'Q633171', 'Q4186834', 'Q1709419', 'Q760926', 'Q613485'] # The French Connection, Bullitt, Serpico, The Seven-Ups, Dirty Harry, The Taking of Pelham One Two Three, Prince of the City, Fort Apache, The Bronx, Cruising, To Live and Die in L.A.

    # ['Q192724', 'Q3820040', 'Q192724', 'Q466611', 'Q205028', 'Q217020', 'Q275120', 'Q3985737', 'Q494985', 'Q182218', 'Q209538', 'Q1201853', 'Q1765358', 'Q14171368', 'Q18407657', 'Q18406872', 'Q5887360', 'Q20001199', 'Q23010088', 'Q22665878', 'Q23780734', 'Q23780914', 'Q23781155', 'Q27985819'] # Iron Man, Iron Man 2, Iron Man 3, The Incredible Hulk, Thor, Captain America: The First Avenger, The Avengers, Iron Man 3, Thor: The Dark World, Captain America: The Winter Soldier, Avengers: Age of Ultron, Captain America: Civil War, Doctor Strange, Guardians of the Galaxy, Guardians of the Galaxy Vol. 2, Spider-Man: Homecoming, Thor: Ragnarok, Black Panther, Avengers: Infinity War, Avengers: Endgame, Spider-Man: Far From Home

    # ['Q221384', 'Q183066', 'Q152531', 'Q16970789', 'Q3258993', 'Q19865453', 'Q632328', 'Q848785', 'Q578312', 'Q1394447', 'Q17093105', 'Q20751325', 'Q63927168', 'Q21463782'] # Black Hawk Down, The Hurt Locker, Zero Dark Thirty, American Sniper, Lone Survivor, 13 Hours: The Secret Soldiers of Benghazi, Green Zone, Jarhead, Body of Lies, Restrepo, Korengal, Hyena Road, Kajaki (also released as Kilo Two Bravo), The Outpost, War Machine

    # most average movies # ['Q26683632', 'Q5932706', 'Q12671094', 'Q444057', 'Q17071466', 'Q554539', 'Q2549142', 'Q6872502', 'Q105441001', 'Q23755528']

    # ['Q41483', 'Q168154', 'Q245208', 'Q20092609', 'Q276769', 'Q1008351', 'Q50714', 'Q3208286', 'Q7596837', 'Q104137', 'Q8061777', 'Q47352417', 'Q994481', 'Q603263', 'Q326114', 'Q232000', 'Q153677', 'Q76479', 'Q241811', 'Q19069', 'Q19983487', 'Q247130', 'Q746029'] # The Good, the Bad and the Ugly, Once Upon a Time in the West, High Noon, The Searchers, Rio Bravo, Stagecoach, Unforgiven, The Wild Bunch, True Grit, Butch Cassidy and the Sundance Kid, For a Few Dollars More, A Fistful of Dollars, Django, The Magnificent Seven, 3:10 to Yuma

    # ["Q103474", "Q184843", "Q21500755", "Q162255", "Q170564", "Q16635326", "Q788822", "Q131191955", "Q221113", "Q83495", "Q189600", "Q207536", "Q200572", "Q504697", "Q626483", "Q18954", "Q244604", "Q1066948", "Q22575835", "Q10384115", "Q3549863", "Q30611788", "Q26751"] # AI Movies

    # ["Q103569", "Q104814", "Q20430699", "Q200804", "Q720357", "Q210756", "Q909749", "Q11621", "Q320588", "Q20382729", "Q1657967", "Q105387", "Q202028", "Q201819", "Q425992", "Q187154", "Q45386", "Q598818", "Q22432", "Q3205861", "Q336517", "Q25136228", "Q5164779", "Q270215", "Q15803822"] # Alien Movies

    results = main(
        anchor_qids=[
            "Q182692",
            "Q190643",
            "Q243439",
            "Q201674",
            "Q585203",
            "Q623502",
            "Q471159",
            "Q1126637",
            "Q180706",
            "Q478333",
            "Q1423020",
            "Q244876",
            "Q1114683",
            "Q15733016",
            "Q85842235",
            "Q1645944",
            "Q639864",
            "Q3520085",
            "Q112226601",
            "Q62277203",
        ],
        epsilon=0.24,
        start_year=1930,
        end_year=2024,
        anchor_method="average",  # or "medoid"
        exclude_anchors=True,
        compare_with_random=True,
        plot_over_time=True,
        plot_distance_dist=True,
        output_dir=f"{BASE_DIR}/figures/epsilon_ball_analysis",
    )
