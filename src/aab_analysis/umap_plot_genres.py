"""
UMAP Trajectory Plot for Genre Clusters Over Time

This script samples 5000 movies, computes UMAP space, and plots trajectories
showing how specified genre clusters move through UMAP space over time by
calculating the mean position for each cluster per year.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from data_utils
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


def plot_cluster_trajectories(
    cluster_names: list,
    n_samples: int = 5000,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    output_path: str = None,
    random_state: int = 42,
):
    """
    Plot trajectories of specified clusters through UMAP space over time.

    Parameters:
    - cluster_names: List of cluster names (e.g., ['drama', 'comedy']) to plot trajectories for
    - n_samples: Number of movies to sample for UMAP calculation (default: 5000)
    - start_year: First year to include (default: 1930)
    - end_year: Last year to include (default: 2024)
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - output_path: Path to save the plot (if None, saves to data_dir)
    - random_state: Random state for reproducibility (default: 42)

    Returns:
    - Dictionary with trajectory data for each cluster
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Plotting trajectories for clusters: {cluster_names}")
    logger.info(f"{'=' * 60}")

    # Load all embeddings and corresponding movie IDs
    logger.info("Loading embeddings...")
    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")

    logger.info(f"Total movies with embeddings: {len(all_movie_ids)}")
    logger.info(f"Embedding shape: {all_embeddings.shape}")

    # Load movie metadata
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

    # Create mapping from movie_id to genre_cluster_names
    movie_to_clusters = {}
    movie_id_to_year = dict(zip(movie_data["movie_id"], movie_data["year"]))

    for idx, row in movie_data.iterrows():
        movie_id = row["movie_id"]
        genre_cluster_names = row.get("genre_cluster_names", "")

        # Extract cluster names (comma-separated)
        if pd.notna(genre_cluster_names) and str(genre_cluster_names).strip():
            clusters = [
                c.strip()
                for c in str(genre_cluster_names).split(",")
                if c.strip() and c.strip() != ""
            ]
            movie_to_clusters[movie_id] = clusters
        else:
            movie_to_clusters[movie_id] = []

    # Filter embeddings to only include movies in the filtered dataset
    valid_movie_ids = set(movie_data["movie_id"].values)
    valid_indices = np.array(
        [i for i, mid in enumerate(all_movie_ids) if mid in valid_movie_ids]
    )

    if len(valid_indices) == 0:
        raise ValueError(
            f"No movies found in the specified year range ({start_year}-{end_year})"
        )

    # Filter embeddings and movie_ids
    filtered_embeddings = all_embeddings[valid_indices]
    filtered_movie_ids = all_movie_ids[valid_indices]

    logger.info(
        f"Filtered to {len(filtered_movie_ids)} movies with embeddings in year range"
    )

    # Sample n_samples movies
    n_samples = min(n_samples, len(filtered_movie_ids))
    logger.info(f"Sampling {n_samples} movies for UMAP calculation...")
    np.random.seed(random_state)
    sample_indices = np.random.choice(
        len(filtered_movie_ids), size=n_samples, replace=False
    )

    sampled_embeddings = filtered_embeddings[sample_indices]
    sampled_movie_ids = filtered_movie_ids[sample_indices]

    # Get years for sampled movies
    sampled_years = np.array(
        [movie_id_to_year.get(mid, -1) for mid in sampled_movie_ids]
    )

    # Remove movies where year lookup failed
    valid_year_mask = sampled_years != -1
    sampled_embeddings = sampled_embeddings[valid_year_mask]
    sampled_movie_ids = sampled_movie_ids[valid_year_mask]
    sampled_years = sampled_years[valid_year_mask]

    logger.info(f"After filtering: {len(sampled_movie_ids)} movies with valid years")

    # Compute UMAP reduction
    logger.info("Computing UMAP reduction...")
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_embedding = reducer.fit_transform(sampled_embeddings)

    logger.info(f"UMAP embedding shape: {umap_embedding.shape}")

    # For each cluster, calculate mean position per year
    trajectories = {}
    years_range = np.arange(start_year, end_year + 1)

    for cluster_name in cluster_names:
        logger.info(f"Processing cluster: {cluster_name}")

        # Find movies that belong to this cluster
        cluster_mask = np.array(
            [
                cluster_name in movie_to_clusters.get(mid, [])
                for mid in sampled_movie_ids
            ]
        )

        if np.sum(cluster_mask) == 0:
            logger.warning(f"No movies found for cluster: {cluster_name}")
            trajectories[cluster_name] = None
            continue

        # Calculate mean position per year
        mean_positions = []
        years_with_data = []

        for year in years_range:
            year_mask = (sampled_years == year) & cluster_mask
            year_indices = np.where(year_mask)[0]

            if len(year_indices) > 0:
                year_umap_positions = umap_embedding[year_indices]
                mean_pos = np.mean(year_umap_positions, axis=0)
                mean_positions.append(mean_pos)
                years_with_data.append(year)
                logger.info(
                    f"  Year {year}: {len(year_indices)} movies, mean position: ({mean_pos[0]:.4f}, {mean_pos[1]:.4f})"
                )

        if len(mean_positions) > 0:
            trajectories[cluster_name] = {
                "positions": np.array(mean_positions),
                "years": np.array(years_with_data),
                "n_movies": np.sum(cluster_mask),
            }
        else:
            logger.warning(f"No data points found for cluster: {cluster_name}")
            trajectories[cluster_name] = None

    # Plot trajectories
    logger.info("Creating plot...")
    plt.figure(figsize=(14, 10))

    # Plot all sampled points as background (light gray)
    plt.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c="lightgray",
        s=5,
        alpha=0.3,
        label="All sampled movies",
    )

    # Plot trajectories for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, cluster_name in enumerate(cluster_names):
        if trajectories[cluster_name] is None:
            continue

        traj_data = trajectories[cluster_name]
        positions = traj_data["positions"]
        years = traj_data["years"]

        # Choose color
        color = colors[i % len(colors)]

        # Plot trajectory line
        plt.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=6,
            label=f"{cluster_name} (n={traj_data['n_movies']})",
            alpha=0.8,
        )

        # Add year labels for first and last points
        if len(positions) > 0:
            # First point
            plt.annotate(
                str(years[0]),
                (positions[0, 0], positions[0, 1]),
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
            )
            # Last point
            if len(positions) > 1:
                plt.annotate(
                    str(years[-1]),
                    (positions[-1, 0], positions[-1, 1]),
                    fontsize=8,
                    alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                )

    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.title(
        f"Trajectories of Genre Clusters in UMAP Space Over Time\n"
        f"(Sample size: {n_samples}, Years: {start_year}-{end_year})",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = os.path.join(data_dir, "umap_cluster_trajectories.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to: {output_path}")

    return trajectories


def main(
    cluster_names: list = None,
    n_samples: int = 5000,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    output_path: str = None,
    random_state: int = 42,
):
    """
    Main function to plot cluster trajectories.

    Parameters:
    - cluster_names: List of cluster names to plot (e.g., ['drama', 'comedy'])
    - n_samples: Number of movies to sample for UMAP calculation
    - start_year: First year to include
    - end_year: Last year to include
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - output_path: Path to save the plot
    - random_state: Random state for reproducibility
    """
    if cluster_names is None:
        cluster_names = ["drama", "comedy"]

    try:
        trajectories = plot_cluster_trajectories(
            cluster_names=cluster_names,
            n_samples=n_samples,
            start_year=start_year,
            end_year=end_year,
            data_dir=data_dir,
            csv_path=csv_path,
            output_path=output_path,
            random_state=random_state,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info("Trajectory summary:")
        for cluster_name, traj_data in trajectories.items():
            if traj_data is not None:
                logger.info(
                    f"  {cluster_name}: {len(traj_data['years'])} years, {traj_data['n_movies']} movies"
                )
            else:
                logger.info(f"  {cluster_name}: No data")
        logger.info(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Error plotting trajectories: {e}")
        raise


if __name__ == "__main__":
    # Example usage - modify cluster_names as needed
    main(
        cluster_names=[
            "drama",
            "comedy",
            "action",
        ],  # Change these to your desired cluster names
        n_samples=5000,
    )
