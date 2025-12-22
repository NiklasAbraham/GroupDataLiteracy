"""
Epsilon ball analysis for finding movies within a distance threshold around anchor movies.

This script finds all movies within an epsilon distance (epsilon ball) around
specified anchor movies. The anchor can be a single movie or the average of
multiple movies. Results include distances, rankings, and temporal analysis.
"""

import hashlib
import logging
import os
import re
import sys

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
):
    """
    Plot the number of movies in the epsilon ball over time.

    Parameters:
    - results_df: DataFrame from analyze_epsilon_ball
    - output_path: Path to save the plot (if None, displays plot)
    - title: Plot title (will have total count appended)
    - figsize: Figure size tuple
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

    # Calculate SMA of 3 and SMA of 8
    year_counts_series = pd.Series(year_counts.values, index=year_counts.index)
    sma_3 = year_counts_series.rolling(window=3, center=False, min_periods=1).mean()
    sma_10 = year_counts_series.rolling(window=10, center=False, min_periods=1).mean()

    # Create plot
    plt.figure(figsize=figsize)
    plt.bar(year_counts.index, year_counts.values, alpha=0.7, edgecolor="black")
    plt.plot(sma_3.index, sma_3.values, color="red", linewidth=2, label="SMA (3)")
    plt.plot(
        sma_10.index, sma_10.values, color="darkred", linewidth=2, label="SMA (10)"
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Movies", fontsize=12)
    plt.title(f"{title} (Total: {total_movies} movies)", fontsize=14, fontweight="bold")
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
    - output_dir: Directory to save plots (if None, uses current directory)
    """
    if anchor_qids is None:
        # Example: James Bond movies
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

    # Run analysis
    results_df = analyze_epsilon_ball(
        anchor_qids=anchor_qids,
        epsilon=epsilon,
        filtered_embeddings=filtered_embeddings,
        filtered_movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        anchor_method=anchor_method,
        exclude_anchors=exclude_anchors,
    )

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
    # 1. QID: Q18602670 (Spectre)
    # 2. QID: Q212145 (The World Is Not Enough)
    # 3. QID: Q207916 (Tomorrow Never Dies)
    # 4. QID: Q151904 (Casino Royale)
    # 5. QID: Q591272 (Casino Royale)
    # 6. QID: Q19089 (GoldenEye)
    # 7. QID: Q181540 (Quantum of Solace)
    # 8. QID: Q107914 (Diamonds Are Forever)
    # 9. QID: Q320423 (The Spy Who Loved Me)
    # 10. QID: Q332368 (A View to a Kill)
    # 11. QID: Q21534241 (No Time to Die)
    # 12. QID: Q30931 (Die Another Day)
    # 13. QID: Q106440 (Goldfinger)
    # ["Q18602670", "Q212145", "Q207916", "Q151904", "Q591272", "Q19089", "Q181540", "Q107914", "Q320423", "Q332368", "Q21534241", "Q30931", "Q106440"]

    results = main(
        anchor_qids=["Q214801", "Q177930"],
        epsilon=0.33,
        start_year=1930,
        end_year=2024,
        anchor_method="average",  # or "medoid"
        exclude_anchors=True,
        plot_over_time=True,
        plot_distance_dist=True,
        output_dir=f"{BASE_DIR}/figures/epsilon_ball_analysis",
    )
