"""
Gaussian distribution analysis for movie embeddings.

This script performs comprehensive Gaussianity analysis on movie embeddings,
including Mahalanobis distance calculations, Q-Q plots, PCA analysis, and
outlier detection.
"""

import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from src.analysis.math_functions import (  # noqa: E402
    gaussian_analysis_with_embeddings,
)
from src.data_utils import (  # noqa: E402
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
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    robust: bool = True,
    alpha: float = 0.05,
    output_dir: str = None,
    n_samples: int = None,
):
    """
    Main function to perform Gaussianity analysis on movie embeddings.

    Parameters:
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - robust: If True, use robust covariance estimator to reduce outlier influence
    - alpha: Significance level for outlier detection (default: 0.05)
    - output_dir: Directory to save plots. If None, uses figures/ directory
    - n_samples: Number of samples to use for testing. If None, uses all available samples.
        If specified, randomly samples this many embeddings.
    """
    logger.info(f"{'=' * 60}")
    logger.info("Gaussianity Analysis for Movie Embeddings")
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

        # Sample subset if n_samples is specified
        if n_samples is not None and n_samples < len(filtered_movie_ids):
            logger.info(
                f"Sampling {n_samples} embeddings from {len(filtered_movie_ids)} available"
            )
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(
                len(filtered_movie_ids), size=n_samples, replace=False
            )
            filtered_embeddings = filtered_embeddings[sample_indices]
            filtered_movie_ids = filtered_movie_ids[sample_indices]
            logger.info(f"Using {len(filtered_movie_ids)} samples for analysis")

        # Perform Gaussianity analysis
        logger.info("Performing Gaussianity analysis...")
        results = gaussian_analysis_with_embeddings(
            embeddings=filtered_embeddings,
            movie_ids=filtered_movie_ids,
            movie_data=movie_data,
            robust=robust,
            alpha=alpha,
            output_dir=output_dir,
            prefix="gaussian_analysis",
        )

        # Print summary results
        logger.info(f"\n{'=' * 60}")
        logger.info("Gaussianity Analysis Results")
        logger.info(f"{'=' * 60}\n")

        logger.info(f"Number of samples: {results['n_samples']}")
        logger.info(f"Number of dimensions: {results['n_dims']}")
        logger.info("\nDistance Statistics:")
        logger.info(f"  Mean Mahalanobis distance²: {results['mean_distance']:.2f}")
        logger.info(f"  Expected distance² (d): {results['expected_distance']:.2f}")
        logger.info(f"  Std of distances²: {results['std_distance']:.2f}")
        logger.info(f"  Expected std: {results['expected_std']:.2f}")

        logger.info("\nGaussianity Assessment:")
        logger.info(
            f"  Is Gaussian likely: {'Yes' if results['is_gaussian_likely'] else 'No'}"
        )
        logger.info(
            f"  Mean representativeness score: {results['mean_representativeness']:.3f}"
        )
        logger.info("    (1.0 = perfect, lower = worse match)")

        logger.info(f"\nOutlier Detection (alpha = {alpha}):")
        logger.info(
            f"  Outlier threshold (distance²): {results['outlier_threshold']:.2f}"
        )
        logger.info(f"  Number of outliers detected: {len(results['outlier_indices'])}")
        logger.info(
            f"  Percentage of outliers: {100 * len(results['outlier_indices']) / results['n_samples']:.2f}%"
        )

        logger.info("\nPCA Analysis:")
        logger.info(
            f"  First 10 components explain: {results['pca_cumulative_variance'][9] * 100:.2f}%"
        )
        logger.info(
            f"  First 50 components explain: {results['pca_cumulative_variance'][min(49, len(results['pca_cumulative_variance']) - 1)] * 100:.2f}%"
        )

        # Print top outliers if available
        if "outlier_details" in results and len(results["outlier_details"]) > 0:
            logger.info("\nTop 10 Outliers (by Mahalanobis distance):")
            sorted_outliers = sorted(
                results["outlier_details"], key=lambda x: x[2], reverse=True
            )[:10]
            for i, (movie_id, title, distance, year) in enumerate(sorted_outliers, 1):
                logger.info(
                    f"  {i}. {title} ({movie_id}, {year}) - Distance: {distance:.2f}"
                )

        # Print plot file locations
        if "plot_files" in results and results["plot_files"]:
            logger.info("\nVisualization plots saved:")
            for plot_name, plot_path in results["plot_files"].items():
                logger.info(f"  {plot_name}: {plot_path}")

        logger.info(f"\n{'=' * 60}")

        return results

    except Exception as e:
        logger.error(f"Error performing Gaussianity analysis: {e}")
        raise


if __name__ == "__main__":
    # Example usage with 10,000 samples for testing
    main(
        start_year=1930,
        end_year=2024,
        robust=True,
        alpha=0.05,
        n_samples=5000,
    )
