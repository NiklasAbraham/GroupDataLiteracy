"""
Gaussian distribution analysis for movie embeddings.

This script performs comprehensive Gaussianity analysis on movie embeddings,
including Mahalanobis distance calculations, Q-Q plots, PCA analysis, and
outlier detection.
"""

import hashlib
import logging
import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from src.utils.data_utils import (  # noqa: E402
    load_final_dataset,
    load_final_dense_embeddings,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "gaussian_analysis_cache")
START_YEAR = 1930
END_YEAR = 2024


def _debias_embeddings(
    embeddings: np.ndarray, k: int = 3, normalize: bool = False
) -> np.ndarray:
    """
    De-bias embeddings using the "All-but-the-top" approach.

    This removes global anisotropy/cone by projecting out the top k principal
    components, while preserving relative covariance and mean differences that
    might encode real temporal structure.

    Steps:
    1. Fit PCA on the full embedding set
    2. Compute global mean μ and top k PCs u1,...,uk
    3. For each embedding x:
       - Mean-center: x ← x - μ
       - Project out dominant directions: x' = x - Σ⟨x, u_i⟩u_i
       - Optionally re-normalize: x' ← x' / ||x'||

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
    - k: Number of top principal components to remove (default: 3, typically 1-5)
    - normalize: If True, re-normalize embeddings to unit length after debiasing

    Returns:
    - De-biased embeddings (shape: [n_samples, embedding_dim])
    """
    from sklearn.decomposition import PCA

    n_samples, n_dims = embeddings.shape

    # Compute global mean
    mean = np.mean(embeddings, axis=0)

    # Mean-center the data
    centered = embeddings - mean

    # Fit PCA on the full embedding set
    pca = PCA(n_components=min(k, n_dims, n_samples - 1))
    pca.fit(centered)

    # Get top k principal components (u1, ..., uk)
    # pca.components_ has shape (n_components, n_features)
    # Each row is a principal component
    top_pcs = pca.components_[:k]  # Shape: (k, n_dims)

    # For each embedding, project out the dominant directions
    # x' = x - Σ⟨x, u_i⟩u_i for i=1 to k
    debiased = centered.copy()
    for i in range(k):
        # Compute projection onto PC i: ⟨x, u_i⟩
        projections = np.dot(centered, top_pcs[i])  # Shape: (n_samples,)
        # Subtract the projection: x - ⟨x, u_i⟩u_i
        debiased = debiased - np.outer(projections, top_pcs[i])

    # Optionally re-normalize to unit length
    if normalize:
        norms = np.linalg.norm(debiased, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        debiased = debiased / norms

    return debiased


def _generate_cache_key(
    start_year: int,
    end_year: int,
    robust: bool,
    n_samples: int,
    random_seed: int,
    movie_ids: np.ndarray,
    debias: bool = False,
    debias_k: int = 3,
    debias_normalize: bool = False,
    whiten: bool = False,
    whiten_n_components: int = None,
    whiten_normalize: bool = True,
) -> str:
    """
    Generate a cache key based on parameters that affect expensive computations.

    Parameters:
    - start_year, end_year: Year filtering
    - robust: Covariance estimation method
    - n_samples: Number of samples (None means all)
    - random_seed: Random seed for sampling
    - movie_ids: Array of movie IDs to ensure same set
    - debias: Whether de-biasing is applied
    - debias_k: Number of top PCs to remove in de-biasing
    - debias_normalize: Whether to normalize after de-biasing
    - whiten: Whether whitening is applied
    - whiten_n_components: Number of PCA components for whitening
    - whiten_normalize: Whether to normalize after whitening

    Returns:
    - Cache key string
    """
    # Create a hash of the movie_ids to ensure same set
    movie_ids_str = "_".join(sorted(map(str, movie_ids)))
    movie_ids_hash = hashlib.md5(movie_ids_str.encode()).hexdigest()[:8]

    # Build cache key from parameters
    n_samples_str = "all" if n_samples is None else str(n_samples)
    robust_str = "robust" if robust else "standard"

    # Determine transformation type
    if whiten:
        norm_str = "norm" if whiten_normalize else "nonorm"
        comp_str = (
            f"ncomp{whiten_n_components}"
            if whiten_n_components is not None
            else "allcomp"
        )
        transform_str = f"whiten_{comp_str}_{norm_str}"
    elif debias:
        norm_str = "norm" if debias_normalize else "nonorm"
        transform_str = f"debias_k{debias_k}_{norm_str}"
    else:
        transform_str = "raw"

    cache_key = (
        f"gaussian_base_"
        f"y{start_year}_{end_year}_"
        f"n{n_samples_str}_"
        f"seed{random_seed}_"
        f"{robust_str}_"
        f"{transform_str}_"
        f"ids{movie_ids_hash}"
    )
    return cache_key


def _load_cached_results(cache_key: str, cache_dir: str) -> dict:
    """Load cached base analysis results if they exist."""
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cached_results(cache_key: str, base_results: dict, cache_dir: str) -> None:
    """Save base analysis results to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    logger.info(f"Saving cached results to {cache_path}")

    # Only save the expensive-to-compute parts
    cache_data = {
        "distances_squared": base_results["distances_squared"],
        "mean": base_results["mean"],
        "cov": base_results["cov"],
        "n_samples": base_results["n_samples"],
        "n_dims": base_results["n_dims"],
        "pca_explained_variance": base_results["pca_explained_variance"],
        "pca_cumulative_variance": base_results["pca_cumulative_variance"],
        "is_gaussian_likely": base_results["is_gaussian_likely"],
        "mean_representativeness": base_results["mean_representativeness"],
        "mean_distance": base_results["mean_distance"],
        "expected_distance": base_results["expected_distance"],
        "std_distance": base_results["std_distance"],
        "expected_std": base_results["expected_std"],
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)


def main(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    robust: bool = True,
    alpha: float = 0.05,
    alphas: list = None,
    output_dir: str = None,
    n_samples: int = None,
    random_seed: int = 42,
    cache_dir: str = None,
    debias: bool = False,
    debias_k: int = 3,
    debias_normalize: bool = False,
    whiten: bool = False,
    whiten_n_components: int = None,
    whiten_normalize: bool = True,
    qq_test_types: list = None,
):
    """
    Main function to perform Gaussianity analysis on movie embeddings.

    Parameters:
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - robust: If True, use robust covariance estimator to reduce outlier influence
    - alpha: Significance level for outlier detection (default: 0.05).
        Ignored if alphas is provided.
    - alphas: List of alpha values to test. If provided, will run analysis for each.
        The expensive computations (distances, PCA) are done once and reused.
    - output_dir: Directory to save plots. If None, uses figures/ directory
    - n_samples: Number of samples to use for testing. If None, uses all available samples.
        If specified, randomly samples this many embeddings.
    - random_seed: Random seed for sampling (default: 42)
    - cache_dir: Directory for caching expensive computations. If None, uses data/gaussian_analysis_cache
    - debias: If True, de-bias embeddings using "All-but-the-top" approach before analysis
    - debias_k: Number of top principal components to remove (default: 3, typically 1-5)
    - debias_normalize: If True, re-normalize embeddings to unit length after de-biasing
    - whiten: If True, whiten embeddings using PCA before analysis (overrides debias if both are True)
    - whiten_n_components: Number of PCA components to keep for whitening. If None, keeps all.
    - whiten_normalize: If True, re-normalize embeddings to unit length after whitening
    - qq_test_types: List of QQ plot test types. Options: 'distances', 'dimensions', 'pca_components'.
        If None, defaults to ['distances']. 'dimensions' tests individual embedding dimensions,
        'pca_components' tests first few PCA components.
    """
    logger.info(f"{'=' * 60}")
    logger.info("Gaussianity Analysis for Movie Embeddings")
    logger.info(f"{'=' * 60}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Set cache directory
    if cache_dir is None:
        cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

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
                f"Sampling {n_samples} embeddings from {len(filtered_movie_ids)} available (seed={random_seed})"
            )
            np.random.seed(random_seed)  # For reproducibility
            sample_indices = np.random.choice(
                len(filtered_movie_ids), size=n_samples, replace=False
            )
            filtered_embeddings = filtered_embeddings[sample_indices]
            filtered_movie_ids = filtered_movie_ids[sample_indices]
            logger.info(f"Using {len(filtered_movie_ids)} samples for analysis")

        # Apply transformation (whitening takes precedence over debias)
        if whiten:
            from src.analysis.math_functions.whitening import whiten_embeddings

            logger.info(
                f"Whitening embeddings (n_components={whiten_n_components}, normalize={whiten_normalize})..."
            )
            filtered_embeddings = whiten_embeddings(
                filtered_embeddings,
                n_components=whiten_n_components,
                normalize=whiten_normalize,
            )
            logger.info("Whitening completed")
        elif debias:
            logger.info(
                f"De-biasing embeddings (removing top {debias_k} PCs, normalize={debias_normalize})..."
            )
            filtered_embeddings = _debias_embeddings(
                filtered_embeddings, k=debias_k, normalize=debias_normalize
            )
            logger.info("De-biasing completed")

        # Determine which alphas to test
        if alphas is not None:
            alpha_list = alphas
            logger.info(f"Testing multiple alpha values: {alpha_list}")
        else:
            alpha_list = [alpha]
            logger.info(f"Using single alpha value: {alpha}")

        # Create base prefix with number of samples and parameter settings
        actual_n_samples = len(filtered_movie_ids)
        robust_str = "robust" if robust else "standard"
        if whiten:
            norm_str = "norm" if whiten_normalize else "nonorm"
            comp_str = (
                f"ncomp{whiten_n_components}"
                if whiten_n_components is not None
                else "allcomp"
            )
            transform_str = f"whiten_{comp_str}_{norm_str}"
        elif debias:
            norm_str = "norm" if debias_normalize else "nonorm"
            transform_str = f"debias_k{debias_k}_{norm_str}"
        else:
            transform_str = "raw"

        # Set default QQ test types if not provided
        if qq_test_types is None:
            qq_test_types = ["distances"]

        # Check for cached results
        cache_key = _generate_cache_key(
            start_year=start_year,
            end_year=end_year,
            robust=robust,
            n_samples=n_samples,
            random_seed=random_seed,
            movie_ids=filtered_movie_ids,
            debias=debias,
            debias_k=debias_k,
            debias_normalize=debias_normalize,
            whiten=whiten,
            whiten_n_components=whiten_n_components,
            whiten_normalize=whiten_normalize,
        )

        # Try to load cached results
        base_results = _load_cached_results(cache_key, cache_dir)

        if base_results is None:
            # Perform expensive analysis once (distances, PCA don't depend on alpha)
            logger.info(
                "Performing Gaussianity analysis (computing distances and PCA)..."
            )
            logger.info(
                "(This may take a while - results will be cached for future use)"
            )

            from src.analysis.math_functions.gaussian_analysis import (
                analyze_gaussianity,
            )

            # Run base analysis with first alpha (just to get the structure)
            base_results = analyze_gaussianity(
                filtered_embeddings, robust=robust, alpha=alpha_list[0]
            )

            # Save to cache
            _save_cached_results(cache_key, base_results, cache_dir)
        else:
            logger.info(
                "Using cached base analysis results (skipping expensive computations)"
            )

        import matplotlib.pyplot as plt
        from scipy import stats
        from sklearn.decomposition import PCA

        from src.analysis.math_functions.gaussian_analysis import (
            create_gaussianity_plots,
        )

        all_results = {}

        # For each alpha, recompute outlier detection and generate plots
        for current_alpha in alpha_list:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing alpha = {current_alpha}")
            logger.info(f"{'=' * 60}")

            # Recompute outlier detection for this alpha (cheap operation)
            n_dims = base_results["n_dims"]
            chi2_threshold = stats.chi2.ppf(1 - current_alpha, n_dims)
            outlier_indices = np.where(
                base_results["distances_squared"] > chi2_threshold
            )[0]

            # Update results with alpha-specific outlier information
            results = base_results.copy()
            results["outlier_indices"] = outlier_indices
            results["outlier_threshold"] = chi2_threshold

            # Create prefix with alpha-specific information
            alpha_str = str(current_alpha).replace(".", "_")
            prefix = f"gaussian_analysis_n{actual_n_samples}_{robust_str}_{transform_str}_alpha{alpha_str}"

            # Generate plots for this alpha
            plot_files = create_gaussianity_plots(
                results, output_dir=output_dir, prefix=prefix
            )

            # Create PCA scatter plot with outlier highlighting
            if filtered_embeddings.shape[0] > 1:
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(filtered_embeddings)

                fig, ax = plt.subplots(figsize=(12, 10))

                outlier_mask = np.zeros(filtered_embeddings.shape[0], dtype=bool)
                outlier_mask[outlier_indices] = True

                ax.scatter(
                    embeddings_2d[~outlier_mask, 0],
                    embeddings_2d[~outlier_mask, 1],
                    alpha=0.5,
                    s=20,
                    c="steelblue",
                    label="Normal",
                )

                if np.any(outlier_mask):
                    ax.scatter(
                        embeddings_2d[outlier_mask, 0],
                        embeddings_2d[outlier_mask, 1],
                        alpha=0.8,
                        s=50,
                        c="red",
                        marker="x",
                        label=f"Outliers (n={np.sum(outlier_mask)})",
                    )

                ax.set_xlabel(
                    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)",
                    fontsize=12,
                )
                ax.set_ylabel(
                    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)",
                    fontsize=12,
                )
                ax.set_title(
                    "PCA Visualization with Outlier Highlighting",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                if output_dir:
                    pca_scatter_path = os.path.join(
                        output_dir, f"{prefix}_pca_scatter.png"
                    )
                    plt.savefig(pca_scatter_path, dpi=300, bbox_inches="tight")
                    plot_files["pca_scatter"] = pca_scatter_path
                plt.close(fig)

            results["plot_files"] = plot_files

            # Add outlier details if movie data is available
            if filtered_movie_ids is not None and movie_data is not None:
                outlier_details = []
                for idx in outlier_indices:
                    movie_id = filtered_movie_ids[idx]
                    distance = np.sqrt(results["distances_squared"][idx])
                    movie_row = movie_data[movie_data["movie_id"] == movie_id]
                    if not movie_row.empty:
                        title = movie_row.iloc[0].get("title", "Unknown")
                        year = movie_row.iloc[0].get("year", None)
                        outlier_details.append((movie_id, title, distance, year))
                    else:
                        outlier_details.append((movie_id, "Unknown", distance, None))
                results["outlier_details"] = outlier_details

            all_results[current_alpha] = results

            # Print summary results for this alpha
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Gaussianity Analysis Results (alpha = {current_alpha})")
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

            logger.info(f"\nOutlier Detection (alpha = {current_alpha}):")
            logger.info(
                f"  Outlier threshold (distance²): {results['outlier_threshold']:.2f}"
            )
            logger.info(
                f"  Number of outliers detected: {len(results['outlier_indices'])}"
            )
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
                for i, (movie_id, title, distance, year) in enumerate(
                    sorted_outliers, 1
                ):
                    logger.info(
                        f"  {i}. {title} ({movie_id}, {year}) - Distance: {distance:.2f}"
                    )

            # Print plot file locations
            if "plot_files" in results and results["plot_files"]:
                logger.info("\nVisualization plots saved:")
                for plot_name, plot_path in results["plot_files"].items():
                    logger.info(f"  {plot_name}: {plot_path}")

        logger.info(f"\n{'=' * 60}")
        logger.info("All analyses completed!")
        logger.info(f"{'=' * 60}")

        # Return single result if only one alpha, otherwise return dict
        if len(alpha_list) == 1:
            return all_results[alpha_list[0]]
        else:
            return all_results

    except Exception as e:
        logger.error(f"Error performing Gaussianity analysis: {e}")
        raise


if __name__ == "__main__":
    # Example usage with recommended whitening test parameters
    # The expensive computations (distances, PCA) are done once and reused

    # Recommended whitening test configurations:
    # 1. Full whitening (all components, normalized) - Best for cosine similarity analysis
    #    - whiten=True, whiten_n_components=None, whiten_normalize=True
    # 2. Reduced whitening (200 components, normalized) - Removes noise in high dimensions
    #    - whiten=True, whiten_n_components=200, whiten_normalize=True
    # 3. Full whitening without normalization - Preserves whitened scale for Euclidean analysis
    #    - whiten=True, whiten_n_components=None, whiten_normalize=False

    # Recommended QQ test types:
    # - ['distances']: Original Mahalanobis distance test (fastest)
    # - ['distances', 'dimensions']: Also test individual embedding dimensions
    # - ['distances', 'dimensions', 'pca_components']: Comprehensive Gaussianity assessment

    main(
        start_year=1930,
        end_year=2024,
        robust=True,
        alphas=[0.01, 0.05, 0.10],  # Test multiple alpha values
        n_samples=5_000,
        random_seed=42,
        debias=False,  # Set to False when using whitening (whitening takes precedence)
        debias_k=3,
        debias_normalize=False,
        # Recommended whitening settings:
        whiten=True,  # Enable whitening
        whiten_n_components=None,  # None = use all components (full whitening)
        # Alternative: 200 for dimensionality reduction
        whiten_normalize=False,  # True = normalize to unit sphere (recommended for cosine similarity)
        # False = preserve whitened scale (for Euclidean distances)
        # Recommended QQ test types:
        qq_test_types=[
            "distances",
            "dimensions",
            "pca_components",
        ],  # Comprehensive testing
        cache_dir=CACHE_DIR,
        output_dir=os.path.join(BASE_DIR, "figures"),
        data_dir=os.path.join(BASE_DIR, "data", "data_final"),
        csv_path=os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv"),
    )

    # Alternative configurations to test:
    #
    # Configuration 1: Reduced dimensionality whitening
    # main(
    #     start_year=1930,
    #     end_year=2024,
    #     robust=True,
    #     alphas=[0.05],
    #     n_samples=5_000,
    #     random_seed=42,
    #     whiten=True,
    #     whiten_n_components=200,  # Keep only top 200 components
    #     whiten_normalize=True,
    #     qq_test_types=["distances", "pca_components"],
    #     cache_dir=CACHE_DIR,
    #     output_dir=os.path.join(BASE_DIR, "figures"),
    #     data_dir=os.path.join(BASE_DIR, "data", "data_final"),
    #     csv_path=os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv"),
    # )
    #
    # Configuration 2: Whitening without normalization
    # main(
    #     start_year=1930,
    #     end_year=2024,
    #     robust=True,
    #     alphas=[0.05],
    #     n_samples=5_000,
    #     random_seed=42,
    #     whiten=True,
    #     whiten_n_components=None,
    #     whiten_normalize=False,  # Preserve whitened scale
    #     qq_test_types=["distances"],
    #     cache_dir=CACHE_DIR,
    #     output_dir=os.path.join(BASE_DIR, "figures"),
    #     data_dir=os.path.join(BASE_DIR, "data", "data_final"),
    #     csv_path=os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv"),
    # )
    #
    # Configuration 3: No transformation (baseline)
    # main(
    #     start_year=1930,
    #     end_year=2024,
    #     robust=True,
    #     alphas=[0.05],
    #     n_samples=5_000,
    #     random_seed=42,
    #     whiten=False,
    #     debias=False,
    #     qq_test_types=["distances", "dimensions", "pca_components"],
    #     cache_dir=CACHE_DIR,
    #     output_dir=os.path.join(BASE_DIR, "figures"),
    #     data_dir=os.path.join(BASE_DIR, "data", "data_final"),
    #     csv_path=os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv"),
    # )

    # nohup python src/analysis/gaussian_fit.py > gaussian_fit.log 2>&1 &
