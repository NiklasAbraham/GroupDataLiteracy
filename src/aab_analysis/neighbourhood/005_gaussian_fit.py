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

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

import importlib.util  # noqa: E402

from src.aab_analysis.math_functions.whitening import whiten_embeddings  # noqa: E402

gaussian_analysis_path = os.path.join(
    os.path.dirname(__file__), "001_gaussian_analysis.py"
)
gaussian_spec = importlib.util.spec_from_file_location(
    "gaussian_analysis", gaussian_analysis_path
)
gaussian_analysis = importlib.util.module_from_spec(gaussian_spec)
sys.modules["gaussian_analysis"] = gaussian_analysis
gaussian_spec.loader.exec_module(gaussian_analysis)
analyze_gaussianity = gaussian_analysis.analyze_gaussianity
create_gaussianity_plots = gaussian_analysis.create_gaussianity_plots
from src.utils.data_utils import load_final_dataset, load_final_dense_embeddings  # noqa: E402

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
    """De-bias embeddings using the 'All-but-the-top' approach."""
    from sklearn.decomposition import PCA

    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    pca = PCA(n_components=min(k, embeddings.shape[1], embeddings.shape[0] - 1))
    pca.fit(centered)

    top_pcs = pca.components_[:k]
    debiased = centered.copy()
    for i in range(k):
        projections = np.dot(centered, top_pcs[i])
        debiased = debiased - np.outer(projections, top_pcs[i])

    if normalize:
        norms = np.linalg.norm(debiased, axis=1, keepdims=True)
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
    """Generate a cache key based on parameters that affect expensive computations."""
    movie_ids_str = "_".join(sorted(map(str, movie_ids)))
    movie_ids_hash = hashlib.md5(movie_ids_str.encode()).hexdigest()[:8]

    n_samples_str = "all" if n_samples is None else str(n_samples)
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
    Perform Gaussianity analysis on movie embeddings.

    Parameters:
    - start_year, end_year: Year filtering
    - data_dir, csv_path: Data paths
    - robust: Use robust covariance estimator
    - alpha: Significance level for outlier detection (default: 0.05). Ignored if alphas is provided.
    - alphas: List of alpha values to test. If provided, will run analysis for each.
    - output_dir: Directory to save plots. If None, uses figures/ directory
    - n_samples: Number of samples to use. If None, uses all available samples.
    - random_seed: Random seed for sampling (default: 42)
    - cache_dir: Directory for caching expensive computations
    - debias: If True, de-bias embeddings using "All-but-the-top" approach
    - debias_k: Number of top principal components to remove (default: 3)
    - debias_normalize: If True, re-normalize embeddings after de-biasing
    - whiten: If True, whiten embeddings using PCA (overrides debias if both are True)
    - whiten_n_components: Number of PCA components to keep for whitening. If None, keeps all.
    - whiten_normalize: If True, re-normalize embeddings after whitening
    - qq_test_types: List of QQ plot test types. Options: 'distances', 'dimensions', 'pca_components'.
    """
    logger.info(f"{'=' * 60}")
    logger.info("Gaussianity Analysis for Movie Embeddings")
    logger.info(f"{'=' * 60}")

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    if cache_dir is None:
        cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

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

    if n_samples is not None and n_samples < len(filtered_movie_ids):
        logger.info(
            f"Sampling {n_samples} embeddings from {len(filtered_movie_ids)} available (seed={random_seed})"
        )
        np.random.seed(random_seed)
        sample_indices = np.random.choice(
            len(filtered_movie_ids), size=n_samples, replace=False
        )
        filtered_embeddings = filtered_embeddings[sample_indices]
        filtered_movie_ids = filtered_movie_ids[sample_indices]
        logger.info(f"Using {len(filtered_movie_ids)} samples for analysis")

    if whiten:
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

    if alphas is not None:
        alpha_list = alphas
        logger.info(f"Testing multiple alpha values: {alpha_list}")
    else:
        alpha_list = [alpha]
        logger.info(f"Using single alpha value: {alpha}")

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

    if qq_test_types is None:
        qq_test_types = ["distances"]

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

    base_results = _load_cached_results(cache_key, cache_dir)

    if base_results is None:
        logger.info("Performing Gaussianity analysis (computing distances and PCA)...")
        logger.info("(This may take a while - results will be cached for future use)")

        base_results = analyze_gaussianity(
            filtered_embeddings, robust=robust, alpha=alpha_list[0]
        )

        _save_cached_results(cache_key, base_results, cache_dir)
    else:
        logger.info(
            "Using cached base analysis results (skipping expensive computations)"
        )

    all_results = {}

    for current_alpha in alpha_list:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing alpha = {current_alpha}")
        logger.info(f"{'=' * 60}")

        n_dims = base_results["n_dims"]
        chi2_threshold = stats.chi2.ppf(1 - current_alpha, n_dims)
        outlier_indices = np.where(base_results["distances_squared"] > chi2_threshold)[
            0
        ]

        results = base_results.copy()
        results["outlier_indices"] = outlier_indices
        results["outlier_threshold"] = chi2_threshold

        alpha_str = str(current_alpha).replace(".", "_")
        prefix = f"gaussian_analysis_n{actual_n_samples}_{robust_str}_{transform_str}_alpha{alpha_str}"

        plot_files = create_gaussianity_plots(
            results,
            output_dir=output_dir,
            prefix=prefix,
            embeddings=filtered_embeddings,
            qq_test_types=qq_test_types,
        )

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
                pca_scatter_path = os.path.join(output_dir, f"{prefix}_pca_scatter.png")
                plt.savefig(pca_scatter_path, dpi=300, bbox_inches="tight")
                plot_files["pca_scatter"] = pca_scatter_path
            plt.close(fig)

        results["plot_files"] = plot_files

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

        logger.info(f"\nOutlier Detection (alpha = {current_alpha}):")
        logger.info(
            f"  Outlier threshold (distance²): {results['outlier_threshold']:.2f}"
        )
        logger.info(f"  Number of outliers detected: {len(results['outlier_indices'])}")
        logger.info(
            f"  Percentage of outliers: {100 * len(results['outlier_indices']) / results['n_samples']:.2f}%"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("All analyses completed!")
    logger.info(f"{'=' * 60}")

    if len(alpha_list) == 1:
        return all_results[alpha_list[0]]
    else:
        return all_results


if __name__ == "__main__":
    main(
        start_year=1930,
        end_year=2024,
        robust=True,
        alphas=[0.01, 0.05, 0.10],
        n_samples=5_000,
        random_seed=42,
        debias=False,
        whiten=True,
        whiten_n_components=None,
        whiten_normalize=False,
        qq_test_types=["distances", "dimensions", "pca_components"],
    )
