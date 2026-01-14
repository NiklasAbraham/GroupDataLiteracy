"""
Gaussian distribution analysis for high-dimensional embeddings.

This module provides functions to test whether embeddings follow a Gaussian
distribution, assess mean representativeness, and detect outliers using
Mahalanobis distance and statistical diagnostics.
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.linalg import LinAlgError
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA


def compute_mahalanobis_distances(
    embeddings: np.ndarray,
    mean: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Mahalanobis distances from the mean for each embedding.

    The Mahalanobis distance measures how many standard deviations away
    from the mean each point is, accounting for the covariance structure.

    Mathematical formulation:
    D_i^2 = (x_i - μ)^T Σ^(-1) (x_i - μ)

    For a true Gaussian in d dimensions:
    - Typical distances cluster around d (the dimensionality)
    - Much larger distances indicate outliers
    - Much smaller distances are also unusual

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
    - mean: Precomputed mean vector (shape: [embedding_dim]).
        If None, computed from embeddings.
    - cov: Precomputed covariance matrix (shape: [embedding_dim, embedding_dim]).
        If None, computed from embeddings.
    - robust: If True, use robust covariance estimator (MinCovDet) to reduce
        outlier influence. If False, use standard sample covariance.

    Returns:
    - distances_squared: Array of squared Mahalanobis distances (shape: [n_samples])
    - mean: The mean vector used (shape: [embedding_dim])
    - cov: The covariance matrix used (shape: [embedding_dim, embedding_dim])

    Raises:
    - LinAlgError: If covariance matrix is singular and cannot be inverted
    """
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided")

    n_samples, n_dims = embeddings.shape

    # Compute mean if not provided
    if mean is None:
        mean = np.mean(embeddings, axis=0)
    else:
        if mean.shape[0] != n_dims:
            raise ValueError(
                f"Mean dimension ({mean.shape[0]}) does not match "
                f"embedding dimension ({n_dims})"
            )

    # Compute covariance if not provided
    if cov is None:
        if robust:
            # Use robust covariance estimator (less sensitive to outliers)
            try:
                # Set support_fraction to avoid numerical instability warnings
                # Higher values (0.8-0.9) are more stable but less robust to outliers
                # Lower values (0.5-0.7) are more robust but can cause numerical issues
                support_fraction = min(0.8, max(0.5, 1.0 - 10.0 / n_samples))
                robust_cov = MinCovDet(
                    support_fraction=support_fraction,
                    random_state=42,
                )
                # Suppress the RuntimeWarning about determinant increase
                # This is a known numerical issue that doesn't affect the final result
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Determinant has increased",
                        category=RuntimeWarning,
                    )
                    robust_cov.fit(embeddings)
                cov = robust_cov.covariance_
            except Exception:
                # Fallback to standard covariance if robust fails
                # Use ddof=1 for sample covariance (unbiased estimator)
                cov = np.cov(embeddings.T, ddof=1)
        else:
            # Use ddof=1 for sample covariance (unbiased estimator)
            cov = np.cov(embeddings.T, ddof=1)
    else:
        if cov.shape != (n_dims, n_dims):
            raise ValueError(
                f"Covariance shape ({cov.shape}) does not match "
                f"expected ({n_dims}, {n_dims})"
            )

    # Check for singular covariance matrix
    try:
        cov_inv = np.linalg.inv(cov)
    except LinAlgError:
        # Try pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov)

    # Compute Mahalanobis distances
    # D^2 = (x - μ)^T Σ^(-1) (x - μ)
    centered = embeddings - mean
    distances_squared = np.sum(centered @ cov_inv * centered, axis=1)

    return distances_squared, mean, cov


def analyze_gaussianity(
    embeddings: np.ndarray,
    robust: bool = True,
    alpha: float = 0.05,
) -> Dict:
    """
    Perform comprehensive Gaussianity analysis on embeddings.

    This function tests whether embeddings follow a Gaussian distribution by:
    1. Computing Mahalanobis distances from the mean
    2. Comparing distances to expected chi-square distribution (Q-Q plot)
    3. Analyzing PCA structure to detect low-dimensional manifolds
    4. Identifying outliers based on distance thresholds

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
    - robust: If True, use robust covariance estimator to reduce outlier influence
    - alpha: Significance level for outlier detection (default: 0.05)

    Returns:
    - Dictionary containing:
        - 'distances_squared': Squared Mahalanobis distances
        - 'mean': Mean vector
        - 'cov': Covariance matrix
        - 'n_samples': Number of samples
        - 'n_dims': Number of dimensions
        - 'outlier_indices': Indices of detected outliers
        - 'outlier_threshold': Distance threshold for outliers
        - 'pca_explained_variance': Explained variance by each PC
        - 'pca_cumulative_variance': Cumulative explained variance
        - 'is_gaussian_likely': Boolean indicating if data appears Gaussian
        - 'mean_representativeness': Score indicating how well mean represents data
    """
    n_samples, n_dims = embeddings.shape

    # Warn if n_samples < n_dims (covariance will be singular)
    if n_samples < n_dims:
        import warnings

        warnings.warn(
            f"n_samples ({n_samples}) < n_dims ({n_dims}). "
            "Covariance matrix will be singular. Using pseudo-inverse.",
            UserWarning,
        )

    # Compute Mahalanobis distances
    distances_squared, mean, cov = compute_mahalanobis_distances(
        embeddings, robust=robust
    )

    # Expected value for chi-square distribution with d degrees of freedom
    # For a true Gaussian, E[D^2] = d
    expected_distance_squared = n_dims

    # Outlier detection using chi-square distribution
    # For d dimensions, D^2 should follow chi-square(d) distribution
    chi2_threshold = stats.chi2.ppf(1 - alpha, n_dims)
    outlier_indices = np.where(distances_squared > chi2_threshold)[0]

    # PCA analysis to check for low-dimensional structure
    pca = PCA()
    pca.fit(embeddings)
    pca_explained_variance = pca.explained_variance_ratio_
    pca_cumulative_variance = np.cumsum(pca_explained_variance)

    # Check if first few components capture too much variance
    # (indicates non-Gaussian structure)
    first_10_pct = (
        pca_cumulative_variance[9] if n_dims > 9 else pca_cumulative_variance[-1]
    )
    is_gaussian_likely = first_10_pct < 0.5  # Rough heuristic

    # Mean representativeness: check if distances cluster around expected value
    mean_distance = np.mean(distances_squared)
    std_distance = np.std(distances_squared)
    # For chi-square(d), variance is 2d, so std is sqrt(2d)
    expected_std = np.sqrt(2 * n_dims)
    # Score: how close is our distribution to expected
    # 1.0 = perfect match, lower = worse match
    mean_representativeness = 1.0 - min(
        1.0, abs(mean_distance - expected_distance_squared) / expected_distance_squared
    )

    results = {
        "distances_squared": distances_squared,
        "mean": mean,
        "cov": cov,
        "n_samples": n_samples,
        "n_dims": n_dims,
        "outlier_indices": outlier_indices,
        "outlier_threshold": chi2_threshold,
        "pca_explained_variance": pca_explained_variance,
        "pca_cumulative_variance": pca_cumulative_variance,
        "is_gaussian_likely": is_gaussian_likely,
        "mean_representativeness": mean_representativeness,
        "mean_distance": mean_distance,
        "expected_distance": expected_distance_squared,
        "std_distance": std_distance,
        "expected_std": expected_std,
    }

    return results


def create_gaussianity_plots(
    analysis_results: Dict,
    output_dir: Optional[str] = None,
    prefix: str = "gaussian_analysis",
    embeddings: Optional[np.ndarray] = None,
    qq_test_types: list = None,
) -> Dict[str, str]:
    """
    Create visualization plots for Gaussianity analysis.

    Creates multiple plots:
    1. Q-Q plots (Mahalanobis distances, dimensions, PCA components)
    2. PCA scatter plot with outlier highlighting
    3. Histogram of distances (mean representativeness)

    Parameters:
    - analysis_results: Dictionary returned by analyze_gaussianity()
    - output_dir: Directory to save plots. If None, plots are not saved.
    - prefix: Prefix for output filenames
    - embeddings: Optional array of embeddings for additional QQ plots
    - qq_test_types: List of QQ plot types to create. Options:
        - 'distances': QQ plot for Mahalanobis distances (default, always created)
        - 'dimensions': QQ plots for individual embedding dimensions
        - 'pca_components': QQ plots for first few PCA components

    Returns:
    - Dictionary mapping plot names to file paths (if saved) or empty dict
    """
    distances_squared = analysis_results["distances_squared"]
    outlier_indices = analysis_results["outlier_indices"]
    n_dims = analysis_results["n_dims"]
    pca_explained_variance = analysis_results["pca_explained_variance"]
    pca_cumulative_variance = analysis_results["pca_cumulative_variance"]

    saved_files = {}

    if qq_test_types is None:
        qq_test_types = ["distances"]

    # 1. Q-Q plot: Compare observed distances to expected chi-square distribution
    if "distances" in qq_test_types:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Theoretical quantiles from chi-square distribution
        theoretical_quantiles = stats.chi2.ppf(
            np.linspace(0.01, 0.99, len(distances_squared)), n_dims
        )
        observed_quantiles = np.sort(distances_squared)

        ax.scatter(
            theoretical_quantiles,
            observed_quantiles,
            alpha=0.6,
            s=20,
            label="Observed vs Expected",
        )

        # Perfect Gaussian line (y = x)
        min_val = min(theoretical_quantiles.min(), observed_quantiles.min())
        max_val = max(theoretical_quantiles.max(), observed_quantiles.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Gaussian (y=x)",
        )

        ax.set_xlabel("Theoretical Quantiles (Chi-square)", fontsize=12)
        ax.set_ylabel("Observed Quantiles (Mahalanobis Distance²)", fontsize=12)
        ax.set_title(
            f"Q-Q Plot: Gaussianity Check (d={n_dims})\n"
            f"Points on line → Gaussian; Deviations → Non-Gaussian",
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_dir:
            import os

            qq_path = os.path.join(output_dir, f"{prefix}_qq_plot_distances.png")
            plt.savefig(qq_path, dpi=300, bbox_inches="tight")
            saved_files["qq_plot_distances"] = qq_path
        plt.close(fig)

    # QQ plots for individual dimensions
    if "dimensions" in qq_test_types and embeddings is not None:
        n_samples = embeddings.shape[0]
        n_dims_to_test = min(10, n_dims)  # Test first 10 dimensions

        # Create subplot grid
        n_cols = 3
        n_rows = (n_dims_to_test + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for dim_idx in range(n_dims_to_test):
            ax = axes[dim_idx]
            dim_data = embeddings[:, dim_idx]

            # Standardize the dimension
            mean_dim = np.mean(dim_data)
            std_dim = np.std(dim_data)
            if std_dim > 1e-10:
                standardized = (dim_data - mean_dim) / std_dim
            else:
                standardized = dim_data - mean_dim

            # Theoretical quantiles from standard normal
            theoretical_quantiles = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(standardized))
            )
            observed_quantiles = np.sort(standardized)

            ax.scatter(
                theoretical_quantiles,
                observed_quantiles,
                alpha=0.5,
                s=10,
            )

            # Perfect Gaussian line
            min_val = min(theoretical_quantiles.min(), observed_quantiles.min())
            max_val = max(theoretical_quantiles.max(), observed_quantiles.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=1.5,
            )

            ax.set_xlabel("Theoretical Quantiles (Normal)", fontsize=9)
            ax.set_ylabel("Observed Quantiles", fontsize=9)
            ax.set_title(f"Dimension {dim_idx + 1}", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_dims_to_test, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Q-Q Plots: Individual Dimensions (First {n_dims_to_test})\n"
            f"Testing Gaussianity of each embedding dimension",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_dir:
            import os

            qq_path = os.path.join(output_dir, f"{prefix}_qq_plot_dimensions.png")
            plt.savefig(qq_path, dpi=300, bbox_inches="tight")
            saved_files["qq_plot_dimensions"] = qq_path
        plt.close(fig)

    # QQ plots for PCA components
    if "pca_components" in qq_test_types and embeddings is not None:
        from sklearn.decomposition import PCA

        # Fit PCA
        pca = PCA()
        pca_components = pca.fit_transform(embeddings)

        n_components_to_test = min(10, pca_components.shape[1])

        # Create subplot grid
        n_cols = 3
        n_rows = (n_components_to_test + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for comp_idx in range(n_components_to_test):
            ax = axes[comp_idx]
            comp_data = pca_components[:, comp_idx]

            # Standardize the component
            mean_comp = np.mean(comp_data)
            std_comp = np.std(comp_data)
            if std_comp > 1e-10:
                standardized = (comp_data - mean_comp) / std_comp
            else:
                standardized = comp_data - mean_comp

            # Theoretical quantiles from standard normal
            theoretical_quantiles = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(standardized))
            )
            observed_quantiles = np.sort(standardized)

            ax.scatter(
                theoretical_quantiles,
                observed_quantiles,
                alpha=0.5,
                s=10,
            )

            # Perfect Gaussian line
            min_val = min(theoretical_quantiles.min(), observed_quantiles.min())
            max_val = max(theoretical_quantiles.max(), observed_quantiles.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=1.5,
            )

            explained_var = pca.explained_variance_ratio_[comp_idx] * 100
            ax.set_xlabel("Theoretical Quantiles (Normal)", fontsize=9)
            ax.set_ylabel("Observed Quantiles", fontsize=9)
            ax.set_title(f"PC{comp_idx + 1} ({explained_var:.2f}% var)", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_components_to_test, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Q-Q Plots: PCA Components (First {n_components_to_test})\n"
            f"Testing Gaussianity of principal components",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_dir:
            import os

            qq_path = os.path.join(output_dir, f"{prefix}_qq_plot_pca_components.png")
            plt.savefig(qq_path, dpi=300, bbox_inches="tight")
            saved_files["qq_plot_pca_components"] = qq_path
        plt.close(fig)

    # 2. PCA scatter plot with outlier highlighting
    # Note: This requires embeddings, which we don't have in results
    # We'll create a plot showing PCA variance structure instead
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Explained variance by component
    n_components_plot = min(50, len(pca_explained_variance))
    ax1.bar(
        range(1, n_components_plot + 1),
        pca_explained_variance[:n_components_plot] * 100,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance (%)", fontsize=12)
    ax1.set_title("PCA Explained Variance (First 50 Components)", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Cumulative explained variance
    ax2.plot(
        range(1, min(100, len(pca_cumulative_variance)) + 1),
        pca_cumulative_variance[: min(100, len(pca_cumulative_variance))] * 100,
        linewidth=2,
        color="steelblue",
    )
    ax2.axhline(y=50, color="r", linestyle="--", label="50% threshold")
    ax2.axhline(y=90, color="orange", linestyle="--", label="90% threshold")
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax2.set_title("Cumulative Explained Variance", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if output_dir:
        import os

        pca_path = os.path.join(output_dir, f"{prefix}_pca_analysis.png")
        plt.savefig(pca_path, dpi=300, bbox_inches="tight")
        saved_files["pca_analysis"] = pca_path
    plt.close(fig)

    # 3. Histogram of distances
    fig, ax = plt.subplots(figsize=(12, 6))

    distances = np.sqrt(distances_squared)  # Convert to actual distances
    expected_distance = np.sqrt(n_dims)

    ax.hist(
        distances,
        bins=50,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Observed Distances",
    )

    # Mark expected value
    ax.axvline(
        expected_distance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Expected (√d = {expected_distance:.2f})",
    )

    # Mark outliers
    outlier_threshold = np.sqrt(analysis_results["outlier_threshold"])
    ax.axvline(
        outlier_threshold,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Outlier Threshold ({outlier_threshold:.2f})",
    )

    ax.set_xlabel("Mahalanobis Distance", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Distribution of Mahalanobis Distances\n"
        f"Mean: {np.mean(distances):.2f}, Expected: {expected_distance:.2f}",
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_dir:
        import os

        hist_path = os.path.join(output_dir, f"{prefix}_distance_histogram.png")
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        saved_files["distance_histogram"] = hist_path
    plt.close(fig)

    return saved_files


def gaussian_analysis_with_embeddings(
    embeddings: np.ndarray,
    movie_ids: Optional[np.ndarray] = None,
    movie_data=None,
    robust: bool = True,
    alpha: float = 0.05,
    output_dir: Optional[str] = None,
    prefix: str = "gaussian_analysis",
) -> Dict:
    """
    Complete Gaussianity analysis with visualizations including PCA scatter plots.

    This is the main function that performs all Gaussianity diagnostics and
    creates comprehensive visualizations, including PCA scatter plots colored
    by outlier status.

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
    - movie_ids: Optional array of movie IDs for labeling outliers
    - movie_data: Optional DataFrame with movie metadata
    - robust: If True, use robust covariance estimator
    - alpha: Significance level for outlier detection
    - output_dir: Directory to save plots. If None, plots are not saved.
    - prefix: Prefix for output filenames

    Returns:
    - Dictionary containing all analysis results plus visualization file paths
    """
    # Perform analysis
    results = analyze_gaussianity(embeddings, robust=robust, alpha=alpha)

    # Create standard plots
    plot_files = create_gaussianity_plots(
        results,
        output_dir=output_dir,
        prefix=prefix,
        embeddings=embeddings,
    )

    # Create PCA scatter plot with outlier highlighting
    if embeddings.shape[0] > 1:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot all points
        outlier_mask = np.zeros(embeddings.shape[0], dtype=bool)
        outlier_mask[results["outlier_indices"]] = True

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
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)", fontsize=12
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)", fontsize=12
        )
        ax.set_title(
            "PCA Visualization with Outlier Highlighting",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_dir:
            import os

            pca_scatter_path = os.path.join(output_dir, f"{prefix}_pca_scatter.png")
            plt.savefig(pca_scatter_path, dpi=300, bbox_inches="tight")
            plot_files["pca_scatter"] = pca_scatter_path
        plt.close(fig)

    # Add plot files to results
    results["plot_files"] = plot_files

    # Add outlier details if movie data is available
    if movie_ids is not None and movie_data is not None:
        outlier_details = []
        for idx in results["outlier_indices"]:
            movie_id = movie_ids[idx]
            distance = np.sqrt(results["distances_squared"][idx])
            movie_row = movie_data[movie_data["movie_id"] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0].get("title", "Unknown")
                year = movie_row.iloc[0].get("year", None)
                outlier_details.append((movie_id, title, distance, year))
            else:
                outlier_details.append((movie_id, "Unknown", distance, None))
        results["outlier_details"] = outlier_details

    return results
