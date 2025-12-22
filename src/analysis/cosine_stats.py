"""
Cosine Statistics Analysis for Hyperspherical Embedding Geometry.

This script performs comprehensive cosine statistics analysis on normalized embeddings
to diagnose global angular dispersion, directional statistics, spectral geometry,
and cluster structure on the hypersphere.
"""

import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from src.analysis.math_functions.whitening import debias_embeddings
from src.data_utils import (
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


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.

    Parameters:
    - embeddings: Array of shape (n_samples, embedding_dim)

    Returns:
    - Normalized embeddings of same shape
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def plot_cosine_distance_normal_fit(
    cosine_similarities: np.ndarray,
    output_path: str = None,
    bins: int = 50,
    n_std: int = 3,
    n_runs: int = 1,
    n_samples: int = None,
) -> None:
    """
    Plot histogram of cosine distances with fitted normal distribution.

    Parameters:
    - cosine_similarities: Array of pairwise cosine similarities
    - output_path: Path to save the plot. If None, displays the plot.
    - bins: Number of bins for the histogram
    - n_std: Number of standard deviations to mark (default: 3)
    - n_runs: Number of runs (for title)
    - n_samples: Number of samples per run (for title)
    """
    # Convert cosine similarities to cosine distances
    # Cosine distance = 1 - cosine similarity
    distances = 1.0 - cosine_similarities

    # Fit normal distribution
    mean = np.mean(distances)
    std = np.std(distances)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram
    n, bins_edges, patches = ax.hist(
        distances,
        bins=bins,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        density=True,
        label="Cosine Distances",
    )

    # Create x values for fitted normal curve
    x_min = distances.min()
    x_max = distances.max()
    x_range = x_max - x_min
    x_curve = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
    y_curve = stats.norm.pdf(x_curve, mean, std)

    # Plot fitted normal distribution
    ax.plot(
        x_curve,
        y_curve,
        "r-",
        linewidth=2,
        label=f"Fitted Normal (μ={mean:.4f}, σ={std:.4f})",
    )

    # Mark mean and standard deviations
    colors_std = ["green", "orange", "purple", "brown"]

    # Mark mean
    ax.axvline(
        mean,
        color=colors_std[0],
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean:.4f}",
    )

    # Mark standard deviations
    for i in range(1, n_std + 1):
        color = colors_std[i] if i < len(colors_std) else "gray"
        ax.axvline(
            mean + i * std,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"+{i}σ = {mean + i * std:.4f}",
        )
        ax.axvline(
            mean - i * std,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"-{i}σ = {mean - i * std:.4f}",
        )

    # Set labels and title
    ax.set_xlabel("Cosine Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Build title with run information
    title_parts = [
        "Cosine Distance Distribution with Fitted Normal",
    ]
    if n_runs > 1:
        title_parts.append(f"({n_runs} runs")
        if n_samples is not None:
            title_parts.append(f", {n_samples} samples/run")
        title_parts.append(")")
    title_parts.append(
        f"\nMean: {mean:.4f}, Std: {std:.4f}, Min: {distances.min():.4f}, "
        f"Max: {distances.max():.4f}, Median: {np.median(distances):.4f}"
    )

    ax.set_title(
        "".join(title_parts),
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cosine distance normal fit plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def compute_mean_pairwise_cosine(
    embeddings: np.ndarray, sample_pairs: int = None, random_seed: int = None
) -> float:
    """
    Compute mean pairwise cosine similarity.

    μ_cos = 2/(n(n-1)) * Σ_{i<j} x_i^T x_j

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - sample_pairs: Number of pairs to sample for large datasets. If None, uses all pairs.
    - random_seed: Random seed for sampling pairs. If None, uses current random state.

    Returns:
    - Mean pairwise cosine similarity
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    if sample_pairs is None or sample_pairs >= n * (n - 1) // 2:
        # Compute full pairwise similarity matrix
        cosine_matrix = cosine_similarity(embeddings)
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu_indices(n, k=1)
        cosines = cosine_matrix[upper_triangle]
    else:
        # Sample pairs for efficiency
        if random_seed is not None:
            np.random.seed(random_seed)
        indices_i = np.random.choice(n, size=sample_pairs, replace=True)
        indices_j = np.random.choice(n, size=sample_pairs, replace=True)
        # Avoid self-pairs
        valid_mask = indices_i != indices_j
        indices_i = indices_i[valid_mask]
        indices_j = indices_j[valid_mask]
        if len(indices_i) == 0:
            return 0.0
        cosines = np.sum(embeddings[indices_i] * embeddings[indices_j], axis=1)

    return np.mean(cosines)


def compute_cosine_variance(
    embeddings: np.ndarray, sample_pairs: int = None, random_seed: int = None
) -> float:
    """
    Compute variance of pairwise cosine similarities.

    σ²_cos = Var(x_i^T x_j)

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - sample_pairs: Number of pairs to sample for large datasets. If None, uses all pairs.
    - random_seed: Random seed for sampling pairs. If None, uses current random state.

    Returns:
    - Variance of pairwise cosine similarities
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    if sample_pairs is None or sample_pairs >= n * (n - 1) // 2:
        cosine_matrix = cosine_similarity(embeddings)
        upper_triangle = np.triu_indices(n, k=1)
        cosines = cosine_matrix[upper_triangle]
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
        indices_i = np.random.choice(n, size=sample_pairs, replace=True)
        indices_j = np.random.choice(n, size=sample_pairs, replace=True)
        valid_mask = indices_i != indices_j
        indices_i = indices_i[valid_mask]
        indices_j = indices_j[valid_mask]
        if len(indices_i) == 0:
            return 0.0
        cosines = np.sum(embeddings[indices_i] * embeddings[indices_j], axis=1)

    return np.var(cosines)


def compute_angular_std(
    embeddings: np.ndarray, sample_pairs: int = None, random_seed: int = None
) -> float:
    """
    Compute angular standard deviation.

    Transform: θ_ij = arccos(x_i^T x_j)
    Return: std(θ_ij)

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - sample_pairs: Number of pairs to sample for large datasets. If None, uses all pairs.
    - random_seed: Random seed for sampling pairs. If None, uses current random state.

    Returns:
    - Standard deviation of angles in radians
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    if sample_pairs is None or sample_pairs >= n * (n - 1) // 2:
        cosine_matrix = cosine_similarity(embeddings)
        upper_triangle = np.triu_indices(n, k=1)
        cosines = cosine_matrix[upper_triangle]
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
        indices_i = np.random.choice(n, size=sample_pairs, replace=True)
        indices_j = np.random.choice(n, size=sample_pairs, replace=True)
        valid_mask = indices_i != indices_j
        indices_i = indices_i[valid_mask]
        indices_j = indices_j[valid_mask]
        if len(indices_i) == 0:
            return 0.0
        cosines = np.sum(embeddings[indices_i] * embeddings[indices_j], axis=1)

    # Clip to [-1, 1] to avoid numerical issues with arccos
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    # Filter out any invalid values (NaN or inf) that might occur due to numerical issues
    valid_angles = angles[np.isfinite(angles)]
    if len(valid_angles) == 0:
        return 0.0
    return np.std(valid_angles)


def compute_mean_resultant_length(embeddings: np.ndarray) -> float:
    """
    Compute Mean Resultant Length (MRL).

    R = |(1/n) Σ_{i=1}^n x_i|

    If uniformly spread on S^(d-1), R ≈ 0.
    If collapsed into narrow region, R → 1.

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)

    Returns:
    - Mean resultant length (scalar)
    """
    n = len(embeddings)
    if n == 0:
        return 0.0
    mean_vector = np.mean(embeddings, axis=0)
    return np.linalg.norm(mean_vector)


def rayleigh_test_uniformity(
    embeddings: np.ndarray, return_p_value: bool = True
) -> tuple:
    """
    Perform Rayleigh test for uniformity on the hypersphere.

    Tests the null hypothesis that embeddings are uniformly distributed on the sphere.

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - return_p_value: If True, return p-value; otherwise just test statistic

    Returns:
    - If return_p_value: (R, test_statistic, p_value)
    - Otherwise: (R, test_statistic)
    """
    n = len(embeddings)
    if n == 0:
        if return_p_value:
            return (0.0, 0.0, 1.0)
        return (0.0, 0.0)

    d = embeddings.shape[1]
    R = compute_mean_resultant_length(embeddings)

    # Rayleigh test statistic for high dimensions
    # Approximate test: n * d * R^2 ~ chi-square(d) under uniformity
    test_statistic = n * d * (R**2)

    if return_p_value:
        # Under uniformity, n*d*R^2 approximately follows chi-square(d)
        p_value = 1 - stats.chi2.cdf(test_statistic, d)
        return (R, test_statistic, p_value)
    else:
        return (R, test_statistic)


def compute_cosine_pca(embeddings: np.ndarray, n_components: int = 50) -> dict:
    """
    Perform PCA on the cosine similarity matrix.

    NOTE: This is DIFFERENT from standard PCA on embeddings!
    - Standard PCA: Analyzes the raw embedding vectors (n_samples × embedding_dim)
      → Measures variance structure within the embedding space
      → Typically shows low explained variance (e.g., 26% for 20 components)

    - Cosine Matrix PCA: Analyzes the pairwise cosine similarity matrix (n_samples × n_samples)
      → Measures structure in how samples relate to each other via cosine similarity
      → Often shows high explained variance (e.g., 74% for 10 components)
      → Indicates few dominant patterns in the relationship structure

    Computes PCA on Cos(X) = [x_i^T x_j]_{i,j}
    Sharp eigenvalue decay means cosine geometry is governed by few latent directions.

    Parameters:
    - embeddings: Normalized embeddings (n_samples, embedding_dim)
    - n_components: Number of components to compute

    Returns:
    - Dictionary with eigenvalues, explained variance, cumulative variance
    """
    n = len(embeddings)
    if n == 0:
        return {
            "eigenvalues": np.array([]),
            "explained_variance": np.array([]),
            "cumulative_variance": np.array([]),
        }

    # Compute cosine similarity matrix
    cosine_matrix = cosine_similarity(embeddings)

    # Center the cosine matrix (subtract mean)
    # This is important for PCA interpretation
    mean_cosine = np.mean(cosine_matrix)
    centered_cosine = cosine_matrix - mean_cosine

    # Perform PCA on centered cosine matrix
    n_components = min(n_components, n - 1, n)
    pca = PCA(n_components=n_components)
    pca.fit(centered_cosine)

    eigenvalues = pca.explained_variance_
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    return {
        "eigenvalues": eigenvalues,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "n_components": n_components,
    }


def analyze_cosine_statistics(
    embeddings: np.ndarray, sample_pairs: int = None, random_seed: int = None
) -> dict:
    """
    Perform comprehensive cosine statistics analysis.

    Parameters:
    - embeddings: Array of embeddings (n_samples, embedding_dim)
    - sample_pairs: Number of pairs to sample for large datasets. If None, uses all pairs.
    - random_seed: Random seed for sampling pairs. If None, uses current random state.

    Returns:
    - Dictionary containing all statistics
    """
    # Normalize embeddings
    normalized_embeddings = normalize_embeddings(embeddings)

    n_samples, n_dims = normalized_embeddings.shape

    # Compute global angular dispersion metrics
    mean_cosine = compute_mean_pairwise_cosine(
        normalized_embeddings, sample_pairs, random_seed=random_seed
    )
    cosine_var = compute_cosine_variance(
        normalized_embeddings, sample_pairs, random_seed=random_seed
    )
    angular_std = compute_angular_std(
        normalized_embeddings, sample_pairs, random_seed=random_seed
    )

    # Directional statistics
    mrl = compute_mean_resultant_length(normalized_embeddings)
    R, rayleigh_stat, rayleigh_p = rayleigh_test_uniformity(
        normalized_embeddings, return_p_value=True
    )

    # Spectral geometry
    cosine_pca = compute_cosine_pca(normalized_embeddings)

    results = {
        "n_samples": n_samples,
        "n_dims": n_dims,
        "mean_cosine": mean_cosine,
        "cosine_variance": cosine_var,
        "angular_std": angular_std,
        "mean_resultant_length": mrl,
        "rayleigh_test_statistic": rayleigh_stat,
        "rayleigh_p_value": rayleigh_p,
        "cosine_pca": cosine_pca,
    }

    return results


def create_cosine_plots(
    results: dict, output_dir: str = None, prefix: str = "cosine_stats"
) -> dict:
    """
    Create visualization plots for cosine statistics analysis.

    Parameters:
    - results: Dictionary returned by analyze_cosine_statistics()
    - output_dir: Directory to save plots. If None, plots are not saved.
    - prefix: Prefix for output filenames

    Returns:
    - Dictionary mapping plot names to file paths
    """
    saved_files = {}

    # 1. Cosine similarity histogram
    # We need to compute pairwise cosines for the histogram
    # For efficiency, sample if dataset is large
    n_samples = results["n_samples"]
    sample_pairs = min(10000, n_samples * (n_samples - 1) // 2)

    # We'll compute this in the main function and pass it

    # 2. Angular distribution histogram
    # Similar - will compute in main

    # 3. PCA on cosine matrix - explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cosine_pca = results["cosine_pca"]
    explained_var = cosine_pca["explained_variance"]
    cumulative_var = cosine_pca["cumulative_variance"]

    n_components_plot = min(50, len(explained_var))
    ax1.bar(
        range(1, n_components_plot + 1),
        explained_var[:n_components_plot] * 100,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance (%)", fontsize=12)
    ax1.set_title("PCA on Cosine Matrix - Explained Variance", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    n_comp_cumulative = min(100, len(cumulative_var))
    ax2.plot(
        range(1, n_comp_cumulative + 1),
        cumulative_var[:n_comp_cumulative] * 100,
        linewidth=2,
        color="steelblue",
    )
    ax2.axhline(y=50, color="r", linestyle="--", label="50% threshold")
    ax2.axhline(y=90, color="orange", linestyle="--", label="90% threshold")
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax2.set_title("PCA on Cosine Matrix - Cumulative Variance", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if output_dir:
        pca_path = os.path.join(output_dir, f"{prefix}_pca_cosine.png")
        plt.savefig(pca_path, dpi=300, bbox_inches="tight")
        saved_files["pca_cosine"] = pca_path
    plt.close(fig)

    return saved_files


def create_raw_vs_whitened_comparison(
    all_raw_embeddings: list,
    raw_cosines: np.ndarray,
    raw_angles: np.ndarray,
    raw_results: list,
    output_dir: str = None,
    prefix: str = "cosine_stats",
    sample_pairs: int = 10000,
    random_seed: int = 42,
) -> dict:
    """
    Create comparison plots between raw and debiased embeddings.

    Uses the "All-but-the-top" debiasing approach (k=3) to remove global anisotropy.

    Creates:
    1. Cosine similarity distributions (raw vs debiased)
    2. Angular distributions (raw vs debiased)
    3. Rayleigh test comparison (bar plot)

    Parameters:
    - all_raw_embeddings: List of raw embedding arrays from each run
    - raw_cosines: Combined pairwise cosines from raw embeddings
    - raw_angles: Combined pairwise angles from raw embeddings
    - raw_results: List of statistics dictionaries from raw embeddings
    - output_dir: Directory to save plots
    - prefix: Prefix for output filenames
    - sample_pairs: Number of pairs to sample for statistics
    - random_seed: Base random seed

    Returns:
    - Dictionary mapping plot names to file paths
    """
    saved_files = {}

    # Concatenate all raw embeddings
    all_raw = np.concatenate(all_raw_embeddings, axis=0)

    # Debiasing all raw embeddings together (using All-but-the-top approach)
    logger.info("Debiasing embeddings for comparison (All-but-the-top, k=3)...")
    debiased_embeddings = debias_embeddings(all_raw, k=3, normalize=True)

    # Compute statistics for debiased embeddings
    logger.info("Computing statistics for debiased embeddings...")
    debiased_normalized = normalize_embeddings(debiased_embeddings)

    # Sample pairs for debiased embeddings (use same number as raw)
    n = len(debiased_normalized)
    if n * (n - 1) // 2 > sample_pairs * len(all_raw_embeddings):
        np.random.seed(random_seed)
        total_sample_pairs = sample_pairs * len(all_raw_embeddings)
        indices_i = np.random.choice(n, size=total_sample_pairs, replace=True)
        indices_j = np.random.choice(n, size=total_sample_pairs, replace=True)
        valid_mask = indices_i != indices_j
        indices_i = indices_i[valid_mask]
        indices_j = indices_j[valid_mask]
        if len(indices_i) > 0:
            debiased_cosines = np.sum(
                debiased_normalized[indices_i] * debiased_normalized[indices_j], axis=1
            )
        else:
            debiased_cosines = np.array([])
    else:
        cosine_matrix = cosine_similarity(debiased_normalized)
        upper_triangle = np.triu_indices(n, k=1)
        debiased_cosines = cosine_matrix[upper_triangle]

    # Compute angles for debiased
    debiased_cosines_clipped = np.clip(debiased_cosines, -1.0, 1.0)
    debiased_angles = np.arccos(debiased_cosines_clipped)
    debiased_angles = debiased_angles[np.isfinite(debiased_angles)]

    # Compute statistics for debiased
    debiased_mean_cosine = np.mean(debiased_cosines)
    debiased_cosine_var = np.var(debiased_cosines)
    debiased_angular_std = np.std(debiased_angles)
    debiased_mrl = compute_mean_resultant_length(debiased_normalized)
    debiased_R, debiased_rayleigh_stat, debiased_rayleigh_p = rayleigh_test_uniformity(
        debiased_normalized, return_p_value=True
    )

    # Raw statistics (averaged across runs)
    raw_mean_cosine = np.mean([r["mean_cosine"] for r in raw_results])
    raw_cosine_var = np.var(raw_cosines)
    raw_angular_std = np.std(raw_angles)
    raw_mrl = np.mean([r["mean_resultant_length"] for r in raw_results])
    raw_rayleigh_p = np.mean([r["rayleigh_p_value"] for r in raw_results])

    # Update variable names for consistency in plots
    whitened_mean_cosine = debiased_mean_cosine
    whitened_cosines = debiased_cosines
    whitened_angles = debiased_angles
    whitened_mrl = debiased_mrl
    whitened_rayleigh_p = debiased_rayleigh_p

    # 1. Cosine similarity distributions - two panel layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Raw embeddings
    ax1.hist(
        raw_cosines,
        bins=50,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Raw Embeddings",
    )
    ax1.axvline(
        raw_mean_cosine,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {raw_mean_cosine:.4f}",
    )
    ax1.set_xlabel("Cosine Similarity", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(
        f"Raw Embeddings\nMean: {raw_mean_cosine:.4f}, Std: {np.std(raw_cosines):.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 1.0)

    # Panel B: Debiased embeddings
    ax2.hist(
        whitened_cosines,
        bins=50,
        alpha=0.7,
        color="orange",
        edgecolor="black",
        label="Debiased Embeddings",
    )
    ax2.axvline(
        whitened_mean_cosine,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {whitened_mean_cosine:.4f}",
    )
    ax2.set_xlabel("Cosine Similarity", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(
        f"Debiased Embeddings (All-but-the-top, k=3)\nMean: {whitened_mean_cosine:.4f}, Std: {np.std(whitened_cosines):.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.2, 1.0)

    plt.suptitle(
        "Cosine Similarity Distributions: Raw vs Debiased",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    cosine_comp_path = os.path.join(output_dir, f"{prefix}_raw_vs_debiased_cosine.png")
    plt.savefig(cosine_comp_path, dpi=300, bbox_inches="tight")
    saved_files["cosine_comparison"] = cosine_comp_path
    plt.close(fig)

    # 2. Angular distributions - two panel layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Convert angles to degrees
    raw_angles_deg = np.degrees(raw_angles)
    whitened_angles_deg = np.degrees(whitened_angles)

    # Panel A: Raw embeddings
    ax1.hist(
        raw_angles_deg,
        bins=50,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Raw Embeddings",
    )
    ax1.axvline(
        np.mean(raw_angles_deg),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {np.mean(raw_angles_deg):.2f}°",
    )
    ax1.set_xlabel("Angle (degrees)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(
        f"Raw Embeddings\nMean: {np.mean(raw_angles_deg):.2f}°, Std: {np.std(raw_angles_deg):.2f}°",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Debiased embeddings
    ax2.hist(
        whitened_angles_deg,
        bins=50,
        alpha=0.7,
        color="orange",
        edgecolor="black",
        label="Debiased Embeddings",
    )
    ax2.axvline(
        np.mean(whitened_angles_deg),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {np.mean(whitened_angles_deg):.2f}°",
    )
    ax2.set_xlabel("Angle (degrees)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(
        f"Debiased Embeddings (All-but-the-top, k=3)\nMean: {np.mean(whitened_angles_deg):.2f}°, Std: {np.std(whitened_angles_deg):.2f}°",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Angular Distributions: Raw vs Debiased", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    angle_comp_path = os.path.join(output_dir, f"{prefix}_raw_vs_debiased_angles.png")
    plt.savefig(angle_comp_path, dpi=300, bbox_inches="tight")
    saved_files["angle_comparison"] = angle_comp_path
    plt.close(fig)

    # 3. Rayleigh test comparison - bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Raw Embeddings", "Debiased Embeddings"]
    mrl_values = [raw_mrl, whitened_mrl]
    colors = ["steelblue", "orange"]

    bars = ax.bar(categories, mrl_values, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for i, (bar, mrl, p_val) in enumerate(
        zip(bars, mrl_values, [raw_rayleigh_p, whitened_rayleigh_p])
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"R = {mrl:.4f}\np = {p_val:.2e}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Mean Resultant Length (R)", fontsize=12)
    ax.set_title(
        "Directional Concentration: Raw vs Debiased\n(Rayleigh Test, All-but-the-top k=3)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(mrl_values) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Add interpretation text
    interpretation = (
        f"Raw: R={raw_mrl:.4f} (uniformity {'rejected' if raw_rayleigh_p < 0.05 else 'not rejected'})\n"
        f"Debiased: R={whitened_mrl:.4f} (uniformity {'rejected' if whitened_rayleigh_p < 0.05 else 'not rejected'})"
    )
    ax.text(
        0.5,
        0.98,
        interpretation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    rayleigh_path = os.path.join(output_dir, f"{prefix}_raw_vs_debiased_rayleigh.png")
    plt.savefig(rayleigh_path, dpi=300, bbox_inches="tight")
    saved_files["rayleigh_comparison"] = rayleigh_path
    plt.close(fig)

    return saved_files


def main(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    output_dir: str = None,
    n_samples: int = 5000,
    n_runs: int = 3,
    random_seed: int = 42,
    sample_pairs: int = 10000,
):
    """
    Main function to perform cosine statistics analysis on movie embeddings.

    Parameters:
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - output_dir: Directory to save plots. If None, uses figures/ directory
    - n_samples: Number of samples to use for testing
    - n_runs: Number of independent runs to perform
    - random_seed: Base random seed (each run will use seed + run_number)
    - sample_pairs: Number of pairs to sample for pairwise statistics on large datasets
    """
    logger.info(f"{'=' * 60}")
    logger.info("Cosine Statistics Analysis for Hyperspherical Embeddings")
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

        all_results = []
        all_cosines = []
        all_angles = []
        all_raw_embeddings = []  # Store raw embeddings for whitening comparison

        # Perform n_runs independent analyses
        for run_idx in range(n_runs):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Run {run_idx + 1} of {n_runs}")
            logger.info(f"{'=' * 60}")

            # Sample subset with different seed for each run
            current_seed = random_seed + run_idx
            if n_samples is not None and n_samples < len(filtered_embeddings):
                logger.info(
                    f"Sampling {n_samples} embeddings from {len(filtered_embeddings)} available (seed={current_seed})"
                )
                np.random.seed(current_seed)
                sample_indices = np.random.choice(
                    len(filtered_embeddings), size=n_samples, replace=False
                )
                sampled_embeddings = filtered_embeddings[sample_indices]
            else:
                sampled_embeddings = filtered_embeddings
                logger.info(f"Using all {len(sampled_embeddings)} embeddings")

            # Perform cosine statistics analysis
            logger.info("Computing cosine statistics...")
            results = analyze_cosine_statistics(
                sampled_embeddings, sample_pairs=sample_pairs, random_seed=current_seed
            )

            # Compute pairwise cosines for histogram (sample if needed)
            normalized_embeddings = normalize_embeddings(sampled_embeddings)
            n = len(normalized_embeddings)
            if n * (n - 1) // 2 > sample_pairs:
                np.random.seed(current_seed)
                indices_i = np.random.choice(n, size=sample_pairs, replace=True)
                indices_j = np.random.choice(n, size=sample_pairs, replace=True)
                valid_mask = indices_i != indices_j
                indices_i = indices_i[valid_mask]
                indices_j = indices_j[valid_mask]
                if len(indices_i) > 0:
                    cosines = np.sum(
                        normalized_embeddings[indices_i]
                        * normalized_embeddings[indices_j],
                        axis=1,
                    )
                else:
                    cosines = np.array([])
            else:
                cosine_matrix = cosine_similarity(normalized_embeddings)
                upper_triangle = np.triu_indices(n, k=1)
                cosines = cosine_matrix[upper_triangle]

            # Compute angles
            cosines_clipped = np.clip(cosines, -1.0, 1.0)
            angles = np.arccos(cosines_clipped)
            # Filter out any invalid values
            valid_mask = np.isfinite(angles)
            angles = angles[valid_mask]
            cosines = cosines[valid_mask]  # Keep cosines aligned with valid angles

            results["pairwise_cosines"] = cosines
            results["pairwise_angles"] = angles
            results["run_idx"] = run_idx

            # Collect data for combined plots
            all_cosines.append(cosines)
            all_angles.append(angles)
            all_results.append(results)
            all_raw_embeddings.append(sampled_embeddings)  # Store raw embeddings

            # Print summary for this run
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Cosine Statistics Results (Run {run_idx + 1})")
            logger.info(f"{'=' * 60}\n")

            logger.info(f"Number of samples: {results['n_samples']}")
            logger.info(f"Number of dimensions: {results['n_dims']}")
            logger.info("\nGlobal Angular Dispersion:")
            logger.info(f"  Mean Pairwise Cosine: {results['mean_cosine']:.6f}")
            logger.info(f"  Cosine Variance: {results['cosine_variance']:.6f}")
            logger.info(f"  Angular Std: {results['angular_std']:.6f} rad")

            logger.info("\nDirectional Statistics:")
            logger.info(
                f"  Mean Resultant Length (R): {results['mean_resultant_length']:.6f}"
            )
            logger.info(
                f"  Rayleigh Test Statistic: {results['rayleigh_test_statistic']:.2f}"
            )
            logger.info(f"  Rayleigh p-value: {results['rayleigh_p_value']:.6f}")
            logger.info(
                f"  Uniformity: {'Rejected (not uniform)' if results['rayleigh_p_value'] < 0.05 else 'Not rejected (may be uniform)'}"
            )

            logger.info("\nSpectral Geometry (PCA on Cosine Matrix):")
            cosine_pca = results["cosine_pca"]
            logger.info(
                f"  First 10 PCs explain: {cosine_pca['cumulative_variance'][min(9, len(cosine_pca['cumulative_variance']) - 1)] * 100:.2f}%"
            )
            logger.info(
                f"  First 50 PCs explain: {cosine_pca['cumulative_variance'][min(49, len(cosine_pca['cumulative_variance']) - 1)] * 100:.2f}%"
            )

        # Combine all data for aggregated plots
        logger.info(f"\n{'=' * 60}")
        logger.info("Creating combined plots from all runs...")
        logger.info(f"{'=' * 60}")

        combined_cosines = np.concatenate(all_cosines)
        combined_angles = np.concatenate(all_angles)

        # Create cosine distance normal fit plot
        logger.info(f"\n{'=' * 60}")
        logger.info("Creating cosine distance normal fit plot...")
        logger.info(f"{'=' * 60}")
        cosine_distance_plot_path = os.path.join(
            output_dir, "cosine_distance_normal_fit.png"
        )
        plot_cosine_distance_normal_fit(
            combined_cosines,
            output_path=cosine_distance_plot_path,
            bins=50,
            n_std=3,
            n_runs=n_runs,
            n_samples=n_samples,
        )
        logger.info(
            f"Cosine distance normal fit plot saved to {cosine_distance_plot_path}"
        )

        # Compute aggregated statistics across runs
        mean_cosine_mean = np.mean([r["mean_cosine"] for r in all_results])
        mean_cosine_std = np.std([r["mean_cosine"] for r in all_results])
        cosine_var_mean = np.mean([r["cosine_variance"] for r in all_results])
        cosine_var_std = np.std([r["cosine_variance"] for r in all_results])
        # Filter out inf/nan values for angular std
        angular_stds = [
            r["angular_std"] for r in all_results if np.isfinite(r["angular_std"])
        ]
        angular_std_mean = np.mean(angular_stds) if len(angular_stds) > 0 else 0.0
        angular_std_std = np.std(angular_stds) if len(angular_stds) > 1 else 0.0
        mrl_mean = np.mean([r["mean_resultant_length"] for r in all_results])
        mrl_std = np.std([r["mean_resultant_length"] for r in all_results])
        rayleigh_p_mean = np.mean([r["rayleigh_p_value"] for r in all_results])

        # Average PCA results across runs
        avg_pca_cumulative = np.mean(
            [r["cosine_pca"]["cumulative_variance"] for r in all_results], axis=0
        )
        pca_10_mean = avg_pca_cumulative[min(9, len(avg_pca_cumulative) - 1)] * 100
        pca_50_mean = avg_pca_cumulative[min(49, len(avg_pca_cumulative) - 1)] * 100

        prefix = f"cosine_stats_combined_n{n_samples}_runs{n_runs}"
        plot_files = {}

        # Create combined PCA plot using average across runs
        # Use the last run's PCA structure as template
        if len(all_results) > 0:
            template_pca = all_results[-1]["cosine_pca"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            explained_var = template_pca["explained_variance"]
            n_components_plot = min(50, len(explained_var))
            ax1.bar(
                range(1, n_components_plot + 1),
                explained_var[:n_components_plot] * 100,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
            )
            ax1.set_xlabel("Principal Component", fontsize=12)
            ax1.set_ylabel("Explained Variance (%)", fontsize=12)
            ax1.set_title(
                "PCA on Cosine Matrix - Explained Variance\n(Average across runs)",
                fontsize=14,
            )
            ax1.grid(True, alpha=0.3, axis="y")

            n_comp_cumulative = min(100, len(avg_pca_cumulative))
            ax2.plot(
                range(1, n_comp_cumulative + 1),
                avg_pca_cumulative[:n_comp_cumulative] * 100,
                linewidth=2,
                color="steelblue",
                label="Average across runs",
            )
            ax2.axhline(y=50, color="r", linestyle="--", label="50% threshold")
            ax2.axhline(y=90, color="orange", linestyle="--", label="90% threshold")
            ax2.set_xlabel("Number of Components", fontsize=12)
            ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
            ax2.set_title(
                "PCA on Cosine Matrix - Cumulative Variance\n(Average across runs)",
                fontsize=14,
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            pca_path = os.path.join(output_dir, f"{prefix}_pca_cosine.png")
            plt.savefig(pca_path, dpi=300, bbox_inches="tight")
            plot_files["pca_cosine"] = pca_path
            plt.close(fig)

        # 1. Combined cosine similarity histogram
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(
            combined_cosines,
            bins=50,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            label="Pairwise Cosines (All Runs)",
        )
        ax.axvline(
            np.mean(combined_cosines),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Overall Mean = {np.mean(combined_cosines):.4f}",
        )
        ax.axvline(
            mean_cosine_mean,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"Mean of Run Means = {mean_cosine_mean:.4f} ± {mean_cosine_std:.4f}",
        )
        ax.set_xlabel("Cosine Similarity", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Pairwise Cosine Similarity Distribution (Combined {n_runs} Runs)\n"
            f"Overall Mean: {np.mean(combined_cosines):.4f}, Overall Var: {np.var(combined_cosines):.4f}\n"
            f"Run Means: {mean_cosine_mean:.4f} ± {mean_cosine_std:.4f}",
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        cosine_hist_path = os.path.join(output_dir, f"{prefix}_cosine_histogram.png")
        plt.savefig(cosine_hist_path, dpi=300, bbox_inches="tight")
        plot_files["cosine_histogram"] = cosine_hist_path
        plt.close(fig)

        # 2. Combined angular distribution histogram
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(
            combined_angles,
            bins=50,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            label="Pairwise Angles (All Runs)",
        )
        ax.axvline(
            np.mean(combined_angles),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Overall Mean = {np.mean(combined_angles):.4f} rad",
        )
        ax.axvline(
            np.std(combined_angles),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Overall Std = {np.std(combined_angles):.4f} rad",
        )
        ax.axvline(
            angular_std_mean,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Mean of Run Stds = {angular_std_mean:.4f} ± {angular_std_std:.4f} rad",
        )
        ax.set_xlabel("Angle (radians)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Pairwise Angular Distribution (Combined {n_runs} Runs)\n"
            f"Overall Mean: {np.mean(combined_angles):.4f} rad, Overall Std: {np.std(combined_angles):.4f} rad\n"
            f"Run Std Means: {angular_std_mean:.4f} ± {angular_std_std:.4f} rad",
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        angle_hist_path = os.path.join(output_dir, f"{prefix}_angle_histogram.png")
        plt.savefig(angle_hist_path, dpi=300, bbox_inches="tight")
        plot_files["angle_histogram"] = angle_hist_path
        plt.close(fig)

        # 3. Combined summary statistics plot
        fig, ax = plt.subplots(figsize=(10, 10))
        stats_text = f"""
Cosine Statistics Summary (Combined {n_runs} Runs)

Global Angular Dispersion:
  Mean Pairwise Cosine: {mean_cosine_mean:.6f} ± {mean_cosine_std:.6f}
  Cosine Variance: {cosine_var_mean:.6f} ± {cosine_var_std:.6f}
  Angular Std: {angular_std_mean:.6f} ± {angular_std_std:.6f} rad

Directional Statistics:
  Mean Resultant Length (R): {mrl_mean:.6f} ± {mrl_std:.6f}
  Average Rayleigh p-value: {rayleigh_p_mean:.6f}
  Uniformity: {"Rejected (not uniform)" if rayleigh_p_mean < 0.05 else "Not Rejected"}

Spectral Geometry:
  First 10 PCs explain: {pca_10_mean:.2f}% (average)
  First 50 PCs explain: {pca_50_mean:.2f}% (average)

Dataset:
  Samples per run: {all_results[0]["n_samples"]}
  Total pairs analyzed: {len(combined_cosines)}
  Dimensions: {all_results[0]["n_dims"]}

Run Statistics:
  Number of runs: {n_runs}
  Samples per run: {n_samples}
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
        ax.set_title(
            "Cosine Statistics Summary (Combined)", fontsize=16, fontweight="bold"
        )

        summary_path = os.path.join(output_dir, f"{prefix}_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plot_files["summary"] = summary_path
        plt.close(fig)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Combined Statistics (Across {n_runs} Runs)")
        logger.info(f"{'=' * 60}\n")

        logger.info(f"Total pairs analyzed: {len(combined_cosines)}")
        logger.info(f"Number of dimensions: {all_results[0]['n_dims']}")
        logger.info("\nGlobal Angular Dispersion (Mean ± Std across runs):")
        logger.info(
            f"  Mean Pairwise Cosine: {mean_cosine_mean:.6f} ± {mean_cosine_std:.6f}"
        )
        logger.info(f"  Cosine Variance: {cosine_var_mean:.6f} ± {cosine_var_std:.6f}")
        logger.info(
            f"  Angular Std: {angular_std_mean:.6f} ± {angular_std_std:.6f} rad"
        )

        logger.info("\nDirectional Statistics (Mean ± Std across runs):")
        logger.info(f"  Mean Resultant Length (R): {mrl_mean:.6f} ± {mrl_std:.6f}")
        logger.info(f"  Average Rayleigh p-value: {rayleigh_p_mean:.6f}")
        logger.info(
            f"  Uniformity: {'Rejected (not uniform)' if rayleigh_p_mean < 0.05 else 'Not rejected (may be uniform)'}"
        )

        logger.info("\nSpectral Geometry (Average across runs):")
        logger.info(f"  First 10 PCs explain: {pca_10_mean:.2f}%")
        logger.info(f"  First 50 PCs explain: {pca_50_mean:.2f}%")

        logger.info("\nCombined visualization plots saved:")
        for plot_name, plot_path in plot_files.items():
            logger.info(f"  {plot_name}: {plot_path}")

        logger.info(f"\n{'=' * 60}")
        logger.info("Creating raw vs debiased comparison plots...")
        logger.info(f"{'=' * 60}")

        # Create raw vs whitened comparison plots
        comparison_plot_files = create_raw_vs_whitened_comparison(
            all_raw_embeddings,
            combined_cosines,
            combined_angles,
            all_results,
            output_dir=output_dir,
            prefix=prefix,
            sample_pairs=sample_pairs,
            random_seed=random_seed,
        )

        logger.info("\nRaw vs Debiased comparison plots saved:")
        for plot_name, plot_path in comparison_plot_files.items():
            logger.info(f"  {plot_name}: {plot_path}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"All {n_runs} runs completed and combined!")
        logger.info(f"{'=' * 60}")

        return all_results

    except Exception as e:
        logger.error(f"Error performing cosine statistics analysis: {e}")
        raise


if __name__ == "__main__":
    main(
        start_year=1930,
        end_year=2024,
        n_samples=5000,
        n_runs=3,
        random_seed=42,
        sample_pairs=10000,
        output_dir=os.path.join(BASE_DIR, "figures"),
        data_dir=os.path.join(BASE_DIR, "data", "data_final"),
        csv_path=os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv"),
    )
