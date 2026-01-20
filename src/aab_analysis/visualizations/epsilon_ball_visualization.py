"""
Visualization functions for epsilon ball analysis.

This module provides plotting functions for visualizing epsilon ball analysis results,
including temporal distributions, distance distributions, and K-S test comparisons.
Uses tueplots (ICML 2024 bundle) for figure size, fonts, and Uni Tuebingen colors.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tueplots.constants.color.rgb as rgb
from tueplots import bundles

logger = logging.getLogger(__name__)


def plot_movies_over_time(
    results_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Movies in Epsilon Ball Over Time",
    random_results_df: pd.DataFrame = None,
):
    """
    Plot the number of movies in the epsilon ball over time.

    Parameters:
    - results_df: DataFrame from analyze_epsilon_ball
    - output_path: Path to save the plot (if None, displays plot). Use .pdf for vector output.
    - title: Plot title (will have total count appended)
    - random_results_df: Optional DataFrame from control group (mean embedding) analysis for comparison
    """
    if results_df.empty:
        logger.warning("No data to plot")
        return

    if "year" not in results_df.columns or results_df["year"].isna().all():
        logger.warning("No year data available for plotting")
        return

    df_with_year = results_df[results_df["year"].notna()].copy()
    df_with_year["year"] = df_with_year["year"].astype(int)

    if df_with_year.empty:
        logger.warning("No movies with valid year data")
        return

    year_counts = df_with_year["year"].value_counts().sort_index()
    total_movies = len(results_df)

    year_counts_series = pd.Series(year_counts.values, index=year_counts.index)
    sma_3 = year_counts_series.rolling(window=3, center=False, min_periods=1).mean()
    sma_10 = year_counts_series.rolling(window=10, center=False, min_periods=1).mean()

    has_random_comparison = (
        random_results_df is not None
        and not random_results_df.empty
        and "year" in random_results_df.columns
        and random_results_df["year"].notna().any()
    )

    if has_random_comparison:
        if "count" in random_results_df.columns:
            random_year_counts_series = pd.Series(
                random_results_df["count"].values,
                index=random_results_df["year"].values,
            ).sort_index()
        else:
            random_df_with_year = random_results_df[
                random_results_df["year"].notna()
            ].copy()
            random_df_with_year["year"] = random_df_with_year["year"].astype(int)
            random_year_counts = random_df_with_year["year"].value_counts().sort_index()
            random_year_counts_series = pd.Series(
                random_year_counts.values, index=random_year_counts.index
            )

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

        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
        fig, ax1 = plt.subplots()

        ax1.bar(
            anchor_normalized.index,
            anchor_normalized.values,
            alpha=0.7,
            edgecolor="black",
            color=rgb.tue_blue,
            label="Anchor Movies (normalized)",
        )
        ax1.plot(
            sma_3_normalized.index,
            sma_3_normalized.values,
            color=rgb.tue_red,
            linewidth=2,
            label="Anchor SMA (3)",
        )
        ax1.plot(
            sma_10_normalized.index,
            sma_10_normalized.values,
            color=rgb.tue_red,
            linewidth=2,
            linestyle="--",
            label="Anchor SMA (10)",
        )
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Normalized Count (Anchor Movies)", color=rgb.tue_blue)
        ax1.tick_params(axis="y", labelcolor=rgb.tue_blue)
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = ax1.twinx()

        all_years = sorted(set(anchor_normalized.index) | set(random_normalized.index))
        random_aligned = pd.Series(0.0, index=all_years, dtype=float)
        random_aligned.loc[random_normalized.index] = random_normalized.values

        ax2.bar(
            random_aligned.index,
            random_aligned.values,
            alpha=0.5,
            edgecolor="black",
            color=rgb.tue_orange,
            label="Control Group (normalized)",
        )
        ax2.plot(
            random_sma_3_normalized.index,
            random_sma_3_normalized.values,
            color=rgb.tue_orange,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label="Control Group SMA (3)",
        )
        ax2.plot(
            random_sma_10_normalized.index,
            random_sma_10_normalized.values,
            color=rgb.tue_lightorange,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label="Control Group SMA (10)",
        )
        ax2.set_ylabel("Normalized Count (Control Group)", color=rgb.tue_orange)
        ax2.tick_params(axis="y", labelcolor=rgb.tue_orange)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        labels2 = [label.replace("Random", "Control Group") for label in labels2]
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"{title} (Total: {total_movies} movies, Normalized)")
        plt.tight_layout()
    else:
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
        plt.figure()
        plt.bar(
            year_counts.index,
            year_counts.values,
            alpha=0.7,
            edgecolor="black",
            color=rgb.tue_blue,
        )
        plt.plot(
            sma_3.index, sma_3.values, color=rgb.tue_red, linewidth=2, label="SMA (3)"
        )
        plt.plot(
            sma_10.index,
            sma_10.values,
            color=rgb.tue_red,
            linewidth=2,
            linestyle="--",
            label="SMA (10)",
        )
        plt.xlabel("Year")
        plt.ylabel("Number of Movies")
        plt.title(f"{title} (Total: {total_movies} movies)")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_distance_distribution(
    results_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Distance Distribution in Epsilon Ball",
):
    """
    Plot the distribution of distances in the epsilon ball.

    Parameters:
    - results_df: DataFrame from analyze_epsilon_ball
    - output_path: Path to save the plot (if None, displays plot). Use .pdf for vector output.
    - title: Plot title (will have total count appended)
    """
    if results_df.empty:
        logger.warning("No data to plot")
        return

    total_movies = len(results_df)
    # Verify we're using ALL movies - get all distance values
    all_distances = results_df["distance"].values
    logger.info(
        f"plot_distance_distribution: Using ALL {len(all_distances)} movies "
        f"(distance range: {all_distances.min():.6f} to {all_distances.max():.6f})"
    )

    plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
    plt.figure()
    # Use ALL movies in epsilon ball for histogram (not limited)
    plt.hist(
        all_distances,
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color=rgb.tue_blue,
    )
    plt.xlabel("Cosine Distance")
    plt.ylabel("Number of Movies")
    plt.title(f"{title} (Total: {total_movies} movies)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.axvline(
        all_distances.mean(),
        color=rgb.tue_red,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {all_distances.mean():.4f}",
    )
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
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
    interpretation: dict = None,
):
    """
    Plot cumulative distribution functions (CDFs) for K-S test visualization.

    Parameters:
    - anchor_distances: Array of cosine distances from anchor epsilon ball
    - random_distances: Array of cosine distances from control group epsilon ball
    - ks_statistic: K-S test statistic
    - p_value: p-value from K-S test
    - output_path: Path to save the plot (if None, displays plot). Use .pdf for vector output.
    - title: Plot title
    - interpretation: Optional interpretation dict from interpret_ks_test
    """
    anchor_sorted = np.sort(anchor_distances)
    random_sorted = np.sort(random_distances)

    anchor_cdf = np.arange(1, len(anchor_sorted) + 1) / len(anchor_sorted)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)

    all_distances = np.sort(np.unique(np.concatenate([anchor_sorted, random_sorted])))
    anchor_cdf_interp = np.interp(all_distances, anchor_sorted, anchor_cdf)
    random_cdf_interp = np.interp(all_distances, random_sorted, random_cdf)
    diff = np.abs(anchor_cdf_interp - random_cdf_interp)
    max_diff_idx = np.argmax(diff)
    max_diff_dist = all_distances[max_diff_idx]
    max_diff_anchor_cdf = anchor_cdf_interp[max_diff_idx]
    max_diff_random_cdf = random_cdf_interp[max_diff_idx]

    plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=2))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(
        anchor_sorted,
        anchor_cdf,
        label="Anchor Movies",
        linewidth=2,
        color=rgb.tue_blue,
    )
    ax1.plot(
        random_sorted,
        random_cdf,
        label="Control Group",
        linewidth=2,
        color=rgb.tue_red,
        linestyle="--",
    )

    ax1.plot(
        [max_diff_dist, max_diff_dist],
        [max_diff_anchor_cdf, max_diff_random_cdf],
        "k-",
        linewidth=2,
        label=f"K-S Statistic = {ks_statistic:.4f}",
    )
    ax1.plot(max_diff_dist, max_diff_anchor_cdf, "ko", markersize=8)
    ax1.plot(max_diff_dist, max_diff_random_cdf, "ko", markersize=8)

    ax1.set_xlabel("Cosine Distance")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("Cumulative Distribution Functions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

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
        color=rgb.tue_blue,
        density=True,
    )
    ax2.hist(
        random_distances,
        bins=bins,
        alpha=0.6,
        label="Control Group",
        color=rgb.tue_red,
        density=True,
    )
    ax2.axvline(
        max_diff_dist,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Max Diff Point (D={ks_statistic:.4f})",
    )
    ax2.set_xlabel("Cosine Distance")
    ax2.set_ylabel("Density")
    ax2.set_title("Distance Distribution Histograms")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{title}\nK-S Statistic: {ks_statistic:.6f}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
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
    interpretation: dict = None,
):
    """
    Plot cumulative distribution functions for temporal K-S test.

    Parameters:
    - anchor_year_counts: Series of movie counts per year for anchor
    - random_year_counts: Series of movie counts per year for control group
    - ks_statistic: K-S test statistic
    - p_value: p-value from K-S test
    - output_path: Path to save the plot (if None, displays plot). Use .pdf for vector output.
    - title: Plot title
    - interpretation: Optional interpretation dict from interpret_ks_test
    """
    anchor_samples = []
    for year, count in anchor_year_counts.items():
        anchor_samples.extend([year] * int(count))

    random_samples = []
    for year, count in random_year_counts.items():
        random_samples.extend([year] * int(count))

    anchor_sorted = np.sort(anchor_samples)
    random_sorted = np.sort(random_samples)

    anchor_cdf = np.arange(1, len(anchor_sorted) + 1) / len(anchor_sorted)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)

    all_years = np.sort(np.unique(np.concatenate([anchor_sorted, random_sorted])))
    anchor_cdf_interp = np.interp(all_years, anchor_sorted, anchor_cdf)
    random_cdf_interp = np.interp(all_years, random_sorted, random_cdf)
    diff = np.abs(anchor_cdf_interp - random_cdf_interp)
    max_diff_idx = np.argmax(diff)
    max_diff_year = all_years[max_diff_idx]
    max_diff_anchor_cdf = anchor_cdf_interp[max_diff_idx]
    max_diff_random_cdf = random_cdf_interp[max_diff_idx]

    plt.rcParams.update(bundles.icml2024(column="half", nrows=2, ncols=1))
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(
        anchor_sorted,
        anchor_cdf,
        label="Anchor Movies",
        linewidth=2,
        color=rgb.tue_blue,
    )
    ax1.plot(
        random_sorted,
        random_cdf,
        label="Control Group",
        linewidth=2,
        color=rgb.tue_red,
        linestyle="--",
    )

    ax1.plot(
        [max_diff_year, max_diff_year],
        [max_diff_anchor_cdf, max_diff_random_cdf],
        "k-",
        linewidth=2,
        label=f"K-S Statistic = {ks_statistic:.4f}",
    )
    ax1.plot(max_diff_year, max_diff_anchor_cdf, "ko", markersize=8)
    ax1.plot(max_diff_year, max_diff_random_cdf, "ko", markersize=8)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("Cumulative Distribution Functions (by Year)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    all_years_aligned = sorted(
        set(anchor_year_counts.index) | set(random_year_counts.index)
    )
    anchor_aligned = pd.Series(0, index=all_years_aligned)
    random_aligned = pd.Series(0, index=all_years_aligned)
    anchor_aligned.loc[anchor_year_counts.index] = anchor_year_counts.values
    random_aligned.loc[random_year_counts.index] = random_year_counts.values

    anchor_max = anchor_aligned.max()
    random_max = random_aligned.max()

    anchor_normalized = (
        anchor_aligned / anchor_max if anchor_max > 0 else anchor_aligned
    )
    random_normalized = (
        random_aligned / random_max if random_max > 0 else random_aligned
    )

    years = np.array(all_years_aligned)
    ax2.scatter(
        years,
        anchor_normalized.values,
        label="Anchor Movies (normalized)",
        alpha=0.7,
        color=rgb.tue_blue,
    )
    ax2.scatter(
        years,
        random_normalized.values,
        label="Control Group (normalized)",
        alpha=0.7,
        color=rgb.tue_red,
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Normalized Count")
    ax2.set_title("Movie Counts per Year (Normalized)")
    ax2.set_ylim([0, 1.05])
    step = max(1, len(years) // 10)
    ax2.set_xticks(years[::step])
    ax2.set_xticklabels(years[::step].tolist(), rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(f"{title}\nK-S Statistic: {ks_statistic:.6f}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"K-S test temporal plot saved to {output_path}")
    else:
        plt.show()

    plt.close()
