"""Statistical test functions for epsilon ball analysis."""

import numpy as np
from scipy import stats


def kolmogorov_smirnov_test(
    anchor_distances: np.ndarray,
    random_distances: np.ndarray,
):
    """Perform Kolmogorov-Smirnov test to compare two distance distributions."""
    if len(anchor_distances) == 0 or len(random_distances) == 0:
        raise ValueError("Both distance arrays must be non-empty")

    statistic, p_value = stats.ks_2samp(anchor_distances, random_distances)

    return statistic, p_value


def kolmogorov_smirnov_test_temporal(
    anchor_year_counts: np.ndarray,
    random_year_counts: np.ndarray,
    years: np.ndarray,
):
    """Perform Kolmogorov-Smirnov test on temporal distributions."""
    if len(anchor_year_counts) != len(random_year_counts):
        raise ValueError("Year count arrays must have the same length")

    if len(years) != len(anchor_year_counts):
        raise ValueError("Years array must match year counts length")

    anchor_samples = []
    for year, count in zip(years, anchor_year_counts):
        anchor_samples.extend([year] * int(count))

    random_samples = []
    for year, count in zip(years, random_year_counts):
        random_samples.extend([year] * int(count))

    if len(anchor_samples) == 0 or len(random_samples) == 0:
        raise ValueError("Cannot perform K-S test with empty samples")

    statistic, p_value = stats.ks_2samp(anchor_samples, random_samples)

    return statistic, p_value


def interpret_ks_test(
    statistic: float,
    p_value: float,
    alpha: float = 0.05,
    sample_size_1: int = None,
    sample_size_2: int = None,
) -> dict:
    """Interpret the results of a Kolmogorov-Smirnov test."""
    if statistic < 0.1:
        effect_size = "negligible"
        practically_different = False
    elif statistic < 0.2:
        effect_size = "small"
        practically_different = False
    elif statistic < 0.3:
        effect_size = "medium"
        practically_different = True
    else:
        effect_size = "large"
        practically_different = True

    large_sample_warning = False
    if sample_size_1 is not None and sample_size_2 is not None:
        min_sample_size = min(sample_size_1, sample_size_2)
        if min_sample_size > 1000:
            large_sample_warning = True

    if large_sample_warning:
        significant = practically_different
        if practically_different:
            interpretation = (
                f"The distributions show a {effect_size} practical difference "
                f"(K-S statistic = {statistic:.4f}). "
                f"Note: With large sample sizes (n1={sample_size_1:,}, n2={sample_size_2:,}), "
                f"p-values become overly sensitive. We focus on effect size instead."
            )
        else:
            interpretation = (
                f"The distributions are practically similar "
                f"(K-S statistic = {statistic:.4f}, effect size: {effect_size}). "
                f"Note: With large sample sizes (n1={sample_size_1:,}, n2={sample_size_2:,}), "
                f"even tiny differences can be statistically significant (p={p_value:.6f}), "
                f"but are not practically meaningful."
            )
    else:
        significant = p_value < alpha
        if significant:
            interpretation = (
                f"The distributions are significantly different (p={p_value:.4f} < {alpha}). "
                f"The K-S statistic of {statistic:.4f} indicates a {effect_size} effect size."
            )
        else:
            interpretation = (
                f"No significant difference between distributions (p={p_value:.4f} >= {alpha}). "
                f"The K-S statistic is {statistic:.4f}."
            )

    return {
        "significant": significant,
        "interpretation": interpretation,
        "effect_size": effect_size,
        "statistic": statistic,
        "p_value": p_value,
        "large_sample_warning": large_sample_warning,
        "practically_different": practically_different,
    }
