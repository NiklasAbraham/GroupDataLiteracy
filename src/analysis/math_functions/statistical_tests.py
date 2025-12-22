"""
Statistical test functions for epsilon ball analysis.

This module provides statistical tests to compare distributions between
anchor movies and random movies in epsilon ball analysis.
"""

import numpy as np
from scipy import stats


def kolmogorov_smirnov_test(
    anchor_distances: np.ndarray,
    random_distances: np.ndarray,
):
    """
    Perform Kolmogorov-Smirnov test to compare two distance distributions.

    The K-S test is a non-parametric test that compares two distributions
    by measuring the maximum distance between their cumulative distribution
    functions (CDFs).

    Mathematical formulation:
    - D = max|CDF_anchor(x) - CDF_random(x)|
    - Under null hypothesis: distributions are identical
    - p-value < 0.05: reject null hypothesis (distributions are different)

    Parameters:
    - anchor_distances: Array of cosine distances from anchor epsilon ball
    - random_distances: Array of cosine distances from random epsilon ball

    Returns:
    - statistic: K-S test statistic (0 to 1)
        - 0: identical distributions
        - 1: completely different distributions
    - p_value: Probability that distributions are the same
        - < 0.05: statistically significant difference
        - >= 0.05: no significant difference

    Example interpretation:
    - statistic=0.15, p=0.03 → Distributions differ significantly
    - statistic=0.08, p=0.42 → No significant difference
    """
    if len(anchor_distances) == 0 or len(random_distances) == 0:
        raise ValueError("Both distance arrays must be non-empty")

    # Perform two-sample Kolmogorov-Smirnov test
    statistic, p_value = stats.ks_2samp(anchor_distances, random_distances)

    return statistic, p_value


def kolmogorov_smirnov_test_temporal(
    anchor_year_counts: np.ndarray,
    random_year_counts: np.ndarray,
    years: np.ndarray,
):
    """
    Perform Kolmogorov-Smirnov test on temporal distributions.

    Compares the distribution of movies across years between anchor
    and random selections.

    Parameters:
    - anchor_year_counts: Array of movie counts per year for anchor
    - random_year_counts: Array of movie counts per year for random
    - years: Array of corresponding years

    Returns:
    - statistic: K-S test statistic (0 to 1)
    - p_value: Probability that distributions are the same

    Note:
    - Requires aligned year arrays (same years for both)
    """
    if len(anchor_year_counts) != len(random_year_counts):
        raise ValueError("Year count arrays must have the same length")

    if len(years) != len(anchor_year_counts):
        raise ValueError("Years array must match year counts length")

    # Create sample arrays weighted by counts
    # For K-S test, we need individual samples, not counts
    anchor_samples = []
    for year, count in zip(years, anchor_year_counts):
        anchor_samples.extend([year] * int(count))

    random_samples = []
    for year, count in zip(years, random_year_counts):
        random_samples.extend([year] * int(count))

    if len(anchor_samples) == 0 or len(random_samples) == 0:
        raise ValueError("Cannot perform K-S test with empty samples")

    # Perform K-S test
    statistic, p_value = stats.ks_2samp(anchor_samples, random_samples)

    return statistic, p_value


def interpret_ks_test(statistic: float, p_value: float, alpha: float = 0.05) -> dict:
    """
    Interpret the results of a Kolmogorov-Smirnov test.

    Parameters:
    - statistic: K-S test statistic
    - p_value: p-value from K-S test
    - alpha: Significance level (default: 0.05)

    Returns:
    - Dictionary with interpretation:
        - significant: Boolean indicating if difference is significant
        - interpretation: Human-readable interpretation
        - effect_size: Descriptive effect size category
    """
    significant = p_value < alpha

    # Interpret effect size based on statistic
    if statistic < 0.1:
        effect_size = "negligible"
    elif statistic < 0.2:
        effect_size = "small"
    elif statistic < 0.3:
        effect_size = "medium"
    else:
        effect_size = "large"

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
    }
