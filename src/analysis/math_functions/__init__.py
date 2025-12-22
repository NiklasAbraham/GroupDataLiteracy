"""
Math utilities for analysis calculations.
"""

from .cosine_distance_util import (
    calculate_average_cosine_distance,
    calculate_average_cosine_distance_between_groups,
)
from .epsilon_ball import (
    compute_anchor_embedding,
    find_movies_in_epsilon_ball,
)
from .find_closest_neighbors import find_n_closest_neighbours
from .find_most_dissimilar import find_most_dissimilar_movies
from .gaussian_analysis import (
    analyze_gaussianity,
    compute_mahalanobis_distances,
    create_gaussianity_plots,
    gaussian_analysis_with_embeddings,
)
from .statistical_tests import (
    interpret_ks_test,
    kolmogorov_smirnov_test,
    kolmogorov_smirnov_test_temporal,
)
from .whitening import (
    debias_embeddings,
    mean_center_embeddings,
    whiten_embeddings,
)

__all__ = [
    "calculate_average_cosine_distance",
    "calculate_average_cosine_distance_between_groups",
    "find_n_closest_neighbours",
    "find_most_dissimilar_movies",
    "analyze_gaussianity",
    "compute_mahalanobis_distances",
    "create_gaussianity_plots",
    "gaussian_analysis_with_embeddings",
    "debias_embeddings",
    "mean_center_embeddings",
    "whiten_embeddings",
    "find_movies_in_epsilon_ball",
    "compute_anchor_embedding",
    "kolmogorov_smirnov_test",
    "kolmogorov_smirnov_test_temporal",
    "interpret_ks_test",
]
