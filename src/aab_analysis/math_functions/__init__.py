"""
Math utilities for analysis calculations.
"""

from .cosine_distance_util import (
    calculate_average_cosine_distance,
    calculate_average_cosine_distance_between_groups,
)
# Note: The following modules are not in math_functions directory:
# - epsilon_ball functions are in ks_test directory
# - find_closest_neighbors and find_most_dissimilar are in neighbourhood/002_neighbor_utils.py
# - gaussian_analysis functions are in neighbourhood/001_gaussian_analysis.py
# - statistical_tests is in ks_test directory
from .whitening import (
    debias_embeddings,
    mean_center_embeddings,
    whiten_embeddings,
)

__all__ = [
    "calculate_average_cosine_distance",
    "calculate_average_cosine_distance_between_groups",
    "debias_embeddings",
    "mean_center_embeddings",
    "whiten_embeddings",
]
