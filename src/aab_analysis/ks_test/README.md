# KS Test Analysis Module

This module provides tools for epsilon ball analysis and statistical testing using Kolmogorov-Smirnov (K-S) tests to compare distributions between anchor movies and control groups.

## Overview

The module performs epsilon ball analysis to find movies within a specified cosine distance threshold around anchor movies, then uses K-S tests to statistically compare:
- Distance distributions between anchor and control groups
- Temporal distributions (year distributions) between anchor and control groups

## Module Structure

### Core Files

- **`epsilon_ball_analysis.py`**: Main analysis script that orchestrates the entire pipeline
- **`epsilon_ball.py`**: Core functions for finding movies within epsilon balls and computing anchor embeddings
- **`statistical_tests.py`**: K-S test implementations and interpretation functions
- **`epsilon_ball_utils.py`**: Utility functions for caching, filename formatting, and hash computation
- **`epsilon_ball_visualization.py`**: Plotting functions for visualizing results

## Usage

### Basic Usage

```python
from src.aab_analysis.ks_test.epsilon_ball_analysis import main

results = main(
    anchor_qids=["Q4941"],  # James Bond movie
    epsilon=0.3,
    start_year=1930,
    end_year=2024,
    compare_with_random=True,  # Compare with control group
    plot_over_time=True,
    plot_distance_dist=True,
    output_dir="./figures_final/epsilon_ball_analysis"
)
```

### Using Individual Functions

```python
from src.aab_analysis.ks_test.epsilon_ball import (
    find_movies_in_epsilon_ball,
    compute_anchor_embedding
)
from src.aab_analysis.ks_test.statistical_tests import (
    kolmogorov_smirnov_test,
    interpret_ks_test
)

# Compute anchor embedding from multiple movies
anchor_embedding = compute_anchor_embedding(
    anchor_qids=["Q4941", "Q212145"],
    embeddings_corpus=embeddings,
    movie_ids=movie_ids,
    method="average"  # or "medoid"
)

# Find movies within epsilon ball
indices, distances, similarities = find_movies_in_epsilon_ball(
    embeddings_corpus=embeddings,
    anchor_embedding=anchor_embedding,
    movie_ids=movie_ids,
    epsilon=0.3,
    exclude_anchor_ids=["Q4941"]
)

# Perform K-S test
statistic, p_value = kolmogorov_smirnov_test(
    anchor_distances=anchor_distances,
    random_distances=random_distances
)

# Interpret results
interpretation = interpret_ks_test(
    statistic=statistic,
    p_value=p_value,
    sample_size_1=len(anchor_distances),
    sample_size_2=len(random_distances)
)
```

## Key Functions

### Epsilon Ball Analysis

- **`find_movies_in_epsilon_ball()`**: Finds all movies within a cosine distance threshold
- **`compute_anchor_embedding()`**: Computes anchor embedding from multiple movies using average or medoid method
- **`analyze_epsilon_ball()`**: Complete analysis pipeline for a single epsilon ball

### Statistical Tests

- **`kolmogorov_smirnov_test()`**: Compares distance distributions between two groups
- **`kolmogorov_smirnov_test_temporal()`**: Compares temporal (year) distributions between two groups
- **`interpret_ks_test()`**: Provides interpretation of K-S test results with effect size analysis

### Visualization

- **`plot_movies_over_time()`**: Plots temporal distribution of movies in epsilon ball
- **`plot_distance_distribution()`**: Plots histogram of distances in epsilon ball
- **`plot_ks_test_cdf()`**: Visualizes K-S test results with CDFs
- **`plot_ks_test_temporal_cdf()`**: Visualizes temporal K-S test results

### Utilities

- **`compute_embeddings_hash()`**: Computes hash for cache validation
- **`load_cached_mean_embedding()`**: Loads cached mean embedding
- **`save_cached_mean_embedding()`**: Saves mean embedding to cache
- **`get_anchor_names_string()`**: Formats anchor movie names for filenames
- **`truncate_filename_component()`**: Truncates long filename components

## Parameters

### Main Function Parameters

- **`anchor_qids`**: List of movie QIDs to use as anchors
- **`epsilon`**: Maximum cosine distance threshold (0 <= epsilon <= 2)
- **`start_year` / `end_year`**: Year range for filtering movies
- **`anchor_method`**: Method to combine anchor embeddings ("average" or "medoid")
- **`exclude_anchors`**: Whether to exclude anchor movies from results
- **`compare_with_random`**: Whether to compare with control group (mean of entire ensemble)
- **`plot_over_time`**: Whether to create temporal distribution plot
- **`plot_distance_dist`**: Whether to create distance distribution plot
- **`output_dir`**: Directory to save plots (PDF format)

## Output

The analysis produces:

1. **DataFrame** with movies in epsilon ball (columns: movie_id, title, year, distance, similarity, rank)
2. **Plots** (if enabled):
   - Temporal distribution over time
   - Distance distribution histogram
   - K-S test CDF comparisons (distance and temporal)
3. **Statistical results**: K-S test statistics, p-values, and interpretations

## Control Group

When `compare_with_random=True`, the module uses the mean embedding of the entire filtered ensemble as a control group. This allows statistical comparison to determine if the anchor-based selection shows significantly different distributions compared to a random selection.

## Caching

Mean embeddings for control groups are cached based on embeddings hash to avoid recomputation. Cache files are stored in `data/cache/mean_embeddings/`.

## Dependencies

- numpy
- pandas
- scipy (for K-S tests)
- matplotlib (for visualization)
- tueplots (for figure styling)
- sklearn (for cosine similarity)
