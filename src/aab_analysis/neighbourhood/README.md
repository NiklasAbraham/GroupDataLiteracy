# Neighbourhood Analysis Module

This module provides tools for analyzing neighborhoods in the movie embedding space, including finding similar/dissimilar movies, Gaussian distribution analysis, and distance-based group comparisons.

## File Organization

The module is organized with numbered files for clear execution order:

### Core Functions

**`neighbourhood_001_gaussian_analysis.py`**
- Core functions for Gaussian distribution analysis
- `compute_mahalanobis_distances()`: Compute Mahalanobis distances from mean
- `analyze_gaussianity()`: Comprehensive Gaussianity analysis
- `create_gaussianity_plots()`: Generate visualization plots (Q-Q plots, PCA, histograms)
- `gaussian_analysis_with_embeddings()`: Complete analysis pipeline with visualizations

**`neighbourhood_002_neighbor_utils.py`**
- Utility functions for neighbor finding
- `find_n_closest_neighbours()`: Find n closest neighbors using cosine similarity
- `find_most_dissimilar_movies()`: Find n most dissimilar movies using cosine distance

### Analysis Scripts

**`003_find_neighbors.py`**
- Script to find closest neighbors to a movie or group of movies
- Supports single QID or multiple QIDs with aggregation (mean/median)
- Filters by year range
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_003_find_neighbors import main
  main(qid="Q1931001", n=10)
  ```

**`004_find_dissimilar.py`**
- Script to find most dissimilar movies
- Can use a specific movie ID, embedding vector, or mean embedding as reference
- Filters by year range
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_004_find_dissimilar import main
  main(reference=None, n=20)  # Uses mean embedding
  ```

**`005_gaussian_fit.py`**
- Comprehensive Gaussianity analysis script with caching
- Supports debiasing and whitening transformations
- Can test multiple alpha values efficiently (caches expensive computations)
- Generates Q-Q plots, PCA analysis, and outlier detection
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_005_gaussian_fit import main
  main(
      start_year=1930,
      end_year=2024,
      robust=True,
      alphas=[0.01, 0.05, 0.10],
      n_samples=5000,
      whiten=True,
      whiten_n_components=None,
      whiten_normalize=False,
      qq_test_types=["distances", "dimensions", "pca_components"]
  )
  ```

**`006_keyword_group_analysis.py`**
- Analyzes average cosine distance within keyword groups vs random movies
- Supports both keyword-based and QID-based group analysis
- Generates bar plots comparing within-group vs random distances
- Calculates global average distance for comparison
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_006_keyword_group_analysis import main
  main(use_qids=False, keywords_list=[["Star Wars"], ["Harry Potter"]])
  ```

**`007_neighbor_distribution.py`**
- Analyzes neighbor distribution using epsilon balls
- Compares distance distributions from anchor vs mean vector
- Plots CDF and histograms of distance distributions
- Finds most "average" movies (closest to mean embedding)
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_007_neighbor_distribution import main
  main(
      anchor_qids=["Q4941"],
      anchor_method="average",
      epsilon_max=2.0,
      epsilon_steps=200
  )
  ```

**`008_cosine_statistics.py`**
- Comprehensive cosine statistics analysis for hyperspherical embedding geometry
- Analyzes global angular dispersion, directional statistics, and spectral geometry
- Performs PCA on cosine similarity matrices
- Tests for uniformity using Rayleigh test
- Compares raw vs debiased embeddings (All-but-the-top approach)
- Supports multiple independent runs for statistical robustness
- Generates histograms, PCA plots, and comparison visualizations
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_008_cosine_statistics import main
  main(
      start_year=1930,
      end_year=2024,
      n_samples=5000,
      n_runs=3,
      random_seed=42,
      sample_pairs=10000
  )
  ```

**`009_extract_high_distance_movies.py`**
- Extracts movies with high Mahalanobis distances from cached Gaussian analysis results
- Works with the caching system from `005_gaussian_fit.py`
- Useful for identifying outliers after Gaussian analysis
- Can save results to CSV
- Usage:
  ```python
  from src.aab_analysis.neighbourhood.neighbourhood_009_extract_high_distance_movies import main
  main(
      distance_threshold=35.0,
      n_samples=5000,
      random_seed=42,
      robust=True,
      whiten=True,
      whiten_n_components=None,
      whiten_normalize=False,
      output_file="high_distance_movies.csv"
  )
  ```

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `matplotlib`: Plotting
- `sklearn`: Machine learning utilities (PCA, covariance estimation, cosine similarity)
- `pandas`: Data manipulation

## Data Requirements

All scripts expect:
- Embedding files in `data/data_final/` directory
- Movie metadata CSV at `data/data_final/final_dataset.csv`
- Embeddings loaded via `load_final_dense_embeddings()`
- Metadata loaded via `load_final_dataset()`

## Common Parameters

Most scripts share these parameters:
- `start_year`, `end_year`: Year range for filtering movies (default: 1930-2024)
- `data_dir`: Directory containing embeddings (default: `data/data_final`)
- `csv_path`: Path to movie metadata CSV (default: `data/data_final/final_dataset.csv`)

## Output

- Scripts generate plots saved to specified output directories
- Gaussian analysis creates multiple visualization files (Q-Q plots, PCA, histograms)
- Keyword group analysis creates bar charts comparing distances
- Neighbor distribution creates CDF and histogram plots

## Notes

- Gaussian analysis uses caching to avoid recomputing expensive operations (distances, PCA) when only alpha values change
- `009_extract_high_distance_movies.py` requires cached results from `005_gaussian_fit.py` - parameters must match exactly
- Cosine statistics analysis supports multiple runs for statistical robustness
- All distance calculations use cosine similarity/distance
- Embeddings are normalized before distance calculations
- Scripts handle missing data gracefully with warnings
