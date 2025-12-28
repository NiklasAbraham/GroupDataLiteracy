# FAISS KNN and Novelty Score Documentation

## Overview

This document explains the implementation of K-Nearest Neighbors (KNN) search using FAISS (Facebook AI Similarity Search) library and the novelty score calculation method used for analyzing movie embeddings. The implementation enables efficient similarity search over large-scale embedding vectors while respecting temporal constraints.

## FAISS (Facebook AI Similarity Search)

FAISS is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It is optimized for:

- **Large-scale similarity search**: Can handle millions or billions of vectors efficiently
- **Multiple distance metrics**: Supports L2 distance, inner product (cosine similarity), and other metrics
- **GPU acceleration**: Can leverage GPU for faster computations
- **Memory efficiency**: Various index types balance speed, accuracy, and memory usage

### FAISS Index Structure

In our implementation, we use `faiss.IndexIDMap(faiss.IndexFlatIP(d))`, which consists of two components:

1. **IndexFlatIP (Inner Product)**: 
   - A brute-force index that computes the inner product (dot product) between query vectors and all indexed vectors
   - Since embeddings are typically normalized, inner product is equivalent to cosine similarity
   - Provides exact search results (no approximation)
   - Time complexity: O(n×d) where n is the number of indexed vectors and d is the embedding dimension

2. **IndexIDMap**:
   - A wrapper that allows assigning custom IDs to vectors instead of using sequential indices
   - Enables mapping back to original movie IDs after search
   - Maintains the mapping between FAISS internal indices and our movie IDs

### Why FAISS?

- **Efficiency**: FAISS is optimized in C++ with efficient vectorized operations
- **Scalability**: Can handle large datasets (92,374 movies in our case) efficiently
- **Flexibility**: Easy to swap different index types depending on accuracy/speed trade-offs
- **Memory management**: Better memory handling than pure NumPy implementations for large datasets

## Temporal K-Nearest Neighbors Implementation

The core function `find_temporal_nearest_neighbors()` in `src/analysis/knn_faiss.py` implements a temporal constraint on the KNN search: for each movie, we only consider neighbors from movies released in strictly earlier years.

### Algorithm Workflow

1. **Data Preparation**:
   ```python
   df_sorted = df.sort_values(by='year', ascending=True)
   all_embeddings = np.stack(df_sorted['embedding'].values).astype('float32')
   ```
   - Movies are sorted by release year in ascending order
   - All embeddings are stacked into a single NumPy array (float32 for FAISS compatibility)

2. **Index Initialization**:
   ```python
   d = all_embeddings.shape[1]  # embedding dimension
   index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
   ```
   - Creates an empty FAISS index with inner product metric
   - Dimension `d` is determined from the embedding shape

3. **Incremental Processing by Year**:
   - Movies are grouped by release year
   - For each year group:
     - If the index is not empty, search for k nearest neighbors among all previously indexed movies (which are from earlier years)
     - Convert similarity scores to distances: `distances = 1 - similarity`
     - Store results (movie_id, neighbor_ids, neighbor_distances)
     - Add the current year's movies to the index for future searches

4. **Temporal Constraint**:
   - Movies from the earliest years have no prior movies to compare with, so they receive placeholder IDs ('Q0') and distance (-1)
   - Each subsequent year can only find neighbors from earlier years, ensuring temporal causality

### Key Implementation Details

- **Cosine Distance Calculation**: Since FAISS IndexFlatIP returns cosine similarity (when embeddings are normalized), we convert to cosine distance: `distance = 1 - similarity`
- **Batch Processing**: Processing by year groups allows efficient batch searches and maintains temporal ordering
- **Memory Efficiency**: Embeddings are processed incrementally, avoiding the need to store all pairwise distances

## Novelty Score Method

The novelty score quantifies how novel or unique a movie is compared to all movies released before it. It is defined as the **cosine distance to the nearest temporal neighbor**.

### Definition

```
novelty_score = min(cosine_distance(movie, neighbor))
```

Where:
- `cosine_distance` ranges from 0 (identical) to 1 (orthogonal/very different)
- Only neighbors from strictly earlier years are considered
- The minimum distance represents the closest match to previous movies

### Interpretation

- **Low novelty score (close to 0)**: The movie is very similar to at least one previous movie, indicating low novelty
- **High novelty score (close to 1)**: The movie is very different from all previous movies, indicating high novelty
- **Score = -1**: Movies from the earliest years with no predecessors (treated as missing data in analysis)

### Calculation Process

1. **Load KNN Results**: The temporal KNN results are loaded from `knn_faiss_novelty.csv`

2. **Extract Nearest Neighbor Distance**:
   ```python
   knn_results["neighbor_distances"] = knn_results["neighbor_distances"].apply(lambda x: json.loads(x))
   knn_results["novelty_score"] = knn_results["neighbor_distances"].apply(lambda x: x[0])
   ```
   - The `neighbor_distances` column contains a list of k distances (sorted by similarity)
   - The first element `x[0]` is the distance to the closest neighbor, which is our novelty score

3. **Merge with Movie Data**: The novelty scores are merged back into the main dataset for analysis

### Applications

The novelty score can be used for:
- **Identifying groundbreaking movies**: Movies with high novelty scores may represent new genres, styles, or themes
- **Temporal analysis**: Analyzing how novelty trends change over time
- **Genre analysis**: Comparing novelty scores across different genres
- **Recommendation systems**: Finding movies that are similar but novel

### Example Usage

```python
# Load data
df = load_final_data_with_embeddings(CSV_PATH, DATA_DIR)
knn_results = pd.read_csv(NOVELTY_PATH)

# Calculate novelty scores
knn_results["neighbor_distances"] = knn_results["neighbor_distances"].apply(lambda x: json.loads(x))
knn_results["novelty_score"] = knn_results["neighbor_distances"].apply(lambda x: x[0])

# Merge with movie data
df_merged = df.merge(knn_results[["movie_id", "novelty_score"]], on="movie_id", how="left")

# Find most novel movies
top_novel_movies = df_merged[df_merged['year'] >= 2000].nlargest(20, 'novelty_score')
```

## Performance Considerations

- **Time Complexity**: O(n²×d) for the full dataset, where n is the number of movies and d is the embedding dimension
- **Space Complexity**: O(n×d) for storing embeddings
- **Optimization Opportunities**: 
  - For larger datasets, approximate indexes (e.g., IndexIVF) could be used
  - Parallel processing across years could be implemented
  - GPU acceleration could be leveraged for larger datasets

## Files and Functions

- **`src/analysis/knn_faiss.py`**: Main implementation of temporal KNN search using FAISS
  - `find_temporal_nearest_neighbors(df, k=10)`: Core function for finding temporal nearest neighbors
  
- **`src/data_utils.py`**: Data loading utilities
  - `load_final_data_with_embeddings()`: Loads movie data with embeddings
  
- **`src/analysis/novelty.ipynb`**: Jupyter notebook demonstrating novelty score calculation and visualization

## Output Format

The KNN results are saved to `data/data_final/knn_faiss_novelty.csv` with columns:
- `movie_id`: Unique identifier for each movie
- `neighbor_ids`: List of k nearest neighbor movie IDs (as JSON string)
- `neighbor_distances`: List of k cosine distances to neighbors (as JSON string, sorted by distance)

The novelty score is then extracted as the first element of `neighbor_distances` for each movie.
