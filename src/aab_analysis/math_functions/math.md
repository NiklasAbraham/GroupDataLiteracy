# Math Module Documentation

This document provides a brief overview of the functions in the `src/analysis/math_functions/` module.

## Module Structure

```
math_functions/
├── __init__.py                    # Module initialization and exports
├── cosine_distance_util.py        # Cosine distance calculation functions
├── find_closest_neighbors.py      # Nearest neighbor search functions
├── find_most_dissimilar.py        # Most dissimilar item search functions
└── boostrapping_embeddings.py     # Bootstrap sampling functions
```

---

## `math_functions/cosine_distance_util.py`

### `calculate_average_cosine_distance(embeddings)`

**Purpose**: Calculates the average pairwise cosine distance within a single group of embeddings.

**Parameters**:
- `embeddings`: numpy array of shape `(n_samples, embedding_dim)`

**Returns**:
- `float`: Average cosine distance within the group (returns `0.0` if fewer than 2 embeddings)

**Use Cases**:
- Measuring cohesion within keyword groups
- Evaluating cluster tightness
- Analyzing semantic similarity within categories

---

### `calculate_average_cosine_distance_between_groups(group1_embeddings, group2_embeddings)`

**Purpose**: Calculates the average cosine distance between two different groups of embeddings.

**Parameters**:
- `group1_embeddings`: numpy array of shape `(n1, embedding_dim)`
- `group2_embeddings`: numpy array of shape `(n2, embedding_dim)`

**Returns**:
- `float`: Average cosine distance between the two groups (returns `0.0` if either group is empty)

**Use Cases**:
- Comparing keyword groups to random movies
- Measuring separation between different categories
- Evaluating inter-cluster distances

---

### `find_nearest_and_furthest_medoids(embeddings)`

**Purpose**: Identifies the nearest and furthest medoids within a set of embeddings.

**Parameters**:
- `embeddings`: numpy array of shape `(n_samples, embedding_dim)`

**Returns**:
- `tuple`: (nearest_medoid_index, furthest_medoid_index)

**Use Cases**:
- Identifying the most and least representative movies
- Analyzing centrality in movie groups

---

## `math_functions/find_closest_neighbors.py`

### `find_n_closest_neighbours(embeddings_corpus, anchor_embedding, movie_ids, movie_data, n=10, anchor_idx=None)`

**Purpose**: Finds the n closest neighbors to an anchor embedding in a corpus of embeddings using cosine similarity.

**Mathematical Formulation**:
- **Cosine Similarity**: 
  ```
  sim(a, b) = (a · b) / (||a|| ||b||)
  ```
  where `a · b` is the dot product and `||a||` is the L2 norm.

- **Cosine Distance**: 
  ```
  dist(a, b) = 1 - sim(a, b)
  ```
  Distance ranges from 0 (identical vectors) to 2 (opposite directions).

- **Normalization**: 
  ```
  a_norm = a / ||a||
  ```
  Normalizing embeddings ensures cosine similarity is computed correctly.

**Parameters**:
- `embeddings_corpus`: numpy array of shape `(n_movies, embedding_dim)` - The corpus of embeddings to search through
- `anchor_embedding`: numpy array of shape `(1, embedding_dim)` - The anchor embedding to find neighbors for
- `movie_ids`: numpy array - Array of movie IDs corresponding to embeddings_corpus
- `movie_data`: pandas DataFrame - Movie metadata (must contain 'movie_id' and 'title' columns)
- `n`: int (default: 10) - Number of closest neighbors to find
- `anchor_idx`: int (optional) - Index of anchor in the corpus, used to exclude it from results

**Returns**:
- `list`: List of tuples `(qid, title, distance, similarity)` for the n closest neighbors, sorted by distance (ascending)

**Raises**:
- `ValueError`: If inputs are invalid, mismatched dimensions, or anchor embedding is a zero vector

**Use Cases**:
- Finding similar movies to a given movie in the embedding space
- Recommender system applications
- Analyzing semantic similarity between items
- Identifying nearest neighbors for clustering or classification tasks

---

## `math_functions/find_most_dissimilar.py`

### `find_most_dissimilar_movies(reference, embeddings, movie_ids, movie_data, n=10)`

**Purpose**: Finds the n most dissimilar items to a reference embedding in a corpus of embeddings using cosine distance.

**Mathematical Formulation**:
- **Cosine Similarity**: 
  ```
  sim(a, b) = (a · b) / (||a|| ||b||)
  ```
  where `a · b` is the dot product and `||a||` is the L2 norm.

- **Cosine Distance**: 
  ```
  dist(a, b) = 1 - sim(a, b)
  ```
  Distance ranges from 0 (identical vectors) to 2 (opposite directions).

- **Normalization**: 
  ```
  a_norm = a / ||a||
  ```
  Normalizing embeddings ensures cosine similarity is computed correctly.

**Parameters**:
- `reference`: str or numpy array - Reference for comparison
  - If `str`: Movie ID (qid) that must exist in `movie_ids`
  - If `numpy.ndarray`: Embedding vector of shape `(embedding_dim,)`
- `embeddings`: numpy array of shape `(n_movies, embedding_dim)` - The corpus of embeddings to search through
- `movie_ids`: numpy array of shape `(n_movies,)` - Array of movie IDs corresponding to embeddings
- `movie_data`: pandas DataFrame - Movie metadata (must contain 'movie_id', 'title', and 'year' columns)
- `n`: int (default: 10) - Number of most dissimilar movies to find

**Returns**:
- `list`: List of tuples `(qid, title, distance, similarity, year)` for the n most dissimilar movies, sorted by distance (descending)

**Raises**:
- `ValueError`: If inputs are invalid, mismatched dimensions, reference movie ID not found, or reference embedding is a zero vector

**Use Cases**:
- Finding movies most dissimilar to a reference movie or embedding
- Identifying outliers in the embedding space
- Analyzing diversity in movie collections
- Finding contrasting examples for analysis
- Comparing items to a mean or centroid embedding

---

## `math_functions/whitening.py`

### `mean_center_embeddings(embeddings)`

**Purpose**: Mean-center embeddings by subtracting the global mean.

**Mathematical Formulation**:
```
μ = (1/n) Σ_{i=1}^n x_i
x_centered = x - μ
```

**Parameters**:
- `embeddings`: Array of embeddings (shape: [n_samples, embedding_dim])

**Returns**:
- Mean-centered embeddings (shape: [n_samples, embedding_dim])

**Use Cases**:
- Preprocessing step before whitening or debiasing
- Removing global bias from embeddings
- Preparing data for PCA-based transformations

---

### `whiten_embeddings(embeddings, n_components=None, normalize=True)`

**Purpose**: Whiten embeddings using PCA to restore isotropy.

**Mathematical Formulation**:
1. Mean-center: `x ← x - μ`
2. Compute PCA: `X_centered = UΣV^T`
3. Whiten: `x_whitened = U / √(eigenvalues)`
4. Optionally normalize: `x' = x_whitened / ||x_whitened||`

Whitening removes correlations and scales variance to 1 in all directions, making the covariance matrix identity.

**Parameters**:
- `embeddings`: Array of embeddings (shape: [n_samples, embedding_dim])
- `n_components`: Number of PCA components to keep. If None, keeps all.
- `normalize`: If True, re-normalize embeddings to unit length after whitening

**Returns**:
- Whitened embeddings (shape: [n_samples, n_components or embedding_dim])

**Use Cases**:
- Restoring isotropy in embedding spaces
- Removing anisotropic structure for clustering
- Preparing embeddings for geometric analysis

---

### `debias_embeddings(embeddings, k=3, normalize=False)`

**Purpose**: De-bias embeddings using the "All-but-the-top" approach.

**Mathematical Formulation**:
```
x' = x - μ - Σ_{i=1}^k ⟨x - μ, u_i⟩ u_i
```

Where:
- `μ` is the global mean
- `u_1, ..., u_k` are the top k principal components
- `⟨x - μ, u_i⟩` is the projection onto PC i

This removes global anisotropy/cone by projecting out the top k principal components, while preserving relative covariance and mean differences that might encode real temporal or semantic structure.

**Parameters**:
- `embeddings`: Array of embeddings (shape: [n_samples, embedding_dim])
- `k`: Number of top principal components to remove (default: 3, typically 1-5)
- `normalize`: If True, re-normalize embeddings to unit length after debiasing

**Returns**:
- De-biased embeddings (shape: [n_samples, embedding_dim])

**Use Cases**:
- Removing directional bias in embeddings (e.g., semantic cone collapse)
- Improving cosine similarity-based analysis
- Demonstrating anisotropy → isotropy transition
- Analyzing embedding geometry before and after debiasing

**References**:
- Mu et al. (ICLR 2017) "All-but-the-Top: Simple and Effective Postprocessing for Word Representations"