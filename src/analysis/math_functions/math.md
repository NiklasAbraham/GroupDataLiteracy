# Math Module Documentation

This document provides a brief overview of the functions in the `src/analysis/math/` module.

## Module Structure

```
math/
├── __init__.py              # Module initialization and exports
└── cosine_distance.py        # Cosine distance calculation functions
```

---

## `math/cosine_distance.py`

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