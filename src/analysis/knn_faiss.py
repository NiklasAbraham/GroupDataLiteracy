import os
import sys

import faiss
import numpy as np
import pandas as pd
import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")

SAVE_PATH = os.path.join(DATA_DIR, "knn_faiss_novelty.csv")

sys.path.insert(0, BASE_DIR)

from src.data_utils import load_final_data_with_embeddings


def find_temporal_nearest_neighbors(df, k=10):
    """
    Finds k-NN for each movie using cosine distance, constrained to movies
    with a strictly smaller release year.

    Returns:
        DataFrame with columns ['movie_id', 'neighbor_ids', 'neighbor_distances']
    """
    df_sorted = df.sort_values(by="year", ascending=True)

    all_embeddings = np.stack(df_sorted["embedding"].values).astype("float32")
    movie_qids_sorted = df_sorted["movie_id"]

    # Normalize embeddings for cosine similarity
    # L2 normalize each embedding vector
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    # Use a small threshold to catch near-zero vectors (floating point precision)
    EPSILON = 1e-8
    norms[norms < EPSILON] = 1.0  # Avoid division by zero or very small values
    all_embeddings_normalized = all_embeddings / norms
    
    # Verify normalization (each vector should have norm ≈ 1)
    # This is just a sanity check - can be removed in production
    normalized_norms = np.linalg.norm(all_embeddings_normalized, axis=1)
    if not np.allclose(normalized_norms, 1.0, atol=1e-6):
        print(f"Warning: Some vectors are not properly normalized. "
              f"Norm range: [{normalized_norms.min():.6f}, {normalized_norms.max():.6f}]")

    d = all_embeddings_normalized.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d))

    df_grouped_by_year = df_sorted.groupby("year", sort=False)

    results = {"movie_id": [], "neighbor_ids": [], "neighbor_distances": []}
    current_idx = 0

    print("Processing movies incrementally...")

    for _, group in tqdm.tqdm(df_grouped_by_year):
        n_samples = len(group)
        group_embeddings = all_embeddings_normalized[current_idx : current_idx + n_samples]
        group_indices = np.arange(current_idx, current_idx + n_samples).astype("int64")
        group_qids = movie_qids_sorted.iloc[
            current_idx : current_idx + n_samples
        ].values.tolist()

        if index.ntotal > 0:
            similarity, neighbour_indices = index.search(group_embeddings, k)

            # Clamp similarity to [-1, 1] to handle numerical precision issues
            # Even with normalized vectors, float32 precision can cause slight deviations
            similarity = np.clip(similarity, -1.0, 1.0)

            # Convert cosine similarity to cosine distance
            # Cosine similarity is in [-1, 1] for normalized vectors
            # Cosine distance = 1 - cosine_similarity, so it's in [0, 2]
            distances = 1 - similarity
            
            # Clamp distances to [0, 2] as a final safety check
            distances = np.clip(distances, 0.0, 2.0)

            results["movie_id"].extend(group_qids)
            results["neighbor_ids"].extend(
                movie_qids_sorted.iloc[neighbour_indices.flatten()]
                .values.reshape(n_samples, k)
                .tolist()
            )
            results["neighbor_distances"].extend(distances.tolist())
        else:
            results["movie_id"].extend(group_qids)
            results["neighbor_ids"].extend([["Q0"] * k] * n_samples)
            results["neighbor_distances"].extend([[-1] * k] * n_samples)

        index.add_with_ids(group_embeddings, group_indices)
        current_idx += n_samples

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    df = load_final_data_with_embeddings(CSV_PATH, DATA_DIR)
    print(df.info())

    knn_results = find_temporal_nearest_neighbors(df, k=500)

    knn_results.to_csv(SAVE_PATH, index=False)
    print(f"Temporal k-NN results saved to {SAVE_PATH}")
