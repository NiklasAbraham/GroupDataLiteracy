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

    d = all_embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d))

    df_grouped_by_year = df_sorted.groupby("year", sort=False)

    results = {"movie_id": [], "neighbor_ids": [], "neighbor_distances": []}
    current_idx = 0

    print("Processing movies incrementally...")

    for _, group in tqdm.tqdm(df_grouped_by_year):
        n_samples = len(group)
        group_embeddings = all_embeddings[current_idx : current_idx + n_samples]
        group_indices = np.arange(current_idx, current_idx + n_samples).astype("int64")
        group_qids = movie_qids_sorted.iloc[
            current_idx : current_idx + n_samples
        ].values.tolist()

        if index.ntotal > 0:
            similarity, neighbour_indices = index.search(group_embeddings, k)

            distances = 1 - similarity

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

    knn_results = find_temporal_nearest_neighbors(df, k=5000)

    knn_results.to_csv(SAVE_PATH, index=False)
    print(f"Temporal k-NN results saved to {SAVE_PATH}")
