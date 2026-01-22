import os
import sys
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from tqdm.auto import tqdm

# Get project root
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.data_utils import load_embeddings_as_dict

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "final_dataset.csv")
BUCKET_SIZE_PERCENTILE = 1.0

def calculate_distance_to_centroid(
    df: pd.DataFrame,
    embeddings_dict: dict,
) -> pd.DataFrame:
    centroid = np.mean(
        np.array(list(embeddings_dict.values())),
        axis=0
    )

    def compute_distance(movie_id: str) -> float:
        embedding = embeddings_dict[movie_id]
        return cosine(embedding, centroid)

    tqdm.pandas()
    df['distance_to_centroid'] = df['movie_id'].progress_apply(compute_distance)

    return df

def bootstrap_cosine_distance_mean_and_std(
    df: pd.DataFrame,
    embeddings_dict: dict,
    num_sample_pairs: int = 1000
) -> pd.DataFrame:
    distances = []

    movie_ids = df['movie_id'].tolist()

    for _ in range(num_sample_pairs):
        id1, id2 = np.random.choice(movie_ids, size=2, replace=False)

        emb1 = embeddings_dict[id1]
        emb2 = embeddings_dict[id2]

        dist = cosine(emb1, emb2)
        distances.append(dist)

    distances_array = np.array(distances)
    mean_distance = np.mean(distances_array)
    std_distance = np.std(distances_array)

    return mean_distance, std_distance

def get_mean_std_distance_per_bucket(
    df: pd.DataFrame,
    embeddings_dict: dict,
    bucket_size_percentile: int = BUCKET_SIZE_PERCENTILE,
    num_sample_pairs: int = 1000,
    length_column: str = 'plot_length_tokens'
) -> pd.DataFrame:
    df = df.sort_values(by=length_column).reset_index(drop=True)
    
    num_buckets = int(100 / bucket_size_percentile)
    bucket_start_indices = [
        int(i * len(df) / num_buckets) for i in range(num_buckets)
    ]
    bucket_end_indices = bucket_start_indices[1:] + [len(df)]

    results = []
    for bucket_index, (start_idx, end_idx) in tqdm(enumerate(zip(bucket_start_indices, bucket_end_indices))):
        bucket_df = df.iloc[start_idx:end_idx]
        mean_dist, std_dist = bootstrap_cosine_distance_mean_and_std(
            bucket_df,
            embeddings_dict,
            num_sample_pairs=num_sample_pairs
        )
        bucket_info = {
            'bucket_start_percentile': bucket_index * bucket_size_percentile,
            'bucket_start_index': start_idx,
            'average_distance_to_centroid': bucket_df['distance_to_centroid'].mean(),
            'max_length_in_bucket': bucket_df[length_column].max(),
            'bucket_end_index': end_idx,
            'num_movies_in_bucket': len(bucket_df),
            'mean_cosine_distance': mean_dist,
            'std_cosine_distance': std_dist
        }
        results.append(bucket_info)
    
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    df = pd.read_csv(CLEAN_DATA_PATH)
    embeddings_dict = load_embeddings_as_dict(data_dir=DATA_DIR, start_year=1930, end_year=2024)
    embedding_sums = [np.sum(v) for v in embeddings_dict.values()]

    print(f"Embedding sums: mean={np.mean(embedding_sums)}, std={np.std(embedding_sums)}")
    
    # df = calculate_distance_to_centroid(
    #     df,
    #     embeddings_dict
    # )
    

    # result_df = get_mean_std_distance_per_bucket(
    #     df,
    #     embeddings_dict,
    #     bucket_size_percentile=BUCKET_SIZE_PERCENTILE,
    #     num_sample_pairs=5000,
    #     length_column='plot_length_tokens'
    # )

    # plt.figure(figsize=(10, 5))
    # plt.title('Mean Cosine Distance per Plot Length Bucket')
    # x = result_df["bucket_start_percentile"]
    # means = result_df['mean_cosine_distance']
    # stds = result_df['std_cosine_distance']

    # plt.errorbar(x, means, yerr=stds, fmt='-o', ecolor='gray', capsize=3)
    # plt.xlabel('Percentile of movies by plot length in tokens')
    # plt.ylabel('Mean Cosine Distance')
    
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure(figsize=(10, 5))
    # plt.plot(x, stds, '-o', color='tab:orange')
    # plt.xlabel('Percentile of movies by plot length in tokens')
    # plt.ylabel('Std Dev of Cosine Distance')
    # plt.title('Std Dev of Cosine Distance per Plot Length Bucket')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure(figsize=(10, 5))
    # plt.plot(x, result_df["average_distance_to_centroid"], '-o', color='tab:orange')
    # plt.xlabel('Percentile of movies by plot length in tokens')
    # plt.ylabel('Average Distance to Centroid')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.show()
