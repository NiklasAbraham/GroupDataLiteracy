import pandas as pd
import asyncio
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import numpy as np

# Add parent directory to path for imports
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.data_utils import load_movie_data, preprocess_genres
from src.api.wikidata_handler import get_wikidata_subclasses


DATA_DIR = os.path.join(base_path, 'data')
CLEAN_DATASET_SAVE_PATH = os.path.join(DATA_DIR, 'final_dataset.csv')
MAX_PLOT_LENGTH = 14000
WRONG_CLASSES_SAVE_PATH = os.path.join(DATA_DIR, 'wrong_wikidata_classes.json')
WRONG_WIKIDATA_CLASSES = {
    "Q24862", # short film
    "Q21191270", # television series episode
    "Q5398426", # television series
    "Q15116915", # show
    "Q18011171", # unfinished or abandoned film project 
    "Q21664088", # two-part episode
    "Q482994", # album
    "Q1030329", # viral video
    "Q21198342", # manga series
    "Q7725634", # literary work
    "Q1261214", # television special
    "Q622550", # trailer
    "Q240862", # director's cut
    "Q1555508", # radio program
    "Q17362920", # Wikimedia duplicated page
    "Q7889", # video game
}
MODEL_NAME = "BAAI/bge-m3"
ENTROPY_THRESHOLD = 4.8398  # determined by cutoff_exploration.py

GENRE_TO_CLUSTER_ID_PATH =  os.path.join(base_path, "src/genre_fix_mapping_new.json")
GENRE_CLUSTER_ID_TO_NAME_PATH = os.path.join(base_path, "src/cluster_to_genre_mapping_01122025.json")
GENRE_CLUSTER_IDS_TO_REMOVE = ["4"]  # exploitation genre cluster ID
STRING_COLUMNS = [
    "country",
    "imdb_id",
    "duration_all",
    "actors_id",
    "actors",
    "directors_id",
    "directors",
    "genre_id",
    "genre",
    "release_date",
    "wikidata_class",
    "wikipedia_link",
    "title",
    "set_in_period",
    "awards",
    "budget",
    "budget_currency",
    "box_office",
    "box_office_currency",
    "box_office_worldwide",
    "box_office_worldwide_currency",
    "popularity",
    "vote_average",
    "vote_count",
    "tmdb_id",
    "plot",
    "plot_section",
    "genre_cluster_ids",
    "genre_cluster_names",
]

async def filter_non_movies(df: pd.DataFrame, wrong_classes_save_path: str = WRONG_CLASSES_SAVE_PATH, new_event_loop: bool = True) -> pd.DataFrame:
    if os.path.exists(wrong_classes_save_path):
        with open(wrong_classes_save_path, 'r') as f:
            wrong_classes = set(json.load(f))
    else:
        wrong_classes = await get_wikidata_subclasses(WRONG_WIKIDATA_CLASSES)

        os.makedirs(os.path.dirname(wrong_classes_save_path), exist_ok=True)
        with open(wrong_classes_save_path, 'w') as f:
            json.dump(list(wrong_classes), f)


    def is_movie(wikidata_class_str: str) -> bool:
        classes = {cls.strip() for cls in wikidata_class_str.split(',') if cls.strip()}
        return len(classes.intersection(wrong_classes)) == 0
    
    filtered_df = df[df['wikidata_class'].apply(is_movie)].reset_index(drop=True)
    return filtered_df

def filter_movies_without_plot(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df[df['plot'].notna() & (df['plot'].str.strip() != "")].reset_index(drop=True)
    return filtered_df

def filter_movies_with_single_occurrence_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove genres that appear only once from each movie's genre column.
    
    Parameters:
    - df: DataFrame with 'genre' column containing comma-separated genre strings
    
    Returns:
    - DataFrame with single-occurrence genres removed from genre strings
    """
    if 'genre' not in df.columns:
        return df
    
    # Extract all individual genres from all movies
    all_genres = []
    for genre_str in df['genre'].dropna():
        if isinstance(genre_str, str) and genre_str.strip():
            # Split by comma and strip whitespace
            genres = [g.strip() for g in genre_str.split(',') if g.strip()]
            all_genres.extend(genres)
    
    # Count genre occurrences
    genre_counts = pd.Series(all_genres).value_counts()
    
    # Find genres that appear only once
    single_occurrence_genres = set(genre_counts[genre_counts == 1].index)
    
    if len(single_occurrence_genres) == 0:
        return df
    
    # Remove single-occurrence genres from each movie's genre string
    def remove_single_occurrence_genres(genre_str):
        if pd.isna(genre_str) or not isinstance(genre_str, str) or not genre_str.strip():
            return genre_str
        genres = [g.strip() for g in genre_str.split(',') if g.strip()]
        # Keep only genres that are not single-occurrence
        filtered_genres = [g for g in genres if g not in single_occurrence_genres]
        # Return empty string if no genres remain, otherwise join with comma
        if not filtered_genres:
            return ""
        return ", ".join(filtered_genres)
    
    df_copy = df.copy()
    df_copy['genre'] = df_copy['genre'].apply(remove_single_occurrence_genres)
    return df_copy

def add_token_features_columns(df: pd.DataFrame, model_name: str = MODEL_NAME) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_token_length(text: str) -> int:
        if not isinstance(text, str):
            return 0
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def get_num_different_tokens(text: str) -> int:
        if not isinstance(text, str):
            return 0
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(set(tokens))
    
    def get_token_shannon_entropy(text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            return 0.0
        token_counts = pd.Series(tokens).value_counts(normalize=True)
        entropy = -sum(token_counts * np.log2(token_counts))
        return entropy

    tqdm.pandas()
    print("Calculating plot lengths in tokens...")
    df['plot_length_tokens'] = df['plot'].progress_apply(get_token_length)
    
    print("Calculating number of different tokens in plots...")
    df['num_different_tokens'] = df['plot'].progress_apply(get_num_different_tokens)
    
    print("Calculating token Shannon entropy in plots...")
    df['token_shannon_entropy'] = df['plot'].progress_apply(get_token_shannon_entropy)
    
    return df

def add_genres_clustered_columns(
    df: pd.DataFrame,
    genre_name_to_cluster_id: dict,
    genre_cluster_id_to_cluster_name: dict
) -> pd.DataFrame:
    def map_genres_to_cluster_ids(genre_str: str) -> str:
        if pd.isna(genre_str) or not isinstance(genre_str, str) or genre_str.strip() == "":
            return ""
        preprocessed_genres = preprocess_genres(genre_str)
        if not preprocessed_genres:
            return ""
        return ",".join(preprocessed_genres.split("|"))
    
    def map_genre_cluster_ids_to_names(cluster_id_str: str) -> str:
        if pd.isna(cluster_id_str) or not isinstance(cluster_id_str, str) or cluster_id_str.strip() == "":
            return ""
        cluster_ids = [cid.strip() for cid in cluster_id_str.split(',') if cid.strip()]
        cluster_names = set()
        for cid in cluster_ids:
            if cid in genre_cluster_id_to_cluster_name:
                cluster_names.add(genre_cluster_id_to_cluster_name[cid])
        return ",".join(sorted(cluster_names))
    
    tqdm.pandas()
    print("Mapping genres to genre cluster IDs...")
    df["genre_cluster_ids"] = df["genre"].progress_apply(map_genres_to_cluster_ids)
    df["genre_cluster_names"] = df["genre_cluster_ids"].progress_apply(map_genre_cluster_ids_to_names)
    return df

def remove_movies_with_genre_cluster(df: pd.DataFrame, genre_cluster_id: str) -> pd.DataFrame:
    def has_genre_cluster(genre_cluster_ids_str: str) -> bool:
        if pd.isna(genre_cluster_ids_str) or not isinstance(genre_cluster_ids_str, str):
            return False
        cluster_ids = {cid.strip() for cid in genre_cluster_ids_str.split(',') if cid.strip()}
        return genre_cluster_id in cluster_ids
    
    tqdm.pandas()
    print(f"Removing movies with genre cluster ID: {genre_cluster_id}...")
    filtered_df = df[~df['genre_cluster_ids'].progress_apply(has_genre_cluster)].reset_index(drop=True)
    return filtered_df

def convert_string_solumns_nans_to_empty_strings(df: pd.DataFrame, string_columns: list) -> pd.DataFrame:
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

async def clean_dataset(
    df: pd.DataFrame,
    genre_to_cluster_id_dict: dict = None,
    cluster_id_to_genre_name_dict: dict = None,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    wrong_classes_save_path: str = WRONG_CLASSES_SAVE_PATH,
    max_plot_length: int = MAX_PLOT_LENGTH,
    filter_single_genres: bool = True,
    new_event_loop: bool = True,
    genre_cluster_ids_to_remove: list = GENRE_CLUSTER_IDS_TO_REMOVE,
    string_columns_for_nan_to_empty: list = STRING_COLUMNS
) -> pd.DataFrame:
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    print(f"Original dataset size: {len(df)}")
    df_filtered = filter_movies_without_plot(df)
    print(f"After filtering movies without plot: {len(df_filtered)}")
    df_filtered = await filter_non_movies(df_filtered, wrong_classes_save_path=wrong_classes_save_path, new_event_loop=new_event_loop)
    print(f"After filtering non-movies: {len(df_filtered)}")
    
    if filter_single_genres:
        df_filtered = filter_movies_with_single_occurrence_genres(df_filtered)
        print(f"After removing single-occurrence genres from genre column: {len(df_filtered)}")

    df_filtered["plot_length_chars"] = df_filtered["plot"].apply(lambda x: len(x.strip()) if isinstance(x, str) else 0)
    df_filtered = df_filtered[df_filtered["plot_length_chars"] <= max_plot_length].reset_index(drop=True)
    print(f"After filtering movies with plot length > {max_plot_length} chars: {len(df_filtered)}")
    
    print(f"Adding token features columns...")
    df_filtered = add_token_features_columns(df_filtered)

    df_filtered = df_filtered[df_filtered["token_shannon_entropy"] >= entropy_threshold].reset_index(drop=True)
    print(f"After filtering movies with token Shannon entropy < {entropy_threshold}: {len(df_filtered)}")

    if genre_to_cluster_id_dict is not None and cluster_id_to_genre_name_dict is not None:
        print("Adding genres clustered columns...")
        df_filtered = add_genres_clustered_columns(df_filtered, genre_to_cluster_id_dict, cluster_id_to_genre_name_dict)
        for genre_cluster_id in genre_cluster_ids_to_remove:
            df_filtered = remove_movies_with_genre_cluster(df_filtered, genre_cluster_id)
            print(f"After removing movies with genre cluster ID {genre_cluster_id}: {len(df_filtered)}")
    else:
        print("Skipping adding genres clustered columns and removing movies with specified genre cluster IDs due to missing mapping dictionaries.")

    print("Converting NaNs to empty strings in string columns...")
    df_filtered = convert_string_solumns_nans_to_empty_strings(df_filtered, string_columns_for_nan_to_empty)

    print("FINAL dataset size:", len(df_filtered))

    return df_filtered

if __name__ == "__main__":
    df = load_movie_data(DATA_DIR, verbose=True)
    genre_to_cluster_id_dict = json.load(open(GENRE_TO_CLUSTER_ID_PATH, 'r'))
    cluster_id_to_genre_name_dict = json.load(open(GENRE_CLUSTER_ID_TO_NAME_PATH, 'r'))

    df = asyncio.run(clean_dataset(
        df,
        genre_to_cluster_id_dict=genre_to_cluster_id_dict,
        cluster_id_to_genre_name_dict=cluster_id_to_genre_name_dict
    ))
    df.to_csv(CLEAN_DATASET_SAVE_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEAN_DATASET_SAVE_PATH}")
