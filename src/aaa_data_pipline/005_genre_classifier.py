"""
Genre Classifier

Classifies movie genres by:
1. Fetching Wikipedia descriptions for genres
2. Embedding descriptions using sentence transformers
3. Clustering genres into main categories
4. Generating a mapping JSON file compatible with data_utils.py
"""

import asyncio
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wikipediaapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))

from src.aaa_data_pipline.api.wikipedia_handler import get_page_from_url  # noqa: E402
from src.utils.data_utils import load_movie_data  # noqa: E402

_cleaning_module = importlib.import_module("src.aaa_data_pipline.004_data_cleaning")
clean_dataset = _cleaning_module.clean_dataset


def extract_unique_genres(movie_df: pd.DataFrame) -> List[str]:
    """Extract unique genres from movie dataframe."""
    if "genre" not in movie_df.columns:
        raise ValueError("DataFrame must have 'genre' column")

    raw_genres = movie_df[movie_df.genre.notna()].genre.unique()
    split_genres = []
    for genre in raw_genres:
        parts = genre.split(",")
        parts = [part.strip() for part in parts]
        split_genres.extend(parts)

    return list(set(split_genres))


def extract_genre_to_id_mapping(movie_df: pd.DataFrame) -> Dict[str, str]:
    """Extract mapping from genre names to their Wikidata IDs."""
    if "genre" not in movie_df.columns or "genre_id" not in movie_df.columns:
        return {}

    genre_to_id = {}
    for _, row in movie_df.iterrows():
        if pd.isna(row.get("genre")) or pd.isna(row.get("genre_id")):
            continue

        genres = [g.strip() for g in str(row["genre"]).split(",")]
        genre_ids = [gid.strip() for gid in str(row["genre_id"]).split(",")]

        for i, genre in enumerate(genres):
            if i < len(genre_ids):
                genre_id = genre_ids[i]
                if genre in genre_to_id:
                    existing_ids = set(genre_to_id[genre].split(","))
                    if genre_id not in existing_ids:
                        genre_to_id[genre] = ",".join(sorted(existing_ids | {genre_id}))
                else:
                    genre_to_id[genre] = genre_id

    return genre_to_id


def clean_description(description: str) -> str:
    """Clean description text."""
    if pd.isna(description) or not isinstance(description, str):
        return ""

    cleaned = re.sub(r"\r\n|\r|\n", " ", description)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)
    return cleaned.strip()


def get_wikipedia_description_from_title(
    genre_title: str, wiki_wiki: wikipediaapi.Wikipedia
) -> str:
    """Fetch Wikipedia description for a genre title."""
    try:
        wiki_url = (
            f"https://en.wikipedia.org/wiki/{genre_title.lower().replace(' ', '_')}"
        )
        genre_page = get_page_from_url(wiki_wiki, wiki_url)
        return clean_description(genre_page.summary)
    except Exception as e:
        print(f"Error fetching genre {genre_title}: {e}")
        return ""


def fetch_genre_descriptions(
    genres: List[str], user_agent: str = "GroupDataLiteracy/1.0 (movie data pipeline)"
) -> Dict[str, str]:
    """Fetch Wikipedia descriptions for a list of genres."""
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")
    genre_descriptions = {}
    for genre in tqdm(genres, desc="Fetching genre descriptions"):
        description = get_wikipedia_description_from_title(genre, wiki_wiki)
        genre_descriptions[genre] = description
    return genre_descriptions


def load_genre_descriptions_csv(csv_path: str) -> pd.DataFrame:
    """Load genre descriptions from CSV file."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["genre", "description"])

    df = pd.read_csv(csv_path)
    if "description" in df.columns:
        df["description"] = df["description"].apply(clean_description)
    return df


def save_genre_descriptions_csv(genre_descriptions: Dict[str, str], csv_path: str):
    """Save genre descriptions to CSV file."""
    df = (
        pd.DataFrame.from_dict(
            genre_descriptions, orient="index", columns=["description"]
        )
        .reset_index()
        .rename(columns={"index": "genre"})
    )
    df.to_csv(csv_path, index=False)


def get_missing_genres(all_genres: List[str], existing_df: pd.DataFrame) -> List[str]:
    """Identify genres missing from existing descriptions."""
    if existing_df.empty:
        return all_genres

    existing_genres = set(existing_df["genre"].values)
    missing_genres = [g for g in all_genres if g not in existing_genres]

    empty_descriptions = existing_df[
        (existing_df["description"].isna())
        | (existing_df["description"].str.strip() == "")
    ]["genre"].tolist()

    missing_genres.extend(empty_descriptions)
    return list(set(missing_genres))


def ensure_genre_descriptions(
    genres: List[str],
    csv_path: str = "genre_description_df.csv",
    user_agent: str = "GroupDataLiteracy/1.0 (movie data pipeline)",
) -> pd.DataFrame:
    """Ensure all genres have descriptions, fetching missing ones from Wikipedia."""
    genre_df = load_genre_descriptions_csv(csv_path)
    missing_genres = get_missing_genres(genres, genre_df)

    if missing_genres:
        print(f"Fetching descriptions for {len(missing_genres)} missing genres...")
        new_descriptions = fetch_genre_descriptions(missing_genres, user_agent)

        for genre, description in new_descriptions.items():
            if genre in genre_df["genre"].values:
                genre_df.loc[genre_df["genre"] == genre, "description"] = description
            else:
                new_row = pd.DataFrame({"genre": [genre], "description": [description]})
                genre_df = pd.concat([genre_df, new_row], ignore_index=True)

        save_genre_descriptions_csv(
            dict(zip(genre_df["genre"], genre_df["description"])), csv_path
        )
    else:
        print("All genre descriptions already available.")

    genre_df = genre_df.dropna(subset=["genre", "description"])
    genre_df = genre_df[genre_df["genre"].apply(lambda x: isinstance(x, str))].copy()

    if "description" in genre_df.columns:
        genre_df["description"] = genre_df["description"].apply(clean_description)
        genre_df = genre_df[genre_df["description"].str.strip() != ""].copy()

    return genre_df


def embed_genre_descriptions(
    genre_df: pd.DataFrame, model_name: str = "Qwen/Qwen3-Embedding-0.6B"
) -> np.ndarray:
    """Embed genre descriptions using sentence transformers."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Embedding genre descriptions...")
    descriptions = genre_df["description"].tolist()
    embeddings = model.encode(descriptions)
    return embeddings


def cluster_genres(
    genre_df: pd.DataFrame,
    embeddings: np.ndarray,
    n_clusters: int = 25,
    random_state: int = 123,
) -> pd.DataFrame:
    """Cluster genres using KMeans based on their embeddings."""
    print(f"Clustering genres into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)

    genre_df = genre_df.copy()
    genre_df["cluster"] = labels
    return genre_df


def create_genre_mapping(
    genre_df: pd.DataFrame, cluster_label_mapping: Optional[Dict[int, str]] = None
) -> Dict[str, str]:
    """Create genre mapping dictionary from clustered genres."""
    if cluster_label_mapping is None:
        genre_df["new_label"] = genre_df["cluster"].astype(str)
    else:
        genre_df["new_label"] = genre_df["cluster"].apply(
            lambda x: cluster_label_mapping.get(x, f"cluster_{x}")
        )

    mapping_dict = {}
    for _, row in genre_df.iterrows():
        if pd.isna(row["genre"]) or not isinstance(row["genre"], str):
            continue

        cleaned_genre = row["genre"].lower().replace("film", "").strip()
        if not cleaned_genre:
            continue

        mapping_dict[cleaned_genre] = row["new_label"]

    return mapping_dict


def save_genre_mapping(
    mapping_dict: Dict[str, str], output_path: str = "src/genre_fix_mapping.json"
):
    """Save genre mapping to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping_dict, f, indent=2)
    print(f"Genre mapping saved to {output_path}")


def classify_genres(
    movie_df: pd.DataFrame,
    descriptions_csv_path: str = "genre_description_df.csv",
    output_json_path: str = "src/genre_fix_mapping.json",
    n_clusters: int = 25,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    cluster_label_mapping: Optional[Dict[int, str]] = None,
    random_state: int = 123,
    apply_data_cleaning: bool = True,
) -> Tuple[Dict[str, str], pd.DataFrame, Dict[str, str]]:
    """
    Main function to classify genres from movie dataframe.

    Steps:
    1. Apply data cleaning (optional)
    2. Extract unique genres
    3. Fetch Wikipedia descriptions (cached in CSV)
    4. Embed descriptions using sentence transformers
    5. Cluster genres
    6. Generate and save mapping JSON

    Returns:
        Tuple of (mapping_dict, genre_df_clustered, genre_to_id)
    """
    print("=" * 60)
    print("Genre Classification Pipeline")
    print("=" * 60)

    if apply_data_cleaning:
        print("\nStep 0: Applying data cleaning...")
        movie_df = asyncio.run(clean_dataset(movie_df, filter_single_genres=True))
        print(f"Dataset size after cleaning: {len(movie_df)}")

    print("\nStep 1: Extracting unique genres...")
    unique_genres = extract_unique_genres(movie_df)
    print(f"Found {len(unique_genres)} unique genres")

    print("\nStep 2: Ensuring genre descriptions...")
    genre_df = ensure_genre_descriptions(unique_genres, descriptions_csv_path)
    print(f"Total genres with descriptions: {len(genre_df)}")

    print("\nStep 3: Embedding genre descriptions...")
    embeddings = embed_genre_descriptions(genre_df, model_name)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nStep 4: Clustering genres...")
    genre_df_clustered = cluster_genres(genre_df, embeddings, n_clusters, random_state)

    print("\nStep 5: Creating genre mapping...")
    mapping_dict = create_genre_mapping(genre_df_clustered, cluster_label_mapping)
    print(f"Created mapping for {len(mapping_dict)} genres")

    print("\nStep 6: Saving genre mapping...")
    save_genre_mapping(mapping_dict, output_json_path)

    print("\nStep 7: Extracting genre to ID mapping...")
    genre_to_id = extract_genre_to_id_mapping(movie_df)
    print(f"Extracted IDs for {len(genre_to_id)} genres")

    print("\n" + "=" * 60)
    print("Genre classification complete!")
    print("=" * 60)

    return mapping_dict, genre_df_clustered, genre_to_id


if __name__ == "__main__":
    data_dir = os.path.join(base_path, "data", "data_final")
    descriptions_csv_path = os.path.join(
        base_path, "src", "aaa_data_pipline", "genre_description_df.csv"
    )
    output_json_path = os.path.join(
        base_path, "src", "config", "genre_fix_mapping_new.json"
    )

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print("Loading movie data...")
    movie_df = load_movie_data(data_dir, verbose=True)

    if movie_df.empty:
        print("Error: No movie data loaded.")
        sys.exit(1)

    print("\nStarting genre classification...")
    mapping_dict, genre_df_clustered, genre_to_id = classify_genres(
        movie_df,
        descriptions_csv_path=descriptions_csv_path,
        output_json_path=output_json_path,
        n_clusters=20,
        apply_data_cleaning=True,
    )
