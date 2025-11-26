"""
Genre Classifier

This module classifies movie genres by:
1. Fetching Wikipedia descriptions for genres
2. Embedding the descriptions using sentence transformers
3. Clustering genres into main categories
4. Generating a mapping JSON file compatible with data_utils.py
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import wikipediaapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import sys
from pathlib import Path

# Add parent directory to path for imports
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))

from src.data_utils import load_movie_data
from src.api.wikipedia_handler import get_page_from_url
from src.data_cleaning import clean_dataset


def extract_unique_genres(movie_df: pd.DataFrame) -> List[str]:
    """
    Extract unique genres from movie dataframe by splitting comma-separated genre strings.
    
    Parameters:
    - movie_df: DataFrame with 'genre' column
    
    Returns:
    - List of unique genre strings
    """
    if "genre" not in movie_df.columns:
        raise ValueError("DataFrame must have 'genre' column")
    
    # Get all non-null genres
    raw_genres = movie_df[movie_df.genre.notna()].genre.unique()
    
    # Split genres by comma and strip whitespace
    split_genres = []
    for genre in raw_genres:
        parts = genre.split(",")
        parts = [part.strip() for part in parts]
        split_genres.extend(parts)
    
    # Get unique genres
    unique_genres = list(set(split_genres))
    
    return unique_genres


def get_wikipedia_description_from_title(
    genre_title: str, 
    wiki_wiki: wikipediaapi.Wikipedia
) -> str:
    """
    Fetch Wikipedia description/summary for a genre title.
    
    Parameters:
    - genre_title: Genre title string (e.g., "horror film")
    - wiki_wiki: Wikipedia API instance
    
    Returns:
    - Wikipedia summary string, or empty string if not found
    """
    try:
        # Convert genre title to Wikipedia URL format
        wiki_url = f"https://en.wikipedia.org/wiki/{genre_title.lower().replace(' ', '_')}"
        genre_page = get_page_from_url(wiki_wiki, wiki_url)
        
        if genre_page is None or not genre_page.exists():
            return ""
        
        return genre_page.summary
    except Exception as e:
        print(f"Error fetching genre {genre_title}: {e}")
        return ""


def fetch_genre_descriptions(
    genres: List[str],
    user_agent: str = 'GroupDataLiteracy/1.0 (movie data pipeline)'
) -> Dict[str, str]:
    """
    Fetch Wikipedia descriptions for a list of genres.
    
    Parameters:
    - genres: List of genre strings
    - user_agent: User agent string for Wikipedia API
    
    Returns:
    - Dictionary mapping genre -> description
    """
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent=user_agent,
        language='en'
    )
    
    genre_descriptions = {}
    for genre in tqdm(genres, desc="Fetching genre descriptions"):
        description = get_wikipedia_description_from_title(genre, wiki_wiki)
        genre_descriptions[genre] = description
    
    return genre_descriptions


def load_genre_descriptions_csv(csv_path: str) -> pd.DataFrame:
    """
    Load genre descriptions from CSV file.
    
    Parameters:
    - csv_path: Path to CSV file
    
    Returns:
    - DataFrame with 'genre' and 'description' columns
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['genre', 'description'])
    
    return pd.read_csv(csv_path)


def save_genre_descriptions_csv(genre_descriptions: Dict[str, str], csv_path: str):
    """
    Save genre descriptions to CSV file.
    
    Parameters:
    - genre_descriptions: Dictionary mapping genre -> description
    - csv_path: Path to save CSV file
    """
    df = pd.DataFrame.from_dict(
        genre_descriptions, 
        orient='index', 
        columns=['description']
    ).reset_index().rename(columns={'index': 'genre'})
    
    df.to_csv(csv_path, index=False)


def get_missing_genres(
    all_genres: List[str],
    existing_df: pd.DataFrame
) -> List[str]:
    """
    Identify genres that are missing from the existing descriptions DataFrame.
    
    Parameters:
    - all_genres: List of all genre strings to check
    - existing_df: DataFrame with existing genre descriptions
    
    Returns:
    - List of genres that are missing or have empty descriptions
    """
    if existing_df.empty:
        return all_genres
    
    existing_genres = set(existing_df['genre'].values)
    missing_genres = [g for g in all_genres if g not in existing_genres]
    
    # Also check for genres with empty descriptions
    empty_descriptions = existing_df[
        (existing_df['description'].isna()) | 
        (existing_df['description'].str.strip() == '')
    ]['genre'].tolist()
    
    missing_genres.extend(empty_descriptions)
    return list(set(missing_genres))


def ensure_genre_descriptions(
    genres: List[str],
    csv_path: str = "genre_description_df.csv",
    user_agent: str = 'GroupDataLiteracy/1.0 (movie data pipeline)'
) -> pd.DataFrame:
    """
    Ensure all genres have descriptions, fetching missing ones from Wikipedia.
    
    Parameters:
    - genres: List of all genre strings
    - csv_path: Path to CSV file for caching descriptions
    - user_agent: User agent string for Wikipedia API
    
    Returns:
    - DataFrame with 'genre' and 'description' columns
    """
    # Load existing descriptions
    genre_df = load_genre_descriptions_csv(csv_path)
    
    # Find missing genres
    missing_genres = get_missing_genres(genres, genre_df)
    
    if missing_genres:
        print(f"Fetching descriptions for {len(missing_genres)} missing genres...")
        # Fetch missing descriptions
        new_descriptions = fetch_genre_descriptions(missing_genres, user_agent)
        
        # Update DataFrame with new descriptions
        for genre, description in new_descriptions.items():
            if genre in genre_df['genre'].values:
                # Update existing row
                genre_df.loc[genre_df['genre'] == genre, 'description'] = description
            else:
                # Add new row
                new_row = pd.DataFrame({'genre': [genre], 'description': [description]})
                genre_df = pd.concat([genre_df, new_row], ignore_index=True)
        
        # Save updated descriptions
        save_genre_descriptions_csv(
            dict(zip(genre_df['genre'], genre_df['description'])),
            csv_path
        )
    else:
        print("All genre descriptions already available.")
    
    # Fill missing descriptions with genre name itself
    genre_df.loc[genre_df['description'].isna() | (genre_df['description'].str.strip() == ''), 'description'] = \
        genre_df.loc[genre_df['description'].isna() | (genre_df['description'].str.strip() == ''), 'genre']
    
    return genre_df


def embed_genre_descriptions(
    genre_df: pd.DataFrame,
    model_name: str = 'Qwen/Qwen3-Embedding-0.6B'
) -> np.ndarray:
    """
    Embed genre descriptions using sentence transformers.
    
    Parameters:
    - genre_df: DataFrame with 'description' column
    - model_name: Name of the sentence transformer model
    
    Returns:
    - Numpy array of embeddings (n_genres x embedding_dim)
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Embedding genre descriptions...")
    descriptions = genre_df['description'].tolist()
    print(f"Descriptions length: {len(descriptions)}")
    print(f"Descriptions avg length: {sum(len(d) for d in descriptions) / len(descriptions)}")
    embeddings = model.encode(descriptions, max_length=512)
    
    return embeddings


def cluster_genres(
    genre_df: pd.DataFrame,
    embeddings: np.ndarray,
    n_clusters: int = 25,
    random_state: int = 123
) -> pd.DataFrame:
    """
    Cluster genres using KMeans based on their embeddings.
    
    Parameters:
    - genre_df: DataFrame with 'genre' column
    - embeddings: Numpy array of genre embeddings
    - n_clusters: Number of clusters
    - random_state: Random state for reproducibility
    
    Returns:
    - DataFrame with added 'cluster' column
    """
    print(f"Clustering genres into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    
    genre_df = genre_df.copy()
    genre_df['cluster'] = labels
    
    return genre_df


def create_genre_mapping(
    genre_df: pd.DataFrame,
    cluster_label_mapping: Optional[Dict[int, str]] = None
) -> Dict[str, str]:
    """
    Create genre mapping dictionary from clustered genres.
    
    Parameters:
    - genre_df: DataFrame with 'genre' and 'cluster' columns
    - cluster_label_mapping: Optional mapping from cluster ID to label name
    
    Returns:
    - Dictionary mapping genre -> cluster_label
    """
    if cluster_label_mapping is None:
        # Use cluster IDs as labels if no mapping provided
        genre_df['new_label'] = genre_df['cluster'].astype(str)
    else:
        genre_df['new_label'] = genre_df['cluster'].apply(
            lambda x: cluster_label_mapping.get(x, f"cluster_{x}")
        )
    
    # Create mapping dictionary
    mapping_dict = {}
    for _, row in genre_df.iterrows():
        # Clean genre name (lowercase, remove "film", strip)
        cleaned_genre = row['genre'].lower().replace("film", "").strip()
        mapping_dict[cleaned_genre] = row['new_label']
    
    return mapping_dict


def save_genre_mapping(
    mapping_dict: Dict[str, str],
    output_path: str = "src/genre_fix_mapping.json"
):
    """
    Save genre mapping to JSON file.
    
    Parameters:
    - mapping_dict: Dictionary mapping genre -> cluster_label
    - output_path: Path to save JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(mapping_dict, f, indent=2)
    
    print(f"Genre mapping saved to {output_path}")


def print_genres_by_cluster(
    genre_df: pd.DataFrame,
    cluster_label_mapping: Optional[Dict[int, str]] = None
):
    """
    Print genres grouped by their cluster labels.
    
    Parameters:
    - genre_df: DataFrame with 'genre' and 'cluster' columns
    - cluster_label_mapping: Optional mapping from cluster ID to label name
    """
    # Create a copy to avoid modifying original
    print_df = genre_df.copy()
    
    # Sort by cluster
    print_df = print_df.sort_values('cluster')
    
    # Get unique clusters
    unique_clusters = sorted(print_df['cluster'].unique())
    
    print("\n" + "=" * 80)
    print("Genres by Cluster:")
    print("=" * 80)
    
    for cluster_id in unique_clusters:
        cluster_genres = print_df[print_df['cluster'] == cluster_id]['genre'].tolist()
        
        # Create label for cluster
        if cluster_label_mapping and cluster_id in cluster_label_mapping:
            cluster_label = f"Cluster {cluster_id} ({cluster_label_mapping[cluster_id]})"
        else:
            cluster_label = f"Cluster {cluster_id}"
        
        print(f"\n{cluster_label}:")
        print(f"  {', '.join(cluster_genres)}")
        print(f"  ({len(cluster_genres)} genres)")
    
    print("=" * 80)


def classify_genres(
    movie_df: pd.DataFrame,
    descriptions_csv_path: str = "genre_description_df.csv",
    output_json_path: str = "src/genre_fix_mapping.json",
    n_clusters: int = 25,
    model_name: str = 'Qwen/Qwen3-Embedding-0.6B',
    cluster_label_mapping: Optional[Dict[int, str]] = None,
    random_state: int = 123
) -> Dict[str, str]:
    """
    Main function to classify genres from movie dataframe.
    
    This function:
    1. Extracts unique genres from movie_df
    2. Fetches Wikipedia descriptions (cached in CSV)
    3. Embeds descriptions using sentence transformers
    4. Clusters genres
    5. Generates and saves mapping JSON
    
    Parameters:
    - movie_df: DataFrame with 'genre' column
    - descriptions_csv_path: Path to CSV file for caching descriptions
    - output_json_path: Path to save output JSON mapping
    - n_clusters: Number of clusters for KMeans
    - model_name: Sentence transformer model name
    - cluster_label_mapping: Optional mapping from cluster ID to label name
    - random_state: Random state for reproducibility
    
    Returns:
    - Dictionary mapping genre -> cluster_label
    """
    print("=" * 60)
    print("Genre Classification Pipeline")
    print("=" * 60)
    
    # Step 1: Extract unique genres
    print("\nStep 1: Extracting unique genres...")
    unique_genres = extract_unique_genres(movie_df)
    print(f"Found {len(unique_genres)} unique genres")
    
    # Step 2: Ensure all genres have descriptions
    print("\nStep 2: Ensuring genre descriptions...")
    genre_df = ensure_genre_descriptions(unique_genres, descriptions_csv_path)
    print(f"Total genres with descriptions: {len(genre_df)}")
    
    # Step 3: Embed descriptions
    print("\nStep 3: Embedding genre descriptions...")
    embeddings = embed_genre_descriptions(genre_df, model_name)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Step 4: Cluster genres
    print("\nStep 4: Clustering genres...")
    genre_df_clustered = cluster_genres(genre_df, embeddings, n_clusters, random_state)
    
    # Step 5: Create mapping
    print("\nStep 5: Creating genre mapping...")
    mapping_dict = create_genre_mapping(genre_df_clustered, cluster_label_mapping)
    print(f"Created mapping for {len(mapping_dict)} genres")
    
    # Step 6: Save mapping
    print("\nStep 6: Saving genre mapping...")
    save_genre_mapping(mapping_dict, output_json_path)

    print("\n" + "=" * 60)
    print("Genre classification complete!")
    print("=" * 60)
    
    return mapping_dict, genre_df_clustered


if __name__ == "__main__":

    # Default paths
    data_dir = os.path.join(base_path, "data", "data_final")
    descriptions_csv_path = os.path.join(base_path, "src", "analysis", "genre_description_df.csv")
    output_json_path = os.path.join(base_path, "src", "genre_fix_mapping_new.json")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please provide the correct data directory path.")
        sys.exit(1)
    
    print("Loading movie data...")
    movie_df = load_movie_data(data_dir, verbose=True)
    movie_df = clean_dataset(movie_df)
    
    if movie_df.empty:
        print("Error: No movie data loaded.")
        sys.exit(1)
    
    # Run classification
    print("\nStarting genre classification...")
    mapping_dict, genre_df_clustered = classify_genres(
        movie_df,
        descriptions_csv_path=descriptions_csv_path,
        output_json_path=output_json_path,
        n_clusters=20
    )

    print_genres_by_cluster(genre_df_clustered, mapping_dict)

    