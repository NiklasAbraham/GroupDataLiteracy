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
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import wikipediaapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add parent directory to path for imports
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))

from src.data_utils import load_movie_data, preprocess_genres
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


def extract_genre_to_id_mapping(movie_df: pd.DataFrame) -> Dict[str, str]:
    """
    Extract mapping from genre names to their Wikidata IDs (wids).
    
    Parameters:
    - movie_df: DataFrame with 'genre' and 'genre_id' columns
    
    Returns:
    - Dictionary mapping genre -> genre_id (comma-separated if multiple IDs)
    """
    if "genre" not in movie_df.columns or "genre_id" not in movie_df.columns:
        return {}
    
    genre_to_id = {}
    
    # Process rows with both genre and genre_id
    for _, row in movie_df.iterrows():
        if pd.isna(row.get('genre')) or pd.isna(row.get('genre_id')):
            continue
        
        genres = [g.strip() for g in str(row['genre']).split(",")]
        genre_ids = [gid.strip() for gid in str(row['genre_id']).split(",")]
        
        # Match genres with their corresponding IDs
        for i, genre in enumerate(genres):
            if i < len(genre_ids):
                genre_id = genre_ids[i]
                # If genre already exists, append ID if different
                if genre in genre_to_id:
                    existing_ids = set(genre_to_id[genre].split(","))
                    if genre_id not in existing_ids:
                        genre_to_id[genre] = ",".join(sorted(existing_ids | {genre_id}))
                else:
                    genre_to_id[genre] = genre_id
    
    return genre_to_id


def clean_description(description: str) -> str:
    """
    Clean description text by removing newlines, normalizing whitespace, and handling special characters.
    
    Parameters:
    - description: Raw description string
    
    Returns:
    - Cleaned description string
    """
    if pd.isna(description) or not isinstance(description, str):
        return ""
    
    # Replace all types of line breaks with spaces
    cleaned = re.sub(r'\r\n|\r|\n', ' ', description)
    
    # Replace multiple consecutive spaces/tabs with single space
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Remove any remaining control characters except spaces
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


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
    - Cleaned Wikipedia summary string, or empty string if not found
    """
    try:
        # Convert genre title to Wikipedia URL format
        wiki_url = f"https://en.wikipedia.org/wiki/{genre_title.lower().replace(' ', '_')}"
        genre_page = get_page_from_url(wiki_wiki, wiki_url)
        
        if genre_page is None or not genre_page.exists():
            return ""
        
        raw_summary = genre_page.summary
        return clean_description(raw_summary)
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
    Load genre descriptions from CSV file and clean them.
    
    Parameters:
    - csv_path: Path to CSV file
    
    Returns:
    - DataFrame with 'genre' and 'description' columns (cleaned)
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['genre', 'description'])
    
    df = pd.read_csv(csv_path)
    
    # Clean descriptions
    if 'description' in df.columns:
        df['description'] = df['description'].apply(clean_description)
    
    return df


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

    # Filter out rows with NaN genres or descriptions
    genre_df = genre_df.dropna(subset=['genre', 'description'])
    # Also filter out non-string genres
    genre_df = genre_df[genre_df['genre'].apply(lambda x: isinstance(x, str))].copy()
    
    # Clean all descriptions in the DataFrame
    if 'description' in genre_df.columns:
        genre_df['description'] = genre_df['description'].apply(clean_description)
        # Remove rows with empty descriptions after cleaning
        genre_df = genre_df[genre_df['description'].str.strip() != ''].copy()

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
    # Only calculate average length for string descriptions, skip non-strings (e.g., floats)
    str_lengths = [len(d) for d in descriptions if isinstance(d, str)]
    avg_length = sum(str_lengths) / len(str_lengths) if str_lengths else 0
    print(f"Descriptions avg length: {avg_length}")
    print(f"First 10 descriptions: {descriptions[:10]}")
    embeddings = model.encode(descriptions)
    
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
        # Skip if genre is NaN or not a string
        if pd.isna(row['genre']) or not isinstance(row['genre'], str):
            continue
        
        # Clean genre name (lowercase, remove "film", strip)
        cleaned_genre = row['genre'].lower().replace("film", "").strip()
        
        # Skip if cleaned genre is empty
        if not cleaned_genre:
            continue
        
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
    mapping_json_path: str = "src/genre_fix_mapping.json",
    cluster_label_mapping: Optional[Dict[int, str]] = None,
    genre_to_id: Optional[Dict[str, str]] = None
):
    """
    Print genres grouped by their cluster labels using the genre mapping from data_utils.py.
    
    Parameters:
    - genre_df: DataFrame with 'genre' and 'cluster' columns
    - mapping_json_path: Path to the genre mapping JSON file
    - cluster_label_mapping: Optional mapping from cluster ID to label name (for display)
    - genre_to_id: Optional mapping from genre name to Wikidata ID (wid)
    """
    # Load genre mapping from JSON file (same as data_utils.py)
    try:
        with open(mapping_json_path, "r") as f:
            genre_mapping = json.loads(f.read())
    except FileNotFoundError:
        print(f"Warning: Genre mapping file not found at {mapping_json_path}")
        print("Falling back to cluster-based grouping...")
        genre_mapping = {}
    except Exception as e:
        print(f"Warning: Error loading genre mapping: {e}")
        print("Falling back to cluster-based grouping...")
        genre_mapping = {}
    
    # Create a copy to avoid modifying original
    print_df = genre_df.copy()
    
    # Filter out NaN values
    print_df = print_df[print_df['genre'].notna() & print_df['genre'].apply(lambda x: isinstance(x, str))].copy()
    
    # Create a mapping from genre to cluster label using the JSON mapping
    # Clean genre names the same way as data_utils.py does
    genre_to_cluster_label = {}
    for _, row in print_df.iterrows():
        genre = row['genre']
        cleaned_genre = genre.lower().replace("film", "").strip()
        
        if cleaned_genre in genre_mapping:
            cluster_label = genre_mapping[cleaned_genre]
        else:
            # If not in mapping, use cluster ID as string
            cluster_label = str(row['cluster'])
        
        if cluster_label not in genre_to_cluster_label:
            genre_to_cluster_label[cluster_label] = []
        genre_to_cluster_label[cluster_label].append(genre)
    
    # Sort cluster labels (try as int first, then as string)
    def sort_key(label):
        try:
            return int(label)
        except ValueError:
            return float('inf')
    
    sorted_labels = sorted(genre_to_cluster_label.keys(), key=sort_key)
    
    print("\n" + "=" * 80)
    print("Genres by Cluster Label (from genre mapping):")
    print("=" * 80)
    
    for cluster_label in sorted_labels:
        genres = sorted(genre_to_cluster_label[cluster_label])
        
        # Create display label
        if cluster_label_mapping:
            # Try to find cluster ID from the label
            try:
                cluster_id = int(cluster_label)
                if cluster_id in cluster_label_mapping:
                    display_label = f"Cluster {cluster_label} ({cluster_label_mapping[cluster_id]})"
                else:
                    display_label = f"Cluster {cluster_label}"
            except ValueError:
                display_label = f"Cluster {cluster_label}"
        else:
            display_label = f"Cluster {cluster_label}"
        
        print(f"\n{display_label}:")
        # Format genres with IDs if available
        if genre_to_id:
            genre_strings = []
            for genre in genres:
                if genre in genre_to_id:
                    genre_strings.append(f"{genre} ({genre_to_id[genre]})")
                else:
                    genre_strings.append(genre)
            print(f"  {', '.join(genre_strings)}")
        else:
            print(f"  {', '.join(genres)}")
        print(f"  ({len(genres)} genres)")
    
    print("=" * 80)


def count_movies_by_cluster(
    movie_df: pd.DataFrame,
    mapping_json_path: str = "src/genre_fix_mapping_new.json",
    cluster_label_mapping: Optional[Dict[int, str]] = None
) -> Dict[str, int]:
    """
    Count how many movies are in each cluster/genre classification using the genre mapping.
    
    Parameters:
    - movie_df: DataFrame with 'genre' column
    - mapping_json_path: Path to the genre mapping JSON file
    - cluster_label_mapping: Optional mapping from cluster ID to label name (for display)
    
    Returns:
    - Dictionary mapping cluster_label -> count of movies
    """
    if "genre" not in movie_df.columns:
        raise ValueError("DataFrame must have 'genre' column")
    
    # Apply preprocess_genres to get new genre classifications
    print("\nApplying genre mapping to movies...")
    movie_df = movie_df.copy()
    movie_df['new_genre'] = movie_df['genre'].apply(preprocess_genres)
    
    # Count movies per cluster
    # Since a movie can have multiple genres (separated by |), we need to count each occurrence
    cluster_counts = {}
    unique_movie_counts = {}  # Count unique movies per cluster
    
    for _, row in movie_df.iterrows():
        new_genre = row['new_genre']
        
        # Skip Unknown genres
        if new_genre == "Unknown" or pd.isna(new_genre):
            continue
        
        # Split by | to get individual cluster labels
        cluster_labels = new_genre.split("|")
        
        # Track unique clusters for this movie
        movie_clusters = set()
        
        for cluster_label in cluster_labels:
            cluster_label = cluster_label.strip()
            if not cluster_label:
                continue
            
            # Count total occurrences
            cluster_counts[cluster_label] = cluster_counts.get(cluster_label, 0) + 1
            movie_clusters.add(cluster_label)
        
        # Count unique movies per cluster
        # Use movie_id if available, otherwise use index
        movie_identifier = row.get('movie_id') if 'movie_id' in row and pd.notna(row.get('movie_id')) else row.name
        
        for cluster_label in movie_clusters:
            if cluster_label not in unique_movie_counts:
                unique_movie_counts[cluster_label] = set()
            unique_movie_counts[cluster_label].add(movie_identifier)
    
    # Convert unique movie sets to counts
    unique_movie_counts_final = {k: len(v) for k, v in unique_movie_counts.items()}
    
    return cluster_counts, unique_movie_counts_final


def print_movie_counts_by_cluster(
    cluster_counts: Dict[str, int],
    unique_movie_counts: Dict[str, int],
    cluster_label_mapping: Optional[Dict[int, str]] = None
):
    """
    Print movie counts per cluster.
    
    Parameters:
    - cluster_counts: Dictionary mapping cluster_label -> total count (including duplicates from multi-genre movies)
    - unique_movie_counts: Dictionary mapping cluster_label -> unique movie count
    - cluster_label_mapping: Optional mapping from cluster ID to label name (for display)
    """
    print("\n" + "=" * 80)
    print("Movie Counts by Cluster (using new genre mapping):")
    print("=" * 80)
    
    # Sort cluster labels (try as int first, then as string)
    def sort_key(label):
        try:
            return int(label)
        except ValueError:
            return float('inf')
    
    sorted_labels = sorted(cluster_counts.keys(), key=sort_key)
    
    print(f"\n{'Cluster Label':<20} {'Total Count':<15} {'Unique Movies':<15}")
    print("-" * 80)
    
    total_movies = 0
    total_unique = 0
    
    for cluster_label in sorted_labels:
        total_count = cluster_counts[cluster_label]
        unique_count = unique_movie_counts.get(cluster_label, 0)
        total_movies += total_count
        total_unique += unique_count
        
        # Create display label
        if cluster_label_mapping:
            try:
                cluster_id = int(cluster_label)
                if cluster_id in cluster_label_mapping:
                    display_label = f"{cluster_label} ({cluster_label_mapping[cluster_id]})"
                else:
                    display_label = cluster_label
            except ValueError:
                display_label = cluster_label
        else:
            display_label = cluster_label
        
        print(f"{display_label:<20} {total_count:<15} {unique_count:<15}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_movies:<15} {total_unique:<15}")
    print("=" * 80)
    print("\nNote: Total Count includes duplicates (movies with multiple genres).")
    print("      Unique Movies counts each movie only once per cluster.")


def plot_movie_counts_by_cluster(
    cluster_counts: Dict[str, int],
    unique_movie_counts: Dict[str, int],
    cluster_label_mapping: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot movie counts per cluster.
    
    Parameters:
    - cluster_counts: Dictionary mapping cluster_label -> total count
    - unique_movie_counts: Dictionary mapping cluster_label -> unique movie count
    - cluster_label_mapping: Optional mapping from cluster ID to label name (for display)
    - figsize: Figure size tuple
    """
    # Sort cluster labels
    def sort_key(label):
        try:
            return int(label)
        except ValueError:
            return float('inf')
    
    sorted_labels = sorted(cluster_counts.keys(), key=sort_key)
    
    # Prepare data for plotting
    labels = []
    total_counts = []
    unique_counts = []
    
    for cluster_label in sorted_labels:
        # Create display label
        if cluster_label_mapping:
            try:
                cluster_id = int(cluster_label)
                if cluster_id in cluster_label_mapping:
                    display_label = f"{cluster_label}\n({cluster_label_mapping[cluster_id]})"
                else:
                    display_label = cluster_label
            except ValueError:
                display_label = cluster_label
        else:
            display_label = cluster_label
        
        labels.append(display_label)
        total_counts.append(cluster_counts[cluster_label])
        unique_counts.append(unique_movie_counts.get(cluster_label, 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Bar chart of total counts
    x_pos = np.arange(len(labels))
    width = 0.6
    ax1.bar(x_pos, total_counts, width, label='Total Count (with duplicates)', color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster Label', fontsize=12)
    ax1.set_ylabel('Number of Movies', fontsize=12)
    ax1.set_title('Total Movie Counts by Cluster\n(including duplicates from multi-genre movies)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Bar chart of unique movie counts
    ax2.bar(x_pos, unique_counts, width, label='Unique Movies', color='coral', alpha=0.7)
    ax2.set_xlabel('Cluster Label', fontsize=12)
    ax2.set_ylabel('Number of Unique Movies', fontsize=12)
    ax2.set_title('Unique Movie Counts by Cluster\n(each movie counted once per cluster)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPlot displayed.")


def classify_genres(
    movie_df: pd.DataFrame,
    descriptions_csv_path: str = "genre_description_df.csv",
    output_json_path: str = "src/genre_fix_mapping.json",
    n_clusters: int = 25,
    model_name: str = 'Qwen/Qwen3-Embedding-0.6B',
    cluster_label_mapping: Optional[Dict[int, str]] = None,
    random_state: int = 123,
    apply_data_cleaning: bool = True
) -> Tuple[Dict[str, str], pd.DataFrame, Dict[str, str]]:
    """
    Main function to classify genres from movie dataframe.
    
    This function:
    1. Applies data cleaning (optional)
    2. Extracts unique genres from movie_df
    3. Fetches Wikipedia descriptions (cached in CSV)
    4. Embeds descriptions using sentence transformers
    5. Clusters genres
    6. Generates and saves mapping JSON
    
    Parameters:
    - movie_df: DataFrame with 'genre' column
    - descriptions_csv_path: Path to CSV file for caching descriptions
    - output_json_path: Path to save output JSON mapping
    - n_clusters: Number of clusters for KMeans
    - model_name: Sentence transformer model name
    - cluster_label_mapping: Optional mapping from cluster ID to label name
    - random_state: Random state for reproducibility
    - apply_data_cleaning: Whether to apply data cleaning from data_cleaning.py (default: True)
    
    Returns:
    - Tuple of (mapping_dict, genre_df_clustered, genre_to_id):
      - mapping_dict: Dictionary mapping genre -> cluster_label
      - genre_df_clustered: DataFrame with 'genre', 'cluster', and 'new_label' columns
      - genre_to_id: Dictionary mapping genre -> Wikidata ID (wid)
    """
    print("=" * 60)
    print("Genre Classification Pipeline")
    print("=" * 60)
    
    # Step 0: Apply data cleaning if requested
    if apply_data_cleaning:
        print("\nStep 0: Applying data cleaning...")
        movie_df = clean_dataset(movie_df, filter_single_genres=True)
        print(f"Dataset size after cleaning: {len(movie_df)}")
    
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
    
    # Step 7: Extract genre to ID mapping
    print("\nStep 7: Extracting genre to ID mapping...")
    genre_to_id = extract_genre_to_id_mapping(movie_df)
    print(f"Extracted IDs for {len(genre_to_id)} genres")
    
    # Step 8: Print genres by cluster
    print("\nStep 8: Printing genres by cluster...")
    print_genres_by_cluster(genre_df_clustered, output_json_path, cluster_label_mapping, genre_to_id)

    print("\n" + "=" * 60)
    print("Genre classification complete!")
    print("=" * 60)
    
    return mapping_dict, genre_df_clustered, genre_to_id


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
    
    if movie_df.empty:
        print("Error: No movie data loaded.")
        sys.exit(1)
    
    # Run classification (data cleaning is applied inside classify_genres)
    print("\nStarting genre classification...")
    mapping_dict, genre_df_clustered, genre_to_id = classify_genres(
        movie_df,
        descriptions_csv_path=descriptions_csv_path,
        output_json_path=output_json_path,
        n_clusters=25,
        apply_data_cleaning=True
    )

    print_genres_by_cluster(genre_df_clustered, output_json_path, None, genre_to_id)
    
    # Step 9: Count and plot movies by cluster
    # Apply the same data cleaning that was used in classify_genres
    print("\nStep 9: Counting movies by cluster...")
    movie_df_for_counting = clean_dataset(movie_df.copy(), filter_single_genres=True)
    print(f"Using {len(movie_df_for_counting)} movies for counting (after data cleaning)")
    
    cluster_label_mapping = None  # Can be set to a dict mapping cluster IDs to names if needed
    cluster_counts, unique_movie_counts = count_movies_by_cluster(
        movie_df_for_counting,
        output_json_path,
        cluster_label_mapping
    )
    
    print_movie_counts_by_cluster(cluster_counts, unique_movie_counts, cluster_label_mapping)
    
    print("\nStep 10: Plotting movie counts by cluster...")
    plot_movie_counts_by_cluster(cluster_counts, unique_movie_counts, cluster_label_mapping)

    