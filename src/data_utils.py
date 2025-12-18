from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os

def load_embeddings_as_dict(data_dir: str, start_year: int = 1930, end_year: int = 2024) -> Dict[str, np.ndarray]:
    """
    Loads all movie embeddings from .npy files in the specified directory and returns them as a dictionary.

    Parameters:
    - data_dir: absolute path for data_final folder
    - start_year: first year to load (inclusive)
    - end_year: last year to load (inclusive)

    Returns:
    - embeddings_dict: dictionary mapping movie_id (str) to embedding (np.ndarray)
    """
    embeddings_dict = {}

    for year in range(start_year, end_year + 1):
        embeddings_path = os.path.join(data_dir, f"movie_embeddings_{year}_cls_token.npy")
        movie_ids_path = os.path.join(data_dir, f"movie_ids_{year}_cls_token.npy")

        if os.path.exists(embeddings_path) and os.path.exists(movie_ids_path):
            embeddings = np.load(embeddings_path)
            movie_ids = np.load(movie_ids_path)

            for movie_id, embedding in zip(movie_ids, embeddings):
                embeddings_dict[str(movie_id)] = embedding

    return embeddings_dict


def load_final_dataset(
    csv_path: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final/final_dataset.csv",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load the final cleaned dataset CSV.
    
    This is the new recommended function for loading movie metadata.
    It loads from a single consolidated CSV file instead of year-based files.
    
    Parameters:
    - csv_path: path to final_dataset.csv
    - verbose: whether to print statistics, default = False
    
    Returns:
    - DataFrame with all movie data
    """
    if verbose:
        print(f"Loading final dataset from {csv_path}...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    if verbose:
        print(f"Loaded {len(df)} movies")
        if 'year' in df.columns:
            year_min = df['year'].min()
            year_max = df['year'].max()
            print(f"Year range: {year_min} to {year_max}")
    
    return df


def load_final_dense_embeddings(
    data_dir: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final",
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dense embeddings from the consolidated final files.
    
    This is the new recommended function for loading dense embeddings.
    It loads from consolidated files instead of year-based files.
    
    Parameters:
    - data_dir: directory containing the final embedding files
    - verbose: whether to print statistics, default = False
    
    Returns:
    - Tuple of (embeddings, movie_ids)
    """
    embeddings_path = os.path.join(data_dir, "final_dense_embeddings.npy")
    movie_ids_path = os.path.join(data_dir, "final_dense_movie_ids.npy")
    
    if verbose:
        print(f"Loading final dense embeddings from {data_dir}...")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(movie_ids_path):
        error_msg = f"Final embedding files not found in {data_dir}. "
        error_msg += "Please run src/consolidate_embeddings.py first."
        raise FileNotFoundError(error_msg)
    
    embeddings = np.load(embeddings_path)
    movie_ids = np.load(movie_ids_path, allow_pickle=True)
    
    if verbose:
        print(f"Loaded {len(movie_ids)} embeddings")
        print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings, movie_ids


def load_final_sparse_embeddings(
    data_dir: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final",
    verbose: bool = False
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]]:
    """
    Load sparse embeddings (lexical weights) from the consolidated final file.
    
    This is the new recommended function for loading sparse embeddings.
    It loads from a consolidated file instead of year-based files.
    
    Parameters:
    - data_dir: directory containing the final embedding file
    - verbose: whether to print statistics, default = False
    
    Returns:
    - Tuple of (token_indices_list, weights_list, movie_ids) or None if file not found
    """
    sparse_path = os.path.join(data_dir, "final_sparse_embeddings.npz")
    
    if verbose:
        print(f"Loading final sparse embeddings from {data_dir}...")
    
    if not os.path.exists(sparse_path):
        if verbose:
            print(f"Final sparse embedding file not found: {sparse_path}")
        return None
    
    try:
        data = np.load(sparse_path, allow_pickle=True)
        
        token_indices_array = data['token_indices']
        weights_array = data['weights']
        movie_ids = data['movie_ids']
        
        # Convert object arrays back to lists of arrays
        token_indices_list = [token_indices_array[i] for i in range(len(token_indices_array))]
        weights_list = [weights_array[i] for i in range(len(weights_array))]
        
        if verbose:
            total_non_zero = sum(len(ti) for ti in token_indices_list)
            print(f"Loaded {len(token_indices_list)} documents")
            print(f"Total non-zero weights: {total_non_zero}")
            print(f"Average non-zero weights per document: {total_non_zero / len(token_indices_list):.1f}")
        
        return token_indices_list, weights_list, movie_ids
        
    except Exception as e:
        if verbose:
            print(f"Error loading sparse embeddings: {e}")
        return None


def load_final_data_with_embeddings(
    csv_path: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final/final_dataset.csv",
    data_dir: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load the final dataset and merge with dense embeddings.
    
    This is the new recommended function for loading movie data with embeddings.
    
    Parameters:
    - csv_path: path to final_dataset.csv
    - data_dir: directory containing final embedding files
    - verbose: whether to print statistics, default = False
    
    Returns:
    - DataFrame with movie data and embeddings
    """
    # Load CSV data
    metadata_df = load_final_dataset(csv_path, verbose=verbose)
    
    # Load embeddings
    embeddings, movie_ids = load_final_dense_embeddings(data_dir, verbose=verbose)
    
    # Create embeddings dataframe
    embeddings_df = pd.DataFrame({
        "movie_id": movie_ids,
        "embedding": list(embeddings)
    })
    
    # Merge
    combined_df = pd.merge(metadata_df, embeddings_df, on="movie_id", how="inner")
    
    if verbose:
        print(f"Combined dataset: {len(combined_df)} movies with embeddings")
    
    # Cluster genres
    genres_clustered_df = cluster_genres(combined_df)
    
    return genres_clustered_df


def load_final_data_limited(
    movies_per_year: int | None,
    csv_path: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final/final_dataset.csv",
    data_dir: str = "/home/nab/Niklas/GroupDataLiteracy/data/data_final",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load a limited number of movies per year from the final dataset.
    
    This is the new recommended function for loading limited movie data.
    
    Parameters:
    - movies_per_year: number of movies to sample per year (None for all)
    - csv_path: path to final_dataset.csv
    - data_dir: directory containing final embedding files
    - verbose: whether to print statistics, default = False
    
    Returns:
    - DataFrame with limited movie data and embeddings
    """
    all_movies = load_final_data_with_embeddings(csv_path, data_dir, verbose=verbose)
    
    if all_movies.empty or movies_per_year is None or movies_per_year <= 0:
        return all_movies
    
    if 'year' not in all_movies.columns:
        if verbose:
            print("Warning: 'year' column not found, returning all movies")
        return all_movies
    
    unique_years = all_movies["year"].unique()
    
    sampled_dfs = []
    for year in unique_years:
        year_movies = all_movies[all_movies["year"] == year]
        
        # Sample up to movies_per_year from this year
        n_sample = min(len(year_movies), movies_per_year)
        sampled = year_movies.sample(n=n_sample, random_state=42)
        sampled_dfs.append(sampled)
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    if verbose:
        print(f"Sampled dataset: {len(result_df)} movies")
    
    return result_df

def preprocess_genres(genre: str) -> str:
    """
    Takes in a raw genre string containing multiple delimetered genres, returns relabelled genres.

    If a movie has multiple genres, it will be delimetered by `|`.
    """
    # Handle NaN, None, or empty values
    if pd.isna(genre) or genre is None or not str(genre).strip():
        return None
    
    # Convert to string if not already
    genre = str(genre)
    
    # Read genre mapping file from src directory
    # Get the directory where this file (data_utils.py) is located
    src_dir = Path(__file__).parent
    mapping_file = src_dir / "genre_fix_mapping_new.json"
    
    # Fallback to old mapping if new one doesn't exist
    if not mapping_file.exists():
        mapping_file = src_dir / "genre_fix_mapping.json"
    
    with open(mapping_file, "r") as f:
        genre_mapping = json.loads(f.read())

    # Split raw string into genres
    split_genres = genre.split(",")

    new_genres = []
    for g in split_genres:
        # Preprocess genres
        new_g = g.lower().replace("film", "").strip()
        
        # Skip empty genres after preprocessing
        if not new_g:
            continue

        # Map them to clustered genre (only use genres that are in the mapping)
        if new_g in genre_mapping:
            mapped_genre = genre_mapping[new_g]
            new_genres.append(mapped_genre)
        # Skip genres that are not in the mapping (don't append them)

    # Remove duplicates
    new_genres = list(set(new_genres))
    
    # Return Unknown if no genres found
    if not new_genres:
        return None

    return "|".join(new_genres)

def cluster_genres(movie_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits and cleans genre strings, embed cleaned genre strings and clusters them into 10 main genres.

    Parameters:
    - dataframe: must have "genre" column

    Returns:
    - dataframe: with new genre label
    """
    if "genre" not in movie_df.columns:
        raise ValueError("Genre column not present in provided dataframe.")

    # Preprocess genre
    movie_df["new_genre"] = movie_df["genre"].apply(preprocess_genres)
    return movie_df

def search_movies_by_keywords(
    movie_df: pd.DataFrame,
    keywords: List[str],
    search_columns: List[str] = None,
    case_sensitive: bool = False
) -> List[str]:
    """
    Search for movies in a dataframe based on keywords in specified columns.
    
    Parameters:
    - movie_df: DataFrame containing movie data
    - keywords: List of keywords to search for (all must match)
    - search_columns: List of column names to search in. If None, searches in 'title' column only.
                     Default is None (searches only in 'title')
    - case_sensitive: Whether search should be case sensitive. Default is False
    
    Returns:
    - List of movie_id (QIDs) that match the search criteria
    """
    if movie_df.empty:
        return []
    
    if search_columns is None:
        search_columns = ['title']
    
    # Validate that all search columns exist in the dataframe
    missing_columns = [col for col in search_columns if col not in movie_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in dataframe: {missing_columns}")
    
    if 'movie_id' not in movie_df.columns:
        raise ValueError("DataFrame must contain 'movie_id' column")
    
    # Filter out rows with missing movie_id
    df_filtered = movie_df[movie_df['movie_id'].notna()].copy()
    
    if df_filtered.empty:
        return []
    
    # Create a mask that starts as all True
    mask = pd.Series([True] * len(df_filtered), index=df_filtered.index)
    
    # For each keyword, check if it appears in any of the search columns
    for keyword in keywords:
        if not keyword or not keyword.strip():
            continue
            
        keyword_clean = keyword.strip()
        
        # Create a column mask: True if keyword appears in any search column for that row
        keyword_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        
        for col in search_columns:
            # Get non-null values in this column
            col_data = df_filtered[col].astype(str)
            
            # Search for keyword in this column
            if case_sensitive:
                col_mask = col_data.str.contains(keyword_clean, na=False, regex=False)
            else:
                col_mask = col_data.str.contains(keyword_clean, na=False, regex=False, case=False)
            
            keyword_mask = keyword_mask | col_mask
        
        # Update overall mask: all keywords must match (AND logic)
        mask = mask & keyword_mask
    
    # Get matching movie_ids
    matching_movie_ids = df_filtered[mask]['movie_id'].unique().tolist()
    
    return matching_movie_ids

def expand_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'new_genre' column containing codes separated
    by '|' and duplicates the movie rows so that each genre code gets its
    own row.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A new DataFrame with expanded rows, where the
                    'new_genre' column holds only one code per row.
    """
    df_expanded = df.copy()
    df_expanded['new_genre'] = df_expanded['new_genre'].str.split('|')
    df_expanded = df_expanded.explode('new_genre')

    return df_expanded

def map_genre_ids_to_strings(df: pd.DataFrame, genre_strings_json) -> pd.DataFrame:
        dict = json.load(open(genre_strings_json))
        df['new_genre'] = df['new_genre'].map(dict)
        df.dropna(subset=['new_genre'], inplace=True)

        return df

def keep_top_n_genres(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if n == 0:
        df["old_genre"] = df["new_genre"]
        df["new_genre"] = "all_genres"
        return df

    genre_counts = df['new_genre'].value_counts()
    top_n_genres = genre_counts.head(n).index.tolist()
    df['top_genre'] = df['new_genre'].isin(top_n_genres)
    df = df[df['top_genre']]

    return df


def keep_selected_top_genres(df: pd.DataFrame, top_indices: List[int]) -> pd.DataFrame:
    """
    Example:
        # Keep the 1st and 4th most common genres
        df = keep_selected_top_genres(df, [0, 3])
    """
    genre_counts = df['new_genre'].value_counts()
    # Get genres at specified indices
    selected_genres = [genre_counts.index[i] for i in top_indices if i < len(genre_counts)]
    # Filter dataframe
    df = df[df['new_genre'].isin(selected_genres)]

    return df


def keep_x_top_genres(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    genre_counts = df['new_genre'].value_counts()
    top_n_genres = genre_counts.head(n).index.tolist()


def drop_nan_in_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Drops rows with NaN values in the specified column."""
    return df.dropna(subset=[column_name])

def filter_movies_with_given_genre(
    df: pd.DataFrame,
    genre_id: str,
    genre_id_column: str = "genre_cluster_ids"
) -> pd.DataFrame:
    def movie_has_given_genre(genre_ids_str: str) -> bool:
        if pd.isna(genre_ids_str):
            return False
        genres = genre_ids_str.split(',')
        return genre_id in genres

    filtered_df = df[df[genre_id_column].apply(movie_has_given_genre)]
    return filtered_df
