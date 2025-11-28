from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os


def load_movie_embeddings(
    data_dir: str, 
    chunking_suffix: str = "_cls_token",
    start_year: int = 1950,
    end_year: int = 2024,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads embeddings from data directory matching the file structure from data_pipeline.py.
    
    Files are expected to be named:
    - movie_embeddings_{year}{chunking_suffix}.npy
    - movie_ids_{year}{chunking_suffix}.npy
    
    For example, with chunking_suffix="_cls_token":
    - movie_embeddings_2024_cls_token.npy
    - movie_ids_2024_cls_token.npy

    Parameters:
    - data_dir: absolute path for data directory
    - chunking_suffix: suffix appended to filename (e.g., "_cls_token", "_mean_pooling", "")
                      Default is "_cls_token" to match current data structure
    - start_year: first year to load (default: 1950)
    - end_year: last year to load (default: 2024)
    - verbose: prints statistics from loading data, default = False

    Returns:
    - all_embeddings: embeddings of all movies, (len movies x embedding size), ordered by all_movie_ids
    - all_movie_ids: ordered array of movie_ids
    """
    DATA_DIR = data_dir

    if verbose:
        print(f"Data directory: {DATA_DIR}")
        print(f"Year range: {start_year} to {end_year}")
        print(f"Chunking suffix: '{chunking_suffix}'")

    # Load all embeddings and corresponding movie IDs
    all_embeddings = []
    all_movie_ids = []
    years_loaded = []

    for year in range(start_year, end_year + 1):
        embeddings_path = os.path.join(DATA_DIR, f"movie_embeddings_{year}{chunking_suffix}.npy")
        movie_ids_path = os.path.join(DATA_DIR, f"movie_ids_{year}{chunking_suffix}.npy")

        if os.path.exists(embeddings_path) and os.path.exists(movie_ids_path):
            try:
                embeddings = np.load(embeddings_path)
                movie_ids = np.load(movie_ids_path)

                all_embeddings.append(embeddings)
                all_movie_ids.append(movie_ids)
                years_loaded.append(year)
                
                if verbose:
                    print(f"Year {year}: Loaded {len(movie_ids)} embeddings (shape: {embeddings.shape})")
            except Exception as e:
                if verbose:
                    print(f"Year {year}: Error loading files - {e}")
                continue
        elif verbose:
            print(f"Year {year}: Files not found (skipping)")

    if not all_embeddings:
        if verbose:
            print("No embeddings found!")
        return np.array([]), np.array([])

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_movie_ids = np.concatenate(all_movie_ids)

    if verbose:
        print(f"\nTotal movies: {len(all_movie_ids)}")
        print(f"Embedding shape: {all_embeddings.shape}")
        print(f"Years loaded: {len(years_loaded)} ({min(years_loaded) if years_loaded else 'N/A'} to {max(years_loaded) if years_loaded else 'N/A'})")

    return (
        all_embeddings,
        all_movie_ids,
    )


def load_lexical_weights(
    data_dir: str,
    chunking_suffix: str = "_cls_token",
    start_year: int = 1950,
    end_year: int = 2024,
    verbose: bool = False
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]]:
    """
    Load lexical weights from data directory matching the file structure from data_pipeline.py.
    
    Files are expected to be named:
    - movie_lexical_weights_{year}{chunking_suffix}.npz
    
    For example, with chunking_suffix="_cls_token":
    - movie_lexical_weights_2024_cls_token.npz
    
    Parameters:
    - data_dir: absolute path for data directory
    - chunking_suffix: suffix appended to filename (e.g., "_cls_token", "_mean_pooling", "")
                      Default is "_cls_token" to match current data structure
    - start_year: first year to load (default: 1950)
    - end_year: last year to load (default: 2024)
    - verbose: prints statistics from loading data, default = False
    
    Returns:
    - Tuple of (token_indices_list, weights_list, movie_ids) or None if no files found
      - token_indices_list: List of numpy arrays, one per document, containing token IDs
      - weights_list: List of numpy arrays, one per document, containing corresponding weights
      - movie_ids: Array of movie IDs corresponding to the lexical weights
      - lexical_weights[i] corresponds to movie_ids[i]
    """
    DATA_DIR = data_dir

    if verbose:
        print(f"Data directory: {DATA_DIR}")
        print(f"Year range: {start_year} to {end_year}")
        print(f"Chunking suffix: '{chunking_suffix}'")

    # Load all lexical weights and corresponding movie IDs
    all_token_indices = []
    all_weights = []
    all_movie_ids = []
    years_loaded = []

    for year in range(start_year, end_year + 1):
        lexical_weights_path = os.path.join(DATA_DIR, f"movie_lexical_weights_{year}{chunking_suffix}.npz")

        if os.path.exists(lexical_weights_path):
            try:
                data = np.load(lexical_weights_path, allow_pickle=True)
                
                # Extract arrays
                token_indices_array = data['token_indices']
                weights_array = data['weights']
                movie_ids = data['movie_ids']
                
                # Convert object arrays back to lists of arrays
                token_indices_list = [token_indices_array[i] for i in range(len(token_indices_array))]
                weights_list = [weights_array[i] for i in range(len(weights_array))]
                
                all_token_indices.extend(token_indices_list)
                all_weights.extend(weights_list)
                all_movie_ids.append(movie_ids)
                years_loaded.append(year)
                
                if verbose:
                    total_non_zero = sum(len(ti) for ti in token_indices_list)
                    print(f"Year {year}: Loaded {len(token_indices_list)} documents ({total_non_zero} non-zero weights)")
            except Exception as e:
                if verbose:
                    print(f"Year {year}: Error loading file - {e}")
                continue
        elif verbose:
            print(f"Year {year}: File not found (skipping)")

    if not all_token_indices:
        if verbose:
            print("No lexical weights found!")
        return None

    # Concatenate movie_ids
    all_movie_ids = np.concatenate(all_movie_ids)

    if verbose:
        total_non_zero = sum(len(ti) for ti in all_token_indices)
        print(f"\nTotal documents: {len(all_token_indices)}")
        print(f"Total non-zero weights: {total_non_zero}")
        print(f"Average non-zero weights per document: {total_non_zero / len(all_token_indices):.1f}")
        print(f"Years loaded: {len(years_loaded)} ({min(years_loaded) if years_loaded else 'N/A'} to {max(years_loaded) if years_loaded else 'N/A'})")

    return all_token_indices, all_weights, all_movie_ids


def load_movie_data_1(data_dir, verbose=False):
    """
    Loads movie features, e.g. Year, Director, Rating, etc.

    DIFFERENCE FROM load_movie_data:
    - Uses hardcoded year range (1950-2024) and expects files named wikidata_movies_YYYY.csv
    - Only loads 3 specific columns: movie_id, genre, title
    - Simpler implementation with less error handling
    - Less flexible: cannot handle year range files (e.g., "1950_to_2024.csv")

    Use load_movie_data() instead for more robust file discovery and full column loading.

    Parameters:
    - data_dir: absolute path of data_final folder
    - verbose: whether to print statistics and debugging statements, default = False

    Returns:
    - movie_data: pd.DataFrame
    """
    START_YEAR = 1950
    END_YEAR = 2024
    DATA_DIR = data_dir
    # Load movie metadata from CSV files to get genres
    movie_data_list = []
    for year in range(START_YEAR, END_YEAR + 1):
        csv_path = os.path.join(DATA_DIR, f"wikidata_movies_{year}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype=str)
            # Select only the relevant columns
            df_year = df[["movie_id", "genre", "title"]].copy()
            df_year["year"] = year
            movie_data_list.append(df_year)

    # Combine all movie data
    movie_data = pd.concat(movie_data_list, ignore_index=True)

    return movie_data


def find_year_files(data_dir: str) -> Dict[int, str]:
    """
    Find all CSV files matching the pattern wikidata_movies_YYYY.csv

    Returns:
        Dictionary mapping year to file path
    """
    year_files = {}
    data_path = Path(data_dir)

    for csv_file in data_path.glob("wikidata_movies_*.csv"):
        # Extract year from filename (e.g., wikidata_movies_1950.csv -> 1950)
        try:
            year_str = csv_file.stem.split("_")[-1]
            # Handle files like "1950_to_2024" by taking first year
            if "to" in year_str:
                year_str = year_str.split("_to_")[0]
            year = int(year_str)
            if year not in year_files:  # Prefer single year files over range files
                year_files[year] = str(csv_file)
        except (ValueError, IndexError):
            continue

    return year_files


def load_movie_data(data_dir: str, verbose: bool = False) -> pd.DataFrame:
    """
    Loads movie features, e.g. Year, Director, Rating, etc.

    DIFFERENCE FROM load_movie_data_1:
    - Dynamically discovers CSV files using find_year_files() instead of hardcoded year range
    - Loads ALL columns from CSV files (not just movie_id, genre, title)
    - Can handle year range files (e.g., "wikidata_movies_1950_to_2024.csv")
    - More robust error handling with try/except blocks
    - Better verbose output showing which files were loaded

    This is the recommended function to use for loading movie data.

    Parameters:
    - data_dir: absolute path of data_final folder
    - verbose: whether to print statistics and debugging statements, default = False

    Returns:
    - movie_data: pd.DataFrame
    """
    year_files = find_year_files(data_dir)

    if not year_files:
        if verbose:
            print(f"No year-specific CSV files found in {data_dir}")
        return pd.DataFrame()

    if verbose:
        print(f"Found {len(year_files)} year files")

    all_dataframes = []

    for year in sorted(year_files.keys()):
        file_path = year_files[year]
        try:
            df = pd.read_csv(file_path, dtype=str, low_memory=False)
            df["year"] = year  # Ensure year column is set
            all_dataframes.append(df)
            if verbose:
                print(
                    f"Year {year}: Loaded {len(df)} movies from {Path(file_path).name}"
                )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        if verbose:
            print(f"Combined dataset: {len(combined_df)} total movies")
    else:
        combined_df = pd.DataFrame()

    return combined_df


def _load_all_data_with_embeddings(
    data_dir: str, 
    chunking_suffix: str = "_cls_token",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Helper to load all metadata and merge with all embeddings.
    
    Parameters:
    - data_dir: absolute path for data directory
    - chunking_suffix: suffix appended to filename (e.g., "_cls_token", "_mean_pooling", "")
                      Default is "_cls_token" to match current data structure
    - verbose: prints statistics from loading data, default = False
    """
    all_embeddings, all_movie_ids = load_movie_embeddings(
        data_dir, 
        chunking_suffix=chunking_suffix,
        verbose=verbose
    )

    embeddings_df = pd.DataFrame(
        {"movie_id": all_movie_ids, "embedding": list(all_embeddings)}
    )

    metadata_df = load_movie_data(data_dir, verbose=verbose)
    combined_df = pd.merge(metadata_df, embeddings_df, on="movie_id", how="inner")

    return combined_df


def load_movie_data_limited(
    data_dir: str, 
    movies_per_year: int | None, 
    chunking_suffix: str = "_cls_token",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load a limited number of movies per year from the dataset, including embeddings.
    
    Parameters:
    - data_dir: absolute path for data directory
    - movies_per_year: number of movies to sample per year (None for all)
    - chunking_suffix: suffix appended to filename (e.g., "_cls_token", "_mean_pooling", "")
                      Default is "_cls_token" to match current data structure
    - verbose: prints statistics from loading data, default = False
    """
    all_movies = _load_all_data_with_embeddings(
        data_dir, 
        chunking_suffix=chunking_suffix,
        verbose=verbose
    )

    if all_movies.empty or movies_per_year is None or movies_per_year <= 0:
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

    return result_df


def preprocess_genres(genre: str) -> str:
    """
    Takes in a raw genre string containing multiple delimetered genres, returns relabelled genres.

    If a movie has multiple genres, it will be delimetered by `|`.
    """
    # Handle NaN, None, or empty values
    if pd.isna(genre) or genre is None or not str(genre).strip():
        return "Unknown"
    
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
        return "Unknown"

    return "|".join(new_genres)

def load_and_preprocess_data(data_path, n_movies_per_year, start_year, top_x_genres):
    """
    Loads movie data, cleans missing values, splits multi-genre entries,
    and filters the data to include only the top X most frequent genres.

    (Use only for non cleaned data, we should use the clustered genres)
    """
    print(f"\nLoading movie data from {data_path}...")
    df = load_movie_data_limited(data_path, n_movies_per_year, verbose=False)
    print(f"Initial number of movies: {df.shape[0]}")

    # Remove rows with missing genre, year, or embedding, and filter by start year
    df_cleaned = df.dropna(subset=['genre', 'year', 'embedding']).copy()
    df_cleaned = df_cleaned[df_cleaned['year'] >= start_year]
    print(f"Number of movies after cleaning: {df_cleaned.shape[0]}")

    # Split and explode multi-genre entries
    df_cleaned['genre_list'] = df_cleaned['genre'].str.split(', ')
    df_exploded = df_cleaned.explode('genre_list')
    df_exploded.rename(columns={'genre_list': 'single_genre'}, inplace=True)
    print(f"Number of movie-genre entries after exploding: {df_exploded.shape[0]}")

    # Filter for the top X genres
    top_genres = df_exploded['single_genre'].value_counts().nlargest(top_x_genres).index.tolist()
    df_filtered = df_exploded[df_exploded['single_genre'].isin(top_genres)].copy()
    print(f"Top {top_x_genres} genres: {df_filtered.shape[0]} total entries")

    return df_filtered

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
