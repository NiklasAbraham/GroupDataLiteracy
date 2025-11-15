from typing import Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os


def load_movie_embeddings(
    data_dir: str, verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data from "data_final" folder (can be downloaded from Google Drive)

    Parameters:
    - data_dir: absolute path for data_final folder
    - verbose: prints statistics from loading data, default = False

    Returns:
    - all_embeddings: embeddings of all movies, (len movies x  embedding size), ordered by all_movie_ids
    - all_movie_ids: ordered array of movie_ids
    """
    # Set up paths - navigate from src/analysis to data directory
    START_YEAR = 1950
    END_YEAR = 2024
    DATA_DIR = data_dir

    if verbose:
        print(f"Data directory: {DATA_DIR}")
        print(f"Year range: {START_YEAR} to {END_YEAR}")

    # Load all embeddings and corresponding movie IDs
    all_embeddings = []
    all_movie_ids = []

    for year in range(START_YEAR, END_YEAR + 1):
        embeddings_path = os.path.join(DATA_DIR, f"movie_embeddings_{year}.npy")
        movie_ids_path = os.path.join(DATA_DIR, f"movie_ids_{year}.npy")

        if os.path.exists(embeddings_path) and os.path.exists(movie_ids_path):
            embeddings = np.load(embeddings_path)
            movie_ids = np.load(movie_ids_path)

            all_embeddings.append(embeddings)
            all_movie_ids.append(movie_ids)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_movie_ids = np.concatenate(all_movie_ids)

    if verbose:
        print(f"\nTotal movies: {len(all_movie_ids)}")
        print(f"Embedding shape: {all_embeddings.shape}")

    return (
        all_embeddings,
        all_movie_ids,
    )


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
    data_dir: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Helper to load all metadata and merge with all embeddings.
    """
    all_embeddings, all_movie_ids = load_movie_embeddings(data_dir, verbose=verbose)

    embeddings_df = pd.DataFrame(
        {"movie_id": all_movie_ids, "embedding": list(all_embeddings)}
    )

    metadata_df = load_movie_data(data_dir, verbose=verbose)
    combined_df = pd.merge(metadata_df, embeddings_df, on="movie_id", how="inner")

    return combined_df


def load_movie_data_limited(
    data_dir: str, movies_per_year: int | None, verbose: bool = False
) -> pd.DataFrame:
    """
    Load a limited number of movies per year from the dataset, including embeddings.
    """
    all_movies = _load_all_data_with_embeddings(data_dir, verbose=verbose)

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
    # Read genre mapping file
    with open("genre_fix_mapping.json", "r") as f:
        genre_mapping = json.loads(f.read())

    # Split raw string into genres
    split_genres = genre.split(",")

    new_genres = []
    for g in split_genres:
        # Preprocess genres
        new_g = g.lower().replace("film", "").strip()

        # Map them to clustered genre
        mapped_genre = genre_mapping[new_g]
        new_genres.append(mapped_genre)

    # Remove duplicates
    new_genres = list(set(new_genres))

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
