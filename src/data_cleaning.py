import pandas as pd
from src.data_utils import load_movie_data
from src.api.wikidata_handler import get_wikidata_subclasses
import asyncio
import json
import os
from src.data_exploration import print_wikidata_column_appearances
import sys
from pathlib import Path

# Add parent directory to path for imports
base_path = Path(__file__).parent.parent

DATA_DIR = os.path.join(base_path, 'data', 'data_final')
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

def filter_non_movies(df: pd.DataFrame, wrong_classes_save_path: str = WRONG_CLASSES_SAVE_PATH) -> pd.DataFrame:
    if os.path.exists(wrong_classes_save_path):
        with open(wrong_classes_save_path, 'r') as f:
            wrong_classes = set(json.load(f))
    else:
        wrong_classes = asyncio.run(get_wikidata_subclasses(WRONG_WIKIDATA_CLASSES))
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

def clean_dataset(df: pd.DataFrame, wrong_classes_save_path: str = WRONG_CLASSES_SAVE_PATH, max_plot_length: int = MAX_PLOT_LENGTH, filter_single_genres: bool = True) -> pd.DataFrame:
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    print(f"Original dataset size: {len(df)}")
    df_filtered = filter_movies_without_plot(df)
    print(f"After filtering movies without plot: {len(df_filtered)}")
    df_filtered = filter_non_movies(df_filtered, wrong_classes_save_path=wrong_classes_save_path)
    print(f"After filtering non-movies: {len(df_filtered)}")
    
    if filter_single_genres:
        df_filtered = filter_movies_with_single_occurrence_genres(df_filtered)
        print(f"After removing single-occurrence genres from genre column: {len(df_filtered)}")

    df_filtered["plot_length_chars"] = df_filtered["plot"].apply(lambda x: len(x.strip()) if isinstance(x, str) else 0)
    df_filtered = df_filtered[df_filtered["plot_length_chars"] <= max_plot_length].reset_index(drop=True)
    print(f"After filtering movies with plot length > {max_plot_length} chars: {len(df_filtered)}")
    
    return df_filtered

if __name__ == "__main__":
    df = load_movie_data(DATA_DIR, verbose=False)

    clean_dataset(df)

