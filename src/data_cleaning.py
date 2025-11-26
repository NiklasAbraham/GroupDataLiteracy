import pandas as pd
from src.data_utils import load_movie_data
from src.api.wikidata_handler import get_wikidata_subclasses
import asyncio
import json
import os
from src.data_exploration import print_wikidata_column_appearances

DATA_DIR = '../all_data_run_2511/data'
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

def clean_dataset(df: pd.DataFrame, wrong_classes_save_path: str = WRONG_CLASSES_SAVE_PATH, max_plot_length: int = MAX_PLOT_LENGTH) -> pd.DataFrame:
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    print(f"Original dataset size: {len(df)}")
    df_filtered = filter_movies_without_plot(df)
    print(f"After filtering movies without plot: {len(df_filtered)}")
    df_filtered = filter_non_movies(df_filtered, wrong_classes_save_path=wrong_classes_save_path)
    print(f"After filtering non-movies: {len(df_filtered)}")

    df_filtered["plot_length_chars"] = df_filtered["plot"].apply(lambda x: len(x.strip()) if isinstance(x, str) else 0)
    df_filtered = df_filtered[df_filtered["plot_length_chars"] <= max_plot_length].reset_index(drop=True)
    print(f"After filtering movies with plot length > {max_plot_length} chars: {len(df_filtered)}")
    
    return df_filtered

if __name__ == "__main__":
    df = load_movie_data(DATA_DIR, verbose=False)

    clean_dataset(df)

