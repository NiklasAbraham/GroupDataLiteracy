import pandas as pd
from data_utils import load_movie_data
import os
import tqdm
import wikipediaapi
import json

DATA_DIR = '../all_data_run_2511/data'
NEW_WIKIPEDIA_DATA_PATH = os.path.join(DATA_DIR, 'wikipedia_fix.json')

def print_data(df: pd.DataFrame) -> None:
    print(df.to_string())

def check_for_duplicates(df: pd.DataFrame) -> None:
    duplicates_id = df.duplicated(subset=['movie_id'], keep='first')
    print(f"Found {duplicates_id.value_counts().get(True, 0)} duplicate entries based on 'movie_id'.")

    duplicates_title = df.duplicated(subset=['title'], keep='first')
    print(f"Found {duplicates_title.value_counts().get(True, 0)} duplicate entries based on 'title'.")
    # print(df["title"][duplicates_title][:50])

    duplicates_wikipedia = df.duplicated(subset=['wikipedia_link'], keep='first')
    print(f"Found {duplicates_wikipedia.value_counts().get(True, 0)} duplicate entries based on 'wikipedia_link'.")
    # print(df["title"][duplicates_wikipedia][:50])

def print_wikidata_column_appearances(df: pd.DataFrame, column_name: str) -> int:
    values_counts = {}
    num_of_movies_with_value_count = {}
    for _, row in df.iterrows():
        row_values = row.get(column_name, "")
        if not isinstance(row_values, str) or not row_values.strip():
            continue
        values_list = [
            cls.strip() for cls in row_values.split(',') if cls.strip()
        ]
        num_of_movies_with_value_count[len(values_list)] = num_of_movies_with_value_count.get(len(values_list), 0) + 1
        for cls in values_list:
            values_counts[cls] = values_counts.get(cls, 0) + 1
    
    sorted_values = sorted(values_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Value Appearances of column {column_name}:")
    for cls, count in sorted_values:
        print(f"{cls}: {count}")
    print(f"Different values of column {column_name} found:", len(values_counts))
    print(f"Number of Movies by Value Count for column {column_name}:")
    for count, num_movies in sorted(num_of_movies_with_value_count.items()):
        print(f"{count} : {num_movies} movies")


def print_rows_with_class(
    df: pd.DataFrame,
    wikidata_class: str,
    columns_to_print = ["movie_id", "title", "wikidata_class", "wikipedia_link"]
) -> None:
    filtered_df = df[df['wikidata_class'].str.contains(wikidata_class, na=False)]
    print("Rows with class: ", wikidata_class)
    print_data(filtered_df[columns_to_print])


if __name__ == "__main__":
    wiki = wikipediaapi.Wikipedia(
        user_agent='GroupDataLiteracy/1.0 (movie data pipeline)',
        language='en'
    )

    pd.set_option('display.width', 300)
    df = load_movie_data(DATA_DIR, verbose=False)
    # check_for_duplicates(df)
    # print(df.info())
    print_wikidata_column_appearances(df, "genre")
    # print_rows_with_class(df, "shirokuban")

    # has_plot_dict = has_actual_wikipedia_plot(df, wiki, save_path=NEW_WIKIPEDIA_DATA_PATH)
    # print(f"Actually have plot: {sum(has_plot_dict.values())} out of {len(has_plot_dict)} movies.")
    # has_plot_dict = json.load(open(NEW_WIKIPEDIA_DATA_PATH, 'r'))
    # first_ten = list(has_plot_dict.items())[:10]

    # for movie_id, has_plot in first_ten:
    #     print(f"Movie ID: {movie_id}, Has Plot: {has_plot}")
    #     print(f"Plot beginning: {df[df['movie_id'] == movie_id]['plot'].values[0][:100]}")