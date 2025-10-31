import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

# === CONFIGURATION ===
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
# Using relative paths, as we discussed
INPUT_FILE = "data\\mock\\wikidata_movies.csv"
OUTPUT_FILE = "data\\mock\\tmdb_movies.csv"

# === FUNCTIONS ===


def get_movie_details(imdb_id):
    """
    Finds a movie on TMDB using its IMDb ID (e.g., "tt0068646")
    and downloads its full details and reviews.
    (This function expects an ALREADY FORMATTED ID)
    """

    imdb_id = str(imdb_id).strip()
    if not imdb_id.startswith("tt"):
        return None  # Skip invalid IDs

    # 1. USE THE /find ENDPOINT TO FIND THE TMDB ID
    find_url = f"{BASE_URL}/find/{imdb_id}?api_key={API_KEY}&language=en-US&external_source=imdb_id"
    response = requests.get(find_url)

    if response.status_code != 200:
        return None  # Error or not found
    find_data = response.json()
    if not find_data.get("movie_results"):
        return None  # No movie found

    # 2. EXTRACT THE NUMERIC TMDB ID (e.g., 238)
    tmdb_id = find_data["movie_results"][0].get("id")
    if not tmdb_id:
        return None

    # 3. USE THE TMDB ID TO GET THE FULL DETAILS
    details_url = f"{BASE_URL}/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
    details_response = requests.get(details_url)

    if details_response.status_code != 200:
        return None

    data = details_response.json()

    # ==========================================================
    # === 4. (NEW) GET REVIEWS FOR THIS MOVIE ===
    # ==========================================================
    reviews_url = f"{BASE_URL}/movie/{tmdb_id}/reviews?api_key={API_KEY}"
    reviews_response = requests.get(reviews_url)

    reviews_content = ""  # Default: empty string
    if reviews_response.status_code == 200:
        reviews_data = reviews_response.json()
        review_list = []
        # Loop through all reviews found
        for review in reviews_data.get("results", []):
            author = review.get("author", "Unknown")
            content = review.get("content", "No content").strip()
            # Format this review
            review_list.append(f"Author: {author}\nReview: {content}")

        # Join all reviews with a clear separator
        reviews_content = "\n\n---\n\n".join(review_list)
    # ==========================================================
    # === END OF NEW BLOCK =====================================
    # ==========================================================

    # Create the final movie object
    movie = {
        "id": data.get("id"),
        "title": data.get("title"),
        "imdb_id_source": imdb_id,
        "reviews": reviews_content,
    }
    return movie


def main():
    # === LOAD CSV ===
    df = pd.read_csv(INPUT_FILE)
    if "imdb_id" not in df.columns:
        raise ValueError("The file must contain a column named 'imdb_id'.")

    # We assume the IDs are already clean
    movie_ids = df["imdb_id"].dropna().astype(str).tolist()
    print(f"🎬 Found {len(movie_ids)} IDs in the file. Starting download...\n")

    # === DOWNLOADING ===
    movies = []
    # We use the ID list directly from the CSV
    for movie_id in tqdm(movie_ids, desc="Downloading details"):
        details = get_movie_details(movie_id)
        if details:
            movies.append(details)
        time.sleep(0.25)  # Be nice to the API

    # === SAVING CSV ===
    if not movies:
        print("❌ No valid movies found. Check your IDs or API key.")
        return

    out_df = pd.DataFrame(movies)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Data saved to '{OUTPUT_FILE}' ({len(out_df)} total movies)")


if __name__ == "__main__":
    main()
