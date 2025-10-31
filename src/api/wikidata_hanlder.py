"""
Wikidata Handler

This module provides functions to query Wikidata for movie information
using SPARQL queries. It can fetch movies by year with various metadata
including plot summaries, genres, directors, and more.
"""

import asyncio
import aiohttp
import os
from typing import List, Dict, Optional

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system environment variables


# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Default headers for Wikidata API
def get_wikidata_headers() -> Dict[str, str]:
    """Get headers for Wikidata API requests with user agent from environment."""
    user_email = os.getenv("WIKIDATA_USER_EMAIL", "anonymous@example.com")
    return {
        "User-Agent": f"MoviesAnalysis/1.0 ({user_email})",
        "Accept": "application/json"
    }


async def get_movies_by_year(
    session: aiohttp.ClientSession,
    year: int,
    limit: int = 50,
    verbose: bool = False
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch movies from Wikidata for a specific year, ordered by popularity (sitelinks).

    Args:
        session: aiohttp ClientSession for making HTTP requests
        year: The release year to query for
        limit: Maximum number of movies to return (default: 50)
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of dictionaries containing movie information:
        - title: Movie title
        - summary: Plot summary
        - release_date: Release date
        - genre: Comma-separated genres
        - director: Comma-separated directors
        - duration: Duration in minutes
        - imdb_id: IMDb identifier
        - country: Country of origin
        - sitelinks: Number of sitelinks (popularity indicator)
        - year: Release year
    """
    query = f"""
    SELECT ?movie ?movieLabel ?plotSummary ?releaseDate ?imdbID ?countryLabel ?duration ?sitelinks
           (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres)
           (GROUP_CONCAT(DISTINCT ?directorLabel; separator=", ") AS ?directors)
    WHERE {{
      ?movie wdt:P31/wdt:P279* wd:Q11424;
             wdt:P577 ?releaseDate;
             wdt:P2437 ?plotSummary;
             wikibase:sitelinks ?sitelinks.
      FILTER(YEAR(?releaseDate) = {year})
      OPTIONAL {{ ?movie wdt:P136 ?genre. }}
      OPTIONAL {{ ?movie wdt:P57 ?director. }}
      OPTIONAL {{ ?movie wdt:P2047 ?duration. }}
      OPTIONAL {{ ?movie wdt:P345 ?imdbID. }}
      OPTIONAL {{ ?movie wdt:P495 ?country. }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?genre rdfs:label ?genreLabel.
        ?director rdfs:label ?directorLabel.
      }}
    }}
    GROUP BY ?movie ?movieLabel ?plotSummary ?releaseDate ?imdbID ?countryLabel ?duration ?sitelinks
    ORDER BY DESC(?sitelinks)
    LIMIT {limit}
    """

    params = {'query': query, 'format': 'json'}
    headers = get_wikidata_headers()

    try:
        async with session.get(WIKIDATA_SPARQL_URL, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get('results', {}).get('bindings', [])
                movies = []
                seen_titles = set()

                for r in results:
                    title = r.get('movieLabel', {}).get('value')
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    movies.append({
                        'title': title,
                        'summary': r.get('plotSummary', {}).get('value'),
                        'release_date': r.get('releaseDate', {}).get('value'),
                        'genre': r.get('genres', {}).get('value'),
                        'director': r.get('directors', {}).get('value'),
                        'duration': r.get('duration', {}).get('value'),
                        'imdb_id': r.get('imdbID', {}).get('value'),
                        'country': r.get('countryLabel', {}).get('value'),
                        'sitelinks': r.get('sitelinks', {}).get('value'),
                        'year': year
                    })
                if verbose:
                    print(f"✓ {year}: {len(movies)} most popular movies")
                return movies
            else:
                if verbose:
                    print(f"✗ {year}: HTTP Error {response.status}")
    except Exception as e:
        if verbose:
            print(f"✗ {year}: Error - {e}")

    return []


async def fetch_movies_for_years(
    movies_per_year: int,
    start_year: int,
    end_year: int,
    verbose: bool = False
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch movies for multiple years concurrently.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of all movies from all years
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for year in range(start_year, end_year + 1):
            tasks.append(get_movies_by_year(session, year, movies_per_year, verbose=verbose))

        results = await asyncio.gather(*tasks)
        all_movies = [movie for year_movies in results for movie in year_movies]
        return all_movies


async def fetch_movies(
    movies_per_year: int = 50,
    start_year: int = 2000,
    end_year: int = 2024,
    verbose: bool = True
) -> List[Dict[str, Optional[str]]]:
    """
    Main function to fetch movies from Wikidata for a range of years.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages

    Returns:
        List of dictionaries containing movie information
    """
    if verbose:
        print(f"Fetching top {movies_per_year} most popular movies per year ({start_year}-{end_year})...\n")

    all_movies = await fetch_movies_for_years(movies_per_year, start_year, end_year, verbose=verbose)

    if verbose:
        print(f"\nFetched {len(all_movies)} movies total")
        print(f"Movies ordered by popularity (sitelinks) per year")

    return all_movies


def save_movies_to_csv(
    movies: List[Dict[str, Optional[str]]],
    filename: str = 'wikidata_movies.csv'
) -> None:
    """
    Save movies data to a CSV file.

    Args:
        movies: List of movie dictionaries
        filename: Output CSV filename
    """
    import csv
    
    fieldnames = [
        'title', 'summary', 'release_date', 'genre', 'director',
        'duration', 'imdb_id', 'country', 'sitelinks', 'year'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(movies)
    
    print(f"Saved {len(movies)} movies to {filename}")


async def main(movies_per_year: int = 50, start_year: int = 1950, end_year: int = 2024):
    """
    Example main function that fetches movies and saves them to CSV.

    Args:
        movies_per_year: Maximum number of movies per year
        start_year: First year to query
        end_year: Last year to query
    """
    movies = await fetch_movies(movies_per_year, start_year, end_year)
    save_movies_to_csv(movies)


if __name__ == "__main__":
    # Example usage
    asyncio.run(main(movies_per_year=50, start_year=1950, end_year=2024))
