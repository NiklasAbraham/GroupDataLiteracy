"""
Wikidata Handler

This module provides functions to query Wikidata for movie information
using SPARQL queries. It can fetch movies by year with various metadata
including plot summaries, genres, directors, and more.
"""

import asyncio
import aiohttp
import os
import logging
import urllib.parse
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        List of dictionaries containing movie information
    """
    query = f"""
    SELECT ?movie ?movieLabel ?plotSummary ?releaseDate ?imdbID ?countryLabel ?duration ?sitelinks
           (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres)
           (GROUP_CONCAT(DISTINCT ?directorLabel; separator=", ") AS ?directors)
    WHERE {{
      ?movie wdt:P31/wdt:P279* wd:Q11424;
             wdt:P577 ?releaseDate;
             wikibase:sitelinks ?sitelinks.
      FILTER(YEAR(?releaseDate) = {year})
      FILTER(?sitelinks > 0)
      OPTIONAL {{ ?movie wdt:P2437 ?plotSummary. }}
      OPTIONAL {{ ?movie wdt:P136 ?genre. }}
      OPTIONAL {{ ?movie wdt:P57 ?director. }}
      OPTIONAL {{ ?movie wdt:P2047 ?duration. }}
      OPTIONAL {{ ?movie wdt:P345 ?imdbID. }}
      OPTIONAL {{ ?movie wdt:P495 ?country. }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?movie rdfs:label ?movieLabel.
        ?genre rdfs:label ?genreLabel.
        ?director rdfs:label ?directorLabel.
      }}
      FILTER(BOUND(?movieLabel) && ?movieLabel != "")
    }}
    GROUP BY ?movie ?movieLabel ?plotSummary ?releaseDate ?imdbID ?countryLabel ?duration ?sitelinks
    ORDER BY DESC(?sitelinks)
    LIMIT {limit}
    """

    params = {'query': query, 'format': 'json'}
    headers = get_wikidata_headers()

    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            async with session.get(WIKIDATA_SPARQL_URL, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data['results']['bindings']
                    movies = []
                    seen_titles = set()
                    
                    # Tracking statistics
                    dropped_no_title = 0
                    dropped_duplicate = 0
                    total_processed = len(results)
                    
                    logger.info(f"Year {year}: Wikidata returned {total_processed} raw results")
                    if len(results) > 0:
                        logger.debug(f"Year {year}: Sample result keys: {list(results[0].keys())}")

                    for idx, r in enumerate(results):
                        title = r.get('movieLabel', {}).get('value')
                        
                        # Try alternative: use movie ID if label missing
                        if not title:
                            movie_uri = r.get('movie', {}).get('value', '')
                            if movie_uri:
                                title = movie_uri.split('/')[-1]
                                logger.debug(f"Year {year}, result {idx+1}: Missing movieLabel, using URI ID: {title}")
                        
                        # Filter: Missing title
                        if not title:
                            dropped_no_title += 1
                            movie_uri = r.get('movie', {}).get('value', 'Unknown')
                            logger.warning(f"Year {year}, result {idx+1}: DROPPED - Missing title (movie URI: {movie_uri})")
                            continue
                        
                        # Filter: Duplicate title
                        if title in seen_titles:
                            dropped_duplicate += 1
                            sitelinks = r.get('sitelinks', {}).get('value', 'N/A')
                            logger.debug(f"Year {year}, result {idx+1}: DROPPED - Duplicate title '{title}' (sitelinks: {sitelinks})")
                            continue
                        
                        seen_titles.add(title)
                        
                        # Extract sitelinks - must be present since we filter by ?sitelinks > 0
                        sitelinks_raw = r.get('sitelinks', {})
                        sitelinks_value = sitelinks_raw.get('value') if sitelinks_raw else None
                        
                        if sitelinks_value is None:
                            logger.error(f"Year {year}, result {idx+1}: CRITICAL - Missing sitelinks for '{title}' despite filter!")
                            sitelinks = '0'  # Fallback - shouldn't happen due to filter
                        else:
                            # Convert to string and ensure it's not empty
                            sitelinks = str(sitelinks_value).strip()
                            if not sitelinks:
                                logger.warning(f"Year {year}, result {idx+1}: Empty sitelinks string for '{title}'")
                                sitelinks = '0'
                        
                        logger.debug(f"Year {year}, result {idx+1}: Added '{title}' (sitelinks: {sitelinks})")

                        # Construct Wikipedia link from title
                        # Since we filter by sitelinks > 0, we know there's a Wikipedia page
                        # URL encode the title for Wikipedia URL (Wikipedia uses spaces as underscores)
                        title_encoded = title.replace(' ', '_')
                        # Handle special characters while preserving underscores and parentheses
                        title_encoded = urllib.parse.quote(title_encoded, safe='_()')
                        wikipedia_link = f"https://en.wikipedia.org/wiki/{title_encoded}"
                        
                        movies.append({
                            'title': title,
                            'summary': r.get('plotSummary', {}).get('value') or '',
                            'release_date': r.get('releaseDate', {}).get('value') or '',
                            'genre': r.get('genres', {}).get('value') or '',
                            'director': r.get('directors', {}).get('value') or '',
                            'duration': r.get('duration', {}).get('value') or '',
                            'imdb_id': r.get('imdbID', {}).get('value') or '',
                            'country': r.get('countryLabel', {}).get('value') or '',
                            'sitelinks': sitelinks,
                            'wikipedia_link': wikipedia_link,
                            'year': year
                        })
                    
                    # Log summary statistics
                    kept_count = len(movies)
                    logger.info(
                        f"Year {year}: Processed {total_processed} results -> "
                        f"Kept: {kept_count}, "
                        f"Dropped (no title): {dropped_no_title}, "
                        f"Dropped (duplicate): {dropped_duplicate}"
                    )
                    
                    if kept_count == 0 and total_processed > 0:
                        logger.warning(
                            f"Year {year}: WARNING - All {total_processed} results were filtered out! "
                            f"Check filters: missing titles ({dropped_no_title}), duplicates ({dropped_duplicate})"
                        )
                    
                    return movies
                elif response.status == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"Year {year}: Rate limit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Year {year}: HTTP Error 429 - Rate limit exceeded after {max_retries} attempts")
                        return []
                else:
                    try:
                        error_text = await response.text()
                    except:
                        error_text = f"Status {response.status}"
                    logger.error(f"Year {year}: HTTP Error {response.status} - {error_text}")
                    return []
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"Year {year}: Error - {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Year {year}: Error after {max_retries} attempts - {e}", exc_info=True)

    return []


async def fetch_movies_for_years(
    movies_per_year: int,
    start_year: int,
    end_year: int,
    verbose: bool = False,
    delay: float = 0.5
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch movies for multiple years with rate limiting to avoid 429 errors.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages (default: False)
        delay: Delay in seconds between year requests (default: 0.5)

    Returns:
        List of all movies from all years
    """
    async with aiohttp.ClientSession() as session:
        all_movies = []
        for year in range(start_year, end_year + 1):
            movies = await get_movies_by_year(session, year, movies_per_year, verbose=verbose)
            all_movies.extend(movies)
            
            # Delay between requests to avoid rate limiting
            if delay > 0 and year < end_year:
                await asyncio.sleep(delay)
        
        return all_movies


async def fetch_movies(
    movies_per_year: int = 50,
    start_year: int = 2000,
    end_year: int = 2024,
    verbose: bool = True,
    delay: float = 0.5
) -> List[Dict[str, Optional[str]]]:
    """
    Main function to fetch movies from Wikidata for a range of years.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages
        delay: Delay in seconds between year requests (default: 0.5)

    Returns:
        List of dictionaries containing movie information
    """
    logger.info(f"Fetching top {movies_per_year} most popular movies per year ({start_year}-{end_year})")
    
    all_movies = await fetch_movies_for_years(movies_per_year, start_year, end_year, verbose=verbose, delay=delay)

    total_years = end_year - start_year + 1
    expected_max = movies_per_year * total_years
    logger.info(
        f"Fetched {len(all_movies)} movies total (max possible: {expected_max}) "
        f"from {total_years} years. Movies ordered by popularity (sitelinks) per year"
    )

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

    # filename in the data folder
    filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)
    
    if not movies:
        logger.warning("No movies to save - empty movie list")
        return
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fieldnames = [
        'title', 'summary', 'release_date', 'genre', 'director',
        'duration', 'imdb_id', 'country', 'sitelinks', 'wikipedia_link', 'year'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(movies)
    
    logger.info(f"Saved {len(movies)} movies to {filename}")


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
    # Example usage - basic mode
    asyncio.run(main(movies_per_year=1000, start_year=2023, end_year=2024))
