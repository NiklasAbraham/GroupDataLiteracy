"""
TMDb (The Movie Database) Handler

This module provides functions to query TMDb API for movie information.
It can fetch movies by year, search for movies, and retrieve detailed metadata
including cast, crew, ratings, and more.

Features:
1. Basic mode: Fetch movies by year with essential metadata
2. Comprehensive mode: Fetch detailed movie information including cast, crew,
   production details, financial data, and ratings
3. Search functionality: Search for movies by title
4. Rate limiting: Respects TMDb API rate limits (40 requests per 10 seconds)
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


# TMDb API configuration
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"


def get_tmdb_api_key() -> str:
    """Get TMDb API key from environment variables."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise ValueError(
            "TMDB_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return api_key


def get_tmdb_headers() -> Dict[str, str]:
    """Get headers for TMDb API requests."""
    return {"Accept": "application/json", "Content-Type": "application/json"}


async def get_movie_by_wiki_id(
    session: aiohttp.ClientSession, wiki_id: int, api_key: str, verbose: bool = False
) -> Optional[Dict]:
    """
    Fetch detailed movie information by Wikipedia ID (e.g. Q123456789)

    Args:
        session: aiohttp ClientSession for making HTTP requests
        wiki_id: Wikipedia ID
        api_key: TMDb API key
        verbose: Whether to print progress messages (default: False)

    Returns:
        Dictionary containing detailed movie information, or None if not found
    """
    url = f"https://api.themoviedb.org/3/find/{wiki_id}?external_source=wikidata_id&language=en&api_key={api_key}"

    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            elif response.status == 404:
                if verbose:
                    print(f"Movie Wiki ID {wiki_id}: Not found")
                return None
            elif response.status == 429:
                if verbose:
                    print(f"Movie Wiki ID {wiki_id}: Rate limit exceeded, waiting...")
                await asyncio.sleep(10)
                return await get_movie_by_wiki_id(session, wiki_id, api_key, verbose)
            else:
                if verbose:
                    error_text = await response.text()
                    print(
                        f"Movie Wiki ID {wiki_id}: HTTP Error {response.status} - {error_text}"
                    )
    except Exception as e:
        if verbose:
            print(f"Movie Wiki ID {wiki_id}: Error - {e}")

    return None


async def get_movie_by_id(
    session: aiohttp.ClientSession,
    movie_id: int,
    api_key: str,
    include_additional: bool = True,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Fetch detailed movie information by TMDb movie ID.

    Args:
        session: aiohttp ClientSession for making HTTP requests
        movie_id: TMDb movie ID
        api_key: TMDb API key
        include_additional: Whether to include credits and external IDs (default: True)
        verbose: Whether to print progress messages (default: False)

    Returns:
        Dictionary containing detailed movie information, or None if not found
    """
    url = f"{TMDB_API_BASE_URL}/movie/{movie_id}"
    params = {"api_key": api_key, "language": "en-US"}

    if include_additional:
        params["append_to_response"] = "credits,external_ids"

    headers = get_tmdb_headers()

    try:
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            elif response.status == 404:
                if verbose:
                    print(f"Movie ID {movie_id}: Not found")
                return None
            elif response.status == 429:
                if verbose:
                    print(f"Movie ID {movie_id}: Rate limit exceeded, waiting...")
                await asyncio.sleep(10)
                return await get_movie_by_id(
                    session, movie_id, api_key, include_additional, verbose
                )
            else:
                if verbose:
                    error_text = await response.text()
                    print(
                        f"Movie ID {movie_id}: HTTP Error {response.status} - {error_text}"
                    )
    except Exception as e:
        if verbose:
            print(f"Movie ID {movie_id}: Error - {e}")

    return None


async def search_movies(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
    year: Optional[int] = None,
    page: int = 1,
    verbose: bool = False,
) -> List[Dict]:
    """
    Search for movies by title.

    Args:
        session: aiohttp ClientSession for making HTTP requests
        query: Movie title to search for
        api_key: TMDb API key
        year: Optional release year to filter results
        page: Page number for pagination (default: 1)
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of movie dictionaries matching the search query
    """
    url = f"{TMDB_API_BASE_URL}/search/movie"
    params = {"api_key": api_key, "query": query, "language": "en-US", "page": page}

    if year:
        params["year"] = year

    headers = get_tmdb_headers()

    try:
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("results", [])
            elif response.status == 429:
                if verbose:
                    print(f"Search '{query}': Rate limit exceeded, waiting...")
                await asyncio.sleep(10)
                return await search_movies(session, query, api_key, year, page, verbose)
            else:
                if verbose:
                    error_text = await response.text()
                    print(
                        f"Search '{query}': HTTP Error {response.status} - {error_text}"
                    )
    except Exception as e:
        if verbose:
            print(f"Search '{query}': Error - {e}")

    return []


async def get_movies_by_year(
    session: aiohttp.ClientSession,
    year: int,
    api_key: str,
    limit: int = 50,
    sort_by: str = "popularity.desc",
    verbose: bool = False,
) -> List[Dict]:
    """
    Fetch movies from TMDb for a specific year, sorted by popularity or other criteria.

    Args:
        session: aiohttp ClientSession for making HTTP requests
        year: The release year to query for
        api_key: TMDb API key
        limit: Maximum number of movies to return (default: 50)
        sort_by: Sort order (default: "popularity.desc")
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of dictionaries containing movie information
    """
    url = f"{TMDB_API_BASE_URL}/discover/movie"
    params = {
        "api_key": api_key,
        "primary_release_year": year,
        "sort_by": sort_by,
        "language": "en-US",
        "page": 1,
        "include_adult": False,
    }

    headers = get_tmdb_headers()

    try:
        movies = []
        page = 1

        while len(movies) < limit:
            params["page"] = page
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])

                    if not results:
                        break

                    movies.extend(results[: limit - len(movies)])

                    # Check if there are more pages
                    total_pages = data.get("total_pages", 1)
                    if page >= total_pages:
                        break

                    page += 1

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.25)

                elif response.status == 429:
                    if verbose:
                        print(f"{year}: Rate limit exceeded, waiting...")
                    await asyncio.sleep(10)
                    continue
                else:
                    if verbose:
                        error_text = await response.text()
                        print(f"{year}: HTTP Error {response.status} - {error_text}")
                    break

        if verbose:
            print(f"{year}: {len(movies)} movies fetched")

        return movies[:limit]

    except Exception as e:
        if verbose:
            print(f"{year}: Error - {e}")

    return []


async def get_movies_comprehensive_by_year(
    session: aiohttp.ClientSession,
    year: int,
    api_key: str,
    limit: int = 50,
    sort_by: str = "popularity.desc",
    verbose: bool = False,
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch comprehensive movie data from TMDb for a specific year with detailed information.

    This function retrieves extensive features useful for data science analysis:
    - Basic info: title, overview, release date, runtime, tagline
    - Ratings: vote average, vote count, popularity
    - Financial: budget, revenue
    - Cast & Crew: cast, directors, producers, writers, etc.
    - Classification: genres, languages, production countries, certification
    - Production: production companies, production countries
    - External IDs: IMDb ID, Facebook, Instagram, Twitter, Wikipedia
    - Technical: status, original language, video availability
    - Media: poster path, backdrop path

    Args:
        session: aiohttp ClientSession for making HTTP requests
        year: The release year to query for
        api_key: TMDb API key
        limit: Maximum number of movies to return (default: 50)
        sort_by: Sort order (default: "popularity.desc")
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of dictionaries containing comprehensive movie information
    """
    # First get the list of movies
    movies_list = await get_movies_by_year(
        session, year, api_key, limit, sort_by, verbose
    )

    if not movies_list:
        return []

    # Then fetch detailed information for each movie
    tasks = []
    for movie in movies_list:
        movie_id = movie.get("id")
        if movie_id:
            tasks.append(
                get_movie_by_id(
                    session, movie_id, api_key, include_additional=True, verbose=False
                )
            )
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.25)

    if verbose:
        print(f"  Fetching detailed information for {len(tasks)} movies...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    comprehensive_movies = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            if verbose:
                print(f"  Error fetching details for movie: {result}")
            continue

        if not result:
            continue

        # Extract comprehensive data
        credits = result.get("credits", {})
        external_ids = result.get("external_ids", {})

        # Extract cast
        cast_list = credits.get("cast", [])
        cast = (
            "|".join([actor.get("name", "") for actor in cast_list[:10]])
            if cast_list
            else None
        )

        # Extract crew
        crew_list = credits.get("crew", [])
        directors = (
            "|".join(
                [
                    person.get("name", "")
                    for person in crew_list
                    if person.get("job", "").lower() == "director"
                ]
            )
            if crew_list
            else None
        )

        producers = (
            "|".join(
                [
                    person.get("name", "")
                    for person in crew_list
                    if person.get("job", "").lower()
                    in ["producer", "executive producer"]
                ]
            )
            if crew_list
            else None
        )

        writers = (
            "|".join(
                [
                    person.get("name", "")
                    for person in crew_list
                    if person.get("job", "").lower()
                    in ["screenplay", "writer", "screenwriter"]
                ]
            )
            if crew_list
            else None
        )

        # Extract genres
        genres_list = result.get("genres", [])
        genres = (
            "|".join([g.get("name", "") for g in genres_list]) if genres_list else None
        )

        # Extract production companies
        production_companies_list = result.get("production_companies", [])
        production_companies = (
            "|".join([company.get("name", "") for company in production_companies_list])
            if production_companies_list
            else None
        )

        # Extract production countries
        production_countries_list = result.get("production_countries", [])
        production_countries = (
            "|".join([country.get("name", "") for country in production_countries_list])
            if production_countries_list
            else None
        )

        # Extract spoken languages
        spoken_languages_list = result.get("spoken_languages", [])
        spoken_languages = (
            "|".join([lang.get("name", "") for lang in spoken_languages_list])
            if spoken_languages_list
            else None
        )

        movie_data = {
            # Basic info
            "title": result.get("title"),
            "original_title": result.get("original_title"),
            "overview": result.get("overview"),
            "tagline": result.get("tagline"),
            "release_date": result.get("release_date"),
            "runtime": result.get("runtime"),
            "year": year,
            # Ratings
            "vote_average": result.get("vote_average"),
            "vote_count": result.get("vote_count"),
            "popularity": result.get("popularity"),
            # Financial
            "budget": result.get("budget"),
            "revenue": result.get("revenue"),
            # Cast & Crew
            "cast": cast,
            "director": directors,
            "producer": producers,
            "writer": writers,
            # Classification
            "genre": genres,
            "language": spoken_languages,
            "production_country": production_countries,
            "certification": (
                result.get("release_dates", {})
                .get("results", [{}])[0]
                .get("release_dates", [{}])[0]
                .get("certification")
                if result.get("release_dates")
                else None
            ),
            # Production
            "production_company": production_companies,
            "status": result.get("status"),
            # External IDs
            "tmdb_id": result.get("id"),
            "imdb_id": external_ids.get("imdb_id") or result.get("imdb_id"),
            "facebook_id": external_ids.get("facebook_id"),
            "instagram_id": external_ids.get("instagram_id"),
            "twitter_id": external_ids.get("twitter_id"),
            "wikipedia_id": external_ids.get("wikipedia_id"),
            # Technical
            "original_language": result.get("original_language"),
            "video": result.get("video"),
            "adult": result.get("adult"),
            # Media
            "poster_path": result.get("poster_path"),
            "backdrop_path": result.get("backdrop_path"),
            "homepage": result.get("homepage"),
        }

        comprehensive_movies.append(movie_data)

    if verbose:
        print(f"{year}: {len(comprehensive_movies)} movies with comprehensive data")

    return comprehensive_movies


async def fetch_movies_for_years(
    movies_per_year: int,
    start_year: int,
    end_year: int,
    api_key: str,
    verbose: bool = False,
    delay: float = 0.25,
) -> List[Dict]:
    """
    Fetch movies for multiple years concurrently.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        api_key: TMDb API key
        verbose: Whether to print progress messages (default: False)
        delay: Delay in seconds between year requests (default: 0.25)

    Returns:
        List of all movies from all years
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for year in range(start_year, end_year + 1):
            tasks.append(
                get_movies_by_year(
                    session, year, api_key, movies_per_year, verbose=verbose
                )
            )
            # Small delay to avoid rate limiting
            if delay > 0:
                await asyncio.sleep(delay)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_movies = []
        for result in results:
            if isinstance(result, Exception):
                if verbose:
                    print(f"Error in fetch: {result}")
                continue
            all_movies.extend(result)
        return all_movies


async def fetch_movies_comprehensive_for_years(
    movies_per_year: int,
    start_year: int,
    end_year: int,
    api_key: str,
    verbose: bool = False,
    delay: float = 0.25,
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch comprehensive movie data for multiple years concurrently.
    Adds delays between requests to respect rate limits.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        api_key: TMDb API key
        verbose: Whether to print progress messages (default: False)
        delay: Delay in seconds between year requests (default: 0.25)

    Returns:
        List of all movies from all years with comprehensive features
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for year in range(start_year, end_year + 1):
            tasks.append(
                get_movies_comprehensive_by_year(
                    session, year, api_key, movies_per_year, verbose=verbose
                )
            )
            # Small delay to avoid rate limiting
            if delay > 0:
                await asyncio.sleep(delay)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_movies = []
        for result in results:
            if isinstance(result, Exception):
                if verbose:
                    print(f"Error in fetch: {result}")
                continue
            all_movies.extend(result)
        return all_movies


async def fetch_movies(
    movies_per_year: int = 50,
    start_year: int = 2000,
    end_year: int = 2024,
    verbose: bool = True,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Main function to fetch movies from TMDb for a range of years.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages
        api_key: TMDb API key (if None, will try to get from environment)

    Returns:
        List of dictionaries containing movie information
    """
    if api_key is None:
        api_key = get_tmdb_api_key()

    if verbose:
        print(
            f"Fetching top {movies_per_year} most popular movies per year ({start_year}-{end_year})...\n"
        )

    all_movies = await fetch_movies_for_years(
        movies_per_year, start_year, end_year, api_key, verbose=verbose
    )

    if verbose:
        print(f"\nFetched {len(all_movies)} movies total")
        print(f"Movies ordered by popularity per year")

    return all_movies


async def fetch_movies_comprehensive(
    movies_per_year: int = 50,
    start_year: int = 2000,
    end_year: int = 2024,
    verbose: bool = True,
    api_key: Optional[str] = None,
) -> List[Dict[str, Optional[str]]]:
    """
    Main function to fetch comprehensive movie data from TMDb for a range of years.
    This version includes extensive features for in-depth data science analysis.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages
        api_key: TMDb API key (if None, will try to get from environment)

    Returns:
        List of dictionaries containing comprehensive movie information
    """
    if api_key is None:
        api_key = get_tmdb_api_key()

    if verbose:
        print(
            f"Fetching comprehensive data for top {movies_per_year} most popular movies per year ({start_year}-{end_year})...\n"
        )
        print("Note: This may take some time due to API rate limits.\n")

    all_movies = await fetch_movies_comprehensive_for_years(
        movies_per_year, start_year, end_year, api_key, verbose=verbose
    )

    if verbose:
        print(f"\nFetched {len(all_movies)} movies total with comprehensive features")
        print(f"Movies ordered by popularity per year")

    return all_movies


def save_movies_to_csv(
    movies: List[Dict[str, Optional[str]]],
    filename: str = "tmdb_movies.csv",
    comprehensive: bool = False,
) -> None:
    """
    Save movies data to a CSV file.

    Args:
        movies: List of movie dictionaries
        filename: Output CSV filename
        comprehensive: If True, uses comprehensive field names (default: False, auto-detects)
    """
    import csv

    if not movies:
        print("No movies to save")
        return

    # Auto-detect if comprehensive format
    if not comprehensive:
        sample_keys = set(movies[0].keys())
        comprehensive = (
            "budget" in sample_keys
            or "cast" in sample_keys
            or "producer" in sample_keys
        )

    if comprehensive:
        fieldnames = [
            "title",
            "original_title",
            "overview",
            "tagline",
            "release_date",
            "runtime",
            "year",
            "vote_average",
            "vote_count",
            "popularity",
            "budget",
            "revenue",
            "cast",
            "director",
            "producer",
            "writer",
            "genre",
            "language",
            "production_country",
            "certification",
            "production_company",
            "status",
            "tmdb_id",
            "imdb_id",
            "facebook_id",
            "instagram_id",
            "twitter_id",
            "wikipedia_id",
            "original_language",
            "video",
            "adult",
            "poster_path",
            "backdrop_path",
            "homepage",
        ]
    else:
        fieldnames = [
            "title",
            "overview",
            "release_date",
            "popularity",
            "vote_average",
            "vote_count",
            "genre",
            "year",
            "tmdb_id",
            "imdb_id",
        ]
        # Use only fields that exist in the data
        available_fields = [f for f in fieldnames if f in movies[0]]
        fieldnames = available_fields

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(movies)

    print(f"Saved {len(movies)} movies to {filename}")


async def main(movies_per_year: int = 50, start_year: int = 2020, end_year: int = 2024):
    """
    Example main function that fetches movies and saves them to CSV.

    Args:
        movies_per_year: Maximum number of movies per year
        start_year: First year to query
        end_year: Last year to query
    """
    movies = await fetch_movies(movies_per_year, start_year, end_year)
    save_movies_to_csv(movies)


async def main_comprehensive(
    movies_per_year: int = 50, start_year: int = 2020, end_year: int = 2024
):
    """
    Example main function that fetches comprehensive movie data and saves to CSV.

    This fetches extensive features useful for data science analysis including:
    - Cast and crew (actors, directors, producers, writers)
    - Financial data (budget, revenue)
    - Ratings and popularity metrics
    - Production details
    - External IDs for further data enrichment

    Args:
        movies_per_year: Maximum number of movies per year
        start_year: First year to query
        end_year: Last year to query
    """
    movies = await fetch_movies_comprehensive(movies_per_year, start_year, end_year)
    save_movies_to_csv(movies, "tmdb_movies_comprehensive.csv")


if __name__ == "__main__":
    # Example usage - basic mode
    asyncio.run(main(movies_per_year=50, start_year=2020, end_year=2024))

    # Example usage - comprehensive mode (recommended for data science)
    # asyncio.run(main_comprehensive(movies_per_year=50, start_year=2020, end_year=2024))
