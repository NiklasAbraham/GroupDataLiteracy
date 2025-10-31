"""
Wikidata Handler

This module provides functions to query Wikidata for movie information
using SPARQL queries. It can fetch movies by year with various metadata
including plot summaries, genres, directors, and more.

This version is optimized to avoid both parse-time StackOverflowErrors
and runtime HTTP 500 errors by simplifying complex queries and using
the optimized property path for subclass traversal.
"""

import asyncio
import aiohttp
import os
import logging
import urllib.parse
import re
import json
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


def get_movie_base_where_clause(year: int) -> str:
    """
    Generate the base WHERE clause for movie queries with required criteria.
    
    REQUIRED criteria (must have):
    - Must be a movie (instance of or subclass of film: wd:Q11424)
    - Must have a release date (wdt:P577)
    - Must have sitelinks (Wikipedia pages) > 0
    - Release year must match the specified year
    - Must have a non-empty English label (movieLabel)
    
    Args:
        year: The release year to filter by
        
    Returns:
        SPARQL WHERE clause fragment with required movie criteria
    """
    # CORRECTION: Use the optimized property path (p:P31/ps:P31/wdt:P279*)
    # to prevent runtime timeouts or StackOverflowErrors.
    return f"""
      ?movie p:P31/ps:P31/wdt:P279* wd:Q11424;
             wdt:P577 ?releaseDate;
             wikibase:sitelinks ?sitelinks.
      FILTER(YEAR(?releaseDate) = {year})
      FILTER(?sitelinks > 0)
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?movie rdfs:label ?movieLabel.
      }}
      FILTER(BOUND(?movieLabel) && ?movieLabel != "")
    """


async def count_movies_by_year(
    session: aiohttp.ClientSession,
    year: int
) -> int:
    """
    Count how many unique movies exist in Wikidata for a specific year that match the criteria.
    
    This uses the same filters as get_movies_by_year (has sitelinks > 0, has label, etc.)
    to give an accurate count of available movies before applying LIMIT.

    Args:
        session: aiohttp ClientSession for making HTTP requests
        year: The release year to query for

    Returns:
        Total count of unique movies matching the criteria
    """
    # Use shared base WHERE clause for consistent criteria
    base_where = get_movie_base_where_clause(year)
    
    count_query = f"""
    SELECT (COUNT(DISTINCT ?movie) AS ?totalMovies)
    WHERE {{
{base_where}
    }}
    """
    
    params = {'query': count_query, 'format': 'json'}
    headers = get_wikidata_headers()
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=timeout) as response:
            if response.status == 200:
                text = await response.text()
                text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
                
                try:
                    data = json.loads(text)
                    count = int(data['results']['bindings'][0]['totalMovies']['value'])
                    return count
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Year {year}: Error parsing count result - {e}")
                    return 0
            else:
                logger.error(f"Year {year}: HTTP Error {response.status} when counting movies")
                return 0
    except Exception as e:
        logger.error(f"Year {year}: Error counting movies - {e}")
        return 0


def get_discovery_query_string(year: int, limit: int = 50) -> str:
    """
    Step 1: Discovery Query - Simple query to find distinct movie IDs.
    
    This query is kept simple to avoid StackOverflowError. It just finds
    movie IDs that match the criteria, ensuring DISTINCT to handle multiple
    release dates per movie.
    
    Args:
        year: The release year to query for
        limit: Maximum number of movie IDs to return
        
    Returns:
        SPARQL query string for discovering movie IDs
    """
    # CORRECTION: Use the optimized property path (p:P31/ps:P31/wdt:P279*)
    # to prevent runtime timeouts or StackOverflowErrors.
    query = f"""
SELECT DISTINCT ?movie ?sitelinks
WHERE {{
    ?movie p:P31/ps:P31/wdt:P279* wd:Q11424;
           wdt:P577 ?releaseDate;
           wikibase:sitelinks ?sitelinks.
    
    FILTER(YEAR(?releaseDate) = {year})
    FILTER(?sitelinks > 0)
    
    # Ensure it has an English label
    ?movie rdfs:label ?movieLabel_en .
    FILTER(LANG(?movieLabel_en) = "en" && ?movieLabel_en != "")
}}
ORDER BY DESC(?sitelinks)
LIMIT {limit}
"""
    return query.strip()


def get_enrichment_query_string(movie_uris: List[str]) -> str:
    """
    Step 2: Enrichment Query - Robust version with pre-aggregation.
    
    This query prevents HTTP 500 errors by avoiding join explosions.
    It uses separate OPTIONAL { SELECT ... } subqueries to aggregate
    each "one-to-many" relationship (actors, directors, genres, etc.)
    *before* joining them, ensuring the main query only joins 1:1 data.
    
    Args:
        movie_uris: List of movie URIs (e.g., ["wd:Q113803868", "wd:Q116181404"])
        
    Returns:
        A robust SPARQL query string for enriching movie data.
    """
    # Build VALUES clause
    values_clause = "VALUES ?movie {\n"
    for uri in movie_uris:
        # Ensure it's a proper wd: URI format
        if not uri.startswith("wd:") and not uri.startswith("http"):
            uri = f"wd:{uri}" if uri.startswith("Q") else uri
        values_clause += f"    {uri}\n"
    values_clause += "}\n"
    
    # This query aggregates all 1-to-1 properties in the main query,
    # and all 1-to-many properties in their own independent subqueries.
    query = f"""
SELECT ?movie ?movieLabel ?sitelinks
       (SAMPLE(?article_) AS ?article)
       (SAMPLE(?plotSummary_) AS ?plotSummary)
       (MIN(?releaseDate_) AS ?releaseDate)
       (SAMPLE(?imdbID_) AS ?imdbID)
       (SAMPLE(?duration_) AS ?duration)
       (MIN(?budget_) AS ?budget)
       (MIN(?boxOffice_) AS ?boxOffice)
       (MIN(?countryLabel_) AS ?countryLabel)
       ?genres ?directors ?actors ?awards ?setPeriods
WHERE {{
    {values_clause}

    # --- Main 1:1 Properties ---
    # Get the 1:1 properties that we will group by
    ?movie wikibase:sitelinks ?sitelinks.
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?movie rdfs:label ?movieLabel. }}
    
    # Get other 1:1-ish properties (we will SAMPLE/MIN these)
    OPTIONAL {{ ?movie wdt:P2437 ?plotSummary_. FILTER(LANG(?plotSummary_) = "en") }}
    OPTIONAL {{ ?movie wdt:P577 ?releaseDate_. }}
    OPTIONAL {{ ?movie wdt:P345 ?imdbID_. }}
    OPTIONAL {{ ?movie wdt:P2047 ?duration_. }}
    OPTIONAL {{ ?movie wdt:P2130 ?budget_. }}
    OPTIONAL {{ ?movie wdt:P2142 ?boxOffice_. }}
    OPTIONAL {{ 
        ?article_ schema:about ?movie;
                  schema:isPartOf <https://en.wikipedia.org/>.
    }}
    OPTIONAL {{ 
        ?movie wdt:P495 ?country_.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?country_ rdfs:label ?countryLabel_. }}
    }}

    # --- Pre-aggregated 1:N Subqueries ---
    # Each of these runs independently and returns 1 row per movie,
    # preventing the Cartesian product (join explosion).
    
    OPTIONAL {{
        SELECT ?movie (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres) WHERE {{
            ?movie wdt:P136 ?genre.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?genre rdfs:label ?genreLabel. }}
        }} GROUP BY ?movie
    }}
    OPTIONAL {{
        SELECT ?movie (GROUP_CONCAT(DISTINCT ?directorLabel; separator=", ") AS ?directors) WHERE {{
            ?movie wdt:P57 ?director.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?director rdfs:label ?directorLabel. }}
        }} GROUP BY ?movie
    }}
    OPTIONAL {{
        SELECT ?movie (GROUP_CONCAT(DISTINCT ?actorLabel; separator=", ") AS ?actors) WHERE {{
            ?movie wdt:P161 ?actor.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?actor rdfs:label ?actorLabel. }}
        }} GROUP BY ?movie
    }}
    OPTIONAL {{
        SELECT ?movie (GROUP_CONCAT(DISTINCT ?awardLabel; separator=", ") AS ?awards) WHERE {{
            ?movie wdt:P166 ?award.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?award rdfs:label ?awardLabel. }}
        }} GROUP BY ?movie
    }}
    OPTIONAL {{
        SELECT ?movie (GROUP_CONCAT(DISTINCT ?setPeriodLabel; separator=", ") AS ?setPeriods) WHERE {{
            ?movie wdt:P2408 ?setPeriod.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?setPeriod rdfs:label ?setPeriodLabel. }}
        }} GROUP BY ?movie
    }}
}}
# Group by the 1:1 properties to collapse results
GROUP BY ?movie ?movieLabel ?sitelinks ?genres ?directors ?actors ?awards ?setPeriods
ORDER BY DESC(?sitelinks)
"""
    return query.strip()


async def get_movies_by_year(
    session: aiohttp.ClientSession,
    year: int,
    limit: int = 50,
    verbose: bool = False
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch movies from Wikidata for a specific year, ordered by popularity (sitelinks).
    
    Uses a two-step process to avoid StackOverflowError:
    1. Discovery query: Simple query to find distinct movie IDs
    2. Enrichment query: Complex query with VALUES clause to get detailed data

    Args:
        session: aiohttp ClientSession for making HTTP requests
        year: The release year to query for
        limit: Maximum number of UNIQUE movies to return (default: 50).
               Returns the top `limit` movies ordered by popularity (sitelinks).
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of dictionaries containing movie information (up to `limit` unique movies)
    """
    headers = get_wikidata_headers()
    max_retries = 3
    retry_delay = 5
    
    # STEP 1: Discovery Query - Get distinct movie URIs
    discovery_query = get_discovery_query_string(year, limit)
    
    if verbose:
        logger.debug(f"Step 1 - Discovery query for year {year}:\n{discovery_query}")
    
    movie_uris = []
    
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            params = {'query': discovery_query, 'format': 'json'}
            async with session.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    # Read as text and clean control characters
                    text = await response.text()
                    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
                    
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Year {year}: Step 1 JSON decode error - {e}")
                        return []
                    
                    # Extract movie URIs from discovery query results
                    results = data['results']['bindings']
                    for r in results:
                        movie_uri = r.get('movie', {}).get('value', '')
                        if movie_uri:
                            # Convert http://www.wikidata.org/entity/Q12345 to wd:Q12345
                            if 'wikidata.org/entity/' in movie_uri:
                                movie_uri = 'wd:' + movie_uri.split('/')[-1]
                            movie_uris.append(movie_uri)
                    
                    logger.info(f"Year {year}: Step 1 found {len(movie_uris)} distinct movie URIs")
                    
                    if not movie_uris:
                        logger.warning(f"Year {year}: No movies found in discovery query")
                        return []
                    
                    # Break out of retry loop on success
                    break
                elif response.status == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"Year {year}: Step 1 rate limit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Year {year}: Step 1 HTTP Error 429 - Rate limit exceeded after {max_retries} attempts")
                        return []
                else:
                    try:
                        error_text = await response.text()
                    except:
                        error_text = f"Status {response.status}"
                    logger.error(f"Year {year}: Step 1 HTTP Error {response.status} - {error_text}")
                    return []
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"Year {year}: Step 1 error - {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Year {year}: Step 1 error after {max_retries} attempts - {e}", exc_info=True)
                return []
    
    # STEP 2: Enrichment Query - Get detailed data for discovered movies
    # Process in batches to avoid URL length limits and query complexity issues
    BATCH_SIZE = 20  # Reduced batch size to avoid 500 errors (server-side query complexity)
    movies = []
    
    for batch_start in range(0, len(movie_uris), BATCH_SIZE):
        batch_uris = movie_uris[batch_start:batch_start + BATCH_SIZE]
        enrichment_query = get_enrichment_query_string(batch_uris)
        
        current_batch_num = batch_start//BATCH_SIZE + 1
        
        if verbose:
            logger.debug(f"Step 2 - Processing batch {current_batch_num}/{(len(movie_uris)-1)//BATCH_SIZE + 1} ({len(batch_uris)} movies) for year {year}")
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=120)  # Longer timeout for complex query
                # Use POST for large queries to avoid URL length limits
                async with session.post(
                    WIKIDATA_SPARQL_URL,
                    data={'query': enrichment_query, 'format': 'json'},
                    headers={**headers, 'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        # Read as text and clean control characters
                        text = await response.text()
                        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
                        
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError as e:
                            logger.error(f"Year {year}: Step 2 batch {current_batch_num} JSON decode error - {e}")
                            break  # Continue with next batch
                        
                        results = data['results']['bindings']
                        
                        # Tracking statistics
                        dropped_no_title = 0
                        total_processed = len(results)
                        
                        if verbose:
                            logger.info(f"Year {year}: Step 2 batch {current_batch_num} returned {total_processed} enriched movie results")
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
                            
                            # Extract sitelinks
                            sitelinks_raw = r.get('sitelinks', {})
                            sitelinks_value = sitelinks_raw.get('value') if sitelinks_raw else None
                            if sitelinks_value is None:
                                sitelinks = '0'
                            else:
                                sitelinks = str(sitelinks_value).strip() or '0'
                            
                            # Extract movie ID (Q-number) from the movie URI
                            movie_uri = r.get('movie', {}).get('value', '')
                            movie_id = ''
                            if movie_uri:
                                # Extract Q-number from URI like http://www.wikidata.org/entity/Q12345
                                movie_id = movie_uri.split('/')[-1] if '/' in movie_uri else movie_uri
                                if movie_id.startswith('Q'):
                                    pass  # Already Q-number
                                elif movie_id.startswith('wd:Q'):
                                    movie_id = movie_id[3:]  # Remove 'wd:' prefix
                            
                            # Get Wikipedia URL from article or construct from title
                            wikipedia_url = r.get('article', {}).get('value', '')
                            if not wikipedia_url:
                                title_encoded = title.replace(' ', '_')
                                title_encoded = urllib.parse.quote(title_encoded, safe='_()')
                                wikipedia_url = f"https://en.wikipedia.org/wiki/{title_encoded}"
                            
                            # CORRECTION: Simplified formatting as gender/nationality are no longer fetched
                            directors_str = r.get('directors', {}).get('value') or ''
                            actors_str = r.get('actors', {}).get('value') or ''
                            
                            movies.append({
                                'movie_id': movie_id,
                                'title': title,
                                'summary': r.get('plotSummary', {}).get('value') or '',
                                'release_date': r.get('releaseDate', {}).get('value') or '',
                                'genre': r.get('genres', {}).get('value') or '',
                                'director': directors_str,
                                'actors': actors_str,
                                'duration': r.get('duration', {}).get('value') or '',
                                'imdb_id': r.get('imdbID', {}).get('value') or '',
                                'country': r.get('countryLabel', {}).get('value') or '',
                                'sitelinks': sitelinks,
                                'wikipedia_link': wikipedia_url,
                                'budget': r.get('budget', {}).get('value') or '',
                                'box_office': r.get('boxOffice', {}).get('value') or '',
                                'awards': r.get('awards', {}).get('value') or '',
                                'set_in_period': r.get('setPeriods', {}).get('value') or '',
                                'year': year
                            })
                        
                        # Log batch statistics
                        batch_kept = len(results) - dropped_no_title
                        if verbose:
                            logger.debug(
                                f"Year {year}: Batch {current_batch_num} processed {total_processed} results -> "
                                f"Kept: {batch_kept}, Dropped (no title): {dropped_no_title}"
                            )
                        
                        # Break on success and continue to next batch
                        break
                    elif response.status == 429:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 1)
                            logger.warning(f"Year {year}: Step 2 batch {current_batch_num} rate limit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Year {year}: Step 2 batch {current_batch_num} HTTP Error 429 - Rate limit exceeded after {max_retries} attempts")
                            # Continue with next batch instead of returning empty
                            break
                    else:
                        try:
                            error_text = await response.text()
                            # Truncate long error messages
                            if len(error_text) > 500:
                                error_text = error_text[:500] + "..."
                        except:
                            error_text = f"Status {response.status}"
                        
                        # Log the full query that failed
                        logger.error(f"Year {year}: Step 2 batch {current_batch_num} HTTP Error {response.status} - {error_text} \n--- Failing Query: ---\n{enrichment_query}\n--- End Query ---")
                        # Continue with next batch instead of returning empty
                        break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(f"Year {year}: Step 2 batch {current_batch_num} error - {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Year {year}: Step 2 batch {current_batch_num} error after {max_retries} attempts - {e}", exc_info=True)
                    # Continue with next batch instead of returning empty
                    break
        
        # Small delay between batches to avoid rate limiting
        if batch_start + BATCH_SIZE < len(movie_uris):
            await asyncio.sleep(0.5)
    
    # Log summary statistics
    kept_count = len(movies)
    logger.info(
        f"Year {year}: Processed all batches -> "
        f"Total movies retrieved: {kept_count}"
    )
    
    return movies


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


async def count_movies_for_years(
    start_year: int,
    end_year: int,
    verbose: bool = True,
    delay: float = 0.5
) -> Dict[int, int]:
    """
    Count how many movies are available in Wikidata for each year in a range.

    Args:
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages
        delay: Delay in seconds between year requests (default: 0.5)

    Returns:
        Dictionary mapping year to count of available movies
    """
    async with aiohttp.ClientSession() as session:
        year_counts = {}
        for year in range(start_year, end_year + 1):
            count = await count_movies_by_year(session, year)
            year_counts[year] = count
            if verbose:
                logger.info(f"Year {year}: {count} movies available")
            
            # Delay between requests
            if delay > 0 and year < end_year:
                await asyncio.sleep(delay)
        
        return year_counts


async def fetch_movies(
    movies_per_year: int = 50,
    start_year: int = 2000,
    end_year: int = 2024,
    verbose: bool = True,
    delay: float = 0.5,
    show_available_counts: bool = False
) -> List[Dict[str, Optional[str]]]:
    """
    Main function to fetch movies from Wikidata for a range of years.

    Args:
        movies_per_year: Maximum number of UNIQUE movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages
        delay: Delay in seconds between year requests (default: 0.5)
        show_available_counts: If True, first count available movies for each year
                              before fetching (default: False)

    Returns:
        List of dictionaries containing movie information
    """
    if show_available_counts:
        logger.info(f"Counting available movies for years {start_year}-{end_year}...")
        year_counts = await count_movies_for_years(start_year, end_year, verbose=verbose, delay=delay)
        
        total_available = sum(year_counts.values())
        logger.info(
            f"\nTotal movies available: {total_available} across {len(year_counts)} years\n"
            f"Fetching up to {movies_per_year} movies per year (you may get fewer if less are available)"
        )
    
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

    # Determine base path (assuming script is in a subdir like 'src/processing')
    # This navigates two levels up to the project root, then into 'data'
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive session)
        base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

    data_dir = os.path.join(base_dir, 'data')
    filename = os.path.join(data_dir, filename)
    
    if not movies:
        logger.warning("No movies to save - empty movie list")
        return
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    fieldnames = [
        'movie_id', 'title', 'summary', 'release_date', 'genre', 'director', 'actors',
        'duration', 'imdb_id', 'country', 'budget', 'box_office', 'awards',
        'set_in_period', 'sitelinks', 'wikipedia_link', 'year'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(movies)
    
    logger.info(f"Saved {len(movies)} movies to {filename}")


async def main(movies_per_year: int = 50, start_year: int = 1950, end_year: int = 2024, show_counts: bool = False):
    """
    Example main function that fetches movies and saves them to CSV.

    Args:
        movies_per_year: Maximum number of UNIQUE movies per year to fetch
        start_year: First year to query
        end_year: Last year to query
        show_counts: If True, show how many movies are available before fetching
    """
    movies = await fetch_movies(
        movies_per_year, 
        start_year, 
        end_year, 
        show_available_counts=show_counts,
        verbose=True # Enable verbose logging for main execution
    )
    save_movies_to_csv(movies, filename=f'wikidata_movies_{start_year}_to_{end_year}.csv')


if __name__ == "__main__":
    # Debug: Print queries for testing online
    # Uncomment the lines below to print the queries and test them at https://query.wikidata.org/
    
    # print("=" * 80)
    # print("STEP 1: Discovery Query (copy and paste into https://query.wikidata.org/):")
    # print("=" * 80)
    # discovery_query = get_discovery_query_string(year=2023, limit=10)
    # print(discovery_query)
    # print("\n" + "=" * 80)
    # print("STEP 2: Enrichment Query (use with movie URIs from Step 1):")
    # print("=" * 80)
    # # Example with sample movie URIs
    # sample_uris = ["wd:Q113803868", "wd:Q116181404", "wd:Q113803869"]
    # enrichment_query = get_enrichment_query_string(sample_uris)
    # print(enrichment_query)
    # print("=" * 80)
    # print("\nTo use this function programmatically, uncomment the code below.\n")
    
    # Fetches up to 1000 movies per year for 2023 and 2024.
    asyncio.run(main(movies_per_year=1000, start_year=2023, end_year=2024))
    
    # Count all years since 1950 and make a plot, then save the figure.
    # import matplotlib.pyplot as plt

    # async def plot_movie_counts_by_year(start_year=1950, end_year=2024):
    #     years = list(range(start_year, end_year + 1))
    #   counts = []
    #    async with aiohttp.ClientSession() as session:
    #        for year in years:
    #            count = await count_movies_by_year(session, year)
    #            counts.append(count)
    #            print(f"Year {year}: {count} movies available in Wikidata")
    #    # Plotting
    #    plt.figure(figsize=(12, 6))
    #    plt.plot(years, counts, marker='o')
    #    plt.xlabel("Year")
    #    plt.ylabel("Number of Movies")
    #    plt.title(f"Wikidata Movie Counts {start_year}-{end_year}")
    #    plt.grid(True)
    #    plt.tight_layout()
    #    plt.savefig(f'wikidata_movie_counts_{start_year}_{end_year}.png')
    #    print(f"Saved plot to wikidata_movie_counts_{start_year}_{end_year}.png")

    # Run the plotting coroutine
    # asyncio.run(plot_movie_counts_by_year(start_year=1950, end_year=2024))

    # total = 1800+1619+1568+1696+1635+1646+1669+1800+1877+1819+1893+1857+1905+1884+2208+2091+2123+2339+2435+2577+2611+2730+2866+2862+3055+2908+2883+3037+3069+3192+3360+3008+3017+2893+3027+3132+3031+3157+3118+3253+3001+2997+2881+2840+2819+2883+2865+3044+3084+3286+3411+3750+3883+4249+4472+5015+5522+5690+5956+6085+6056+6048+6444+6718+6690+6725+8589+6179+6045+5583+3667+3822+4062+3721+3200
    # total = 259932
