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
import pandas as pd
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
EN_WIKIPEDIA_URL = "https://en.wikipedia.org/"

FILM_ITEM_ID = "wd:Q11424"
RELEASE_DATE_ID = "wdt:P577"

TELEVISION_SERIES_EPISODE_ID = "wd:Q21191270"
SHORT_FILM_ID = "wd:Q24862"
EXCLUDED_CLASSES = [TELEVISION_SERIES_EPISODE_ID, SHORT_FILM_ID]

VALID_IN_PLACE_ID = "P3005"
WORLDWIDE_BOX_OFFICE_PLACE_ID = "wd:Q13780930"

IS_INSTANCE = "wdt:P31"
IS_SUBCLASS = "wdt:P279"
IS_INSTANCE_OF_SOME_SUBCLASS = f"{IS_INSTANCE}/{IS_SUBCLASS}*"

def get_wikidata_subclasses_query_string(parent_classes_qids: List[str]) -> str:
    values_clause = "VALUES ?parentClass {\n"
    for parent_class in parent_classes_qids:
        values_clause += f"    wd:{parent_class}\n"
    values_clause += "}\n"

    query = f"""
    SELECT DISTINCT ?subclassLabel WHERE {{
        {values_clause}
        ?subclass {IS_SUBCLASS}* ?parentClass.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    return query.strip()

async def get_wikidata_subclasses(parent_classes: List[str]) -> List[str]:
    query = get_wikidata_subclasses_query_string(parent_classes)
    headers = get_wikidata_headers()
    timeout = aiohttp.ClientTimeout(total=60)
    params = {'query': query, 'format': 'json'}
    async with aiohttp.ClientSession() as session:
        async with session.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=timeout) as response:
            if response.status == 200:
                text = await response.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON response for subclasses: {e}")
                    return []
                
                subclasses = []
                for result in data['results']['bindings']:
                    subclass = result.get('subclassLabel', {}).get('value', None)
                    if subclass:
                        subclasses.append(subclass)
                
                return subclasses
            else:
                logger.error(f"HTTP Error {response.status} when fetching subclasses")
                return []

def get_exclude_deprecated_filter(feature_variable: str) -> str:
    return f"""
        ?{feature_variable}_intermediate_0 wikibase:rank ?rank.
        FILTER(?rank != wikibase:DeprecatedRank)
    """

def get_box_office_filter(feature_variable: str) -> str:
    return f"""
        {get_exclude_deprecated_filter(feature_variable)}
        FILTER NOT EXISTS {{
            ?{feature_variable}_intermediate_0 pq:{VALID_IN_PLACE_ID} ?place.
            FILTER(?place != {WORLDWIDE_BOX_OFFICE_PLACE_ID})
        }}
    """

def get_worldwide_box_office_filter(feature_variable: str) -> str:
    return f"""
        {get_exclude_deprecated_filter(feature_variable)}
        ?{feature_variable}_intermediate_0 pq:{VALID_IN_PLACE_ID} {WORLDWIDE_BOX_OFFICE_PLACE_ID}.
    """

def get_qid_from_uri(uri: str) -> str:
    """Convert a full Wikidata URI to its QID format (e.g., Q12345)."""
    if uri.startswith("http://www.wikidata.org/entity/"):
        qid = uri.split('/')[-1]
        return qid
    elif uri.startswith("wd:Q"):
        return uri[3:] 
    return uri

def extract_qids_from_uris(uris: str) -> str:
    list_of_qids = [
        get_qid_from_uri(x.strip()) for x in uris.split(",")
    ]
    return ",".join(list_of_qids)

def convert_to_minutes(duration_str: str) -> str:
    if not duration_str or len(duration_str.strip()) == 0:
        return ""
    list_of_durations_in_seconds = [
        str(int(float(x.strip()) / 60)) for x in duration_str.split(",")
    ]
    return ",".join(list_of_durations_in_seconds)


FEATURE_SPECIFICATIONS = pd.DataFrame.from_records(
    columns=["feature", "QID_list", "useLabels", "aggregationFunction", "extraFilter", "postprocessingFunction"],
    data=[
        ("movie_id", [], False, None, None, extract_qids_from_uris),
        ("title", [], True, None, None, None),
        ("wikidata_class", [IS_INSTANCE], True, "GROUP_CONCAT", None, None),
        ("summary", None, None, None, None, None),
        ("release_date", [RELEASE_DATE_ID], False, "MIN", None, None),

        ("genre", ["wdt:P136"], True, "GROUP_CONCAT", None, None),
        ("genre_id", ["wdt:P136"], False, "GROUP_CONCAT", None, extract_qids_from_uris),

        ("directors", ["wdt:P57"], True, "GROUP_CONCAT", None, None),
        ("directors_id", ["wdt:P57"], False, "GROUP_CONCAT", None, extract_qids_from_uris),

        ("actors", ["wdt:P161"], True, "GROUP_CONCAT", None, None),
        ("actors_id", ["wdt:P161"], False, "GROUP_CONCAT", None, extract_qids_from_uris),

        ("duration_all", ["p:P2047", "psn:P2047", "wikibase:quantityAmount"], False, "GROUP_CONCAT",
         get_exclude_deprecated_filter("duration_all"), convert_to_minutes),
        ("duration", ["p:P2047", "psn:P2047", "wikibase:quantityAmount"], False, "MIN", 
         get_exclude_deprecated_filter("duration"), convert_to_minutes),
        
        ("imdb_id", ["wdt:P345"], False, "SAMPLE", None, None),
        ("country", ["wdt:P495"], True, "GROUP_CONCAT", None, None),
        ("set_in_period", ["wdt:P2408"], True, "GROUP_CONCAT", None, None),
        ("awards", ["wdt:P166"], True, "GROUP_CONCAT", None, None),
        ("wikipedia_link", [], False, "SAMPLE", None, None),

        ("budget", ["p:P2130", "psv:P2130", "wikibase:quantityAmount"], False, "NONDISTINCT_GROUP_CONCAT", None, None),
        ("budget_currency", ["p:P2130", "psv:P2130", "wikibase:quantityUnit"], True, "NONDISTINCT_GROUP_CONCAT", None, None),

        ("box_office", ["p:P2142", "psv:P2142", "wikibase:quantityAmount"], False, "NONDISTINCT_GROUP_CONCAT", 
         get_box_office_filter("box_office"), None),
        ("box_office_currency", ["p:P2142", "psv:P2142", "wikibase:quantityUnit"], True, "NONDISTINCT_GROUP_CONCAT", 
         get_box_office_filter("box_office_currency"), None),

        ("box_office_worldwide", ["p:P2142", "psv:P2142", "wikibase:quantityAmount"], False, "NONDISTINCT_GROUP_CONCAT",
         get_worldwide_box_office_filter("box_office_worldwide"), None),
        ("box_office_worldwide_currency", ["p:P2142", "psv:P2142", "wikibase:quantityUnit"], True, "NONDISTINCT_GROUP_CONCAT",
         get_worldwide_box_office_filter("box_office_worldwide_currency"), None)
    ]
)


def get_wikidata_headers() -> Dict[str, str]:
    """Get headers for Wikidata API requests with user agent from environment."""
    user_email = os.getenv("WIKIDATA_USER_EMAIL", "anonymous@example.com")
    return {
        "User-Agent": f"MoviesAnalysis/1.0 ({user_email})",
        "Accept": "application/json"
    }


def get_movie_base_where_clause(year: int, excluded_classes: list[str] = EXCLUDED_CLASSES) -> str:
    """Returns WHERE part of SparQL query for movies released in a given year.

    Movies that are instances of some excluded subclass (or subclass of one of these) are excluded.
    Only movies with english Wikipedia link and english label are included.

    Args:
        year: The release year to query for.
        excluded_classes: Movie classes to exclude. Defaults to EXCLUDED_CLASSES.

    Returns:
        String containing WHERE part of query.
    """
    exclude_classes_query_part = ""
    for excluded_class in excluded_classes:
        exclude_classes_query_part += f"MINUS {{?movieSubclass {IS_SUBCLASS}* {excluded_class}}}\n"

    return f"""WHERE {{
        ?movieSubclass {IS_SUBCLASS}* {FILM_ITEM_ID}.

        {exclude_classes_query_part}

        ?movie {IS_INSTANCE} ?movieSubclass;
                {RELEASE_DATE_ID} ?releaseDate;
                wikibase:sitelinks ?sitelinks.

        FILTER(?sitelinks > 0)
        ?sitelink schema:about ?movie;
            schema:isPartOf <{EN_WIKIPEDIA_URL}>.

        FILTER(YEAR(?releaseDate) = {year})
        FILTER NOT EXISTS {{
            ?movie {RELEASE_DATE_ID} ?dateBeforeYear.
            FILTER(YEAR(?dateBeforeYear) < {year})
        }}

        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en".
            ?movie rdfs:label ?movieLabel.
        }}
        FILTER(BOUND(?movieLabel) && ?movieLabel != "")
    }}
    """


def sanitize_json_text(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)


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
    base_where = get_movie_base_where_clause(year)
    
    count_query = f"""
        SELECT (COUNT(DISTINCT ?movie) AS ?totalMovies)
        {base_where}
    """
    
    params = {'query': count_query, 'format': 'json'}
    headers = get_wikidata_headers()
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=timeout) as response:
            if response.status == 200:
                text = await response.text()
                text = sanitize_json_text(text)
                
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
    base_where_query = get_movie_base_where_clause(year)
    query = f"""
        SELECT DISTINCT ?movie ?sitelinks
        {base_where_query}
        ORDER BY DESC(?sitelinks)
        LIMIT {limit}
    """
    return query.strip()


def get_aggregation_function_string(aggregation_function: Optional[str], feature_variable: str) -> str:
    if aggregation_function == "GROUP_CONCAT":
        return f"GROUP_CONCAT(DISTINCT ?{feature_variable}; separator=\", \")"
    elif aggregation_function == "NONDISTINCT_GROUP_CONCAT":
        return f"GROUP_CONCAT(?{feature_variable}; separator=\", \")"
    return f"{aggregation_function}(?{feature_variable})"


def get_enrichment_feature_query(
    feature: str,
    qid_list: list[str],
    use_labels: bool,
    aggregation_function: Optional[str],
    extra_filter: str
) -> str:
    if qid_list is None or len(qid_list) == 0:
        return ""

    feature_var = feature + ("Label" if use_labels else "_")
    labels_query = f"""
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?{feature}_ rdfs:label ?{feature}Label. }}
    """ if use_labels else ""

    qid_link = ""
    previous_var = "?movie_id"
    for iteration, qid in enumerate(qid_list[:-1]):
        intermediate_var = f"?{feature}_intermediate_{iteration}"
        qid_link += f"""
            {previous_var} {qid} {intermediate_var}.
        """
        previous_var = intermediate_var
    qid_link += f"""
        {previous_var} {qid_list[-1]} ?{feature}_.
    """
    aggregation_function_string = get_aggregation_function_string(aggregation_function, feature_var)

    feature_enrichment_query = f"""
        OPTIONAL {{
            SELECT ?movie_id ({aggregation_function_string} AS ?{feature}) WHERE {{
                {qid_link}
                {extra_filter if extra_filter else ""}
                {labels_query}
            }} GROUP BY ?movie_id
        }}
    """
    return feature_enrichment_query


def get_enrichment_query_string(movie_uris: List[str], feature_specifications: pd.DataFrame) -> str:
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
    values_clause = "VALUES ?movie_id {\n"
    for uri in movie_uris:
        if not uri.startswith("wd:") and not uri.startswith("http"):
            uri = f"wd:{uri}" if uri.startswith("Q") else uri
        values_clause += f"    {uri}\n"
    values_clause += "}\n"
    
    features = feature_specifications["feature"].tolist()
    featureVariables = [
        f"?{feature}" for feature in features
    ]
    select_features = " ".join(featureVariables)
    feature_enrichments = ""

    for feature, qid_list, use_labels, aggregation_function, extra_filter in zip(
        feature_specifications["feature"],
        feature_specifications["QID_list"],
        feature_specifications["useLabels"],
        feature_specifications["aggregationFunction"],
        feature_specifications["extraFilter"]
    ):
        if qid_list is not None and aggregation_function is not None:
            enrichment_subquery = get_enrichment_feature_query(
                feature, qid_list, use_labels, aggregation_function, extra_filter
            )
            feature_enrichments += enrichment_subquery + "\n"

    query = f"""
SELECT {select_features}
WHERE {{
    {values_clause}

    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?movie_id rdfs:label ?title. }}
    OPTIONAL {{ 
        ?wikipedia_link schema:about ?movie_id;
                           schema:isPartOf <{EN_WIKIPEDIA_URL}>.
    }}

    {feature_enrichments}
}}
"""
    return query.strip()


def postprocess_query_result(
    result: Dict,
    feature_specifications: pd.DataFrame = FEATURE_SPECIFICATIONS
) -> Dict:
    for feature, postprocess_func in zip(
        feature_specifications["feature"].tolist(),
        feature_specifications["postprocessingFunction"].tolist()
    ):
        postprocess_func = postprocess_func if postprocess_func is not None else (lambda x: x)
        result[feature] = postprocess_func(result.get(feature, {}).get('value', ""))
    return result


async def get_movies_by_year(
    session: aiohttp.ClientSession,
    year: int,
    limit: int = 50,
    verbose: bool = False,
    feature_specifications: pd.DataFrame = FEATURE_SPECIFICATIONS
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
                    text = await response.text()
                    text = sanitize_json_text(text)
                    
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
                            movie_uris.append("wd:" + get_qid_from_uri(movie_uri))
                    
                    logger.info(f"Year {year}: Step 1 found {len(movie_uris)} distinct movie URIs")
                    
                    if not movie_uris:
                        logger.warning(f"Year {year}: No movies found in discovery query")
                        return []                   
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
    BATCH_SIZE = 20
    movies = []
    
    for batch_start in range(0, len(movie_uris), BATCH_SIZE):
        batch_uris = movie_uris[batch_start:batch_start + BATCH_SIZE]
        enrichment_query = get_enrichment_query_string(batch_uris, feature_specifications=feature_specifications)
        
        current_batch_num = batch_start//BATCH_SIZE + 1
        
        if verbose:
            logger.debug(f"Step 2 - Processing batch {current_batch_num}/{(len(movie_uris)-1)//BATCH_SIZE + 1} ({len(batch_uris)} movies) for year {year}")
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=120)

                async with session.post(
                    WIKIDATA_SPARQL_URL,
                    data={'query': enrichment_query, 'format': 'json'},
                    headers={**headers, 'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        text = sanitize_json_text(text)
                        
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError as e:
                            logger.error(f"Year {year}: Step 2 batch {current_batch_num} JSON decode error - {e}")
                            break
                        
                        results = data['results']['bindings']
                        
                        dropped_no_title = 0
                        total_processed = len(results)
                        
                        if verbose:
                            logger.info(f"Year {year}: Step 2 batch {current_batch_num} returned {total_processed} enriched movie results")
                        if len(results) > 0:
                            logger.debug(f"Year {year}: Sample result keys: {list(results[0].keys())}")

                        for result in results:
                            movies.append(
                                postprocess_query_result(result, feature_specifications=feature_specifications)
                            )
                        
                        if verbose:
                            logger.debug(
                                f"Year {year}: Batch {current_batch_num} processed {total_processed} results."
                            )                        
                        break
                    elif response.status == 429:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 1)
                            logger.warning(f"Year {year}: Step 2 batch {current_batch_num} rate limit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Year {year}: Step 2 batch {current_batch_num} HTTP Error 429 - Rate limit exceeded after {max_retries} attempts")
                            break
                    else:
                        try:
                            error_text = await response.text()
                            if len(error_text) > 500:
                                error_text = error_text[:500] + "..."
                        except:
                            error_text = f"Status {response.status}"
                        
                        logger.error(f"Year {year}: Step 2 batch {current_batch_num} HTTP Error {response.status} - {error_text} \n--- Failing Query: ---\n{enrichment_query}\n--- End Query ---")
                        break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(f"Year {year}: Step 2 batch {current_batch_num} error - {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Year {year}: Step 2 batch {current_batch_num} error after {max_retries} attempts - {e}", exc_info=True)
                    break
        
        # Small delay between batches to avoid rate limiting
        if batch_start + BATCH_SIZE < len(movie_uris):
            await asyncio.sleep(0.5)
    
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
    feature_specifications: pd.DataFrame = FEATURE_SPECIFICATIONS,
    verbose: bool = False,
    delay: float = 0.5,
    save_per_year: bool = False
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch movies for multiple years with rate limiting to avoid 429 errors.

    Args:
        movies_per_year: Maximum number of movies to fetch per year
        start_year: First year to query (inclusive)
        end_year: Last year to query (inclusive)
        verbose: Whether to print progress messages (default: False)
        delay: Delay in seconds between year requests (default: 0.5)
        save_per_year: If True, save a CSV file after each year is processed (default: False)

    Returns:
        List of all movies from all years
    """
    async with aiohttp.ClientSession() as session:
        all_movies = []
        for year in range(start_year, end_year + 1):
            movies = await get_movies_by_year(
                session, year, movies_per_year, verbose=verbose, feature_specifications=feature_specifications
            )
            all_movies.extend(movies)
            
            # Save CSV file for this year if requested
            if save_per_year and movies:
                filename = f'wikidata_movies_{year}.csv'
                save_movies_to_csv(movies, filename=filename)
                logger.info(f"Saved {len(movies)} movies for year {year} to {filename}")
            
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
    show_available_counts: bool = False,
    save_per_year: bool = False,
    feature_specifications: pd.DataFrame = FEATURE_SPECIFICATIONS
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
        save_per_year: If True, save a CSV file after each year is processed (default: False)

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
    
    all_movies = await fetch_movies_for_years(
        movies_per_year,
        start_year,
        end_year,
        feature_specifications=feature_specifications,
        verbose=verbose,
        delay=delay,
        save_per_year=save_per_year
    )

    total_years = end_year - start_year + 1
    expected_max = movies_per_year * total_years
    logger.info(
        f"Fetched {len(all_movies)} movies total (max possible: {expected_max}) "
        f"from {total_years} years. Movies ordered by popularity (sitelinks) per year"
    )

    return all_movies


def save_movies_to_csv(
    movies: List[Dict[str, Optional[str]]],
    filename: str = 'wikidata_movies.csv',
    feature_specifications: pd.DataFrame = FEATURE_SPECIFICATIONS
) -> None:
    """
    Save movies data to a CSV file.

    Args:
        movies: List of movie dictionaries
        filename: Output CSV filename
    """
    import csv

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

    data_dir = os.path.join(base_dir, 'data')
    filename = os.path.join(data_dir, filename)
    
    if not movies:
        logger.warning("No movies to save - empty movie list")
        return
    
    os.makedirs(data_dir, exist_ok=True)
    
    fieldnames = feature_specifications["feature"].tolist()
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(movies)
    
    logger.info(f"Saved {len(movies)} movies to {filename}")


async def main(movies_per_year: int = 50, start_year: int = 1950, end_year: int = 2024, show_counts: bool = False, save_per_year: bool = True):
    """
    Example main function that fetches movies and saves them to CSV.

    Args:
        movies_per_year: Maximum number of UNIQUE movies per year to fetch
        start_year: First year to query
        end_year: Last year to query
        show_counts: If True, show how many movies are available before fetching
        save_per_year: If True, save a CSV file for each year (default: True)
    """
    movies = await fetch_movies(
        movies_per_year, 
        start_year, 
        end_year, 
        show_available_counts=show_counts,
        verbose=True,
        save_per_year=save_per_year
    )
    
    if save_per_year:
        save_movies_to_csv(movies, filename=f'wikidata_movies_{start_year}_to_{end_year}.csv')
        logger.info(f"Also saved combined file with {len(movies)} movies from all years")


if __name__ == "__main__":
    asyncio.run(main(movies_per_year=5000, start_year=2004, end_year=2004))
