"""
Data Pipeline for Movie Data Collection and Processing

This pipeline orchestrates three main steps:
1. Step 1: Wikidata handler - collects movie data from Wikidata by year
2. Step 2: MovieDB handler - enriches data with popularity and votes via Wikidata ID
3. Step 3: Wikipedia handler - retrieves movie plots via sitelinks
4. Step 4: Embeddings - generates embeddings for plots and saves per year

All steps check for existing files and expand CSV files with new data.
"""

import asyncio
import aiohttp
import os
import sys
import csv
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Set
from pathlib import Path

# Add src directory to Python path for imports
try:
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
except NameError:
    # Fallback if __file__ is not defined
    SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), 'src'))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

# Import handlers
from api.wikidata_handler import (
    fetch_movies_for_years,
    save_movies_to_csv as save_wikidata_to_csv
)
from api.moviedb_handler import (
    get_movie_by_wiki_id,
    get_movie_by_id,
    get_tmdb_api_key
)
from api.wikipedia_handler import (
    get_page_from_url,
    get_plot_section
)
from embedding.embedding import EmbeddingService
from embedding.factory import verify_gpu_setup, verify_embeddings
import wikipediaapi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory paths
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))

DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================
# Modify these variables to configure the pipeline behavior

START_YEAR = 1950
END_YEAR = 2024
MOVIES_PER_YEAR = 8000

# Skip steps (set to True to skip)
SKIP_WIKIDATA = False
SKIP_MOVIEDB = False
SKIP_WIKIPEDIA = False
SKIP_EMBEDDINGS = False

# Force refresh existing files (set to True to re-fetch even if files exist)
FORCE_REFRESH = False

# Verbose logging (set to False for less output)
VERBOSE = True

# Embedding configuration
MODEL_NAME = 'BAAI/bge-m3'
BATCH_SIZE = 128
# Target devices for embeddings (None = auto-detect, or specify like ['cuda:0', 'cuda:1'])
TARGET_DEVICES = None


def get_csv_path(year: int) -> str:
    """Get the CSV file path for a specific year."""
    return os.path.join(DATA_DIR, f'wikidata_movies_{year}.csv')


def load_existing_csv(year: int) -> pd.DataFrame:
    """Load existing CSV file for a year, return empty DataFrame if doesn't exist."""
    csv_path = get_csv_path(year)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype=str)
            logger.info(f"Loaded existing CSV for year {year}: {len(df)} movies")
            return df
        except Exception as e:
            logger.warning(f"Error loading CSV for year {year}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, year: int) -> None:
    """Save DataFrame to CSV file for a specific year."""
    csv_path = get_csv_path(year)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved {len(df)} movies to {csv_path}")


def get_existing_movie_ids(df: pd.DataFrame) -> Set[str]:
    """Extract set of existing movie IDs from DataFrame."""
    if 'movie_id' in df.columns:
        return set(df['movie_id'].dropna().astype(str))
    return set()


# ============================================================================
# STEP 1: Wikidata Handler
# ============================================================================

async def step1_wikidata(
    start_year: int,
    end_year: int,
    movies_per_year: int = 50,
    verbose: bool = True,
    force_refresh: bool = False
) -> Dict[int, pd.DataFrame]:
    """
    Step 1: Collect data from Wikidata.
    
    Args:
        start_year: First year to process
        end_year: Last year to process
        movies_per_year: Number of movies per year to fetch
        verbose: Enable verbose logging
        force_refresh: If True, re-fetch even if files exist
        
    Returns:
        Dictionary mapping year to DataFrame
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Wikidata Data Collection")
    logger.info("=" * 80)
    
    year_dataframes = {}
    
    for year in range(start_year, end_year + 1):
        csv_path = get_csv_path(year)
        
        if not force_refresh and os.path.exists(csv_path):
            logger.info(f"Year {year}: CSV file already exists, skipping...")
            df = load_existing_csv(year)
            year_dataframes[year] = df
            continue
        
        logger.info(f"Year {year}: Fetching movies from Wikidata...")
        
        async with aiohttp.ClientSession() as session:
            movies = await fetch_movies_for_years(
                movies_per_year=movies_per_year,
                start_year=year,
                end_year=year,
                verbose=verbose,
                delay=0.5,
                save_per_year=True
            )
        
        if movies:
            # Convert to DataFrame
            df = pd.DataFrame(movies)
            year_dataframes[year] = df
            logger.info(f"Year {year}: Fetched {len(movies)} movies")
        else:
            logger.warning(f"Year {year}: No movies fetched")
            year_dataframes[year] = pd.DataFrame()
    
    logger.info(f"Step 1 completed: Processed {len(year_dataframes)} years")
    return year_dataframes


# ============================================================================
# STEP 2: MovieDB Handler
# ============================================================================

async def step2_moviedb(
    year_dataframes: Dict[int, pd.DataFrame],
    verbose: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    Step 2: Enrich data with TMDb popularity and votes via Wikidata ID.
    
    Args:
        year_dataframes: Dictionary mapping year to DataFrame
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping year to enriched DataFrame
    """
    logger.info("=" * 80)
    logger.info("STEP 2: MovieDB Data Enrichment")
    logger.info("=" * 80)
    
    try:
        api_key = get_tmdb_api_key()
    except ValueError as e:
        logger.error(f"TMDb API key not found: {e}")
        logger.warning("Skipping Step 2: MovieDB enrichment")
        return year_dataframes
    
    enriched_dataframes = {}
    
    async with aiohttp.ClientSession() as session:
        for year, df in year_dataframes.items():
            if df.empty:
                logger.info(f"Year {year}: No movies to enrich")
                enriched_dataframes[year] = df
                continue
            
            logger.info(f"Year {year}: Enriching {len(df)} movies with TMDb data...")
            
            # Check which movies already have TMDb data
            has_tmdb_data = (
                df['popularity'].notna() if 'popularity' in df.columns 
                else pd.Series([False] * len(df), index=df.index)
            )
            
            # Process movies that don't have TMDb data
            movies_to_process = df[~has_tmdb_data]
            
            if movies_to_process.empty:
                logger.info(f"Year {year}: All movies already have TMDb data")
                enriched_dataframes[year] = df
                continue
            
            logger.info(f"Year {year}: Processing {len(movies_to_process)} movies without TMDb data")
            
            # Initialize new columns if they don't exist
            if 'popularity' not in df.columns:
                df['popularity'] = None
            if 'vote_average' not in df.columns:
                df['vote_average'] = None
            if 'vote_count' not in df.columns:
                df['vote_count'] = None
            if 'tmdb_id' not in df.columns:
                df['tmdb_id'] = None
            
            # Process each movie
            for idx, row in movies_to_process.iterrows():
                movie_id = row.get('movie_id', '')
                if not movie_id or pd.isna(movie_id):
                    continue
                
                # Extract Q-number - TMDb API expects the full Wikidata ID including 'Q'
                wiki_id = str(movie_id).strip()
                if not wiki_id.startswith('Q'):
                    continue
                
                try:
                    # Get TMDb data using Wikidata ID
                    # Note: wiki_id is passed as string (e.g., "Q155653"), function signature might say int but accepts string
                    tmdb_data = await get_movie_by_wiki_id(
                        session, str(wiki_id), api_key, verbose=verbose
                    )
                    
                    if tmdb_data and tmdb_data.get('movie_results'):
                        movie_result = tmdb_data['movie_results'][0]
                        tmdb_id = movie_result.get('id')
                        
                        if tmdb_id:
                            df.at[idx, 'tmdb_id'] = tmdb_id
                            
                            # Fetch full movie details to get popularity, vote_average, and vote_count
                            full_details = await get_movie_by_id(
                                session, int(tmdb_id), api_key, include_additional=False, verbose=False
                            )
                            if full_details:
                                df.at[idx, 'popularity'] = full_details.get('popularity')
                                df.at[idx, 'vote_average'] = full_details.get('vote_average')
                                df.at[idx, 'vote_count'] = full_details.get('vote_count')
                                
                                if verbose:
                                    logger.debug(
                                        f"Year {year}, {row.get('title', 'Unknown')}: "
                                        f"popularity={full_details.get('popularity')}, "
                                        f"votes={full_details.get('vote_count')}"
                                    )
                    else:
                        if verbose:
                            logger.debug(f"Year {year}, {row.get('title', 'Unknown')}: No TMDb data found")
                    
                    # Rate limiting - be nice to the API (0.25s per movie, but we make 2 calls per movie)
                    await asyncio.sleep(0.15)
                    
                except Exception as e:
                    logger.warning(f"Year {year}, Movie {movie_id}: Error fetching TMDb data - {e}")
                    continue
            
            # Save updated CSV
            save_csv(df, year)
            enriched_dataframes[year] = df
            logger.info(f"Year {year}: Enrichment complete")
    
    logger.info("Step 2 completed: MovieDB enrichment finished")
    return enriched_dataframes


# ============================================================================
# STEP 3: Wikipedia Handler
# ============================================================================

def step3_wikipedia(
    year_dataframes: Dict[int, pd.DataFrame],
    verbose: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    Step 3: Retrieve movie plots from Wikipedia via sitelinks.
    
    Args:
        year_dataframes: Dictionary mapping year to DataFrame
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping year to DataFrame with plots
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Wikipedia Plot Retrieval")
    logger.info("=" * 80)
    
    # Initialize Wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='GroupDataLiteracy/1.0 (movie data pipeline)',
        language='en'
    )
    
    enriched_dataframes = {}
    
    for year, df in year_dataframes.items():
        if df.empty:
            logger.info(f"Year {year}: No movies to process")
            enriched_dataframes[year] = df
            continue
        
        logger.info(f"Year {year}: Retrieving plots for {len(df)} movies...")
        
        # Initialize plot column if it doesn't exist
        if 'plot' not in df.columns:
            df['plot'] = None
        
        # Check which movies already have plots
        has_plot = df['plot'].notna() & (df['plot'] != '')
        
        # Process movies that don't have plots
        movies_to_process = df[~has_plot]
        
        if movies_to_process.empty:
            logger.info(f"Year {year}: All movies already have plots")
            enriched_dataframes[year] = df
            continue
        
        logger.info(f"Year {year}: Processing {len(movies_to_process)} movies without plots")
        
        # Process each movie
        for idx, row in movies_to_process.iterrows():
            wikipedia_link = row.get('wikipedia_link', '')
            
            if not wikipedia_link or pd.isna(wikipedia_link):
                continue
            
            try:
                # Get Wikipedia page
                page = get_page_from_url(wiki_wiki, str(wikipedia_link))
                
                # Extract plot section
                plot = get_plot_section(page)
                
                if plot:
                    df.at[idx, 'plot'] = plot
                    if verbose:
                        logger.debug(
                            f"Year {year}, {row.get('title', 'Unknown')}: "
                            f"Retrieved plot ({len(plot)} chars)"
                        )
                else:
                    # Fallback to summary if plot not found
                    if hasattr(page, 'summary') and page.summary:
                        df.at[idx, 'plot'] = page.summary
                        if verbose:
                            logger.debug(
                                f"Year {year}, {row.get('title', 'Unknown')}: "
                                f"Using summary instead of plot"
                            )
                
            except ValueError as e:
                if verbose:
                    logger.debug(f"Year {year}, {row.get('title', 'Unknown')}: {e}")
            except Exception as e:
                logger.warning(f"Year {year}, Movie {row.get('title', 'Unknown')}: Error - {e}")
        
        # Save updated CSV
        save_csv(df, year)
        enriched_dataframes[year] = df
        logger.info(f"Year {year}: Plot retrieval complete")
    
    logger.info("Step 3 completed: Wikipedia plot retrieval finished")
    return enriched_dataframes


# ============================================================================
# STEP 4: Embeddings
# ============================================================================

def step4_embeddings(
    year_dataframes: Dict[int, pd.DataFrame],
    model_name: str = 'BAAI/bge-m3',
    target_devices: Optional[List[str]] = None,
    batch_size: int = 128,
    verbose: bool = True
) -> None:
    """
    Step 4: Generate embeddings for movie plots and save per year.
    Uses the factory functions for GPU setup and verification.
    
    Args:
        year_dataframes: Dictionary mapping year to DataFrame
        model_name: Name of the embedding model
        target_devices: List of CUDA devices (e.g., ['cuda:0'])
        batch_size: Batch size for encoding
        verbose: Enable verbose logging
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Embedding Generation")
    logger.info("=" * 80)
    
    # Use factory's GPU setup verification
    # Setup target_devices first
    try:
        import torch
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            else:
                target_devices = ['cpu']
                logger.warning("No CUDA devices found, using CPU (this will be slow)")
        
        # Verify GPU setup using factory (only if not using CPU)
        # Note: verify_gpu_setup may modify TARGET_DEVICES in factory module, but we use our own variable
        if target_devices != ['cpu']:
            # Temporarily set factory's TARGET_DEVICES for verification
            import embedding.factory as factory_module
            original_devices = getattr(factory_module, 'TARGET_DEVICES', None)
            factory_module.TARGET_DEVICES = target_devices
            try:
                # verify_gpu_setup may adjust the devices, so update our target_devices if needed
                verify_gpu_setup()
                # If factory adjusted devices, use the adjusted list
                if factory_module.TARGET_DEVICES != target_devices:
                    target_devices = factory_module.TARGET_DEVICES
                    logger.info(f"GPU setup adjusted devices to: {target_devices}")
            except SystemExit:
                # verify_gpu_setup calls sys.exit(1) if CUDA is required but not available
                # In pipeline, we'll fall back to CPU instead
                logger.warning("Factory GPU verification failed, falling back to CPU")
                target_devices = ['cpu']
            finally:
                # Restore original if it existed
                if original_devices is not None:
                    factory_module.TARGET_DEVICES = original_devices
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        target_devices = ['cpu']
    except Exception as e:
        logger.warning(f"Error during GPU setup verification: {e}. Continuing with available devices...")
    
    # Initialize embedding service
    try:
        embedding_service = EmbeddingService(model_name=model_name)
        logger.info(f"Initialized embedding service with model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        logger.warning("Skipping Step 4: Embedding generation")
        return
    
    for year, df in year_dataframes.items():
        if df.empty:
            logger.info(f"Year {year}: No movies to embed")
            continue
        
        # Check if embeddings already exist
        embeddings_path = os.path.join(DATA_DIR, f'movie_embeddings_{year}.npy')
        if os.path.exists(embeddings_path):
            logger.info(f"Year {year}: Embeddings file already exists, skipping...")
            continue
        
        # Extract plots
        if 'plot' not in df.columns:
            logger.warning(f"Year {year}: No 'plot' column found, skipping embeddings")
            continue
        
        # Filter out movies without plots
        plots = df['plot'].dropna().astype(str).tolist()
        
        if not plots:
            logger.warning(f"Year {year}: No plots available for embedding")
            continue
        
        logger.info(f"Year {year}: Generating embeddings for {len(plots)} plots...")
        
        try:
            import time
            start_time = time.time()
            
            # Generate embeddings using factory approach
            if 'cpu' in target_devices or (len(target_devices) == 1 and target_devices[0] == 'cpu'):
                # Single device encoding for CPU
                logger.info(f"Year {year}: Encoding on CPU...")
                embeddings = embedding_service.model.encode(
                    plots,
                    batch_size=batch_size,
                    show_progress_bar=verbose
                )
            else:
                # Multi-GPU encoding using factory's parallel approach
                logger.info(f"Year {year}: Encoding on {len(target_devices)} device(s): {target_devices}")
                embeddings = embedding_service.encode_parallel(
                    corpus=plots,
                    target_devices=target_devices,
                    batch_size=batch_size
                )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Verify embeddings using factory's verification function
            if verify_embeddings(embeddings, plots):
                # Save embeddings
                np.save(embeddings_path, embeddings)
                logger.info(f"Year {year}: Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")
                logger.info(f"Year {year}: Encoding completed in {duration:.2f} seconds ({len(plots)/duration:.2f} docs/sec)")
            else:
                logger.error(f"Year {year}: Embedding verification failed, not saving")
            
        except Exception as e:
            logger.error(f"Year {year}: Error generating embeddings - {e}", exc_info=verbose)
            continue
    
    logger.info("Step 4 completed: Embedding generation finished")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

async def run_pipeline(
    start_year: int = 1950,
    end_year: int = 2024,
    movies_per_year: int = 50,
    skip_wikidata: bool = False,
    skip_moviedb: bool = False,
    skip_wikipedia: bool = False,
    skip_embeddings: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
    model_name: str = 'BAAI/bge-m3',
    target_devices: Optional[List[str]] = None,
    batch_size: int = 128
) -> None:
    """
    Run the complete data pipeline.
    
    Args:
        start_year: First year to process
        end_year: Last year to process
        movies_per_year: Number of movies per year to fetch
        skip_wikidata: Skip Step 1 (Wikidata collection)
        skip_moviedb: Skip Step 2 (MovieDB enrichment)
        skip_wikipedia: Skip Step 3 (Wikipedia plots)
        skip_embeddings: Skip Step 4 (Embeddings)
        force_refresh: Force re-fetching even if files exist
        verbose: Enable verbose logging
        model_name: Name of the embedding model
        target_devices: List of CUDA devices
        batch_size: Batch size for embeddings
    """
    logger.info("=" * 80)
    logger.info("MOVIE DATA PIPELINE - Starting")
    logger.info("=" * 80)
    logger.info(f"Years: {start_year} to {end_year}")
    logger.info(f"Movies per year: {movies_per_year}")
    logger.info("")
    
    # Step 1: Wikidata
    if skip_wikidata:
        logger.info("Skipping Step 1: Wikidata collection")
        # Load existing data
        year_dataframes = {}
        for year in range(start_year, end_year + 1):
            df = load_existing_csv(year)
            year_dataframes[year] = df
    else:
        year_dataframes = await step1_wikidata(
            start_year=start_year,
            end_year=end_year,
            movies_per_year=movies_per_year,
            verbose=verbose,
            force_refresh=force_refresh
        )
    
    # Step 2: MovieDB
    if not skip_moviedb:
        year_dataframes = await step2_moviedb(
            year_dataframes=year_dataframes,
            verbose=verbose
        )
    else:
        logger.info("Skipping Step 2: MovieDB enrichment")
    
    # Step 3: Wikipedia
    if not skip_wikipedia:
        year_dataframes = step3_wikipedia(
            year_dataframes=year_dataframes,
            verbose=verbose
        )
    else:
        logger.info("Skipping Step 3: Wikipedia plot retrieval")
    
    # Step 4: Embeddings
    if not skip_embeddings:
        step4_embeddings(
            year_dataframes=year_dataframes,
            model_name=model_name,
            target_devices=target_devices,
            batch_size=batch_size,
            verbose=verbose
        )
    else:
        logger.info("Skipping Step 4: Embedding generation")
    
    logger.info("=" * 80)
    logger.info("MOVIE DATA PIPELINE - Completed")
    logger.info("=" * 80)


async def main():
    """Main entry point for the pipeline."""
    await run_pipeline(
        start_year=START_YEAR,
        end_year=END_YEAR,
        movies_per_year=MOVIES_PER_YEAR,
        skip_wikidata=SKIP_WIKIDATA,
        skip_moviedb=SKIP_MOVIEDB,
        skip_wikipedia=SKIP_WIKIPEDIA,
        skip_embeddings=SKIP_EMBEDDINGS,
        force_refresh=FORCE_REFRESH,
        verbose=VERBOSE,
        model_name=MODEL_NAME,
        target_devices=TARGET_DEVICES,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    asyncio.run(main())

    # nohup python data_pipeline.py > data_pipeline.log 2>&1 &

