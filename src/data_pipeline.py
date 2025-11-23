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
import re
import pandas as pd
import numpy as np
import logging
import atexit
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from api.wikipedia_handler import fetch_plot_from_url
from embedding.embedding import EmbeddingService
from embedding.util_embeddings import verify_gpu_setup, verify_embeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import chunking classes
try:
    from analysis.chunking import CLSToken, MeanPooling, ChunkFirstEmbed, LateChunking
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False
    logger.warning("Chunking classes not available. Install required dependencies or check path.")

# Base directory paths
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))

DATA_DIR = os.path.join(BASE_DIR, 'data')

# Global reference for cleanup
_embedding_service_instance = None

def _cleanup_multiprocessing_resources():
    """Cleanup function called at exit to free multiprocessing resources."""
    global _embedding_service_instance
    if _embedding_service_instance is not None:
        try:
            _embedding_service_instance.cleanup()
        except Exception:
            pass
    
    # Clean up any remaining multiprocessing resources
    try:
        import multiprocessing
        # Force cleanup of any remaining shared resources
        multiprocessing.active_children()  # Wait for any remaining processes
    except Exception:
        pass

# Register cleanup function
atexit.register(_cleanup_multiprocessing_resources)

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================
# Modify these variables to configure the pipeline behavior

START_YEAR = 1950
END_YEAR = 2024
MOVIES_PER_YEAR = 8000

# Skip steps (set to True to skip)
SKIP_WIKIDATA = True
SKIP_MOVIEDB = True
SKIP_WIKIPEDIA = True
SKIP_EMBEDDINGS = False

# Force refresh existing files (set to True to re-fetch even if files exist)
FORCE_REFRESH = False

# Verbose logging (set to False for less output)
VERBOSE = True

# Embedding configuration
MODEL_NAME = 'BAAI/bge-m3'
BATCH_SIZE = 35
# Target devices for embeddings (None = auto-detect, or specify like ['cuda:0', 'cuda:1'])
TARGET_DEVICES = None

# Chunking configuration
# Options: None, 'cls_token', 'mean_pooling', 'chunk_first_embed', 'late_chunking'
CHUNKING_METHOD = None
# Chunking parameters (only used for chunk_first_embed and late_chunking)
CHUNKING_CHUNK_SIZE = 512  # For chunk_first_embed
CHUNKING_WINDOW_SIZE = 512  # For late_chunking
CHUNKING_STRIDE = 256  # For both chunk_first_embed and late_chunking

# Save lexical weights (only available for BGE-M3 model)
SAVE_LEXICAL_WEIGHTS = False


def get_csv_path(year: int) -> str:
    """Get the CSV file path for a specific year."""
    return os.path.join(DATA_DIR, f'wikidata_movies_{year}.csv')


def load_lexical_weights(year: int, chunking_suffix: str = "", data_dir: Optional[str] = None) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]]:
    """
    Load lexical weights from file.
    
    Args:
        year: Year to load lexical weights for
        chunking_suffix: Suffix to append to filename (e.g., "_cls_token", "_mean_pooling")
        data_dir: Directory containing the files (default: uses DATA_DIR)
    
    Returns:
        Tuple of (token_indices_list, weights_list, movie_ids) or None if file doesn't exist
        - token_indices_list: List of numpy arrays, one per document, containing token IDs
        - weights_list: List of numpy arrays, one per document, containing corresponding weights
        - movie_ids: Array of movie IDs corresponding to the lexical weights
        - lexical_weights[i] corresponds to movie_ids[i]
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    lexical_weights_path = os.path.join(data_dir, f'movie_lexical_weights_{year}{chunking_suffix}.npz')
    
    if not os.path.exists(lexical_weights_path):
        logger.warning(f"Lexical weights file not found: {lexical_weights_path}")
        return None
    
    try:
        data = np.load(lexical_weights_path, allow_pickle=True)
        
        # Extract arrays
        token_indices_array = data['token_indices']
        weights_array = data['weights']
        movie_ids = data['movie_ids']
        
        # Convert object arrays back to lists of arrays
        token_indices_list = [token_indices_array[i] for i in range(len(token_indices_array))]
        weights_list = [weights_array[i] for i in range(len(weights_array))]
        
        logger.info(f"Loaded lexical weights for year {year}: {len(token_indices_list)} documents")
        logger.info(f"Total non-zero weights: {sum(len(ti) for ti in token_indices_list)}")
        
        return token_indices_list, weights_list, movie_ids
    
    except Exception as e:
        logger.error(f"Error loading lexical weights from {lexical_weights_path}: {e}")
        return None


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


# Note: clean_plot_text has been moved to api.wikipedia_handler
# Import it if needed elsewhere, but it's used within fetch_plot_from_url


def save_csv(df: pd.DataFrame, year: int) -> None:
    """Save DataFrame to CSV file for a specific year."""
    from api.wikipedia_handler import clean_plot_text
    
    csv_path = get_csv_path(year)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Clean plot column if it exists (remove line breaks, normalize whitespace)
    # Make a copy to avoid modifying the original DataFrame
    df_to_save = df.copy() if 'plot' in df.columns else df
    if 'plot' in df.columns:
        df_to_save['plot'] = df_to_save['plot'].apply(lambda x: clean_plot_text(x) if pd.notna(x) else x)
    
    # pandas default quoting (QUOTE_MINIMAL) handles commas correctly by quoting fields that contain commas
    df_to_save.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved {len(df_to_save)} movies to {csv_path}")


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
            logger.info(f"Year {year}: CSV file already exists, skipping Wikidata fetch...")
            df = load_existing_csv(year)
            year_dataframes[year] = df
            continue
        
        logger.info(f"Year {year}: CSV file not found, fetching movies from Wikidata...")
        
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
            
            # Count movies with and without TMDb data
            num_with_tmdb = has_tmdb_data.sum()
            num_without_tmdb = (~has_tmdb_data).sum()
            
            # Process movies that don't have TMDb data
            movies_to_process = df[~has_tmdb_data]
            
            if movies_to_process.empty:
                logger.info(f"Year {year}: All {len(df)} movies already have TMDb data, skipping...")
                enriched_dataframes[year] = df
                continue
            
            logger.info(
                f"Year {year}: {num_with_tmdb} movies already have TMDb data, "
                f"processing {num_without_tmdb} movies without TMDb data"
            )
            
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

def _process_single_movie_plot(
    idx: int,
    row: pd.Series,
    wikipedia_link: str,
    year: int,
    verbose: bool = True
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Process a single movie to retrieve its plot from Wikipedia.
    This function is designed to be called in parallel.
    Delegates to wikipedia_handler.fetch_plot_from_url() for the actual work.
    
    Args:
        idx: DataFrame index of the movie
        row: DataFrame row with movie data
        wikipedia_link: Wikipedia URL for the movie
        year: Year for logging purposes
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (index, plot_text, error_message)
        - index: DataFrame index
        - plot_text: Retrieved plot text (or None if failed)
        - error_message: Error message if failed (or None if success)
    """
    if not wikipedia_link or pd.isna(wikipedia_link):
        return (idx, None, None)
    
    # Use the handler function - this encapsulates all Wikipedia API logic
    plot_text, error_msg = fetch_plot_from_url(str(wikipedia_link))
    
    # Log results if verbose
    if plot_text and verbose:
        logger.debug(
            f"Year {year}, {row.get('title', 'Unknown')}: "
            f"Retrieved plot ({len(plot_text)} chars)"
        )
    elif error_msg and verbose:
        if error_msg == "No plot or summary found":
            logger.debug(f"Year {year}, {row.get('title', 'Unknown')}: {error_msg}")
        else:
            logger.debug(f"Year {year}, {row.get('title', 'Unknown')}: {error_msg}")
    
    if error_msg and error_msg != "No plot or summary found":
        # Log warnings for actual errors (but not for "no plot found" which is expected)
        logger.warning(f"Year {year}, Movie {row.get('title', 'Unknown')}: Error - {error_msg}")
    
    return (idx, plot_text, error_msg)


def step3_wikipedia(
    year_dataframes: Dict[int, pd.DataFrame],
    verbose: bool = True,
    max_workers: int = 8
) -> Dict[int, pd.DataFrame]:
    """
    Step 3: Retrieve movie plots from Wikipedia via sitelinks.
    
    This function processes movies in parallel using ThreadPoolExecutor
    for faster Wikipedia API requests.
    
    Args:
        year_dataframes: Dictionary mapping year to DataFrame
        verbose: Enable verbose logging
        max_workers: Number of parallel threads (default: 4)
        
    Returns:
        Dictionary mapping year to DataFrame with plots
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Wikipedia Plot Retrieval")
    logger.info("=" * 80)
    logger.info(f"Using {max_workers} parallel threads for Wikipedia requests")
    
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
        # Ensure we filter out empty strings and non-string values
        has_plot = df['plot'].notna() & (df['plot'] != '') & (df['plot'].astype(str).str.strip() != '')
        
        # Count movies with and without plots
        num_with_plot = has_plot.sum()
        num_without_plot = (~has_plot).sum()
        
        # Process movies that don't have plots
        movies_to_process = df[~has_plot]
        
        if movies_to_process.empty:
            logger.info(f"Year {year}: All {len(df)} movies already have plots, skipping...")
            enriched_dataframes[year] = df
            continue
        
        logger.info(
            f"Year {year}: {num_with_plot} movies already have plots, "
            f"processing {num_without_plot} movies without plots"
        )
        
        # Prepare tasks for parallel processing
        tasks = []
        for idx, row in movies_to_process.iterrows():
            wikipedia_link = row.get('wikipedia_link', '')
            tasks.append((idx, row, wikipedia_link))
        
        # Process movies in parallel using ThreadPoolExecutor
        results = {}
        completed_count = 0
        total_count = len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    _process_single_movie_plot,
                    idx,
                    row,
                    wikipedia_link,
                    year,
                    verbose
                ): (idx, row.get('title', 'Unknown'))
                for idx, row, wikipedia_link in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                idx, title = future_to_task[future]
                completed_count += 1
                
                try:
                    result_idx, plot_text, error_msg = future.result()
                    results[result_idx] = plot_text
                    
                    # Log progress periodically
                    if completed_count % 10 == 0 or completed_count == total_count:
                        logger.info(
                            f"Year {year}: Progress: {completed_count}/{total_count} movies processed "
                            f"({completed_count/total_count*100:.1f}%)"
                        )
                        
                except Exception as e:
                    logger.error(f"Year {year}, Movie {title}: Unexpected error in thread - {e}")
                    results[idx] = None
        
        # Update DataFrame with results (thread-safe: all updates happen sequentially)
        plots_retrieved = 0
        for idx, plot_text in results.items():
            if plot_text:
                df.at[idx, 'plot'] = plot_text
                plots_retrieved += 1
        
        logger.info(
            f"Year {year}: Retrieved plots for {plots_retrieved}/{num_without_plot} movies "
            f"({plots_retrieved/num_without_plot*100:.1f}% success rate)"
        )
        
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
    verbose: bool = True,
    force_refresh: bool = False,
    chunking_method: Optional[str] = None,
    chunking_chunk_size: int = 512,
    chunking_window_size: int = 512,
    chunking_stride: int = 256,
    save_lexical_weights: bool = False
) -> None:
    """
    Step 4: Generate embeddings for movie plots and save per year.
    Uses the factory functions for GPU setup and verification.
    
    This function properly checks if embeddings already exist by:
    1. Checking if the embeddings file exists
    2. Loading the file and verifying it matches the current number of plots
    3. Verifying that movie_ids match (not just counts)
    4. Only skipping regeneration if embeddings are complete and match current data
    5. Regenerating if counts don't match, movie_ids have changed, or if force_refresh=True
    
    Embeddings are saved with movie_id indexing:
    - Saves embeddings array to 'movie_embeddings_{year}[_chunking_suffix].npy'
    - Saves corresponding movie_ids array to 'movie_ids_{year}[_chunking_suffix].npy'
    - embedding[i] corresponds to movie_ids[i]
    - This ensures embeddings can be matched back to movies by movie_id
    
    Args:
        year_dataframes: Dictionary mapping year to DataFrame
        model_name: Name of the embedding model
        target_devices: List of CUDA devices (e.g., ['cuda:0'])
        batch_size: Batch size for embedding generation
        verbose: Enable verbose logging
        force_refresh: If True, regenerate embeddings even if file exists
        chunking_method: Chunking method to use. Options: None, 'cls_token', 'mean_pooling', 
                        'chunk_first_embed', 'late_chunking'. If None, uses direct embedding.
        chunking_chunk_size: Chunk size for chunk_first_embed (default: 512)
        chunking_window_size: Window size for late_chunking (default: 512)
        chunking_stride: Stride for chunk_first_embed and late_chunking (default: 256)
        save_lexical_weights: If True, save lexical weights (only available for BGE-M3 model)
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Embedding Generation")
    logger.info("=" * 80)
    
    # Initialize chunking method if specified
    chunking_instance = None
    chunking_suffix = ""
    if chunking_method and chunking_method.lower() not in ['none', '']:
        if not CHUNKING_AVAILABLE:
            logger.error("Chunking requested but chunking classes are not available. Skipping chunking.")
            chunking_method = None
        else:
            chunking_method_lower = chunking_method.lower()
            try:
                if chunking_method_lower == 'cls_token':
                    chunking_instance = CLSToken(embedding_service=None, model_name=model_name)
                    chunking_suffix = "_cls_token"
                    logger.info("Using chunking method: CLSToken")
                elif chunking_method_lower == 'mean_pooling':
                    chunking_instance = MeanPooling(embedding_service=None, model_name=model_name)
                    chunking_suffix = "_mean_pooling"
                    logger.info("Using chunking method: MeanPooling")
                elif chunking_method_lower == 'chunk_first_embed':
                    chunking_instance = ChunkFirstEmbed(
                        embedding_service=None,
                        model_name=model_name,
                        chunk_size=chunking_chunk_size,
                        stride=chunking_stride
                    )
                    chunking_suffix = f"_chunk_first_{chunking_chunk_size}_{chunking_stride}"
                    logger.info(f"Using chunking method: ChunkFirstEmbed (chunk_size={chunking_chunk_size}, stride={chunking_stride})")
                elif chunking_method_lower == 'late_chunking':
                    chunking_instance = LateChunking(
                        embedding_service=None,
                        model_name=model_name,
                        window_size=chunking_window_size,
                        stride=chunking_stride
                    )
                    chunking_suffix = f"_late_chunking_{chunking_window_size}_{chunking_stride}"
                    logger.info(f"Using chunking method: LateChunking (window_size={chunking_window_size}, stride={chunking_stride})")
                else:
                    logger.warning(f"Unknown chunking method: {chunking_method}. Available options: 'cls_token', 'mean_pooling', 'chunk_first_embed', 'late_chunking'. Using direct embedding.")
                    chunking_method = None
            except Exception as e:
                logger.error(f"Error initializing chunking method {chunking_method}: {e}. Using direct embedding.")
                chunking_method = None
                chunking_instance = None
    
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
        
        # Verify GPU setup using utility function (only if not using CPU)
        if target_devices != ['cpu']:
            try:
                # verify_gpu_setup returns adjusted devices
                adjusted_devices = verify_gpu_setup(target_devices)
                if adjusted_devices != target_devices:
                    target_devices = adjusted_devices
                    logger.info(f"GPU setup adjusted devices to: {target_devices}")
            except SystemExit:
                # verify_gpu_setup calls sys.exit(1) if CUDA is required but not available
                # In pipeline, we'll fall back to CPU instead
                logger.warning("GPU verification failed, falling back to CPU")
                target_devices = ['cpu']
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        target_devices = ['cpu']
    except Exception as e:
        logger.warning(f"Error during GPU setup verification: {e}. Continuing with available devices...")
    
    # Initialize embedding service
    global _embedding_service_instance
    try:
        embedding_service = EmbeddingService(model_name=model_name)
        _embedding_service_instance = embedding_service  # Store for cleanup
        logger.info(f"Initialized embedding service with model: {model_name}")
        
        # If using chunking, share the embedding service with the chunking instance
        if chunking_instance is not None:
            chunking_instance.embedding_service = embedding_service
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        logger.warning("Skipping Step 4: Embedding generation")
        return
    
    for year, df in year_dataframes.items():
        if df.empty:
            logger.info(f"Year {year}: No movies to embed")
            continue
        
        # Extract plots first to check count
        if 'plot' not in df.columns:
            logger.warning(f"Year {year}: No 'plot' column found, skipping embeddings")
            continue
        
        # Ensure movie_id column exists
        if 'movie_id' not in df.columns:
            logger.warning(f"Year {year}: No 'movie_id' column found, skipping embeddings")
            continue
        
        # Get indices of movies with plots and extract plots
        has_plot_mask = df['plot'].notna() & (df['plot'].astype(str).str.strip() != '') & (df['plot'].astype(str) != 'nan')
        movies_with_plots = df[has_plot_mask].copy()
        plots = movies_with_plots['plot'].astype(str).tolist()
        movie_ids_with_plots = movies_with_plots['movie_id'].tolist()
        
        if not plots:
            logger.warning(f"Year {year}: No plots available for embedding")
            continue
        
        # Check if embeddings already exist and verify they match current data
        embeddings_path = os.path.join(DATA_DIR, f'movie_embeddings_{year}{chunking_suffix}.npy')
        movie_ids_path = os.path.join(DATA_DIR, f'movie_ids_{year}{chunking_suffix}.npy')
        lexical_weights_path = os.path.join(DATA_DIR, f'movie_lexical_weights_{year}{chunking_suffix}.npz') if save_lexical_weights else None
        embeddings_exist = os.path.exists(embeddings_path) and os.path.exists(movie_ids_path)
        
        if embeddings_exist and not force_refresh:
            try:
                existing_embeddings = np.load(embeddings_path)
                existing_movie_ids = np.load(movie_ids_path)
                
                current_plot_count = len(plots)
                existing_embedding_count = existing_embeddings.shape[0] if existing_embeddings.ndim > 0 else 0
                
                # Check if counts match and movie_ids match
                if current_plot_count == existing_embedding_count:
                    if len(existing_movie_ids) == len(movie_ids_with_plots):
                        if np.array_equal(existing_movie_ids, np.array(movie_ids_with_plots)):
                            logger.info(f"Year {year}: Embeddings file exists and matches current data ({current_plot_count} plots), skipping...")
                            continue
                        else:
                            logger.warning(
                                f"Year {year}: Embeddings file exists but movie_ids have changed. "
                                f"Regenerating embeddings..."
                            )
                    else:
                        logger.warning(
                            f"Year {year}: Embeddings file exists but movie_id count mismatch. "
                            f"Regenerating embeddings..."
                        )
                else:
                    logger.warning(
                        f"Year {year}: Embeddings file exists but count mismatch "
                        f"(CSV has {current_plot_count} plots, embeddings have {existing_embedding_count}). "
                        f"Regenerating embeddings..."
                    )
            except Exception as e:
                logger.warning(f"Year {year}: Error loading existing embeddings file: {e}. Regenerating...")
        elif embeddings_exist and force_refresh:
            logger.info(f"Year {year}: force_refresh=True, regenerating embeddings even though file exists")
        
        logger.info(f"Year {year}: Generating embeddings for {len(plots)} plots...")
        
        try:
            import time
            start_time = time.time()
            
            # Generate embeddings using chunking if specified, otherwise use direct embedding
            lexical_weights_list = None
            if chunking_instance is not None:
                # Use chunking method
                logger.info(f"Year {year}: Using chunking method {chunking_method}...")
                embeddings = chunking_instance.embed_batch(plots, batch_size=batch_size)
                
                # If lexical weights are requested, extract them separately
                # Note: chunking methods don't return lexical weights, so we need a separate call
                if save_lexical_weights:
                    logger.info(f"Year {year}: Extracting lexical weights separately...")
                    # Extract lexical weights using encode_corpus
                    # Process in batches to manage memory
                    lexical_weights_list = []
                    for i in range(0, len(plots), batch_size):
                        batch_plots = plots[i:i+batch_size]
                        batch_results = embedding_service.encode_corpus(batch_plots, batch_size=len(batch_plots))
                        if 'lexical_weights' in batch_results:
                            # lexical_weights is a list where each element corresponds to one plot
                            batch_lexical_weights = batch_results['lexical_weights']
                            if isinstance(batch_lexical_weights, list):
                                lexical_weights_list.extend(batch_lexical_weights)
                            else:
                                # If it's not a list, try to convert
                                logger.warning(f"Year {year}: Unexpected lexical_weights format: {type(batch_lexical_weights)}")
                    if not lexical_weights_list:
                        logger.warning(f"Year {year}: No lexical weights found in results. They may not be available for this model or chunking method.")
            else:
                # Generate embeddings using factory approach (direct embedding)
                if save_lexical_weights:
                    # Use encode_corpus to get both embeddings and lexical weights
                    logger.info(f"Year {year}: Encoding with lexical weights extraction...")
                    if 'cpu' in target_devices or (len(target_devices) == 1 and target_devices[0] == 'cpu'):
                        # For CPU, we need to use encode_corpus
                        results = embedding_service.encode_corpus(plots, batch_size=batch_size)
                        # Extract dense embeddings
                        if 'dense_vecs' in results:
                            embeddings = results['dense_vecs']
                        elif 'dense' in results:
                            embeddings = results['dense']
                        else:
                            # Fallback to model.encode if dense not found
                            logger.warning(f"Year {year}: Dense embeddings not found in results, using model.encode...")
                            embeddings = embedding_service.model.encode(
                                plots,
                                batch_size=batch_size,
                                show_progress_bar=verbose
                            )
                        # Extract lexical weights
                        if 'lexical_weights' in results:
                            lexical_weights_list = results['lexical_weights']
                            if not isinstance(lexical_weights_list, list):
                                logger.warning(f"Year {year}: lexical_weights is not a list: {type(lexical_weights_list)}")
                                lexical_weights_list = None
                    else:
                        # For multi-GPU, encode_parallel doesn't return lexical weights
                        # So we need to use encode_corpus for lexical weights
                        logger.info(f"Year {year}: Encoding on {len(target_devices)} device(s): {target_devices}")
                        # First get embeddings
                        embeddings = embedding_service.encode_parallel(
                            corpus=plots,
                            target_devices=target_devices,
                            batch_size=batch_size
                        )
                        # Then extract lexical weights separately
                        logger.info(f"Year {year}: Extracting lexical weights separately...")
                        lexical_weights_list = []
                        for i in range(0, len(plots), batch_size):
                            batch_plots = plots[i:i+batch_size]
                            batch_results = embedding_service.encode_corpus(batch_plots, batch_size=len(batch_plots))
                            if 'lexical_weights' in batch_results:
                                batch_lexical_weights = batch_results['lexical_weights']
                                if isinstance(batch_lexical_weights, list):
                                    lexical_weights_list.extend(batch_lexical_weights)
                                else:
                                    logger.warning(f"Year {year}: Unexpected lexical_weights format: {type(batch_lexical_weights)}")
                        if not lexical_weights_list:
                            logger.warning(f"Year {year}: No lexical weights found in results.")
                else:
                    # Standard encoding without lexical weights
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
            
            # Verify embeddings using utility function
            if verify_embeddings(embeddings, plots):
                # Save embeddings with corresponding movie_ids
                # Embeddings[i] corresponds to movie_ids_with_plots[i]
                np.save(embeddings_path, embeddings)
                np.save(movie_ids_path, np.array(movie_ids_with_plots))
                logger.info(f"Year {year}: Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")
                logger.info(f"Year {year}: Saved movie_ids to {movie_ids_path} ({len(movie_ids_with_plots)} movie_ids)")
                logger.info(f"Year {year}: Encoding completed in {duration:.2f} seconds ({len(plots)/duration:.2f} docs/sec)")
                logger.info(f"Year {year}: Embeddings are indexed by movie_id - embedding[i] corresponds to movie_ids[i] (e.g., first embedding for movie_id: {movie_ids_with_plots[0] if movie_ids_with_plots else 'N/A'})")
                
                # Save lexical weights if available
                if save_lexical_weights and lexical_weights_list is not None and len(lexical_weights_list) == len(movie_ids_with_plots):
                    try:
                        # Convert lexical weights to sparse format (only non-zero weights)
                        # Save as two arrays: token_indices and weights
                        token_indices_list = []
                        weights_list = []
                        
                        for doc_idx, lw in enumerate(lexical_weights_list):
                            token_indices = []
                            weights = []
                            
                            if isinstance(lw, dict):
                                # Dict format: {token_id_str: weight, ...}
                                for token_id_str, weight in lw.items():
                                    weight_float = float(weight)
                                    if abs(weight_float) > 1e-10:  # Only save non-zero weights
                                        try:
                                            token_id_int = int(token_id_str)
                                            token_indices.append(token_id_int)
                                            weights.append(weight_float)
                                        except (ValueError, TypeError):
                                            logger.warning(f"Year {year}: Document {doc_idx}: Invalid token_id '{token_id_str}', skipping")
                            elif isinstance(lw, np.ndarray):
                                # Array format: indices are token IDs
                                non_zero_mask = np.abs(lw) > 1e-10
                                if non_zero_mask.any():
                                    token_indices_arr = np.where(non_zero_mask)[0]
                                    weights_arr = lw[non_zero_mask]
                                    token_indices.extend(token_indices_arr.tolist())
                                    weights.extend(weights_arr.tolist())
                            elif isinstance(lw, (list, tuple)):
                                # List format: indices are token IDs
                                lw_array = np.array(lw, dtype=np.float32)
                                non_zero_mask = np.abs(lw_array) > 1e-10
                                if non_zero_mask.any():
                                    token_indices_arr = np.where(non_zero_mask)[0]
                                    weights_arr = lw_array[non_zero_mask]
                                    token_indices.extend(token_indices_arr.tolist())
                                    weights.extend(weights_arr.tolist())
                            else:
                                logger.warning(f"Year {year}: Document {doc_idx}: Unexpected lexical weight type: {type(lw)}")
                            
                            # Convert to numpy arrays
                            token_indices_list.append(np.array(token_indices, dtype=np.int64) if token_indices else np.array([], dtype=np.int64))
                            weights_list.append(np.array(weights, dtype=np.float32) if weights else np.array([], dtype=np.float32))
                        
                        # Save using npz format
                        # Use object arrays to store variable-length arrays per document
                        np.savez(lexical_weights_path,
                                token_indices=np.array(token_indices_list, dtype=object),
                                weights=np.array(weights_list, dtype=object),
                                movie_ids=np.array(movie_ids_with_plots))
                        
                        # Log statistics
                        total_non_zero = sum(len(ti) for ti in token_indices_list)
                        avg_non_zero = total_non_zero / len(token_indices_list) if token_indices_list else 0
                        logger.info(f"Year {year}: Saved lexical weights to {lexical_weights_path}")
                        logger.info(f"Year {year}: Saved {len(lexical_weights_list)} lexical weight entries")
                        logger.info(f"Year {year}: Total {total_non_zero} non-zero weights (avg {avg_non_zero:.1f} per document)")
                        logger.info(f"Year {year}: Lexical weights are indexed by movie_id - token_indices[i] and weights[i] correspond to movie_ids[i]")
                    except Exception as e:
                        logger.error(f"Year {year}: Error saving lexical weights: {e}")
                elif save_lexical_weights:
                    if lexical_weights_list is None:
                        logger.warning(f"Year {year}: Lexical weights requested but not available (model may not support them)")
                    elif len(lexical_weights_list) != len(movie_ids_with_plots):
                        logger.warning(f"Year {year}: Lexical weights count mismatch: {len(lexical_weights_list)} vs {len(movie_ids_with_plots)}")
            else:
                logger.error(f"Year {year}: Embedding verification failed, not saving")
            
            # Clear intermediate variables to free memory
            del embeddings
            del plots
            del movie_ids_with_plots
            del movies_with_plots
            if lexical_weights_list is not None:
                del lexical_weights_list
            
        except Exception as e:
            logger.error(f"Year {year}: Error generating embeddings - {e}", exc_info=verbose)
            # Clear variables even on error
            if 'embeddings' in locals():
                del embeddings
            if 'plots' in locals():
                del plots
            if 'movie_ids_with_plots' in locals():
                del movie_ids_with_plots
            if 'movies_with_plots' in locals():
                del movies_with_plots
            continue
        
        # Aggressive memory clearing between years to prevent accumulation
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                # Log memory before clearing
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
                if verbose:
                    logger.info(f"Year {year}: GPU memory before clearing - allocated: {allocated_before:.2f} GB, reserved: {reserved_before:.2f} GB")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Clear inter-process cache (important for multi-GPU)
                if len(target_devices) > 1 and all(d.startswith('cuda') for d in target_devices):
                    torch.cuda.ipc_collect()
                
                # Force garbage collection to free Python objects
                gc.collect()
                
                # Clear cache again after GC
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Log memory after clearing (always show clearing message, detailed stats if verbose)
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Year {year}: GPU memory cleared - allocated: {allocated_after:.2f} GB, reserved: {reserved_after:.2f} GB (freed {allocated_before - allocated_after:.2f} GB)")
        except Exception as e:
            logger.warning(f"Year {year}: Error clearing GPU memory: {e}")
    
    # Clean up embedding service resources
    try:
        if chunking_instance is not None:
            chunking_instance.cleanup()
        embedding_service.cleanup()
        logger.info("Cleaned up embedding service resources")
    except Exception as e:
        logger.warning(f"Error during embedding service cleanup: {e}")
    
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
    batch_size: int = 128,
    wikipedia_max_workers: int = 8,
    chunking_method: Optional[str] = None,
    chunking_chunk_size: int = 512,
    chunking_window_size: int = 512,
    chunking_stride: int = 256,
    save_lexical_weights: bool = False
) -> None:
    """
    Run the complete data pipeline.
    
    The pipeline is designed to be flexible and incremental:
    - Step 1 (Wikidata): Only fetches years where CSV files don't exist (unless force_refresh=True)
    - Step 2 (MovieDB): Only processes movies that don't already have TMDb data (checks 'popularity' column)
    - Step 3 (Wikipedia): Only processes movies that don't already have plots
    - Step 4 (Embeddings): Only processes years where embedding files don't exist
    
    Example: If you delete CSV files for 1950 and 1951, only those years will be
    re-fetched from Wikidata and enriched with MovieDB data. Years 1952+ that already
    have MovieDB data will be skipped.
    
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
        wikipedia_max_workers: Number of parallel threads for Wikipedia requests (default: 4)
        chunking_method: Chunking method to use. Options: None, 'cls_token', 'mean_pooling', 
                        'chunk_first_embed', 'late_chunking'. If None, uses direct embedding.
        chunking_chunk_size: Chunk size for chunk_first_embed (default: 512)
        chunking_window_size: Window size for late_chunking (default: 512)
        chunking_stride: Stride for chunk_first_embed and late_chunking (default: 256)
        save_lexical_weights: If True, save lexical weights (only available for BGE-M3 model)
    """
    logger.info("=" * 80)
    logger.info("MOVIE DATA PIPELINE - Starting")
    logger.info("=" * 80)
    logger.info(f"Years: {start_year} to {end_year}")
    logger.info(f"Movies per year: {movies_per_year}")
    logger.info("")
    
    # Pre-scan: Check which years have existing CSV files
    existing_years = []
    missing_years = []
    for year in range(start_year, end_year + 1):
        csv_path = get_csv_path(year)
        if os.path.exists(csv_path):
            existing_years.append(year)
        else:
            missing_years.append(year)
    
    if existing_years:
        logger.info(f"Found existing CSV files for {len(existing_years)} years: {existing_years[:5]}{'...' if len(existing_years) > 5 else ''}")
    if missing_years:
        logger.info(f"Missing CSV files for {len(missing_years)} years (will be fetched): {missing_years}")
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
            verbose=verbose,
            max_workers=wikipedia_max_workers
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
            verbose=verbose,
            force_refresh=force_refresh,
            chunking_method=chunking_method,
            chunking_chunk_size=chunking_chunk_size,
            chunking_window_size=chunking_window_size,
            chunking_stride=chunking_stride,
            save_lexical_weights=save_lexical_weights
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
        batch_size=BATCH_SIZE,
        chunking_method=CHUNKING_METHOD,
        chunking_chunk_size=CHUNKING_CHUNK_SIZE,
        chunking_window_size=CHUNKING_WINDOW_SIZE,
        chunking_stride=CHUNKING_STRIDE,
        save_lexical_weights=SAVE_LEXICAL_WEIGHTS
    )


if __name__ == "__main__":
    asyncio.run(main())

    # nohup python data_pipeline.py > data_pipeline.log 2>&1 &


