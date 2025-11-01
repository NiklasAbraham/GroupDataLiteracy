# -*- coding: utf-8 -*-
"""
run_embedding_factory.py

This is the main execution script ("factory") for running the
movie plot embedding pipeline.

It performs the following steps:
1. Configures the model, target devices, and file paths.
2. Verifies GPU availability.
3. Loads the corpus of movie plots from a CSV file.
4. Initializes the EmbeddingService.
5. Runs the parallel encoding process.
6. Saves the resulting embeddings to a NumPy file.
"""

import time
import logging
import sys
import os
import csv
import numpy as np
import torch

# Handle imports - try relative import first (when used as module), then absolute (when run directly)
try:
    from .embedding import EmbeddingService
except ImportError:
    # If relative import fails, try absolute import (when run as script)
    try:
        from embedding.embedding import EmbeddingService
    except ImportError:
        # Add parent directory to path if needed
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from embedding.embedding import EmbeddingService

# --- Configuration ---
# Specify the model you wish to use. 'BAAI/bge-m3' is recommended
# for long documents (movie plots) due to its 8192 token context.
MODEL_NAME = 'BAAI/bge-m3'

# Specify the CUDA devices
TARGET_DEVICES = ['cuda:0']

# Tune this based on VRAM and model size. For 24GB cards and bge-m3,
# 128 or 256 is a reasonable starting point.
BATCH_SIZE = 128

# Input and Output files
CORPUS_FILE = '../../data/mock/mock_movies_100.csv'  # Path to CSV input file
CSV_COLUMN = 'synopsis'  # Column name in CSV to extract
OUTPUT_FILE = 'movie_embeddings.npy'
# ---------------------

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s] %(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def load_corpus_from_csv(filepath: str, column_name: str) -> list[str]:
    """
    Loads a corpus from a CSV file by extracting a specific column.
    
    Args:
        filepath (str): Path to the CSV file.
        column_name (str): Name of the column to extract.
    
    Returns:
        list[str]: A list of documents from the specified column.
    """
    logger.info(f"Loading corpus from CSV file: {filepath}")
    logger.info(f"Extracting column: {column_name}")
    
    corpus = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Verify column exists
            if column_name not in reader.fieldnames:
                logger.error(f"Column '{column_name}' not found in CSV. Available columns: {reader.fieldnames}")
                sys.exit(1)
            
            # Extract data from the specified column
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                value = row.get(column_name, '').strip()
                if value:
                    corpus.append(value)
                else:
                    logger.warning(f"Row {row_num}: Empty value in column '{column_name}', skipping.")
        
        if not corpus:
            logger.warning(f"CSV file {filepath} contains no valid data in column '{column_name}'.")
            return []
        
        logger.info(f"Successfully loaded {len(corpus)} documents from CSV.")
        return corpus
    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred loading CSV corpus: {e}")
        sys.exit(1)


def load_corpus(filepath: str, column_name: str) -> list[str]:
    """
    Loads a corpus from a CSV file by extracting a specific column.
    
    Args:
        filepath (str): Path to the CSV input file.
        column_name (str): Column name to extract from CSV.
    
    Returns:
        list[str]: A list of documents.
    """
    # Convert relative path to absolute if needed
    if not os.path.isabs(filepath):
        # Assume relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filepath)
        # Normalize path (resolve .. and .)
        filepath = os.path.normpath(filepath)
    
    return load_corpus_from_csv(filepath, column_name)


def verify_gpu_setup():
    """
    Verifies that CUDA is available and the number of specified
    devices matches what torch can detect.
    """
    global TARGET_DEVICES
    
    logger.info("Verifying GPU setup...")
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU setup.")
        sys.exit(1)
    
    available_gpus = torch.cuda.device_count()
    logger.info(f"Found {available_gpus} available CUDA devices.")
    
    if available_gpus < len(TARGET_DEVICES):
        logger.warning(f"Warning: Configured for {len(TARGET_DEVICES)} devices, "
                       f"but only {available_gpus} are available.")
        # We can proceed, but only with the available devices
        TARGET_DEVICES = TARGET_DEVICES[:available_gpus]
        if not TARGET_DEVICES:
            logger.error("No target devices are available to run on.")
            sys.exit(1)
        logger.info(f"Adjusting to use {len(TARGET_DEVICES)} devices: {TARGET_DEVICES}")
    else:
        logger.info("GPU configuration matches available devices.")


def verify_embeddings(embeddings: np.ndarray, corpus: list[str]) -> bool:
    """
    Verifies that embeddings were created correctly.
    
    Args:
        embeddings (np.ndarray): The embedding matrix.
        corpus (list[str]): The original corpus.
    
    Returns:
        bool: True if embeddings are valid, False otherwise.
    """
    logger.info("--- Verifying Embeddings ---")
    
    if embeddings.size == 0:
        logger.error("Embeddings array is empty.")
        return False
    
    if len(corpus) != embeddings.shape[0]:
        logger.error(f"Mismatch: {len(corpus)} documents but {embeddings.shape[0]} embeddings.")
        return False
    
    if embeddings.shape[1] == 0:
        logger.error("Embedding dimension is 0.")
        return False
    
    # Check for NaN or Inf values
    if np.isnan(embeddings).any():
        logger.error("Embeddings contain NaN values.")
        return False
    
    if np.isinf(embeddings).any():
        logger.error("Embeddings contain Inf values.")
        return False
    
    logger.info(f"✓ Embeddings verified: shape {embeddings.shape}, dtype {embeddings.dtype}")
    logger.info(f"✓ Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    logger.info(f"✓ Embedding mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    
    return True


def main():
    """
    Main execution function.
    """
    logger.info("--- Starting Embedding Factory ---")
    logger.info(f"Input file: {CORPUS_FILE}")
    logger.info(f"CSV column: {CSV_COLUMN}")
    
    # 1. Verify hardware
    verify_gpu_setup()
    
    # 2. Load data from CSV
    corpus = load_corpus(CORPUS_FILE, CSV_COLUMN)
    
    if not corpus:
        logger.info("No documents to process. Exiting.")
        return

    # 3. Initialize the embedding service
    # This will automatically select the appropriate strategy based on the model name
    # and load the model into memory
    service = EmbeddingService(model_name=MODEL_NAME, target_devices=TARGET_DEVICES)

    # 4. Run the parallel encoding process
    logger.info("--- Starting Parallel Encoding ---")
    global_start_time = time.time()
    
    embeddings = service.encode_parallel(
        corpus=corpus,
        target_devices=TARGET_DEVICES,
        batch_size=BATCH_SIZE
    )
    
    global_end_time = time.time()
    logger.info("--- Parallel Encoding Finished ---")

    # 5. Verify embeddings
    if not verify_embeddings(embeddings, corpus):
        logger.error("Embedding verification failed. Exiting.")
        return

    # 6. Save the results
    if embeddings.size > 0:
        logger.info(f"Saving embedding matrix to {OUTPUT_FILE}...")
        try:
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..', 'data', OUTPUT_FILE), embeddings)
            logger.info("Embeddings saved successfully.")
        except IOError as e:
            logger.error(f"Failed to save embeddings to disk: {e}")
            return
    else:
        logger.warning("Embedding process returned no data. Nothing to save.")
        return

    # 7. Final Report
    logger.info("--- Final Report ---")
    logger.info(f"Model Used:         {MODEL_NAME}")
    logger.info(f"GPUs Utilized:      {len(TARGET_DEVICES)} ({', '.join(TARGET_DEVICES)})")
    logger.info(f"Documents Encoded:  {len(corpus)}")
    logger.info(f"Embedding Shape:    {embeddings.shape}")
    logger.info(f"Embedding Dim:      {embeddings.shape[1]}")
    logger.info(f"Output File:        {OUTPUT_FILE}")
    logger.info(f"Total Time (incl. pool): {global_end_time - global_start_time:.2f} seconds")
    logger.info("--- Embedding Factory Finished ---")


if __name__ == "__main__":
    # The if __name__ == "__main__" guard is CRITICAL for
    # multiprocessing on CUDA. It prevents child processes
    # from re-executing the script upon import.
    main()
