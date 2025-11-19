# -*- coding: utf-8 -*-
"""
manager.py

Orchestrates the embedding bias-variance experiment across all chunking methods.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import torch

# Add parent directories to path for imports
# manager.py is in src/analysis/chunking/
# So we go up 4 levels to get to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from data_utils import load_movie_data, cluster_genres
from analysis.chunking.chunk_base_class import ChunkBase
from analysis.chunking import calculations
from embedding.embedding import EmbeddingService
from embedding.util_embeddings import verify_gpu_setup

# Import all chunking methods
from analysis.chunking.chunk_mean_pooling import MeanPooling
from analysis.chunking.chunk_no_chunking_cls_token import CLSToken
from analysis.chunking.chunk_first_then_embed import ChunkFirstEmbed
from analysis.chunking.chunk_late_chunking import LateChunking

# ============================================================================
# Configuration Parameters
# ============================================================================
DATA_DIR = str(BASE_DIR / "data")  # Path to data directory containing movie CSVs
OUTPUT_DIR = None  # Path to output directory (None = auto-generate timestamped directory)
MODEL_NAME = "BAAI/bge-m3" #"Qwen/Qwen3-Embedding-0.6B"# "BAAI/bge-m3"  # Model name to use
N_MOVIES = 5000  # Number of movies to process
RANDOM_SEED = 42  # Random seed for reproducibility
BATCH_SIZE = 100  # Batch size for embedding processing
# ============================================================================


def main():
    """Main entry point for running the experiment."""
    print("\n" + "="*80)
    print("Starting Embedding Bias-Variance Experiment")
    print("="*80)
    
    # Set up output directory
    if OUTPUT_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_DIR / "outputs" / f"experiment_{timestamp}"
    else:
        output_dir = Path(OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Use single GPU only for Qwen3 (simple and stable)
    target_devices = ['cuda:0']
    print(f"Using single GPU: {target_devices[0]}")
    
    # Create shared EmbeddingService for all methods
    embedding_service = EmbeddingService(MODEL_NAME, target_devices)
    
    # Initialize chunking methods with shared EmbeddingService
    methods = {
        'MeanPooling': MeanPooling(embedding_service=embedding_service, 
                                   model_name=MODEL_NAME),
        'CLSToken': CLSToken(embedding_service=embedding_service,
                            model_name=MODEL_NAME),
        # ChunkFirstEmbed - processes text in chunks before embedding
        'ChunkFirstEmbed_512_256': ChunkFirstEmbed(embedding_service=embedding_service,
                                                     model_name=MODEL_NAME,
                                                     chunk_size=512,
                                                     stride=256),
        'ChunkFirstEmbed_1024_512': ChunkFirstEmbed(embedding_service=embedding_service,
                                                     model_name=MODEL_NAME,
                                                     chunk_size=1024,
                                                     stride=512),
        'ChunkFirstEmbed_2048_1024': ChunkFirstEmbed(embedding_service=embedding_service,
                                                     model_name=MODEL_NAME,
                                                     chunk_size=2048,
                                                     stride=1024),
        'LateChunking_512_256': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=512,
                                            stride=256),
        'LateChunking_1024_512': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=1024,
                                            stride=512),
        'LateChunking_2048_1024': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=2048,
                                            stride=1024),
        'LateChunking_2048_512': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=2048,
                                            stride=512),
        'LateChunking_512_0': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=512,
                                            stride=0),
        'LateChunking_1024_0': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=1024,
                                            stride=0),
        'LateChunking_2048_0': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=2048,
                                            stride=0),
    }
    
    print(f"Initialized {len(methods)} chunking methods")
    
    # Load tokenizer for counting tokens (separate from EmbeddingService)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load movie data
    print(f"\nLoading movie data from {DATA_DIR}...")
    df = load_movie_data(DATA_DIR, verbose=True)
    print(f"Loaded {len(df)} movies")
    
    # Filter to movies with plot data, and a genre
    df = df[df['plot'].notna() & (df['plot'].str.len() > 200) & (df['genre'].notna())].copy()
    print(f"Movies with plot data and genre: {len(df)}")
    
    # Apply genre clustering/filtering from data_utils
    # This uses the genre_fix_mapping.json to map and clean genres
    # Note: cluster_genres expects genre_fix_mapping.json in src directory
    original_cwd = os.getcwd()
    try:
        os.chdir(str(SRC_DIR))
        print("Processing genres using cluster_genres from data_utils...")
        df = cluster_genres(df)
    finally:
        os.chdir(original_cwd)
    
    # Filter to movies with valid processed genres (new_genre column)
    if 'new_genre' in df.columns:
        df = df[df['new_genre'].notna() & (df['new_genre'] != 'Unknown')].copy()
        print(f"Movies with processed genres: {len(df)}")
    
    # Sample movies
    np.random.seed(RANDOM_SEED)
    if len(df) > N_MOVIES:
        sampled_df = df.sample(n=N_MOVIES, random_state=RANDOM_SEED)
    else:
        sampled_df = df.copy()
        print(f"Warning: Only {len(df)} movies available, using all of them")
    
    print(f"\nProcessing {len(sampled_df)} movies...")
    
    # Extract data
    plots = sampled_df['plot'].tolist()
    movie_ids = sampled_df['movie_id'].values
    years = sampled_df['year'].values if 'year' in sampled_df.columns else None
    
    # Use new_genre column (processed genres) instead of raw genre column
    # new_genre contains genres separated by | (from cluster_genres)
    if 'new_genre' in sampled_df.columns:
        genres_raw = sampled_df['new_genre'].values
        # Convert genres to lists of genre strings (multi-label format)
        # Each movie can have multiple genres separated by |, e.g., "Action|Drama|Thriller" -> ["Action", "Drama", "Thriller"]
        genres_list = [[g.strip() for g in str(genre).split('|') if g.strip() and g.strip() != 'Unknown'] for genre in genres_raw]
        # Convert to numpy array with object dtype to preserve list structure
        genres = np.array(genres_list, dtype=object) if genres_list else None
    else:
        # Fallback to raw genre column if new_genre is not available
        genres_raw = sampled_df['genre'].values if 'genre' in sampled_df.columns else None
        if genres_raw is not None:
            # Convert genres to lists of genre strings (multi-label format)
            # Each movie can have multiple genres, e.g., "Action, Drama, Thriller" -> ["Action", "Drama", "Thriller"]
            genres_list = [[g.strip() for g in str(genre).split(',') if g.strip()] for genre in genres_raw]
            # Convert to numpy array with object dtype to preserve list structure
            genres = np.array(genres_list, dtype=object) if genres_list else None
        else:
            genres = None
    
    # Count tokens for each plot
    print("Counting tokens...")
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        if not isinstance(text, str):
            return 0
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return 0
    
    text_lengths = np.array([count_tokens(plot) for plot in plots])
    
    # Store embeddings and metrics for each method
    all_embeddings = {}
    all_metrics = {}
    
    # Helper function to check embeddings sanity
    def sanity_check(name, X):
        """Check that embeddings have proper unit norms."""
        norms = np.linalg.norm(X, axis=1)
        print(f"    Sanity check [{name}]: unit-norm (min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f} zeros={(norms<1e-8).sum()}/{len(norms)})")
        if not np.allclose(norms, 1.0, rtol=1e-3):
            print(f"    WARNING: Not all embeddings are unit-norm!")
        return norms
    
    # Run each method
    
    
    for method_name, method_instance in methods.items():
        print(f"\n{'='*80}")
        print(f"Processing method: {method_name}")
        print(f"{'='*80}")
        
        print(f"  Embedding {len(plots)} movies in batches of {BATCH_SIZE}...")
        
        # Batch processing is mandatory - all methods must implement embed_batch
        if not hasattr(method_instance, 'embed_batch'):
            raise ValueError(f"Method {method_name} does not implement embed_batch(). Batch processing is required.")
        
        try:
            import time
            
            # Enable pre-L2 norm collection
            method_instance.enable_preL2_collection(True)
            
            start_time = time.time()
            embeddings = method_instance.embed_batch(plots, batch_size=BATCH_SIZE)
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Only print if it took more than 1 second
                print(f"  ✓ Embedding completed in {elapsed_time:.2f} seconds ({len(plots)/elapsed_time:.1f} movies/sec)")
            
            # Fetch pre-L2 norms and run sanity check
            pre_norms = method_instance.fetch_preL2_norms()
            sanity_check(method_name, embeddings)
            
            # Verify embeddings shape matches texts
            assert embeddings.shape[0] == len(plots), f"Embedding count mismatch: {embeddings.shape[0]} vs {len(plots)}"
            
        except Exception as e:
            print(f"  ERROR: Batch processing failed for {method_name}: {e}")
            raise RuntimeError(f"Batch processing failed for {method_name}. No individual processing fallback allowed.") from e
        all_embeddings[method_name] = embeddings
        
        # Clear CUDA cache after each method to free memory
        import torch
        import gc
        if torch.cuda.is_available():
            # Clean up separate transformer model if it exists (for Qwen3)
            if hasattr(embedding_service.strategy, 'cleanup_transformer_model'):
                try:
                    embedding_service.strategy.cleanup_transformer_model()
                except Exception as e:
                    print(f"  Warning: Could not cleanup transformer model: {e}")
            
            # For single GPU with large models, we need to be very aggressive
            # The model stays loaded on GPU, so we need to free as much as possible
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            
            # Aggressive memory clearing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()  # Clear inter-process cache
            
            # Force garbage collection to free Python objects
            gc.collect()
            
            # Clear cache again after GC
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log memory usage
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory: {allocated_before:.2f}→{allocated_after:.2f} GB allocated, {reserved_before:.2f}→{reserved_after:.2f} GB reserved")
            
            # If memory is still high (>20GB), warn about potential issues
            if allocated_after > 20.0:
                print(f"  WARNING: GPU memory still very high ({allocated_after:.2f} GB). Next method may fail with OOM.")
        
        # Save embeddings
        embeddings_path = output_dir / f"{method_name}_embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"  Saved embeddings to {embeddings_path}")
        
        # Compute metrics
        print(f"  Computing metrics for {method_name}...")
        metrics = calculations.evaluate_method(
            embeddings=embeddings,
            text_lengths=text_lengths,
            film_ids=movie_ids,
            genres=genres,
            years=years,
            method_name=method_name,
            texts=np.array(plots),
            embedding_service=embedding_service,
            batch_size=BATCH_SIZE,
            pre_norms=pre_norms  # Pass pre-L2 norms for accurate length correlation
        )
        all_metrics[method_name] = metrics
    
    # Generate comparison plots
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")
    
    # Plot combined length-norm correlation for all methods
    calculations.plot_length_norm_correlation_combined(
        all_embeddings, text_lengths, str(output_dir / "length_norm_corr.png")
    )
    print(f"Saved combined length-norm correlation plot")
    
    calculations.plot_isotropy(all_metrics, str(output_dir / "pca_isotropy.png"))
    
    if genres is not None:
        calculations.plot_genre_silhouette(all_metrics, str(output_dir / "genre_silhouette.png"))
    
    if years is not None:
        calculations.plot_drift_stability(all_embeddings, years, 
                                        str(output_dir / "drift_stability.png"))
    
    # Save metrics CSV
    metrics_path = output_dir / "metrics.csv"
    calculations.save_metrics_csv(all_metrics, str(metrics_path))
    
    # Create summary table
    summary_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=True)
    print(f"Summary table saved to {summary_path}")
    
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("\nSummary of Results:")
    print(summary_df.to_string())
    
    # Cleanup EmbeddingService
    embedding_service.cleanup()
    
    return all_metrics, all_embeddings


if __name__ == "__main__":
    main()
