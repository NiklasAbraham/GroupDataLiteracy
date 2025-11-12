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

# Add parent directories to path for imports
# manager.py is in src/analysis/chunking/
# So we go up 4 levels to get to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from data_utils import load_movie_data
from analysis.chunking.chunk_base_class import ChunkBase
from analysis.chunking import calculations
from embedding.embedding import EmbeddingService

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
MODEL_NAME = "BAAI/bge-m3"  # Model name to use
N_MOVIES = 500  # Number of movies to process
RANDOM_SEED = 42  # Random seed for reproducibility
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
    
    # Create shared EmbeddingService for all methods
    embedding_service = EmbeddingService(MODEL_NAME, None)
    
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
        'ChunkFirstEmbed_2048_512': ChunkFirstEmbed(embedding_service=embedding_service,
                                                    model_name=MODEL_NAME,
                                                    chunk_size=2048,
                                                    stride=512),
        'ChunkFirstEmbed_4096_2048': ChunkFirstEmbed(embedding_service=embedding_service,
                                                    model_name=MODEL_NAME,
                                                    chunk_size=4096,
                                                    stride=2048),
        
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
        'LateChunking_4096_2048': LateChunking(embedding_service=embedding_service,
                                            model_name=MODEL_NAME,
                                            window_size=4096,
                                            stride=2048),
    }
    
    print(f"Initialized {len(methods)} chunking methods")
    
    # Load tokenizer for counting tokens (separate from EmbeddingService)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load movie data
    print(f"\nLoading movie data from {DATA_DIR}...")
    df = load_movie_data(DATA_DIR, verbose=True)
    print(f"Loaded {len(df)} movies")
    
    # Filter to movies with plot data
    df = df[df['plot'].notna() & (df['plot'].str.len() > 0)].copy()
    print(f"Movies with plot data: {len(df)}")
    
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
    genres = sampled_df['genre'].values if 'genre' in sampled_df.columns else None
    
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
    
    # Run each method
    BATCH_SIZE = 256  # Batch size for embedding processing
    
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
            start_time = time.time()
            embeddings = method_instance.embed_batch(plots, batch_size=BATCH_SIZE)
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Only print if it took more than 1 second
                print(f"  ✓ Embedding completed in {elapsed_time:.2f} seconds ({len(plots)/elapsed_time:.1f} movies/sec)")
        except Exception as e:
            print(f"  ERROR: Batch processing failed for {method_name}: {e}")
            raise RuntimeError(f"Batch processing failed for {method_name}. No individual processing fallback allowed.") from e
        all_embeddings[method_name] = embeddings
        
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
            batch_size=BATCH_SIZE
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
    calculations.plot_variance_boxplot(all_metrics, str(output_dir / "variance_boxplot.png"))
    
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
