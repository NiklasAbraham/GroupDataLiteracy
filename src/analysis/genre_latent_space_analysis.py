"""
Analyze genre separation and spread in the latent embedding space.

This script calculates inter and intra genre cosine distances and provides
multiple measures to assess how well genres are separated in the latent space.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.spatial.distance import cdist
from scipy import stats

from data_utils import load_final_data_with_embeddings
from analysis.math_functions.cosine_distance_util import (
    calculate_average_cosine_distance,
    calculate_average_cosine_distance_between_groups
)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def extract_primary_genre(genre_str: str) -> Optional[str]:
    """Extract primary genre from pipe-separated genre string."""
    if pd.isna(genre_str) or genre_str is None:
        return None
    genres = str(genre_str).split("|")
    if genres and genres[0].strip():
        return genres[0].strip()
    return None


def calculate_intra_genre_distances(
    embeddings: np.ndarray,
    genres: np.ndarray,
    min_samples: int = 2
) -> Dict[str, Dict]:
    """
    Calculate intra-genre cosine distances for each genre.
    
    Parameters:
    - embeddings: Array of embeddings [n_samples, embedding_dim]
    - genres: Array of genre labels [n_samples]
    - min_samples: Minimum number of samples required per genre
    
    Returns:
    - Dictionary with intra-genre statistics per genre
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # Extract primary genres
    primary_genres = np.array([extract_primary_genre(g) for g in genres])
    valid_mask = np.array([g is not None for g in primary_genres])
    valid_embeddings = normalized_embeddings[valid_mask]
    valid_genres = primary_genres[valid_mask]
    
    unique_genres = np.unique(valid_genres)
    genre_stats = {}
    
    for genre in unique_genres:
        genre_mask = valid_genres == genre
        genre_embeddings = valid_embeddings[genre_mask]
        
        if len(genre_embeddings) < min_samples:
            continue
        
        # Calculate pairwise cosine distances
        avg_distance = calculate_average_cosine_distance(genre_embeddings)
        
        # Calculate additional statistics
        if len(genre_embeddings) > 1:
            distance_matrix = cosine_distances(genre_embeddings)
            upper_triangle = np.triu_indices(len(genre_embeddings), k=1)
            distances = distance_matrix[upper_triangle]
            
            # Calculate centroid
            centroid = np.mean(genre_embeddings, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            
            # Calculate distances from centroid
            centroid_distances = cosine_distances([centroid], genre_embeddings)[0]
            
            genre_stats[genre] = {
                'n_samples': len(genre_embeddings),
                'avg_intra_distance': avg_distance,
                'std_intra_distance': np.std(distances),
                'min_intra_distance': np.min(distances),
                'max_intra_distance': np.max(distances),
                'median_intra_distance': np.median(distances),
                'avg_distance_to_centroid': np.mean(centroid_distances),
                'std_distance_to_centroid': np.std(centroid_distances),
                'centroid': centroid
            }
    
    return genre_stats


def calculate_inter_genre_distances(
    embeddings: np.ndarray,
    genres: np.ndarray,
    genre_stats: Dict[str, Dict],
    min_samples: int = 2
) -> Dict[str, float]:
    """
    Calculate inter-genre cosine distances between all genre pairs.
    
    Parameters:
    - embeddings: Array of embeddings [n_samples, embedding_dim]
    - genres: Array of genre labels [n_samples]
    - genre_stats: Dictionary from calculate_intra_genre_distances
    - min_samples: Minimum number of samples required per genre
    
    Returns:
    - Dictionary with inter-genre distances
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # Extract primary genres
    primary_genres = np.array([extract_primary_genre(g) for g in genres])
    valid_mask = np.array([g is not None for g in primary_genres])
    valid_embeddings = normalized_embeddings[valid_mask]
    valid_genres = primary_genres[valid_mask]
    
    # Get genres with sufficient samples
    valid_genre_names = [g for g in genre_stats.keys() if genre_stats[g]['n_samples'] >= min_samples]
    
    inter_genre_distances = {}
    
    # Calculate pairwise distances between genres
    for genre1, genre2 in combinations(valid_genre_names, 2):
        genre1_mask = valid_genres == genre1
        genre2_mask = valid_genres == genre2
        
        genre1_embeddings = valid_embeddings[genre1_mask]
        genre2_embeddings = valid_embeddings[genre2_mask]
        
        # Calculate average distance between groups
        avg_distance = calculate_average_cosine_distance_between_groups(
            genre1_embeddings, genre2_embeddings
        )
        
        # Calculate centroid distances
        centroid1 = genre_stats[genre1]['centroid']
        centroid2 = genre_stats[genre2]['centroid']
        centroid_distance = cosine_distances([centroid1], [centroid2])[0, 0]
        
        inter_genre_distances[f"{genre1}__{genre2}"] = {
            'avg_distance': avg_distance,
            'centroid_distance': centroid_distance,
            'genre1': genre1,
            'genre2': genre2
        }
    
    return inter_genre_distances


def calculate_separation_metrics(
    embeddings: np.ndarray,
    genres: np.ndarray,
    min_samples: int = 2
) -> Dict[str, float]:
    """
    Calculate separation metrics using sklearn clustering metrics.
    
    Parameters:
    - embeddings: Array of embeddings [n_samples, embedding_dim]
    - genres: Array of genre labels [n_samples]
    - min_samples: Minimum number of samples required per genre
    
    Returns:
    - Dictionary with separation metrics
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # Extract primary genres
    primary_genres = np.array([extract_primary_genre(g) for g in genres])
    valid_mask = np.array([g is not None for g in primary_genres])
    valid_embeddings = normalized_embeddings[valid_mask]
    valid_genres = primary_genres[valid_mask]
    
    # Filter genres with sufficient samples
    genre_counts = Counter(valid_genres)
    valid_genre_names = [g for g, count in genre_counts.items() if count >= min_samples]
    
    if len(valid_genre_names) < 2:
        return {
            'silhouette_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'davies_bouldin_score': np.nan,
            'n_genres': len(valid_genre_names),
            'n_samples': len(valid_embeddings)
        }
    
    # Create label mapping
    genre_to_label = {genre: idx for idx, genre in enumerate(valid_genre_names)}
    labels = np.array([genre_to_label.get(g, -1) for g in valid_genres])
    
    # Filter to only valid labels
    valid_label_mask = labels >= 0
    filtered_embeddings = valid_embeddings[valid_label_mask]
    filtered_labels = labels[valid_label_mask]
    
    if len(np.unique(filtered_labels)) < 2:
        return {
            'silhouette_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'davies_bouldin_score': np.nan,
            'n_genres': len(np.unique(filtered_labels)),
            'n_samples': len(filtered_embeddings)
        }
    
    # Calculate metrics
    # Note: silhouette_score can be slow for large datasets, so we use cosine distance
    try:
        silhouette = silhouette_score(
            filtered_embeddings,
            filtered_labels,
            metric='cosine'
        )
    except Exception as e:
        print(f"Warning: Could not calculate silhouette score: {e}")
        silhouette = np.nan
    
    try:
        calinski_harabasz = calinski_harabasz_score(
            filtered_embeddings,
            filtered_labels
        )
    except Exception as e:
        print(f"Warning: Could not calculate Calinski-Harabasz score: {e}")
        calinski_harabasz = np.nan
    
    try:
        davies_bouldin = davies_bouldin_score(
            filtered_embeddings,
            filtered_labels
        )
    except Exception as e:
        print(f"Warning: Could not calculate Davies-Bouldin score: {e}")
        davies_bouldin = np.nan
    
    return {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski_harabasz,
        'davies_bouldin_score': davies_bouldin,
        'n_genres': len(valid_genre_names),
        'n_samples': len(filtered_embeddings)
    }


def calculate_overall_metrics(
    intra_genre_stats: Dict[str, Dict],
    inter_genre_distances: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Calculate overall metrics from intra and inter genre distances.
    
    Parameters:
    - intra_genre_stats: Dictionary from calculate_intra_genre_distances
    - inter_genre_distances: Dictionary from calculate_inter_genre_distances
    
    Returns:
    - Dictionary with overall metrics
    """
    if not intra_genre_stats or not inter_genre_distances:
        return {
            'overall_intra_distance': np.nan,
            'overall_inter_distance': np.nan,
            'separation_ratio': np.nan,
            'separation_gap': np.nan
        }
    
    # Overall intra-genre distance (weighted by sample size)
    total_samples = sum(stats['n_samples'] for stats in intra_genre_stats.values())
    weighted_intra = sum(
        stats['avg_intra_distance'] * stats['n_samples']
        for stats in intra_genre_stats.values()
    ) / total_samples if total_samples > 0 else np.nan
    
    # Overall inter-genre distance
    inter_distances = [d['avg_distance'] for d in inter_genre_distances.values()]
    overall_inter = np.mean(inter_distances) if inter_distances else np.nan
    
    # Separation ratio (higher is better - inter should be much larger than intra)
    separation_ratio = overall_inter / weighted_intra if weighted_intra > 0 else np.nan
    
    # Separation gap (absolute difference)
    separation_gap = overall_inter - weighted_intra
    
    return {
        'overall_intra_distance': weighted_intra,
        'overall_inter_distance': overall_inter,
        'separation_ratio': separation_ratio,
        'separation_gap': separation_gap,
        'n_genres': len(intra_genre_stats),
        'n_genre_pairs': len(inter_genre_distances)
    }


def calculate_genre_spread_metrics(
    embeddings: np.ndarray,
    genres: np.ndarray,
    genre_stats: Dict[str, Dict],
    min_samples: int = 2
) -> Dict[str, Dict]:
    """
    Calculate spread metrics for each genre (how spread out movies are within genre).
    
    Parameters:
    - embeddings: Array of embeddings [n_samples, embedding_dim]
    - genres: Array of genre labels [n_samples]
    - genre_stats: Dictionary from calculate_intra_genre_distances
    - min_samples: Minimum number of samples required per genre
    
    Returns:
    - Dictionary with spread metrics per genre
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # Extract primary genres
    primary_genres = np.array([extract_primary_genre(g) for g in genres])
    valid_mask = np.array([g is not None for g in primary_genres])
    valid_embeddings = normalized_embeddings[valid_mask]
    valid_genres = primary_genres[valid_mask]
    
    spread_metrics = {}
    
    for genre, stats_dict in genre_stats.items():
        if stats_dict['n_samples'] < min_samples:
            continue
        
        genre_mask = valid_genres == genre
        genre_embeddings = valid_embeddings[genre_mask]
        centroid = stats_dict['centroid']
        
        # Calculate variance in embedding space
        # Distance from centroid for each embedding
        distances_to_centroid = cosine_distances([centroid], genre_embeddings)[0]
        
        # Calculate spread as standard deviation of distances
        spread_std = np.std(distances_to_centroid)
        spread_range = np.max(distances_to_centroid) - np.min(distances_to_centroid)
        
        # Calculate radius (95th percentile distance from centroid)
        spread_radius_95 = np.percentile(distances_to_centroid, 95)
        spread_radius_99 = np.percentile(distances_to_centroid, 99)
        
        spread_metrics[genre] = {
            'spread_std': spread_std,
            'spread_range': spread_range,
            'spread_radius_95': spread_radius_95,
            'spread_radius_99': spread_radius_99,
            'compactness': 1.0 / (1.0 + spread_std)  # Higher = more compact
        }
    
    return spread_metrics


def print_summary_report(
    intra_genre_stats: Dict[str, Dict],
    inter_genre_distances: Dict[str, Dict],
    separation_metrics: Dict[str, float],
    overall_metrics: Dict[str, float],
    spread_metrics: Dict[str, Dict]
):
    """Print a comprehensive summary report."""
    print("\n" + "="*80)
    print("GENRE LATENT SPACE ANALYSIS - SUMMARY REPORT")
    print("="*80)
    
    print("\n--- OVERALL METRICS ---")
    print(f"Number of genres analyzed: {overall_metrics.get('n_genres', 'N/A')}")
    print(f"Number of genre pairs: {overall_metrics.get('n_genre_pairs', 'N/A')}")
    print(f"Overall intra-genre distance: {overall_metrics.get('overall_intra_distance', np.nan):.4f}")
    print(f"Overall inter-genre distance: {overall_metrics.get('overall_inter_distance', np.nan):.4f}")
    print(f"Separation ratio (inter/intra): {overall_metrics.get('separation_ratio', np.nan):.4f}")
    print(f"Separation gap (inter - intra): {overall_metrics.get('separation_gap', np.nan):.4f}")
    
    print("\n--- SEPARATION QUALITY METRICS ---")
    print(f"Silhouette Score: {separation_metrics.get('silhouette_score', np.nan):.4f}")
    print(f"  (Range: -1 to 1, higher is better)")
    print(f"Calinski-Harabasz Score: {separation_metrics.get('calinski_harabasz_score', np.nan):.4f}")
    print(f"  (Higher is better)")
    print(f"Davies-Bouldin Score: {separation_metrics.get('davies_bouldin_score', np.nan):.4f}")
    print(f"  (Lower is better)")
    
    print("\n--- TOP 10 GENRES BY SAMPLE SIZE ---")
    sorted_genres = sorted(
        intra_genre_stats.items(),
        key=lambda x: x[1]['n_samples'],
        reverse=True
    )[:10]
    
    for genre, stats in sorted_genres:
        print(f"\n{genre}:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Avg intra-distance: {stats['avg_intra_distance']:.4f}")
        print(f"  Std intra-distance: {stats['std_intra_distance']:.4f}")
        if genre in spread_metrics:
            print(f"  Spread (std): {spread_metrics[genre]['spread_std']:.4f}")
            print(f"  Compactness: {spread_metrics[genre]['compactness']:.4f}")
    
    print("\n--- TOP 10 CLOSEST GENRE PAIRS ---")
    sorted_pairs = sorted(
        inter_genre_distances.items(),
        key=lambda x: x[1]['avg_distance'],
    )[:10]
    
    for pair_key, dist_info in sorted_pairs:
        genre1 = dist_info['genre1']
        genre2 = dist_info['genre2']
        print(f"{genre1} <-> {genre2}: {dist_info['avg_distance']:.4f}")
    
    print("\n--- TOP 10 MOST DISTANT GENRE PAIRS ---")
    sorted_pairs = sorted(
        inter_genre_distances.items(),
        key=lambda x: x[1]['avg_distance'],
        reverse=True
    )[:10]
    
    for pair_key, dist_info in sorted_pairs:
        genre1 = dist_info['genre1']
        genre2 = dist_info['genre2']
        print(f"{genre1} <-> {genre2}: {dist_info['avg_distance']:.4f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("""
    Separation Ratio: Measures how much larger inter-genre distances are compared
                      to intra-genre distances. A ratio > 2 suggests good separation.
    
    Silhouette Score: Measures how well samples are separated into their genres.
                      Values > 0.5 indicate good separation.
    
    Calinski-Harabasz: Measures ratio of between-cluster to within-cluster variance.
                       Higher values indicate better-defined clusters.
    
    Davies-Bouldin: Measures average similarity ratio of each cluster to its most
                    similar cluster. Lower values indicate better separation.
    
    Compactness: Measures how tightly grouped movies are within each genre.
                 Higher values (closer to 1) indicate more compact genres.
    """)


def main(
    csv_path: str = "/home/niklas/Desktop/Uni_Niklas/MASTER/Semester1/Data_Literacy/GroupDataLiteracy/data/data_final/final_dataset.csv",
    data_dir: str = "/home/niklas/Desktop/Uni_Niklas/MASTER/Semester1/Data_Literacy/GroupDataLiteracy/data/data_final",
    min_samples_per_genre: int = 10,
    verbose: bool = True
):
    """
    Main function to analyze genre separation in latent space.
    
    Parameters:
    - csv_path: Path to final_dataset.csv
    - data_dir: Directory containing embedding files
    - min_samples_per_genre: Minimum number of movies required per genre
    - verbose: Whether to print detailed output
    """
    if verbose:
        print("Loading dataset with embeddings...")
    
    # Load data
    df = load_final_data_with_embeddings(csv_path=csv_path, data_dir=data_dir, verbose=verbose)
    
    # Filter out movies without genres
    df = df[df['new_genre'].notna()].copy()
    
    if verbose:
        print(f"Loaded {len(df)} movies with genres")
    
    # Extract embeddings
    embeddings = np.vstack(df['embedding'].values)
    genres = df['new_genre'].values
    
    if verbose:
        print(f"Embedding shape: {embeddings.shape}")
        print("Calculating intra-genre distances...")
    
    # Calculate intra-genre distances
    intra_genre_stats = calculate_intra_genre_distances(
        embeddings, genres, min_samples=min_samples_per_genre
    )
    
    if verbose:
        print(f"Analyzed {len(intra_genre_stats)} genres")
        print("Calculating inter-genre distances...")
    
    # Calculate inter-genre distances
    inter_genre_distances = calculate_inter_genre_distances(
        embeddings, genres, intra_genre_stats, min_samples=min_samples_per_genre
    )
    
    if verbose:
        print(f"Analyzed {len(inter_genre_distances)} genre pairs")
        print("Calculating separation metrics...")
    
    # Calculate separation metrics
    separation_metrics = calculate_separation_metrics(
        embeddings, genres, min_samples=min_samples_per_genre
    )
    
    if verbose:
        print("Calculating overall metrics...")
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(intra_genre_stats, inter_genre_distances)
    
    if verbose:
        print("Calculating genre spread metrics...")
    
    # Calculate spread metrics
    spread_metrics = calculate_genre_spread_metrics(
        embeddings, genres, intra_genre_stats, min_samples=min_samples_per_genre
    )
    
    # Print summary
    if verbose:
        print_summary_report(
            intra_genre_stats,
            inter_genre_distances,
            separation_metrics,
            overall_metrics,
            spread_metrics
        )
    
    # Return all results
    return {
        'intra_genre_stats': intra_genre_stats,
        'inter_genre_distances': inter_genre_distances,
        'separation_metrics': separation_metrics,
        'overall_metrics': overall_metrics,
        'spread_metrics': spread_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze genre separation in latent space")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/niklas/Desktop/Uni_Niklas/MASTER/Semester1/Data_Literacy/GroupDataLiteracy/data/data_final/final_dataset.csv",
        help="Path to final_dataset.csv"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/niklas/Desktop/Uni_Niklas/MASTER/Semester1/Data_Literacy/GroupDataLiteracy/data/data_final",
        help="Directory containing embedding files"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="Minimum number of movies required per genre"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    results = main(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        min_samples_per_genre=args.min_samples,
        verbose=not args.quiet
    )
