# -*- coding: utf-8 -*-
"""
calculations.py

Metrics and analysis functions for evaluating embedding methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the general predictor function
# calculations.py is in src/analysis/chunking/
# predictor_logistic_regression.py is in src/predictions/
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))
#from predictions.predictor_logistic_regression import predict_genres_logistic_regression


def compute_length_norm_correlation(embeddings: np.ndarray, text_lengths: np.ndarray, pre_norms: np.ndarray = None) -> float:
    """
    Compute Pearson correlation between text length and embedding norm.
    If pre_norms provided, uses those (pre-L2 magnitudes). Otherwise uses post-L2 norms.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        text_lengths (np.ndarray): Array of text lengths in tokens [n_samples]
        pre_norms (np.ndarray, optional): Pre-L2 magnitudes [n_samples]. If provided, used instead of post-L2.
        
    Returns:
        float: Pearson correlation coefficient
    """
    # Use pre-L2 norms if provided, otherwise compute from embeddings
    if pre_norms is not None and len(pre_norms) == len(embeddings):
        norms = pre_norms
    else:
        # Compute L2 norms from embeddings (post-L2, should be ~1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Guard against constant-norm degeneracy
        if np.allclose(norms, norms[0], rtol=1e-6, atol=1e-8):
            return 0.0
    
    # Compute correlation
    correlation, _ = pearsonr(text_lengths, norms)
    
    return correlation


def _postprocess_abtt(X: np.ndarray, n_pc: int = 2) -> np.ndarray:
    """
    Apply All-but-the-Top (ABTT) post-processing to remove top principal components.
    
    This improves isotropy by removing dominant directions while preserving
    discriminative power. Standard method from Mu et al. (2017).
    
    Args:
        X (np.ndarray): Embeddings [n_samples, embedding_dim]
        n_pc (int): Number of top PCs to remove (default: 2)
        
    Returns:
        np.ndarray: Post-processed embeddings [n_samples, embedding_dim]
    """
    if X.shape[0] < 2:
        return X
    
    # Convert to float32 for SVD (float16 is not supported by linalg)
    X = X.astype(np.float32) if X.dtype == np.float16 else X
    
    # Center the embeddings
    Xc = X - X.mean(axis=0, keepdims=True)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    
    # Remove top n_pc directions
    if n_pc > 0 and n_pc < Vt.shape[0]:
        comps = Vt[:n_pc].T  # [embedding_dim, n_pc]
        # Project out the top components
        Xp = Xc - Xc @ comps @ comps.T
    else:
        Xp = Xc
    
    return Xp


def compute_isotropy(embeddings: np.ndarray, abtt_pc: int = 0) -> float:
    """
    Compute isotropy: percentage of variance explained by first principal component.
    Lower values indicate more isotropic (uniform) embeddings.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        abtt_pc (int): Number of top PCs to remove via ABTT before computing isotropy (default: 0)
        
    Returns:
        float: Percentage of variance explained by first PC
    """
    if embeddings.shape[0] < 2:
        return 0.0
    
    # Convert to float32 if needed (float16 may cause issues with PCA)
    X = embeddings.astype(np.float32) if embeddings.dtype == np.float16 else embeddings
    
    # Apply ABTT post-processing if requested
    if abtt_pc > 0:
        X = _postprocess_abtt(X, n_pc=abtt_pc)
    
    pca = PCA(n_components=1)
    pca.fit(X)
    
    # Percentage of variance explained by first PC
    variance_explained = pca.explained_variance_ratio_[0] * 100
    
    return variance_explained


def compute_between_film_distance(embeddings: np.ndarray, film_ids: np.ndarray, n_samples: int = 1000) -> float:
    """
    Compute mean cosine distance between random pairs of different films.
    Optimized with vectorized operations.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        film_ids (np.ndarray): Array of film IDs [n_samples]
        n_samples (int): Number of random pairs to sample
        
    Returns:
        float: Mean cosine distance between films
    """
    unique_films = np.unique(film_ids)
    
    if len(unique_films) < 2:
        return 0.0
    
    # Pre-compute mean embedding per film (vectorized)
    film_embeddings_dict = {}
    for film_id in unique_films:
        film_mask = film_ids == film_id
        film_emb = embeddings[film_mask]
        # Use mean if multiple embeddings, otherwise use single embedding
        film_embeddings_dict[film_id] = film_emb[0] if len(film_emb) == 1 else film_emb.mean(axis=0)
    
    # Convert to array for vectorized operations
    film_ids_array = np.array(list(film_embeddings_dict.keys()))
    film_embeddings_array = np.array([film_embeddings_dict[fid] for fid in film_ids_array])
    
    # Sample random pairs upfront (vectorized)
    np.random.seed(42)  # For reproducibility
    n_pairs = min(n_samples, len(unique_films) * (len(unique_films) - 1) // 2)
    
    if n_pairs == 0:
        return 0.0
    
    # Generate all possible pairs and sample from them
    if len(unique_films) <= 100:  # For small number of films, use all pairs
        from itertools import combinations
        all_pairs = list(combinations(range(len(film_ids_array)), 2))
        if len(all_pairs) > n_pairs:
            pair_indices = np.random.choice(len(all_pairs), size=n_pairs, replace=False)
            selected_pairs = [all_pairs[i] for i in pair_indices]
        else:
            selected_pairs = all_pairs
    else:  # For large number of films, sample randomly
        selected_pairs = []
        for _ in range(n_pairs):
            idx1, idx2 = np.random.choice(len(film_ids_array), size=2, replace=False)
            selected_pairs.append((idx1, idx2))
    
    # Vectorized cosine distance computation
    # Cosine distance = 1 - cosine similarity
    # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
    # For normalized vectors: cosine similarity = dot(a, b)
    
    # Normalize embeddings (L2 normalization)
    norms = np.linalg.norm(film_embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = film_embeddings_array / norms
    
    # Vectorized cosine distance computation for all pairs at once
    if len(selected_pairs) > 0:
        # Extract embeddings for all pairs
        idx1_array = np.array([p[0] for p in selected_pairs])
        idx2_array = np.array([p[1] for p in selected_pairs])
        
        # Compute cosine similarities for all pairs at once (vectorized)
        # Cosine similarity = dot product of normalized vectors
        emb1_batch = normalized_embeddings[idx1_array]  # [n_pairs, emb_dim]
        emb2_batch = normalized_embeddings[idx2_array]  # [n_pairs, emb_dim]
        
        # Element-wise multiplication and sum along embedding dimension
        cosine_sims = np.sum(emb1_batch * emb2_batch, axis=1)  # [n_pairs]
        
        # Cosine distance = 1 - cosine similarity
        distances = 1.0 - cosine_sims
    else:
        distances = []
    
    if len(distances) == 0:
        return 0.0
    
    return np.mean(distances)


def compute_genre_clustering_quality(embeddings: np.ndarray, genres: np.ndarray) -> float:
    """
    Compute silhouette score for genre clustering.
    For multi-label genres, uses the first genre as the primary label.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        genres (np.ndarray): Array of genre labels [n_samples]
                           Can be list of lists (multi-label) or array of strings (single-label)
        
    Returns:
        float: Silhouette score (higher is better, range: -1 to 1)
    """
    # Handle both multi-label (list of lists) and single-label (array of strings) formats
    # For silhouette score, we need single labels, so we use the first genre for multi-label cases
    valid_genres = []
    valid_indices = []
    
    for idx, genre in enumerate(genres):
        if genre is None or (isinstance(genre, float) and np.isnan(genre)):
            continue
        if isinstance(genre, str):
            # Single label
            genre_str = genre.strip()
        elif isinstance(genre, list):
            # Multi-label: use first genre as primary label
            if len(genre) > 0:
                genre_str = str(genre[0]).strip()
            else:
                continue
        else:
            # Try to convert to string
            genre_str = str(genre).strip()
            # If it contains commas, take the first one
            if ',' in genre_str:
                genre_str = genre_str.split(',')[0].strip()
        
        if genre_str:
            valid_genres.append(genre_str)
            valid_indices.append(idx)
    
    if len(valid_genres) < 2:
        return 0.0
    
    valid_embeddings = embeddings[valid_indices]
    valid_genres_array = np.array(valid_genres)
    
    unique_genres = np.unique(valid_genres_array)
    if len(unique_genres) < 2:
        return 0.0
    
    if valid_embeddings.shape[0] < 2:
        return 0.0
    
    # Compute silhouette score
    try:
        score = silhouette_score(valid_embeddings, valid_genres_array)
        return score
    except Exception as e:
        print(f"Error computing silhouette score: {e}")
        return 0.0


def compute_genre_classification_metrics(embeddings: np.ndarray, genres: np.ndarray, 
                                       test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
    """
    Train a multi-label logistic regression classifier to predict genres from embeddings.
    Each movie can have multiple genres. Compute subset accuracy and F1 score.
    
    This function now uses the general predictor from predictor_logistic_regression.py.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        genres (np.ndarray): Array of genre labels [n_samples]
                           Can be list of lists (multi-label) or array of strings (single-label)
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        Dict[str, float]: Dictionary with 'genre_accuracy' (subset accuracy) and 'genre_f1_score' (micro-averaged)
    """
    # Use the general predictor function
    return predict_genres_logistic_regression(
        embeddings=embeddings,
        genres=genres,
        test_size=test_size,
        random_state=random_state,
        return_model=False
    )

def compute_cosine_distance(row):
    if np.all(pd.isna(row['next_avg_embedding'])):
        return np.nan

    return cosine(row['avg_embedding'], row['next_avg_embedding'])

def calculate_drift_vector(row):
    """
    Simple stuff
    """
    try:
        return row['next_avg_embedding'] - row['avg_embedding']
    except TypeError:
        return np.nan

def compute_simple_difference(v_t: float, v_t_plus_1: float) -> float:
    """
    Computes the simple difference between two values: (Value at t+1) - (Value at t).
    """

    # Check if the next value is missing (np.nan or pd.NA)
    if pd.isna(v_t_plus_1):
        return np.nan

    return v_t_plus_1 - v_t

def compute_cosine_similarity_variance(embeddings: np.ndarray, n_samples: int = 5000, random_state: int = 42, abtt_pc: int = 0) -> Dict[str, float]:
    """
    Compute variance and statistics of cosine similarity across all pairs of embeddings.
    This measures the spread/diversity of the embedding space.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        n_samples (int): Maximum number of pairs to sample for efficiency (default: 5000)
        random_state (int): Random seed for reproducibility (default: 42)
        abtt_pc (int): Number of top PCs to remove via ABTT before computing (default: 0)
        
    Returns:
        Dict[str, float]: Dictionary with cosine similarity statistics
    """
    if embeddings.shape[0] < 2:
        return {
            'cosine_sim_variance': np.nan,
            'cosine_sim_mean': np.nan,
            'cosine_sim_std': np.nan,
            'cosine_sim_min': np.nan,
            'cosine_sim_max': np.nan,
            'cosine_sim_median': np.nan
        }
    
    # Apply ABTT post-processing if requested
    if abtt_pc > 0:
        embeddings = _postprocess_abtt(embeddings, n_pc=abtt_pc)
    
    if embeddings.shape[0] < 2:
        return {
            'cosine_sim_variance': np.nan,
            'cosine_sim_mean': np.nan,
            'cosine_sim_std': np.nan,
            'cosine_sim_min': np.nan,
            'cosine_sim_max': np.nan,
            'cosine_sim_median': np.nan
        }
    
    # Normalize embeddings for efficient cosine similarity computation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    # Sample pairs for efficiency
    n_total = embeddings.shape[0]
    max_pairs = n_total * (n_total - 1) // 2
    
    if max_pairs <= n_samples:
        # Use all pairs
        from itertools import combinations
        pairs = list(combinations(range(n_total), 2))
    else:
        # Sample random pairs
        np.random.seed(random_state)
        pairs = []
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(n_total, size=2, replace=False)
            pairs.append((idx1, idx2))
    
    # Compute cosine similarities for all pairs (vectorized)
    idx1_array = np.array([p[0] for p in pairs])
    idx2_array = np.array([p[1] for p in pairs])
    
    emb1_batch = normalized_embeddings[idx1_array]
    emb2_batch = normalized_embeddings[idx2_array]
    
    # Cosine similarity = dot product of normalized vectors
    cosine_sims = np.sum(emb1_batch * emb2_batch, axis=1)
    
    return {
        'cosine_sim_variance': float(np.var(cosine_sims)),
        'cosine_sim_mean': float(np.mean(cosine_sims)),
        'cosine_sim_std': float(np.std(cosine_sims)),
        'cosine_sim_min': float(np.min(cosine_sims)),
        'cosine_sim_max': float(np.max(cosine_sims)),
        'cosine_sim_median': float(np.median(cosine_sims))
    }


def compute_separation_power(embeddings: np.ndarray, genres: np.ndarray, 
                            n_samples_per_class: int = 100, random_state: int = 42, abtt_pc: int = 0) -> Dict[str, float]:
    """
    Compute separation power metrics for the latent space based on genre classes.
    Measures how well-separated different genre classes are in the embedding space.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        genres (np.ndarray): Array of genre labels [n_samples], can be list of lists (multi-label)
        n_samples_per_class (int): Maximum number of samples per class for efficiency (default: 100)
        random_state (int): Random seed for reproducibility (default: 42)
        abtt_pc (int): Number of top PCs to remove via ABTT before computing (default: 0)
        
    Returns:
        Dict[str, float]: Dictionary with separation power metrics
    """
    # Apply ABTT post-processing if requested
    if abtt_pc > 0:
        embeddings = _postprocess_abtt(embeddings, n_pc=abtt_pc)
    # Extract primary genre for each sample (for multi-label, use first genre)
    primary_genres = []
    valid_indices = []
    
    for idx, genre in enumerate(genres):
        if genre is None or (isinstance(genre, float) and np.isnan(genre)):
            continue
        if isinstance(genre, str):
            primary_genre = genre.strip()
        elif isinstance(genre, list):
            if len(genre) > 0:
                primary_genre = str(genre[0]).strip()
            else:
                continue
        else:
            primary_genre = str(genre).strip()
            if ',' in primary_genre:
                primary_genre = primary_genre.split(',')[0].strip()
        
        if primary_genre:
            primary_genres.append(primary_genre)
            valid_indices.append(idx)
    
    if len(primary_genres) < 2:
        return {
            'intra_class_sim_mean': np.nan,
            'inter_class_sim_mean': np.nan,
            'separation_ratio': np.nan,
            'separation_gap': np.nan,
            'n_classes': np.nan,
            'n_samples': np.nan
        }
    
    valid_embeddings = embeddings[valid_indices]
    primary_genres_array = np.array(primary_genres)
    
    # Normalize embeddings
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_embeddings = valid_embeddings / norms
    
    # Get unique classes
    unique_classes = np.unique(primary_genres_array)
    n_classes = len(unique_classes)
    
    if n_classes < 2:
        return {
            'intra_class_sim_mean': np.nan,
            'inter_class_sim_mean': np.nan,
            'separation_ratio': np.nan,
            'separation_gap': np.nan,
            'n_classes': n_classes,
            'n_samples': len(valid_embeddings)
        }
    
    # Sample pairs for efficiency
    np.random.seed(random_state)
    
    intra_class_sims = []
    inter_class_sims = []
    
    # Compute intra-class similarities (within same genre)
    for class_label in unique_classes:
        class_mask = primary_genres_array == class_label
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) < 2:
            continue
        
        # Sample pairs within this class
        n_class_pairs = min(n_samples_per_class, len(class_indices) * (len(class_indices) - 1) // 2)
        
        if len(class_indices) <= 100:
            # Use all pairs for small classes
            from itertools import combinations
            class_pairs = list(combinations(class_indices, 2))
            if len(class_pairs) > n_class_pairs:
                class_pairs = [class_pairs[i] for i in np.random.choice(len(class_pairs), size=n_class_pairs, replace=False)]
        else:
            # Sample pairs for large classes
            class_pairs = []
            for _ in range(n_class_pairs):
                idx1, idx2 = np.random.choice(class_indices, size=2, replace=False)
                class_pairs.append((idx1, idx2))
        
        # Compute cosine similarities
        for idx1, idx2 in class_pairs:
            sim = np.dot(normalized_embeddings[idx1], normalized_embeddings[idx2])
            intra_class_sims.append(sim)
    
    # Compute inter-class similarities (between different genres)
    # Sample pairs from different classes
    n_inter_pairs = min(n_samples_per_class * n_classes, len(valid_embeddings) * 10)
    inter_pairs = []
    
    for _ in range(n_inter_pairs):
        # Pick two different classes
        class1, class2 = np.random.choice(unique_classes, size=2, replace=False)
        idx1 = np.random.choice(np.where(primary_genres_array == class1)[0])
        idx2 = np.random.choice(np.where(primary_genres_array == class2)[0])
        inter_pairs.append((idx1, idx2))
    
    # Compute cosine similarities
    for idx1, idx2 in inter_pairs:
        sim = np.dot(normalized_embeddings[idx1], normalized_embeddings[idx2])
        inter_class_sims.append(sim)
    
    if len(intra_class_sims) == 0 or len(inter_class_sims) == 0:
        return {
            'intra_class_sim_mean': np.nan,
            'inter_class_sim_mean': np.nan,
            'separation_ratio': np.nan,
            'separation_gap': np.nan,
            'n_classes': n_classes,
            'n_samples': len(valid_embeddings)
        }
    
    intra_class_sims = np.array(intra_class_sims)
    inter_class_sims = np.array(inter_class_sims)
    
    intra_mean = np.mean(intra_class_sims)
    inter_mean = np.mean(inter_class_sims)
    
    # Separation ratio: higher means better separation (inter-class similarity should be lower)
    # We use (1 - inter_mean) / (1 - intra_mean) to get a ratio where >1 means good separation
    # Or simpler: inter_mean / intra_mean, where lower is better (we want inter < intra)
    # Actually, let's use: (intra_mean - inter_mean) / (1 + intra_mean) as a normalized gap
    separation_gap = intra_mean - inter_mean  # Positive means good separation
    separation_ratio = inter_mean / intra_mean if intra_mean > 0 else np.nan  # Lower is better
    
    return {
        'intra_class_sim_mean': float(intra_mean),
        'inter_class_sim_mean': float(inter_mean),
        'separation_ratio': float(separation_ratio),
        'separation_gap': float(separation_gap),
        'n_classes': int(n_classes),
        'n_samples': int(len(valid_embeddings))
    }


def compute_temporal_drift(embeddings: np.ndarray, years: np.ndarray, 
                          anchor_film_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Compute temporal drift: track cosine shifts of anchor films across decades.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        years (np.ndarray): Array of years [n_samples]
        anchor_film_indices (Optional[List[int]]): Indices of anchor films to track
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with decade-wise cosine distances
    """
    if anchor_film_indices is None or len(anchor_film_indices) == 0:
        # Select a few random films as anchors
        np.random.seed(42)
        anchor_film_indices = np.random.choice(len(embeddings), size=min(5, len(embeddings)), replace=False).tolist()
    
    # Group by decade
    decades = (years // 10) * 10
    unique_decades = np.sort(np.unique(decades))
    
    if len(unique_decades) < 2:
        return {}
    
    # Compute mean embedding per decade (vectorized)
    decade_embeddings_list = []
    for decade in unique_decades:
        decade_mask = decades == decade
        if np.sum(decade_mask) > 0:
            decade_embeddings_list.append(embeddings[decade_mask].mean(axis=0))
    
    if len(decade_embeddings_list) < 2:
        return {}
    
    decade_embeddings_array = np.array(decade_embeddings_list)
    
    # Normalize embeddings for efficient cosine distance computation
    norms = np.linalg.norm(decade_embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = decade_embeddings_array / norms
    
    # Compute cosine distances between consecutive decades (vectorized)
    drift_distances = []
    decade_labels = []
    
    for i in range(len(unique_decades) - 1):
        decade1 = unique_decades[i]
        decade2 = unique_decades[i + 1]
        
        # Cosine distance = 1 - cosine similarity
        cosine_sim = np.dot(normalized_embeddings[i], normalized_embeddings[i + 1])
        cosine_dist = 1.0 - cosine_sim
        drift_distances.append(cosine_dist)
        decade_labels.append(f"{decade1}-{decade2}")
    
    return {
        'decades': np.array(decade_labels),
        'distances': np.array(drift_distances)
    }


def evaluate_method(embeddings: np.ndarray, 
                    text_lengths: np.ndarray,
                    film_ids: np.ndarray,
                    genres: Optional[np.ndarray] = None,
                    years: Optional[np.ndarray] = None,
                    method_name: str = "unknown",
                    texts: Optional[np.ndarray] = None,
                    embedding_service = None,
                    batch_size: int = 8,
                    pre_norms: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate a single embedding method and compute all metrics.
    Metrics are computed in parallel where possible.
    
    Args:
        embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
        text_lengths (np.ndarray): Array of text lengths in tokens [n_samples]
        film_ids (np.ndarray): Array of film IDs [n_samples]
        genres (Optional[np.ndarray]): Array of genre labels [n_samples]
        years (Optional[np.ndarray]): Array of years [n_samples]
        method_name (str): Name of the method
        texts (Optional[np.ndarray]): Array of original texts [n_samples] (unused, kept for compatibility)
        embedding_service: EmbeddingService instance (unused, kept for compatibility)
        batch_size (int): Batch size (unused, kept for compatibility)
        pre_norms (Optional[np.ndarray]): Pre-L2 magnitudes [n_samples] for accurate length correlation
        
    Returns:
        Dict[str, float]: Dictionary of metric names to values
    """
    print(f"    Computing metrics (parallelized)...")
    start_metrics = time.time()
    
    # Compute independent metrics in parallel
    # Increased max_workers to accommodate all parallel tasks including genre metrics
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all independent metric computations
        future_length_norm = executor.submit(compute_length_norm_correlation, embeddings, text_lengths, pre_norms)
        future_isotropy_raw = executor.submit(compute_isotropy, embeddings, abtt_pc=0)
        future_isotropy_abtt2 = executor.submit(compute_isotropy, embeddings, abtt_pc=2)
        future_between = executor.submit(compute_between_film_distance, embeddings, film_ids)
        future_cosine_sim_stats = executor.submit(compute_cosine_similarity_variance, embeddings, abtt_pc=0)
        future_cosine_sim_stats_abtt2 = executor.submit(compute_cosine_similarity_variance, embeddings, abtt_pc=2)
        
        # Submit genre-related metrics in parallel if genres are available
        if genres is not None:
            future_silhouette = executor.submit(compute_genre_clustering_quality, embeddings, genres)
            future_genre_classification = executor.submit(compute_genre_classification_metrics, embeddings, genres)
            future_separation_power = executor.submit(compute_separation_power, embeddings, genres, abtt_pc=0)
            future_separation_power_abtt2 = executor.submit(compute_separation_power, embeddings, genres, abtt_pc=2)
        else:
            future_silhouette = None
            future_genre_classification = None
            future_separation_power = None
        
        # Get results from parallel computations
        length_norm_corr = future_length_norm.result()
        isotropy_firstPC = future_isotropy_raw.result()
        isotropy_firstPC_abtt2 = future_isotropy_abtt2.result()
        mean_between_distance = future_between.result()
        cosine_sim_stats = future_cosine_sim_stats.result()
        cosine_sim_stats_abtt2 = future_cosine_sim_stats_abtt2.result()
        
        # Get genre-related results if available
        if future_silhouette is not None and future_genre_classification is not None:
            silhouette_score_val = future_silhouette.result()
            genre_classification_metrics = future_genre_classification.result()
            genre_accuracy = genre_classification_metrics.get('genre_accuracy', np.nan)
            genre_f1_score = genre_classification_metrics.get('genre_f1_score', np.nan)
            genre_f1_macro = genre_classification_metrics.get('genre_f1_macro', np.nan)
            genre_hamming_loss = genre_classification_metrics.get('genre_hamming_loss', np.nan)
            separation_power_metrics = future_separation_power.result()
            intra_class_sim = separation_power_metrics.get('intra_class_sim_mean', np.nan)
            inter_class_sim = separation_power_metrics.get('inter_class_sim_mean', np.nan)
            separation_ratio = separation_power_metrics.get('separation_ratio', np.nan)
            separation_gap = separation_power_metrics.get('separation_gap', np.nan)
            separation_power_metrics_abtt2 = future_separation_power_abtt2.result()
            intra_class_sim_abtt2 = separation_power_metrics_abtt2.get('intra_class_sim_mean', np.nan)
            inter_class_sim_abtt2 = separation_power_metrics_abtt2.get('inter_class_sim_mean', np.nan)
            separation_ratio_abtt2 = separation_power_metrics_abtt2.get('separation_ratio', np.nan)
            separation_gap_abtt2 = separation_power_metrics_abtt2.get('separation_gap', np.nan)
        else:
            silhouette_score_val = np.nan
            genre_accuracy = np.nan
            genre_f1_score = np.nan
            genre_f1_macro = np.nan
            genre_hamming_loss = np.nan
            intra_class_sim = np.nan
            inter_class_sim = np.nan
            separation_ratio = np.nan
            separation_gap = np.nan
            intra_class_sim_abtt2 = np.nan
            inter_class_sim_abtt2 = np.nan
            separation_ratio_abtt2 = np.nan
            separation_gap_abtt2 = np.nan
    
    metrics = {
        'method': method_name,
        'length_norm_corr': length_norm_corr,
        'isotropy_firstPC': isotropy_firstPC,
        'isotropy_firstPC_abtt2': isotropy_firstPC_abtt2,
        'mean_between_distance': mean_between_distance,
        'cosine_sim_variance': cosine_sim_stats.get('cosine_sim_variance', np.nan),
        'cosine_sim_mean': cosine_sim_stats.get('cosine_sim_mean', np.nan),
        'cosine_sim_std': cosine_sim_stats.get('cosine_sim_std', np.nan),
        'cosine_sim_min': cosine_sim_stats.get('cosine_sim_min', np.nan),
        'cosine_sim_max': cosine_sim_stats.get('cosine_sim_max', np.nan),
        'cosine_sim_median': cosine_sim_stats.get('cosine_sim_median', np.nan),
        'cosine_sim_variance_abtt2': cosine_sim_stats_abtt2.get('cosine_sim_variance', np.nan),
        'cosine_sim_mean_abtt2': cosine_sim_stats_abtt2.get('cosine_sim_mean', np.nan),
        'cosine_sim_std_abtt2': cosine_sim_stats_abtt2.get('cosine_sim_std', np.nan),
        'cosine_sim_min_abtt2': cosine_sim_stats_abtt2.get('cosine_sim_min', np.nan),
        'cosine_sim_max_abtt2': cosine_sim_stats_abtt2.get('cosine_sim_max', np.nan),
        'cosine_sim_median_abtt2': cosine_sim_stats_abtt2.get('cosine_sim_median', np.nan),
        'silhouette_score': silhouette_score_val,
        'genre_accuracy': genre_accuracy,  # Subset accuracy (exact match for all labels)
        'genre_f1_score': genre_f1_score,  # Micro-averaged F1
        'genre_f1_macro': genre_f1_macro,  # Macro-averaged F1
        'genre_hamming_loss': genre_hamming_loss,  # Hamming loss (lower is better)
        'intra_class_sim_mean': intra_class_sim,  # Mean cosine similarity within same genre
        'inter_class_sim_mean': inter_class_sim,  # Mean cosine similarity between different genres
        'separation_ratio': separation_ratio,  # inter_class / intra_class (lower is better)
        'separation_gap': separation_gap,  # intra_class - inter_class (higher is better)
        'intra_class_sim_mean_abtt2': intra_class_sim_abtt2,  # Mean cosine similarity within same genre (ABTT-2)
        'inter_class_sim_mean_abtt2': inter_class_sim_abtt2,  # Mean cosine similarity between different genres (ABTT-2)
        'separation_ratio_abtt2': separation_ratio_abtt2,  # inter_class / intra_class (lower is better, ABTT-2)
        'separation_gap_abtt2': separation_gap_abtt2,  # intra_class - inter_class (higher is better, ABTT-2)
    }
    
    elapsed = time.time() - start_metrics
    print(f"    Metrics computed in {elapsed:.2f}s")
    
    return metrics


def plot_length_norm_correlation(embeddings: np.ndarray, text_lengths: np.ndarray,
                                 method_name: str, output_path: str) -> None:
    """Plot correlation between text length and embedding norm (single method)."""
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Sample to maximum 5000 points for performance
    max_points = 5000
    if len(text_lengths) > max_points:
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(text_lengths), size=max_points, replace=False)
        text_lengths_plot = text_lengths[sample_indices]
        norms_plot = norms[sample_indices]
    else:
        text_lengths_plot = text_lengths
        norms_plot = norms
    
    # Compute correlation on sampled data
    correlation, _ = pearsonr(text_lengths_plot, norms_plot)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(text_lengths_plot, norms_plot, alpha=0.5, s=20)
    plt.xlabel('Text Length (tokens)', fontsize=12)
    plt.ylabel('Embedding L2 Norm', fontsize=12)
    title_suffix = f' (sampled {len(text_lengths_plot)}/{len(text_lengths)} points)' if len(text_lengths) > max_points else ''
    plt.title(f'{method_name}: Length-Norm Correlation = {correlation:.3f}{title_suffix}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_length_norm_correlation_combined(embeddings_dict: Dict[str, np.ndarray], 
                                          text_lengths: np.ndarray,
                                          output_path: str) -> None:
    """Plot correlation between text length and embedding norm for all methods in one plot."""
    # Sample to maximum 5000 points for performance
    max_points = 5000
    if len(text_lengths) > max_points:
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(text_lengths), size=max_points, replace=False)
        text_lengths_plot = text_lengths[sample_indices]
    else:
        sample_indices = None
        text_lengths_plot = text_lengths
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for idx, (method_name, embeddings) in enumerate(embeddings_dict.items()):
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Use same sample indices for all methods
        if sample_indices is not None:
            norms_plot = norms[sample_indices]
        else:
            norms_plot = norms
        
        # Compute correlation on sampled data
        correlation, _ = pearsonr(text_lengths_plot, norms_plot)
        
        plt.scatter(text_lengths_plot, norms_plot, alpha=0.6, s=30, 
                   label=f'{method_name} (r={correlation:.3f})',
                   color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)])
    
    plt.xlabel('Text Length (tokens)', fontsize=12)
    plt.ylabel('Embedding L2 Norm', fontsize=12)
    title_suffix = f' (sampled {len(text_lengths_plot)}/{len(text_lengths)} points)' if len(text_lengths) > max_points else ''
    plt.title(f'Length-Norm Correlation Comparison{title_suffix}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_isotropy(metrics_dict: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Plot isotropy (first PC variance) for all methods, showing both raw and ABTT-2."""
    methods = list(metrics_dict.keys())
    
    # Get raw and ABTT-2 isotropy values
    isotropy_raw = [metrics_dict[m].get('isotropy_firstPC', 0.0) for m in methods]
    isotropy_abtt2 = [metrics_dict[m].get('isotropy_firstPC_abtt2', 0.0) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, isotropy_raw, width, label='Raw', color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, isotropy_abtt2, width, label='ABTT-2', color='#ff7f0e', alpha=0.8)
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('% Variance Explained by First PC', fontsize=12)
    plt.title('Isotropy: Lower is Better (Raw vs ABTT-2)', fontsize=14)
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_genre_silhouette(metrics_dict: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Plot silhouette scores for genre clustering."""
    methods = list(metrics_dict.keys())
    silhouette_scores = [metrics_dict[m].get('silhouette_score', np.nan) for m in methods]
    
    # Filter out NaN values
    valid_data = [(m, s) for m, s in zip(methods, silhouette_scores) if not np.isnan(s)]
    if not valid_data:
        print("No valid silhouette scores to plot")
        return
    
    methods_valid, scores_valid = zip(*valid_data)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods_valid, scores_valid, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods_valid)])
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Genre Clustering Quality (Higher is Better)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([-1, 1])
    
    # Add value labels
    for bar, val in zip(bars, scores_valid):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_drift_stability(embeddings_dict: Dict[str, np.ndarray], 
                        years: np.ndarray,
                        output_path: str) -> None:
    """Plot temporal drift stability across methods."""
    plt.figure(figsize=(12, 6))
    
    # Define line styles for differentiation within same method type
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
    
    # Extract base method names (before first underscore)
    def get_base_method_name(method_name: str) -> str:
        """Extract base method name before first underscore."""
        if '_' in method_name:
            return method_name.split('_')[0]
        return method_name
    
    # Get unique base method names and assign colors
    base_methods = set(get_base_method_name(name) for name in embeddings_dict.keys())
    base_method_list = sorted(list(base_methods))
    
    # Assign distinct colors to each base method type
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    base_method_colors = {}
    for idx, base_method in enumerate(base_method_list):
        base_method_colors[base_method] = color_palette[idx % len(color_palette)]
    
    # Track line style index per base method (for variants)
    base_method_style_idx = {base: 0 for base in base_method_list}
    
    for method_name, embeddings in embeddings_dict.items():
        drift_data = compute_temporal_drift(embeddings, years)
        if 'distances' in drift_data and len(drift_data['distances']) > 0:
            base_method = get_base_method_name(method_name)
            color = base_method_colors[base_method]
            
            # Use different line styles for variants of the same method
            style_idx = base_method_style_idx[base_method]
            line_style = line_styles[style_idx % len(line_styles)]
            base_method_style_idx[base_method] += 1
            
            plt.plot(drift_data['decades'], drift_data['distances'], 
                    marker='o', label=method_name, 
                    linewidth=3.5,  # Thicker lines
                    markersize=8,
                    linestyle=line_style,  # Different line styles for variants
                    color=color,
                    alpha=0.3)  # Transparency
    
    plt.xlabel('Decade Transition', fontsize=12)
    plt.ylabel('Cosine Distance', fontsize=12)
    plt.title('Temporal Drift Stability (Lower is Better)', fontsize=14)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_csv(metrics_dict: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Save metrics to CSV file."""
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

