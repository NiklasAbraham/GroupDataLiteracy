# -*- coding: utf-8 -*-
"""
predictor_logistic_regression.py

General-purpose logistic regression predictor for multi-label genre classification from embeddings.
Can load embeddings from data folder or accept embeddings directly.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Add parent directories to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from src.utils.data_utils import cluster_genres, load_movie_data, load_movie_embeddings


def _normalize_genres(genres: np.ndarray) -> Tuple[List[List[str]], np.ndarray]:
    """
    Normalize genre labels to list of lists format.

    Args:
        genres: Array of genre labels [n_samples]
               Can be list of lists (multi-label) or array of strings (single-label)

    Returns:
        Tuple of (genres_list, valid_indices):
        - genres_list: List of lists of genre strings
        - valid_indices: Array of indices that have valid genres
    """
    genres_list = []
    valid_indices = []

    for idx, genre in enumerate(genres):
        if genre is None or (isinstance(genre, float) and np.isnan(genre)):
            continue

        if isinstance(genre, str):
            # Handle pipe-separated (from cluster_genres) or comma-separated genres
            if "|" in genre:
                genre_list = [
                    g.strip()
                    for g in genre.split("|")
                    if g.strip() and g.strip() != "Unknown"
                ]
            else:
                genre_list = [g.strip() for g in genre.split(",") if g.strip()]
        elif isinstance(genre, list):
            # Already a list
            genre_list = [
                g.strip() if isinstance(g, str) else str(g).strip()
                for g in genre
                if g and str(g).strip()
            ]
        else:
            # Try to convert to string and split
            genre_str = str(genre)
            if "|" in genre_str:
                genre_list = [
                    g.strip()
                    for g in genre_str.split("|")
                    if g.strip() and g.strip() != "Unknown"
                ]
            else:
                genre_list = [g.strip() for g in genre_str.split(",") if g.strip()]

        if len(genre_list) > 0:
            genres_list.append(genre_list)
            valid_indices.append(idx)

    return genres_list, np.array(valid_indices)


def predict_genres_logistic_regression(
    embeddings: np.ndarray,
    genres: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    return_model: bool = False,
) -> Union[
    Dict[str, float],
    Tuple[Dict[str, float], MultiOutputClassifier, MultiLabelBinarizer],
]:
    """
    Train a multi-label logistic regression classifier to predict genres from embeddings.
    Each movie can have multiple genres. Compute subset accuracy and F1 score.

    This is a general-purpose function that can be used by manager.py or standalone.

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        genres: Array of genre labels [n_samples]
               Can be list of lists (multi-label) or array of strings (single-label)
               Supports both pipe-separated (|) and comma-separated (,) formats
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        return_model: If True, also return the trained model and label binarizer (default: False)

    Returns:
        If return_model=False:
            Dict[str, float]: Dictionary with classification metrics:
                - 'genre_accuracy': Subset accuracy (exact match for all labels)
                - 'genre_f1_score': Micro-averaged F1 (primary metric)
                - 'genre_f1_macro': Macro-averaged F1 (additional metric)
                - 'genre_hamming_loss': Hamming loss (lower is better)

        If return_model=True:
            Tuple of (metrics_dict, trained_model, label_binarizer)
    """
    # Normalize genres to list of lists format
    genres_list, valid_indices = _normalize_genres(genres)

    if len(genres_list) < 2:
        metrics = {
            "genre_accuracy": np.nan,
            "genre_f1_score": np.nan,
            "genre_f1_macro": np.nan,
            "genre_hamming_loss": np.nan,
        }
        if return_model:
            return metrics, None, None
        return metrics

    valid_embeddings = embeddings[valid_indices]

    # Convert to binary indicator matrix using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(genres_list)  # [n_samples, n_labels]

    if y_binary.shape[1] < 1:
        metrics = {
            "genre_accuracy": np.nan,
            "genre_f1_score": np.nan,
            "genre_f1_macro": np.nan,
            "genre_hamming_loss": np.nan,
        }
        if return_model:
            return metrics, None, None
        return metrics

    if valid_embeddings.shape[0] < 2:
        metrics = {
            "genre_accuracy": np.nan,
            "genre_f1_score": np.nan,
            "genre_f1_macro": np.nan,
            "genre_hamming_loss": np.nan,
        }
        if return_model:
            return metrics, None, None
        return metrics

    # Split data into train and test sets
    # For multi-label, we can't use standard stratification, so we use shuffle split
    X_train, X_test, y_train, y_test = train_test_split(
        valid_embeddings,
        y_binary,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Train multi-label logistic regression classifier
    # sklearn's LogisticRegression already uses efficient internal batching
    # We optimize solver choice based on dataset size for better performance
    try:
        # Determine best solver based on dataset size
        n_samples = X_train.shape[0]

        # For large datasets, use 'lbfgs' which is memory efficient and handles batching well
        # For smaller datasets, 'liblinear' is faster
        if n_samples > 10000:
            solver = "lbfgs"
        else:
            solver = "liblinear"

        # Use MultiOutputClassifier to handle multi-label classification
        # Each label gets its own binary classifier
        base_clf = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver=solver,
            n_jobs=1,  # sklearn handles internal batching, we parallelize at higher level
        )
        clf = MultiOutputClassifier(base_clf, n_jobs=1)
        clf.fit(X_train, y_train)

        # Predict on test set (predictions are already batched internally by sklearn)
        y_pred = clf.predict(X_test)

        # Compute metrics for multi-label classification
        # Subset accuracy: exact match (all labels must be correct)
        subset_accuracy = accuracy_score(y_test, y_pred)

        # Micro-averaged F1: treats each label prediction as an individual binary prediction
        f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)

        # Also compute macro-averaged F1 for additional insight
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Hamming loss: fraction of labels that are incorrectly predicted
        hamming = hamming_loss(y_test, y_pred)

        metrics = {
            "genre_accuracy": subset_accuracy,  # Subset accuracy (exact match)
            "genre_f1_score": f1_micro,  # Micro-averaged F1 (primary metric)
            "genre_f1_macro": f1_macro,  # Macro-averaged F1 (additional metric)
            "genre_hamming_loss": hamming,  # Hamming loss (lower is better)
        }

        if return_model:
            return metrics, clf, mlb
        return metrics

    except Exception as e:
        print(f"Error computing genre classification metrics: {e}")
        metrics = {
            "genre_accuracy": np.nan,
            "genre_f1_score": np.nan,
            "genre_f1_macro": np.nan,
            "genre_hamming_loss": np.nan,
        }
        if return_model:
            return metrics, None, None
        return metrics


def load_embeddings_and_genres(
    data_dir: str,
    chunking_suffix: str = "_cls_token",
    start_year: int = 1950,
    end_year: int = 2024,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings and corresponding genres from data directory.

    This function loads embeddings using load_movie_embeddings and then
    matches them with movie metadata to get genres.

    Args:
        data_dir: Absolute path to data directory
        chunking_suffix: Suffix appended to filename (e.g., "_cls_token", "_mean_pooling", "")
        start_year: First year to load (default: 1950)
        end_year: Last year to load (default: 2024)
        verbose: Print loading statistics (default: False)

    Returns:
        Tuple of (embeddings, genres, movie_ids):
        - embeddings: Array of embeddings [n_movies, embedding_dim]
        - genres: Array of genre labels [n_movies] (pipe-separated format from cluster_genres)
        - movie_ids: Array of movie IDs [n_movies]
    """
    # Load embeddings
    embeddings, movie_ids = load_movie_embeddings(
        data_dir=data_dir,
        chunking_suffix=chunking_suffix,
        start_year=start_year,
        end_year=end_year,
        verbose=verbose,
    )

    if len(embeddings) == 0:
        if verbose:
            print("No embeddings found!")
        return np.array([]), np.array([]), np.array([])

    # Load movie metadata
    if verbose:
        print(f"Loading movie metadata from {data_dir}...")
    metadata_df = load_movie_data(data_dir, verbose=verbose)

    # Process genres using cluster_genres
    original_cwd = os.getcwd()
    try:
        os.chdir(str(SRC_DIR))
        if verbose:
            print("Processing genres using cluster_genres from data_utils...")
        metadata_df = cluster_genres(metadata_df)
    finally:
        os.chdir(original_cwd)

    # Merge embeddings with metadata
    embeddings_df = pd.DataFrame({"movie_id": movie_ids})

    # Select relevant columns from metadata
    if "new_genre" in metadata_df.columns:
        genre_col = "new_genre"
    elif "genre" in metadata_df.columns:
        genre_col = "genre"
    else:
        if verbose:
            print("Warning: No genre column found in metadata")
        return embeddings, np.array([None] * len(movie_ids)), movie_ids

    metadata_subset = metadata_df[["movie_id", genre_col]].copy()
    combined_df = pd.merge(embeddings_df, metadata_subset, on="movie_id", how="inner")

    if verbose:
        print(
            f"Matched {len(combined_df)}/{len(embeddings_df)} embeddings with metadata"
        )

    # Filter out movies with missing or unknown genres
    if genre_col == "new_genre":
        combined_df = combined_df[
            combined_df[genre_col].notna() & (combined_df[genre_col] != "Unknown")
        ].copy()
    else:
        combined_df = combined_df[combined_df[genre_col].notna()].copy()

    if verbose:
        print(f"Movies with valid genres: {len(combined_df)}")

    # Get indices of matched movies
    matched_indices = combined_df.index.values
    matched_movie_ids = combined_df["movie_id"].values
    matched_genres = combined_df[genre_col].values

    # Filter embeddings to only include matched movies
    # Create a mapping from movie_id to embedding index
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    # Get embedding indices for matched movies
    embedding_indices = [
        movie_id_to_idx[mid] for mid in matched_movie_ids if mid in movie_id_to_idx
    ]

    if len(embedding_indices) == 0:
        if verbose:
            print("Warning: No matching embeddings found after genre filtering")
        return np.array([]), np.array([]), np.array([])

    filtered_embeddings = embeddings[embedding_indices]
    filtered_genres = matched_genres[: len(embedding_indices)]  # Ensure same length
    filtered_movie_ids = matched_movie_ids[: len(embedding_indices)]

    return filtered_embeddings, filtered_genres, filtered_movie_ids


def main():
    """
    Main entry point demonstrating how to use the predictor.
    Loads embeddings from data folder and runs genre prediction.
    """
    print("\n" + "=" * 80)
    print("Genre Prediction with Logistic Regression")
    print("=" * 80)

    # Configuration
    DATA_DIR = str(BASE_DIR / "data" / "data_final")
    CHUNKING_SUFFIX = "_cls_token"  # Change this to match your embedding files
    START_YEAR = 1930
    END_YEAR = 2024
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    print("\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Chunking suffix: {CHUNKING_SUFFIX}")
    print(f"  Year range: {START_YEAR} to {END_YEAR}")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  Random state: {RANDOM_STATE}")

    # Load embeddings and genres
    print("\nLoading embeddings and genres...")
    embeddings, genres, movie_ids = load_embeddings_and_genres(
        data_dir=DATA_DIR,
        chunking_suffix=CHUNKING_SUFFIX,
        start_year=START_YEAR,
        end_year=END_YEAR,
        verbose=True,
    )

    if len(embeddings) == 0:
        print(
            "ERROR: No embeddings loaded. Please check your data directory and chunking suffix."
        )
        return

    print(f"\nLoaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    print(f"Loaded {len(genres)} genre labels")

    # Run prediction
    print("\nTraining logistic regression classifier...")
    metrics = predict_genres_logistic_regression(
        embeddings=embeddings,
        genres=genres,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        return_model=False,
    )

    # Print results
    print(f"\n{'=' * 80}")
    print("Results:")
    print(f"{'=' * 80}")
    print(f"  Genre Accuracy (Subset):     {metrics['genre_accuracy']:.4f}")
    print(f"  Genre F1 Score (Micro):      {metrics['genre_f1_score']:.4f}")
    print(f"  Genre F1 Score (Macro):      {metrics['genre_f1_macro']:.4f}")
    print(f"  Genre Hamming Loss:          {metrics['genre_hamming_loss']:.4f}")
    print(f"{'=' * 80}\n")

    return metrics


if __name__ == "__main__":
    main()
