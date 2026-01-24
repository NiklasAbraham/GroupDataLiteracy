"""Utility functions for epsilon ball analysis."""

import hashlib
import json
import logging
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_movie_name_for_filename(title: str) -> str:
    """Format a movie title for use in filenames."""
    formatted = title.replace(" ", "_")
    formatted = re.sub(r'[<>:"/\\|?*]', "", formatted)
    formatted = re.sub(r"_+", "_", formatted)
    formatted = formatted.strip("_")
    return formatted


def get_anchor_names_string(anchor_qids: list, movie_data: pd.DataFrame) -> str:
    """Get a formatted string of anchor movie names for use in filenames."""
    anchor_names = []
    for qid in anchor_qids:
        anchor_movie = movie_data[movie_data["movie_id"] == qid]
        if not anchor_movie.empty:
            title = anchor_movie.iloc[0]["title"]
            formatted_name = format_movie_name_for_filename(title)
            anchor_names.append(formatted_name)
        else:
            anchor_names.append(qid)

    return "__".join(anchor_names)


def truncate_filename_component(component: str, max_length: int = 120) -> str:
    """Truncate a filename component if it's too long, adding a hash suffix for uniqueness."""
    if len(component) <= max_length:
        return component

    hash_suffix = hashlib.md5(component.encode()).hexdigest()[:8]
    truncated = component[: max_length - 9]
    return f"{truncated}__{hash_suffix}"


def compute_embeddings_hash(
    filtered_embeddings: np.ndarray,
    start_year: int,
    end_year: int,
) -> str:
    """Compute a hash of the filtered embeddings and parameters to verify cache validity."""
    hash_data = {
        "shape": filtered_embeddings.shape,
        "dtype": str(filtered_embeddings.dtype),
        "start_year": start_year,
        "end_year": end_year,
    }

    n_samples = min(100, len(filtered_embeddings))
    if len(filtered_embeddings) > 0:
        sample_indices = [0]
        if len(filtered_embeddings) > 1:
            sample_indices.append(len(filtered_embeddings) - 1)
        if len(filtered_embeddings) > 2:
            sample_indices.append(len(filtered_embeddings) // 2)
        if len(filtered_embeddings) > n_samples:
            np.random.seed(42)
            random_indices = np.random.choice(
                len(filtered_embeddings),
                size=n_samples - len(sample_indices),
                replace=False,
            )
            sample_indices.extend(random_indices.tolist())

        sample_data = filtered_embeddings[sample_indices].tobytes()
        hash_data["sample_hash"] = hashlib.md5(sample_data).hexdigest()

        hash_data["sum"] = float(np.sum(filtered_embeddings))

    hash_string = json.dumps(hash_data, sort_keys=True)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def load_cached_mean_embedding(
    cache_dir: str,
    embeddings_hash: str,
) -> Tuple[Optional[np.ndarray], bool]:
    """Load cached mean embedding if it exists and hash matches."""
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.npy")
    metadata_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.json")

    if not os.path.exists(cache_file) or not os.path.exists(metadata_file):
        return None, False

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if metadata.get("hash") != embeddings_hash:
            logger.warning("Cache hash mismatch, will recompute mean embedding")
            return None, False

        mean_embedding = np.load(cache_file)
        logger.info(
            f"Loaded cached mean embedding from {cache_file} "
            f"(computed from {metadata.get('n_movies', 'unknown')} movies)"
        )
        return mean_embedding, True

    except Exception as e:
        logger.warning(f"Error loading cached mean embedding: {e}, will recompute")
        return None, False


def save_cached_mean_embedding(
    mean_embedding: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    n_movies: int,
) -> None:
    """Save mean embedding to cache with metadata."""
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.npy")
    metadata_file = os.path.join(cache_dir, f"mean_embedding_{embeddings_hash}.json")

    try:
        np.save(cache_file, mean_embedding)

        metadata = {
            "hash": embeddings_hash,
            "n_movies": n_movies,
            "shape": list(mean_embedding.shape),
            "dtype": str(mean_embedding.dtype),
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Cached mean embedding to {cache_file}")

    except Exception as e:
        logger.warning(f"Error saving cached mean embedding: {e}")
