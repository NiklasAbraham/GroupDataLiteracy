"""
PCA analysis on movie embeddings with concept space projection.

Performs exploratory PCA on movie embeddings to identify latent semantic dimensions,
then projects concept words from the concept space onto the PCA space for interpretation.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.aab_analysis.concept_words.concept_space import (  # noqa: E402
    DEFAULT_CONCEPT_DIR,
    DEFAULT_CONCEPT_MODEL,
    ConceptSpace,
    get_concept_space_filenames,
)

warnings.filterwarnings('ignore')


def extract_embeddings_from_df(df: pd.DataFrame) -> np.ndarray:
    """Extract embeddings from a DataFrame that has an 'embedding' column."""
    if 'embedding' not in df.columns:
        raise ValueError("DataFrame must have 'embedding' column")
    
    embeddings_list = df['embedding'].tolist()
    return np.vstack(embeddings_list).astype(np.float32)


def perform_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Perform PCA on embeddings with column-wise centering only (no scaling).
    
    Returns:
        Tuple of (pc_scores, eigenvalues, principal_directions, pca_model)
    """
    mu = np.mean(embeddings, axis=0)
    X_centered = embeddings - mu
    
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(X_centered)
    
    eigenvalues = pca.explained_variance_
    principal_directions = pca.components_
    
    if verbose:
        explained_variance = pca.explained_variance_ratio_
        print("\nPCA Results:")
        for i in range(n_components):
            print(f"  PC{i+1} explained variance: {explained_variance[i]:.4f} ({explained_variance[i]*100:.2f}%)")
        print(f"  Total explained variance: {explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)")
        print("  Eigenvalues: ", end="")
        for i in range(n_components):
            print(f"λ{i+1}={eigenvalues[i]:.4f}", end=", " if i < n_components - 1 else "\n")
    
    return pc_scores, eigenvalues, principal_directions, pca


def select_relevant_concepts(
    concept_space: ConceptSpace,
    embeddings: np.ndarray,
    top_k: int = 30,
    verbose: bool = True
) -> List[str]:
    """Select most relevant concepts from concept space based on similarity to movie embeddings."""
    mean_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
    
    similarities = mean_embedding @ concept_space.concept_vecs.T
    top_indices = np.argsort(-similarities)[:top_k]
    
    selected_words = [concept_space.concept_words[idx] for idx in top_indices]
    
    if verbose:
        print(f"\nSelected {len(selected_words)} most relevant concepts from concept space")
        print("Top 10 selected concepts:")
        for i, word in enumerate(selected_words[:10], 1):
            sim = similarities[top_indices[i-1]]
            print(f"  {i}. {word:20s}: similarity={sim:.4f}")
    
    return selected_words


def project_concept_words(
    concept_space: ConceptSpace,
    principal_directions: np.ndarray,
    embeddings: np.ndarray,
    top_k_concepts: int = 30,
    top_per_dimension: int = 10,
    n_dimensions: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], Dict[int, Dict[str, List[Tuple[str, float]]]]]:
    """
    Project concept words onto PCA space and find balanced top concepts per dimension.
    For each dimension, returns equal numbers of concepts from positive and negative tails.
    """
    words = select_relevant_concepts(concept_space, embeddings, top_k=top_k_concepts, verbose=verbose)
    concept_indices = [concept_space.word2idx[word] for word in words]
    
    concept_vecs = concept_space.concept_vecs[concept_indices]
    projections = concept_vecs @ principal_directions.T
    
    n_components = projections.shape[1]
    if n_dimensions is None:
        n_dimensions = n_components
    else:
        n_dimensions = min(n_dimensions, n_components)
    
    n_side = top_per_dimension // 2
    
    top_per_dim = {}
    for dim in range(n_dimensions):
        dim_projections = projections[:, dim]
        
        pos_indices = np.argsort(-dim_projections)[:n_side]
        top_positive = [(words[i], float(dim_projections[i])) for i in pos_indices]
        
        neg_indices = np.argsort(dim_projections)[:n_side]
        top_negative = [(words[i], float(dim_projections[i])) for i in neg_indices]
        
        top_per_dim[dim] = {
            "positive": top_positive,
            "negative": top_negative,
        }
        
        if verbose:
            print(f"\nTop {n_side} POSITIVE concepts for PC{dim+1}:")
            for i, (word, score) in enumerate(top_positive, 1):
                print(f"  +{i:2d}. {word:20s}: {score:8.4f}")
            
            print(f"\nTop {n_side} NEGATIVE concepts for PC{dim+1}:")
            for i, (word, score) in enumerate(top_negative, 1):
                print(f"  -{i:2d}. {word:20s}: {score:8.4f}")
    
    return projections, words, top_per_dim


def visualize_pca(
    pc_scores: np.ndarray,
    metadata_df: pd.DataFrame,
    concept_projections: Optional[np.ndarray] = None,
    concept_words: Optional[List[str]] = None,
    explained_variance: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """Create 2D scatterplot of PCA results with optional concept word overlays."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'new_genre' in metadata_df.columns:
        first_genres = metadata_df['new_genre'].str.split('|').str[0]
        unique_genres = first_genres.unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
        genre_to_color = dict(zip(unique_genres, colors))
        
        for genre in unique_genres:
            mask = first_genres == genre
            if mask.sum() > 0:
                ax.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                          label=genre, alpha=0.6, s=30, c=[genre_to_color[genre]])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.6, s=30)
    
    if concept_projections is not None and concept_words is not None:
        for word, proj in zip(concept_words, concept_projections):
            ax.scatter(proj[0], proj[1], marker='*', s=200, c='red',
                      edgecolors='black', linewidths=1, zorder=10)
            ax.annotate(word, (proj[0], proj[1]), xytext=(5, 5),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       zorder=11)
    
    if explained_variance is not None:
        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
        if len(explained_variance) > 1:
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
    else:
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
    
    ax.set_title('PCA of Movie Embeddings with Concept Word Projections', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
    
    plt.show()


def run_pca_analysis(
    df: pd.DataFrame,
    n_components: int = 2,
    top_k_concepts: int = 30,
    top_per_dimension: int = 10,
    n_dimensions_to_analyze: Optional[int] = None,
    concept_model: str = DEFAULT_CONCEPT_MODEL,
    min_zipf_vocab: float = 2.5,
    max_vocab: int = 20000,
    concept_dir: Optional[Path] = None,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, ConceptSpace]:
    """Complete PCA analysis pipeline on movie embeddings with concept space projection."""
    embeddings = extract_embeddings_from_df(df)
    
    if verbose:
        print(f"Dataset: {len(embeddings)} movies, embedding dimension: {embeddings.shape[1]}")
    
    pc_scores, eigenvalues, principal_directions, pca_model = perform_pca(
        embeddings,
        n_components=n_components,
        verbose=verbose
    )
    
    concept_dir = Path(concept_dir or DEFAULT_CONCEPT_DIR)
    words_filename, vecs_filename = get_concept_space_filenames(
        min_zipf_vocab, max_vocab, concept_model
    )
    concept_words_path = concept_dir / words_filename
    concept_vecs_path = concept_dir / vecs_filename
    
    if not concept_words_path.exists() or not concept_vecs_path.exists():
        raise FileNotFoundError(
            f"Concept space files not found:\n  {concept_words_path}\n  {concept_vecs_path}\n"
            f"Please build the concept space first using build_wordnet_concept_vocab and embed_and_save_concept_vocab."
        )
    
    concept_space = ConceptSpace(concept_words_path, concept_vecs_path, model_name=concept_model)
    
    if verbose:
        print(f"\nLoaded concept space with {len(concept_space.concept_words)} concepts")
    
    concept_projections, all_concept_words, top_per_dim_dict = project_concept_words(
        concept_space,
        principal_directions,
        embeddings=embeddings,
        top_k_concepts=top_k_concepts,
        top_per_dimension=top_per_dimension,
        n_dimensions=n_dimensions_to_analyze if n_dimensions_to_analyze is not None else n_components,
        verbose=verbose
    )
    
    if n_components >= 2:
        n_viz_per_side = min(5, top_per_dimension // 2)
        pc1_top = ([word for word, _ in top_per_dim_dict[0]["positive"][:n_viz_per_side]] +
                   [word for word, _ in top_per_dim_dict[0]["negative"][:n_viz_per_side]])
        pc2_top = ([word for word, _ in top_per_dim_dict[1]["positive"][:n_viz_per_side]] +
                   [word for word, _ in top_per_dim_dict[1]["negative"][:n_viz_per_side]])
        viz_words = list(dict.fromkeys(pc1_top + pc2_top))
        
        viz_indices = [all_concept_words.index(w) for w in viz_words if w in all_concept_words]
        viz_projections = concept_projections[viz_indices, :2]
        viz_words_final = [all_concept_words[i] for i in viz_indices]
    else:
        n_viz_per_side = min(5, top_per_dimension // 2)
        viz_words_final = ([word for word, _ in top_per_dim_dict[0]["positive"][:n_viz_per_side]] +
                           [word for word, _ in top_per_dim_dict[0]["negative"][:n_viz_per_side]])
        viz_indices = [all_concept_words.index(w) for w in viz_words_final if w in all_concept_words]
        viz_projections = concept_projections[viz_indices, :min(2, n_components)]
    
    visualize_pca(
        pc_scores,
        df,
        concept_projections=viz_projections,
        concept_words=viz_words_final,
        explained_variance=pca_model.explained_variance_ratio_,
        output_path=output_path
    )
    
    return pc_scores, eigenvalues, principal_directions, pca_model, concept_space
