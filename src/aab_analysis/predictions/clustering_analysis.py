# -*- coding: utf-8 -*-
"""
clustering_analysis.py

Comprehensive clustering analysis for genre prediction diagnostics and improvement.
Analyzes how well embeddings cluster by genre and provides insights for improving prediction.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

# Add parent directories to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def _normalize_genres(genres: np.ndarray) -> Tuple[List[List[str]], np.ndarray]:
    """Normalize genre labels to list of lists format."""
    genres_list = []
    valid_indices = []

    for idx, genre in enumerate(genres):
        if genre is None or (isinstance(genre, float) and np.isnan(genre)):
            continue

        if isinstance(genre, str):
            if "|" in genre:
                genre_list = [
                    g.strip()
                    for g in genre.split("|")
                    if g.strip() and g.strip() != "Unknown"
                ]
            else:
                genre_list = [g.strip() for g in genre.split(",") if g.strip()]
        elif isinstance(genre, list):
            genre_list = [
                g.strip() if isinstance(g, str) else str(g).strip()
                for g in genre
                if g and str(g).strip()
            ]
        else:
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


def analyze_genre_separability(
    embeddings: np.ndarray,
    genres: np.ndarray,
    n_clusters_range: Tuple[int, int] = (5, 50),
    step: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Analyze how well embeddings separate by genre using clustering.

    Tests different numbers of clusters and measures how well they align with genre labels.

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        genres: Array of genre labels [n_samples]
        n_clusters_range: Range of cluster numbers to test (min, max)
        step: Step size for cluster number range
        random_state: Random seed

    Returns:
        Dictionary with clustering metrics for different numbers of clusters
    """
    genres_list, valid_indices = _normalize_genres(genres)

    if len(genres_list) < 2:
        return {"error": "Not enough valid genres"}

    valid_embeddings = embeddings[valid_indices]

    # Get primary genre for each movie (for single-label clustering metrics)
    primary_genres = [g[0] if len(g) > 0 else "Unknown" for g in genres_list]
    unique_genres = set(primary_genres)

    print("\nAnalyzing genre separability...")
    print(f"  Movies: {len(valid_embeddings)}")
    print(f"  Unique genres: {len(unique_genres)}")

    results = {}
    min_k, max_k = n_clusters_range

    for n_clusters in range(min_k, max_k + 1, step):
        if n_clusters > len(valid_embeddings):
            break

        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(valid_embeddings)

        # Compute metrics comparing clusters to genres
        # Silhouette score (higher is better, -1 to 1)
        silhouette = silhouette_score(valid_embeddings, cluster_labels)

        # Adjusted Rand Index (higher is better, -1 to 1, measures cluster-label agreement)
        ari = adjusted_rand_score(primary_genres, cluster_labels)

        # Normalized Mutual Information (higher is better, 0 to 1)
        nmi = normalized_mutual_info_score(primary_genres, cluster_labels)

        # Homogeneity: each cluster contains only members of a single class
        homogeneity = homogeneity_score(primary_genres, cluster_labels)

        # Completeness: all members of a given class are assigned to the same cluster
        completeness = completeness_score(primary_genres, cluster_labels)

        # V-measure: harmonic mean of homogeneity and completeness
        v_measure = v_measure_score(primary_genres, cluster_labels)

        results[n_clusters] = {
            "silhouette": silhouette,
            "ari": ari,
            "nmi": nmi,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "inertia": kmeans.inertia_,
        }

        if n_clusters % 10 == 0 or n_clusters == min_k:
            print(
                f"  k={n_clusters:2d}: Silhouette={silhouette:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}, V-measure={v_measure:.3f}"
            )

    return results


def analyze_genre_cluster_purity(
    embeddings: np.ndarray,
    genres: np.ndarray,
    n_clusters: int = 20,
    random_state: int = 42,
) -> Dict:
    """
    Analyze cluster purity: how well do clusters align with genres?

    For each cluster, compute:
    - Dominant genre and its percentage
    - Genre distribution within cluster
    - Purity score (fraction of cluster that belongs to dominant genre)

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        genres: Array of genre labels [n_samples]
        n_clusters: Number of clusters to create
        random_state: Random seed

    Returns:
        Dictionary with cluster purity analysis
    """
    genres_list, valid_indices = _normalize_genres(genres)

    if len(genres_list) < 2:
        return {"error": "Not enough valid genres"}

    valid_embeddings = embeddings[valid_indices]

    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_embeddings)

    # Get primary genre for each movie
    primary_genres = np.array([g[0] if len(g) > 0 else "Unknown" for g in genres_list])

    # Analyze each cluster
    cluster_analysis = {}
    overall_purity = 0.0

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = cluster_mask.sum()

        if cluster_size == 0:
            continue

        cluster_genres = primary_genres[cluster_mask]
        genre_counts = Counter(cluster_genres)
        dominant_genre, dominant_count = genre_counts.most_common(1)[0]
        purity = dominant_count / cluster_size

        overall_purity += purity * cluster_size

        cluster_analysis[cluster_id] = {
            "size": cluster_size,
            "dominant_genre": dominant_genre,
            "dominant_count": dominant_count,
            "purity": purity,
            "genre_distribution": dict(genre_counts),
        }

    overall_purity /= len(valid_embeddings)

    # Find best and worst clusters
    cluster_purities = [
        (cid, info["purity"], info["size"]) for cid, info in cluster_analysis.items()
    ]
    cluster_purities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nCluster Purity Analysis (k={n_clusters}):")
    print(f"  Overall purity: {overall_purity:.3f}")
    print("  Best clusters (purity > 0.7):")
    for cid, purity, size in cluster_purities[:5]:
        if purity > 0.7:
            info = cluster_analysis[cid]
            print(
                f"    Cluster {cid}: {info['dominant_genre']} ({purity:.2%}, n={size})"
            )

    print("  Worst clusters (purity < 0.3):")
    for cid, purity, size in cluster_purities[-5:]:
        if purity < 0.3:
            info = cluster_analysis[cid]
            print(
                f"    Cluster {cid}: {info['dominant_genre']} ({purity:.2%}, n={size})"
            )

    return {
        "overall_purity": overall_purity,
        "cluster_analysis": cluster_analysis,
        "n_clusters": n_clusters,
    }


def analyze_genre_overlap_in_embedding_space(
    embeddings: np.ndarray, genres: np.ndarray, top_n_genres: int = 10
) -> Dict:
    """
    Analyze how much genres overlap in embedding space.

    For each genre, compute:
    - Mean embedding (centroid)
    - Distance to other genre centroids
    - Overlap with other genres (movies that are close to multiple genre centroids)

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        genres: Array of genre labels [n_samples]
        top_n_genres: Number of top genres to analyze in detail

    Returns:
        Dictionary with genre overlap analysis
    """
    genres_list, valid_indices = _normalize_genres(genres)

    if len(genres_list) < 2:
        return {"error": "Not enough valid genres"}

    valid_embeddings = embeddings[valid_indices]

    # Get all genres and their frequencies
    all_genres_flat = [g for genre_list in genres_list for g in genre_list]
    genre_counts = Counter(all_genres_flat)
    top_genres = [g for g, _ in genre_counts.most_common(top_n_genres)]

    # Compute genre centroids
    genre_centroids = {}
    genre_indices = defaultdict(list)

    for idx, genre_list in enumerate(genres_list):
        for genre in genre_list:
            if genre in top_genres:
                genre_indices[genre].append(idx)

    for genre in top_genres:
        if len(genre_indices[genre]) > 0:
            genre_embeddings = valid_embeddings[genre_indices[genre]]
            genre_centroids[genre] = genre_embeddings.mean(axis=0)

    # Compute pairwise distances between genre centroids
    genre_names = list(genre_centroids.keys())
    n_genres = len(genre_names)
    centroid_distances = np.zeros((n_genres, n_genres))

    for i, genre1 in enumerate(genre_names):
        for j, genre2 in enumerate(genre_names):
            if i == j:
                centroid_distances[i, j] = 0.0
            else:
                # Cosine distance
                vec1 = genre_centroids[genre1]
                vec2 = genre_centroids[genre2]
                vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
                vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
                cosine_sim = np.dot(vec1_norm, vec2_norm)
                centroid_distances[i, j] = 1.0 - cosine_sim

    # Find closest genre pairs
    closest_pairs = []
    for i in range(n_genres):
        for j in range(i + 1, n_genres):
            closest_pairs.append(
                (genre_names[i], genre_names[j], centroid_distances[i, j])
            )
    closest_pairs.sort(key=lambda x: x[2])

    print(f"\nGenre Overlap Analysis (top {top_n_genres} genres):")
    print("  Closest genre pairs (most similar in embedding space):")
    for genre1, genre2, dist in closest_pairs[:5]:
        print(f"    {genre1} <-> {genre2}: {dist:.3f}")

    print("  Most distant genre pairs (most distinct):")
    for genre1, genre2, dist in closest_pairs[-5:]:
        print(f"    {genre1} <-> {genre2}: {dist:.3f}")

    return {
        "genre_centroids": genre_centroids,
        "centroid_distances": centroid_distances,
        "genre_names": genre_names,
        "closest_pairs": closest_pairs[:10],
        "most_distant_pairs": closest_pairs[-10:],
    }


def visualize_genre_clusters(
    embeddings: np.ndarray,
    genres: np.ndarray,
    n_clusters: int = 20,
    output_path: Optional[str] = None,
    random_state: int = 42,
) -> None:
    """
    Visualize genre clusters using PCA/TSNE.

    Creates 2D visualization showing:
    - How movies cluster in embedding space
    - How clusters align with genres
    - Genre separability

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        genres: Array of genre labels [n_samples]
        n_clusters: Number of clusters to create
        output_path: Path to save visualization (optional)
        random_state: Random seed
    """
    genres_list, valid_indices = _normalize_genres(genres)

    if len(genres_list) < 2:
        print("Not enough valid genres for visualization")
        return

    valid_embeddings = embeddings[valid_indices]
    primary_genres = np.array([g[0] if len(g) > 0 else "Unknown" for g in genres_list])

    # Reduce dimensionality for visualization
    print("\nReducing dimensionality for visualization...")

    # Use PCA first (faster)
    pca = PCA(n_components=50, random_state=random_state)
    embeddings_pca = pca.fit_transform(valid_embeddings)

    # Then use t-SNE for final 2D projection
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_pca)

    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_embeddings)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Colored by cluster
    scatter1 = axes[0].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap="tab20",
        alpha=0.6,
        s=10,
    )
    axes[0].set_title(f"Embeddings Clustered (k={n_clusters})", fontsize=14)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster ID")

    # Plot 2: Colored by primary genre (top genres only)
    top_genres = Counter(primary_genres).most_common(10)
    top_genre_set = set([g for g, _ in top_genres])

    # Create color mapping for top genres
    genre_to_color = {g: i for i, (g, _) in enumerate(top_genres)}
    genre_colors = np.array([genre_to_color.get(g, -1) for g in primary_genres])

    # Plot top genres
    for i, (genre, _) in enumerate(top_genres):
        mask = genre_colors == i
        axes[1].scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=genre, alpha=0.6, s=10
        )

    # Plot others
    others_mask = genre_colors == -1
    if others_mask.sum() > 0:
        axes[1].scatter(
            embeddings_2d[others_mask, 0],
            embeddings_2d[others_mask, 1],
            label="Other",
            alpha=0.3,
            s=5,
            c="gray",
        )

    axes[1].set_title("Embeddings by Primary Genre (Top 10)", fontsize=14)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def comprehensive_clustering_analysis(
    embeddings: np.ndarray,
    genres: np.ndarray,
    output_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict:
    """
    Run comprehensive clustering analysis for genre prediction diagnostics.

    This function runs all clustering analyses and provides insights for improving
    genre prediction performance.

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        genres: Array of genre labels [n_samples]
        output_dir: Directory to save visualizations (optional)
        random_state: Random seed

    Returns:
        Dictionary with all analysis results
    """
    print("\n" + "=" * 80)
    print("Comprehensive Clustering Analysis for Genre Prediction")
    print("=" * 80)

    results = {}

    # 1. Genre separability analysis
    print("\n" + "-" * 80)
    print("1. Genre Separability Analysis")
    print("-" * 80)
    separability_results = analyze_genre_separability(
        embeddings, genres, n_clusters_range=(5, 50), step=5, random_state=random_state
    )
    results["separability"] = separability_results

    # Find optimal number of clusters
    if separability_results and "error" not in separability_results:
        best_k = max(
            separability_results.keys(),
            key=lambda k: separability_results[k]["v_measure"],
        )
        print(f"\n  Optimal number of clusters (by V-measure): k={best_k}")
        results["optimal_k"] = best_k
    else:
        best_k = 20

    # 2. Cluster purity analysis
    print("\n" + "-" * 80)
    print("2. Cluster Purity Analysis")
    print("-" * 80)
    purity_results = analyze_genre_cluster_purity(
        embeddings, genres, n_clusters=best_k, random_state=random_state
    )
    results["purity"] = purity_results

    # 3. Genre overlap analysis
    print("\n" + "-" * 80)
    print("3. Genre Overlap Analysis")
    print("-" * 80)
    overlap_results = analyze_genre_overlap_in_embedding_space(
        embeddings, genres, top_n_genres=15
    )
    results["overlap"] = overlap_results

    # 4. Visualization
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 80)
        print("4. Creating Visualizations")
        print("-" * 80)
        viz_path = output_path / "genre_clusters_visualization.png"
        visualize_genre_clusters(
            embeddings,
            genres,
            n_clusters=best_k,
            output_path=str(viz_path),
            random_state=random_state,
        )
        results["visualization_path"] = str(viz_path)

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("Summary and Recommendations")
    print("=" * 80)

    if "purity" in results and "overall_purity" in results["purity"]:
        overall_purity = results["purity"]["overall_purity"]
        print(f"\nOverall cluster purity: {overall_purity:.3f}")

        if overall_purity < 0.3:
            print(
                "  WARNING: Very low cluster purity. Embeddings may not contain strong genre signals."
            )
            print("  Recommendations:")
            print("    - Consider using different embedding model or fine-tuning")
            print("    - Try feature engineering (e.g., add metadata features)")
            print("    - Consider hierarchical classification (coarse-to-fine genres)")
        elif overall_purity < 0.5:
            print(
                "  MODERATE: Moderate cluster purity. Some genre information present."
            )
            print("  Recommendations:")
            print("    - Filter rare genres to reduce noise")
            print("    - Use ensemble methods or more complex models")
            print("    - Consider genre hierarchy (group similar genres)")
        else:
            print(
                "  GOOD: High cluster purity. Embeddings contain strong genre signals."
            )
            print("  Recommendations:")
            print("    - Current approach should work well")
            print("    - Consider using cluster assignments as additional features")

    print("\n" + "=" * 80)

    return results


def main():
    """Main entry point for clustering analysis."""
    # Configuration
    DATA_DIR = str(BASE_DIR / "data" / "data_final")
    CHUNKING_SUFFIX = "_cls_token"
    START_YEAR = 1930
    END_YEAR = 2024
    OUTPUT_DIR = str(BASE_DIR / "outputs" / "clustering_analysis")
    RANDOM_STATE = 42

    print("\n" + "=" * 80)
    print("Clustering Analysis for Genre Prediction")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Chunking suffix: {CHUNKING_SUFFIX}")
    print(f"  Year range: {START_YEAR} to {END_YEAR}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Load embeddings and genres
    print("\nLoading embeddings and genres...")
    from predictions.predictor_logistic_regression import load_embeddings_and_genres

    embeddings, genres, movie_ids = load_embeddings_and_genres(
        data_dir=DATA_DIR,
        chunking_suffix=CHUNKING_SUFFIX,
        start_year=START_YEAR,
        end_year=END_YEAR,
        verbose=True,
    )

    if len(embeddings) == 0:
        print("ERROR: No embeddings loaded.")
        return

    print(f"\nLoaded {len(embeddings)} embeddings with shape {embeddings.shape}")

    # Run comprehensive analysis
    results = comprehensive_clustering_analysis(
        embeddings=embeddings,
        genres=genres,
        output_dir=OUTPUT_DIR,
        random_state=RANDOM_STATE,
    )

    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    main()
