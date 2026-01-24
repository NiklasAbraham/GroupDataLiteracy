"""
Whitening and debiasing functions for embeddings.

This module provides functions to transform embeddings to restore isotropy,
remove directional bias, and improve geometric properties.
"""

import numpy as np
from sklearn.decomposition import PCA


def mean_center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Mean-center embeddings by subtracting the global mean.

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])

    Returns:
    - Mean-centered embeddings (shape: [n_samples, embedding_dim])
    """
    mean = np.mean(embeddings, axis=0)
    return embeddings - mean


def whiten_embeddings(
    embeddings: np.ndarray, n_components: int = None, normalize: bool = True
) -> np.ndarray:
    """
    Whiten embeddings using PCA to restore isotropy.

    Whitening removes correlations and scales variance to 1 in all directions.
    This transforms the data so that the covariance matrix becomes identity,
    effectively making the distribution isotropic (uniform in all directions).

    Mathematical formulation:
    1. Mean-center: x ← x - μ
    2. Compute PCA: X_centered = UΣV^T
    3. Whiten: x_whitened = U / √(eigenvalues)
    4. Optionally normalize: x' = x_whitened / ||x_whitened||

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
    - n_components: Number of PCA components to keep. If None, keeps all.
    - normalize: If True, re-normalize embeddings to unit length after whitening

    Returns:
    - Whitened embeddings (shape: [n_samples, n_components or embedding_dim])
    """
    n_samples, n_dims = embeddings.shape

    centered = mean_center_embeddings(embeddings)

    if n_components is None:
        n_components = min(n_samples, n_dims)
    else:
        n_components = min(n_components, n_samples, n_dims)

    pca = PCA(n_components=n_components, whiten=True)
    whitened = pca.fit_transform(centered)

    if normalize:
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        whitened = whitened / norms

    return whitened


def debias_embeddings(
    embeddings: np.ndarray, k: int = 3, normalize: bool = False
) -> np.ndarray:
    """
    De-bias embeddings using the "All-but-the-top" approach.

    This removes global anisotropy/cone by projecting out the top k principal
    components, while preserving relative covariance and mean differences that
    might encode real temporal or semantic structure.

    Steps:
    1. Fit PCA on the full embedding set
    2. Compute global mean μ and top k PCs u1,...,uk
    3. For each embedding x:
       - Mean-center: x ← x - μ
       - Project out dominant directions: x' = x - Σ⟨x, u_i⟩u_i
       - Optionally re-normalize: x' ← x' / ||x'||

    Mathematical formulation:
    x' = x - μ - Σ_{i=1}^k ⟨x - μ, u_i⟩ u_i

    Parameters:
    - embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
    - k: Number of top principal components to remove (default: 3, typically 1-5)
    - normalize: If True, re-normalize embeddings to unit length after debiasing

    Returns:
    - De-biased embeddings (shape: [n_samples, embedding_dim])
    """
    n_samples, n_dims = embeddings.shape

    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    pca = PCA(n_components=min(k, n_dims, n_samples - 1))
    pca.fit(centered)

    top_pcs = pca.components_[:k]

    debiased = centered.copy()
    for i in range(k):
        projections = np.dot(centered, top_pcs[i])
        debiased = debiased - np.outer(projections, top_pcs[i])

    if normalize:
        norms = np.linalg.norm(debiased, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        debiased = debiased / norms

    return debiased
