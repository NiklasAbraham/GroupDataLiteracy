from abc import ABC
from typing import Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


class DriftNormalizationStrategy(ABC):
    """
    Abstract base class for drift normalization strategies.
    """

    def __init__(self):
        self.sigma_per_dim: Optional[np.ndarray] = None
        self.weighted_mean_per_dim: Optional[np.ndarray] = None

    def apply_preprocessing(self, df: pd.DataFrame, columns_to_group: list, random_seed: int = 42) -> pd.DataFrame:
        """
        Modify dataframe before grouping (e.g., downsampling).
        Also estimates sigma and weighted mean per dimension.
        """
        self.sigma_per_dim, self.weighted_mean_per_dim = self._estimate_pooled_sigma_and_mean_per_dim(df, columns_to_group)
        return df

    def _estimate_pooled_sigma_and_mean_per_dim(self, df: pd.DataFrame, columns_to_group: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate pooled standard deviation and weighted mean for each embedding dimension.

        Returns:
            sigma_per_dim: Pooled std per dimension (from within-group variance)
            weighted_mean_per_dim: Weighted mean per dimension (weighted by group size)
        """
        all_embeddings = np.vstack(df['embedding'].values).astype(np.float64)
        n_dims = all_embeddings.shape[1]

        weighted_var_sum = np.zeros(n_dims)
        df_sum = 0
        weighted_sum = np.zeros(n_dims)
        total_n = 0

        for _, group_df in df.groupby(columns_to_group):
            embeddings = np.vstack(group_df['embedding'].values).astype(np.float64)
            n = len(embeddings)

            group_mean = embeddings.mean(axis=0)
            weighted_sum += n * group_mean
            total_n += n

            if n < 2:
                continue

            group_var = embeddings.var(axis=0, ddof=1)
            weighted_var_sum += (n - 1) * group_var
            df_sum += (n - 1)

        if df_sum == 0:
            raise ValueError("Not enough data to estimate sigma per dimension")

        pooled_sigma = np.sqrt(weighted_var_sum / df_sum)
        pooled_sigma[pooled_sigma == 0] = 1.0
        weighted_mean = weighted_sum / total_n

        return pooled_sigma, weighted_mean

    def normalize_embedding_values(self, df: pd.DataFrame, embedding_col: str,
                                   dim_index: int) -> pd.DataFrame:
        """
        Standard z-score normalization: (value - mean) / sigma
        Does NOT account for sampling variance (group size).
        """
        if self.sigma_per_dim is None or self.weighted_mean_per_dim is None:
            raise ValueError("Must call apply_preprocessing first")

        df = df.copy()
        df['dim_value'] = df[embedding_col].apply(lambda emb: emb[dim_index])

        yearly_stats = df.groupby('year_group').agg(
            mean_value=('dim_value', 'mean'),
            total_n=('group_size', 'sum')
        ).reset_index()

        # Simple z-score: (value - mean) / sigma
        sigma_dim = self.sigma_per_dim[dim_index]
        global_mean = self.weighted_mean_per_dim[dim_index]

        yearly_stats['dim_value_zscore'] = (yearly_stats['mean_value'] - global_mean) / sigma_dim

        return yearly_stats

    def normalize_drift(self,
                        drift_values: np.ndarray,
                        n_t: tuple,
                        n_t_plus_1: tuple) -> np.ndarray:
        """
        Normalize drift values after calculation.
        Default: no-op, returns values unchanged.
        """
        return drift_values


class NoNormalization(DriftNormalizationStrategy):
    """Standard z-score normalization (inherited from base)."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "No normalization"


class MinGroupSizeDownsampling(DriftNormalizationStrategy):
    """Downsamples all groups to match the smallest group size."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Downsampling"

    def apply_preprocessing(self, df: pd.DataFrame, columns_to_group: list, random_seed: int = 42) -> pd.DataFrame:
        group_sizes = df.groupby(columns_to_group).size()
        min_size = group_sizes.min()
        print(f"[MinGroupSizeDownsampling] Downsampling all groups to {min_size} samples")

        # Keep only min_size random samples per group
        df = (
            df.groupby(columns_to_group)
            .apply(lambda x: x.sample(n=min(len(x), min_size), random_state=random_seed))
            .reset_index(drop=True)
        )

        # Call parent method to estimate sigma and mean
        self.sigma_per_dim, self.weighted_mean_per_dim = self._estimate_pooled_sigma_and_mean_per_dim(df, columns_to_group)

        return df


class ZScoreNormalization(DriftNormalizationStrategy):
    """
    Normalizes accounting for sampling variance.
    For embeddings: z = (value - weighted_mean) / (sigma / sqrt(n))
    For drift: z = drift / (sigma * sqrt(1/n_t + 1/n_{t+1}))
    """

    def __init__(self):
        super().__init__()
        self.sigma: Optional[float] = None  # For drift (cosine distance)

    def __str__(self):
        return "Sampling variance normalization"

    def apply_preprocessing(self, df: pd.DataFrame, columns_to_group: list, random_seed: int = 42) -> pd.DataFrame:
        self.sigma = self._estimate_pooled_sigma(df, columns_to_group)
        self.sigma_per_dim, self.weighted_mean_per_dim = self._estimate_pooled_sigma_and_mean_per_dim(df, columns_to_group)
        print(f"[ZScoreNormalization] Estimated pooled sigma (cosine): {self.sigma:.6f}")
        return df

    def _estimate_pooled_sigma(self, df: pd.DataFrame, columns_to_group: list) -> float:
        """
        Estimate pooled sigma for cosine distance (used for drift normalization).
        """
        weighted_var_sum = 0.0
        df_sum = 0

        for _, group_df in df.groupby(columns_to_group):
            embeddings = np.vstack(group_df['embedding'].values).astype(np.float64)
            n = len(embeddings)

            if n < 2:
                continue

            group_mean = embeddings.mean(axis=0)
            group_mean_normalized = group_mean / np.linalg.norm(group_mean)
            cosine_distances = np.array([cosine(e, group_mean_normalized) for e in embeddings])
            group_var = np.mean(cosine_distances ** 2)

            weighted_var_sum += (n - 1) * group_var
            df_sum += (n - 1)

        if df_sum == 0:
            raise ValueError("Not enough data to estimate sigma")

        return np.sqrt(weighted_var_sum / df_sum)

    def normalize_drift(self, drift_values: np.ndarray,
                        n_t: np.ndarray,
                        n_t_plus_1: np.ndarray) -> np.ndarray:
        """
        Normalizes drift values by standard error.

        Args:
            drift_values: Array of arrays, each containing N bootstrap samples
            n_t: Array of group sizes at time t
            n_t_plus_1: Array of group sizes at time t+1

        Returns:
            Array of arrays with normalized drift values
        """
        if self.sigma is None or self.sigma == 0:
            return drift_values

        n_t = np.asarray(n_t, dtype=float)
        n_t_plus_1 = np.asarray(n_t_plus_1, dtype=float)

        se_drift = self.sigma * np.sqrt(1 / n_t + 1 / n_t_plus_1)

        result = []
        for dv, se in zip(drift_values, se_drift):
            dv = np.atleast_1d(dv)
            if np.isnan(se) or se == 0:
                result.append(dv)
            else:
                result.append(dv / se)

        return np.array(result, dtype=object)

    def normalize_embedding_values(self, df: pd.DataFrame, embedding_col: str,
                                   dim_index: int) -> pd.DataFrame:
        """
        Z-score normalization accounting for sampling variance.
        z = (value - weighted_mean) / (sigma / sqrt(n))

        Args:
            df: DataFrame with embedding column and 'group_size' column
            embedding_col: Name of column containing embeddings
            dim_index: Which embedding dimension to normalize

        Returns:
            DataFrame with columns: year_group, mean_value, total_n, dim_value_zscore
        """
        if self.sigma_per_dim is None or self.weighted_mean_per_dim is None:
            raise ValueError("Must call apply_preprocessing first")

        df = df.copy()
        df['dim_value'] = df[embedding_col].apply(lambda emb: emb[dim_index])

        yearly_stats = df.groupby('year_group').agg(
            mean_value=('dim_value', 'mean'),
            total_n=('group_size', 'sum')
        ).reset_index()

        # Z-score with SE: (value - mean) / (sigma / sqrt(n))
        sigma_dim = self.sigma_per_dim[dim_index]
        global_mean = self.weighted_mean_per_dim[dim_index]
        se = sigma_dim / np.sqrt(yearly_stats['total_n'].values)

        yearly_stats['dim_value_zscore'] = (yearly_stats['mean_value'] - global_mean) / se

        return yearly_stats