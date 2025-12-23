import plt

from analysis.normalization_strategy import DriftNormalizationStrategy, NoNormalization
from typing import Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.chunking.calculations import compute_scalar_difference, \
    compute_vector_difference
from analysis.math_functions import calculate_average_cosine_distance_between_groups
from analysis.math_functions.boostrapping_embeddings import get_bootstrapped_embeddings
from analysis.math_functions.cosine_distance_util import get_average_embedding

def create_yearly_grouping_column(df: pd.DataFrame, n_years_per_group: int) -> pd.DataFrame:
    """
    Group movies by n_years creating a new "year_group" column. For example movies between 1950-1955 are grouped in group year 1950 if n_years_per_group = 5.
    """
    df["year_group"] = (df["year"] // n_years_per_group) * n_years_per_group
    return df

def group_movies_by_columns(df: pd.DataFrame, columns_to_group: list, max_per_group: int = None,
                            grouping_function: Callable[[Any], Any] = get_average_embedding,
                            n_boostrap_samples: int = None, verbose: bool = False) -> pd.DataFrame:
    """
    Groups movies by columns, calculating average embeddings per group and storing it in 'avg_embedding' column
    It reduces each movie group to one row

    Args:
        columns_to_group: with respect to which column names to group
        max_per_group: fixed max size for each group
    """
    # Count num of movies per group and cap it to the max
    group_sizes = df.groupby(columns_to_group).size().reset_index(name='group_size')
    if max_per_group:
        group_sizes['group_size'] = group_sizes['group_size'].clip(upper=max_per_group)

    # Cap the groups to max size
    if max_per_group is not None:
        df = (
            df.groupby(columns_to_group)
            .head(max_per_group)
            .reset_index(drop=True)
        )

    if n_boostrap_samples:
        # If bootstrapping is requested, define the function to return N embeddings
        aggregation_func = lambda x: get_bootstrapped_embeddings(
            embeddings_series=x,
            n_bootstrap_samples=n_boostrap_samples,
            grouping_function=grouping_function
        )
    else:
        aggregation_func = lambda x: grouping_function(np.vstack(x))

    grouped_embeddings = df.groupby(columns_to_group)['embedding'].apply(
        lambda x: aggregation_func(x)
    ).reset_index(name="avg_embedding")

    # Create a column for the num of movies in each group (debugging purposes)
    result = grouped_embeddings.merge(group_sizes, on=columns_to_group, how='left')

    if verbose:
        for index, row in result.iterrows():
            print(f"genre {row['new_genre']}: {row['year_group']}: {row['group_size']}")
    return result


def create_yearly_grouping_column(df: pd.DataFrame, n_years_per_group: int) -> pd.DataFrame:
    """
    Group movies by n_years creating a new "year_group" column. For example movies between 1950-1955 are grouped in group year 1950 if n_years_per_group = 5.
    """
    df["year_group"] = (df["year"] // n_years_per_group) * n_years_per_group
    return df

def calculate_bootstrapped_change(prev_embeddings: np.ndarray, next_embeddings: np.ndarray,
                                  change_func: Callable) -> np.ndarray:
    """
    Calculates the change (drift/acceleration) for N pairs of bootstrapped values.

    Args:
        prev_embeddings: N embeddings for the previous time period (2D array: [n_samples, n_dims]).
        next_embeddings: N embeddings for the next time period (2D array: [n_samples, n_dims]).
        change_func: The function (e.g., cosine distance, scalar difference) to apply to each pair.

    Returns:
        A 1D array of N change metrics.
    """
    if not isinstance(next_embeddings, np.ndarray) or next_embeddings.ndim == 0:
        return np.array([np.nan])

    # If the inputs are single embeddings (1D), make them N=1 arrays (2D)
    if prev_embeddings.ndim == 1:
        prev_embeddings = prev_embeddings[np.newaxis, :]
        next_embeddings = next_embeddings[np.newaxis, :]

    n_samples = prev_embeddings.shape[0]
    change_values = []
    # Apply the change function to each pair
    for i in range(n_samples):
        result = change_func(prev_embeddings[i], next_embeddings[i])
        # Extract scalar if result is an array with single element
        if isinstance(result, np.ndarray) and result.size == 1:
            result = result.item()
        change_values.append(result)

    return np.array(change_values)

def calculate_rate_of_change(df: pd.DataFrame, value_col: str, change_col_name: str,
                             change_func) -> pd.DataFrame:
    """
    Calculates the change (drift/acceleration) for a column, grouped by a set of columns, for each specified time period.

    So for genre drift velocity: calculate change of avg_embeddings, grouped by genres and time period. The avg_embeddings and time periods need to be calculated beforehand.

    Returns:
        The DataFrame with a new column for the calculated change.
    """
    # 1. Shift the value column within each group -> important group by genre, else the last next_value of one group will have the first of the next genre instead of None, It should have None
    df[f'next_{value_col}'] = df.groupby("new_genre")[value_col].shift(-1)

    tqdm.pandas(desc=f"Calculating {change_col_name}")
    df[change_col_name] = df.progress_apply(
        lambda x: calculate_bootstrapped_change(
            prev_embeddings=x[value_col],
            next_embeddings=x[f'next_{value_col}'],
            change_func=change_func
        ),
        axis=1
    )

    # Drop the temporary 'next_' column
    df = df.drop(columns=[f'next_{value_col}'])

    return df

def calculate_movies_drift_velocity(df: pd.DataFrame, normalization_strategy: DriftNormalizationStrategy):
    """
    For each year group, how much on average (measured with cosine similarity) did it change with respect to the previous year group
    """
    # First calculate raw drift velocity
    df = calculate_rate_of_change(
        df=df,
        value_col="avg_embedding",
        change_col_name="drift_velocity",
        change_func=calculate_average_cosine_distance_between_groups
    )

    # Add next_group_size for normalization
    df['next_group_size'] = df.groupby('new_genre')['group_size'].shift(-1)

    # Normalize drift
    df['drift_velocity'] = normalization_strategy.normalize_drift(
        drift_values=df['drift_velocity'].values,
        n_t=df['group_size'].values,
        n_t_plus_1=df['next_group_size'].values
    )

    # Clean up
    df = df.drop(columns=['next_group_size'])

    return df

def calculate_movies_drift_acceleration(df: pd.DataFrame):
    df_with_acceleration = calculate_rate_of_change(
        df=df,
        value_col="drift_velocity",
        change_col_name="drift_acceleration",
        change_func=compute_scalar_difference
    )
    return df_with_acceleration

def calculate_movies_drift(df: pd.DataFrame):
    """
    For each year group, of much on average (measured with a vector that averages the drift in each dimension) did it change with respect to the previous year group
    """
    df_with_drift = calculate_rate_of_change(
        df=df,
        value_col="avg_embedding",
        change_col_name="drift",
        change_func=compute_vector_difference
    )
    return df_with_drift

def cumsum_arrays(series):
    # Filter out NaN arrays (from last time period in each group)
    valid_arrays = [arr for arr in series.values if not (len(arr) == 1 and np.isnan(arr[0]))]

    # Stack all arrays in the series into a 2D array
    # Each row is a time point, each column is a bootstrap sample
    stacked = np.stack(valid_arrays)

    # Compute cumulative sum along time axis (axis=0)
    cumsum_result = np.cumsum(stacked, axis=0)

    result = [cumsum_result[i] for i in range(len(cumsum_result))]
    # Add NaN for the last time period
    result.append(np.array([np.nan]))

    return result

def calculate_cumulative_change_of_column(df: pd.DataFrame, column_to_cum: str,
                                          columns_to_group_by: list[str]) -> pd.DataFrame:
    """
    Calculate cumulative sum for a column containing arrays (bootstrap samples).
    Handles both scalar arrays (like drift_velocity) and vector arrays properly.
    """
    df[f"{column_to_cum}_cum"] = (
        df.groupby(columns_to_group_by)[column_to_cum]
        .transform(cumsum_arrays)
    )
    return df

def calculate_drift_metrics(df: pd.DataFrame, n_years_per_group: int = 1, max_movies_per_group: int = None,
                            grouping_function: Callable[[Any], Any] = get_average_embedding, n_boostrap_samples: int = None,
                            normalization_strategy: DriftNormalizationStrategy = NoNormalization()):
    """
    Groups by genre and a specified number of years (n_years_per_group).
    Calculates change Velocity, Acceleration, and cumulative Change of velocity.
    """
    random_seed = 42

    df = create_yearly_grouping_column(df, n_years_per_group)
    columns_to_group = ["new_genre", "year_group"]
    df = normalization_strategy.apply_preprocessing(df, columns_to_group=columns_to_group, random_seed=random_seed)

    df = group_movies_by_columns(df=df, columns_to_group=columns_to_group, grouping_function=grouping_function, n_boostrap_samples=n_boostrap_samples)
    df = df.sort_values(by=columns_to_group)
    # Downsize to group year with least amounts of movies

    # Velocity -> Calculate embedding change in group per year
    df = calculate_movies_drift_velocity(df=df, normalization_strategy=normalization_strategy)

    # Drift -> how much do embeddings change on average per year (velocity but for each dimension)
    calculate_movies_drift(df=df)

    # Acceleration -> Calculate velocity change in group per year
    #df = calculate_movies_drift_acceleration(df=df)

    # Cumulative change of velocity
    #df = calculate_cumulative_change_of_column(df=df, column_to_cum="drift_velocity", columns_to_group_by=["new_genre"])

    return df


def plot_dimension_averaged_across_genres(df: pd.DataFrame, embedding_col: str, dim_index: int,
                                          normalization_strategy: DriftNormalizationStrategy):
    """Average the embedding dimension across all genres for each year."""

    # Apply preprocessing
    df = normalization_strategy.apply_preprocessing(
        df=df,
        columns_to_group=["new_genre", "year_group"],
        random_seed=42
    )

    columns_to_group = ["new_genre", "year_group"]
    df = group_movies_by_columns(df=df, columns_to_group=columns_to_group, )

    # Normalize and get yearly stats
    yearly_stats = normalization_strategy.normalize_embedding_values(
        df=df,
        embedding_col=embedding_col,
        dim_index=dim_index
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_stats['year_group'], yearly_stats['dim_value_zscore'], marker='o')
    plt.xlabel('Year')
    plt.ylabel(f'Z-score of Embedding[{dim_index}]')
    plt.title(f'Embedding Dimension {dim_index} Averaged Across All Genres with {normalization_strategy.__str__()}')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.show()