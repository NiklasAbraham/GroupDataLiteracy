import numpy as np
import pandas as pd

from analysis.chunking.calculations import compute_simple_difference
from analysis.math_functions import calculate_average_cosine_distance_between_groups

def group_movies_by_columns(df: pd.DataFrame, columns_to_group: list, max_per_group: int = None) -> pd.DataFrame:
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

    grouped_embeddings = df.groupby(columns_to_group)['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index(name="avg_embedding")

    # Create a column for the num of movies in each group (debugging purposes)
    result = grouped_embeddings.merge(group_sizes, on=columns_to_group, how='left')

    for index, row in result.iterrows():
        print(f"genre {row['new_genre']}: {row['year_group']}: {row['group_size']}")
    return result

def create_yearly_grouping_column(df: pd.DataFrame, n_years_per_group: int) -> pd.DataFrame:
    """
    Group movies by n_years creating a new "year_group" column. For example movies between 1950-1955 are grouped in group year 1950 if n_years_per_group = 5.
    """
    df["year_group"] = (df["year"] // n_years_per_group) * n_years_per_group
    return df

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

    # 2. Calculate the change
    df[change_col_name] = df.apply(
        lambda x: change_func(np.array(x[value_col]), np.array(x[f'next_{value_col}'])),
        axis=1
    )

    # Drop the temporary 'next_' column
    df = df.drop(columns=[f'next_{value_col}'])

    return df

def calculate_movies_drift_velocity(df: pd.DataFrame):
    df_with_velocity = calculate_rate_of_change(
        df=df,
        value_col="avg_embedding",
        change_col_name="drift_velocity",
        change_func=calculate_average_cosine_distance_between_groups
    )
    return df_with_velocity

def calculate_movies_drift_acceleration(df: pd.DataFrame):
    df_with_acceleration = calculate_rate_of_change(
        df=df,
        value_col="drift_velocity",
        change_col_name="drift_acceleration",
        change_func=compute_simple_difference
    )
    return df_with_acceleration

def calculate_cumulative_change_of_column(df: pd.DataFrame, column_to_cum: str, columns_to_group_by: list[str]) -> pd.DataFrame:
    df[f"{column_to_cum}_cum"] = df.groupby(columns_to_group_by)[column_to_cum].cumsum()
    return df

def calculate_drift_metrics(df: pd.DataFrame, n_years_per_group: int = 1, max_movies_per_group: int = None):
    """
    Groups by genre and a specified number of years (n_years_per_group).
    Calculates change Velocity, Acceleration, and cumulative Change of velocity.
    """

    df = create_yearly_grouping_column(df, n_years_per_group)
    df = group_movies_by_columns(df, ["year_group", "new_genre"], max_per_group=max_movies_per_group)
    df = df.sort_values(by=['new_genre', 'year_group'])

    # Velocity -> Calculate embedding change in group per year
    df = calculate_movies_drift_velocity(df=df)

    # Acceleration -> Calculate velocity change in group per year
    df = calculate_movies_drift_acceleration(df=df)

    # 4. Cumulative change of velocity
    df = calculate_cumulative_change_of_column(df=df, column_to_cum="drift_velocity", columns_to_group_by=["new_genre"])

    return df
