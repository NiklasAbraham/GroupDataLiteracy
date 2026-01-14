from typing import Callable

import numpy as np
import pandas as pd


def get_bootstrapped_embeddings(n_bootstrap_samples: int, embeddings_series: pd.Series, grouping_function: Callable):
    """
    Takes an embedding series and returns the bootstrapped embeddings
    It is a series to call it in the .apply function

    :param grouping_function: how to mean embeddings -> average or medoid
    """
    all_embeddings = np.vstack(embeddings_series.values)
    num_movies = all_embeddings.shape[0]

    bootstrapped_results = []
    for _ in range(n_bootstrap_samples):
        # Create a bootstrap sample (resample with replacement)
        indices = np.random.choice(num_movies, num_movies, replace=True)
        sample_embeddings = all_embeddings[indices]

        # Calculate the representative embedding for the sample
        bootstrapped_embedding = grouping_function(sample_embeddings)
        bootstrapped_results.append(bootstrapped_embedding)

    return np.array(bootstrapped_results)