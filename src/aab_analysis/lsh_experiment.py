import os
import sys
from pathlib import Path

base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils.data_utils import *
from data_cleaning import clean_dataset

movie_df = load_movie_data(os.path.join(base_path, "data", "data_final"))
cleaned_movie_df = clean_dataset(movie_df)
print(movie_df.shape)
print(cleaned_movie_df.shape)
cleaned_movie_df.to_csv(
    os.path.join(base_path, "data", "data_final", "cleaned_movie_data.csv"), index=False
)
