import importlib.util
import os
import sys
from pathlib import Path

base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

from src.utils.data_utils import load_movie_data  # noqa: E402

# Import clean_dataset from file with numeric prefix
data_cleaning_path = base_path / "aaa_data_pipline" / "004_data_cleaning.py"
spec = importlib.util.spec_from_file_location("data_cleaning", data_cleaning_path)
data_cleaning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaning)
clean_dataset = data_cleaning.clean_dataset

movie_df = load_movie_data(os.path.join(base_path, "data", "data_final"))
cleaned_movie_df = clean_dataset(movie_df)
print(movie_df.shape)
print(cleaned_movie_df.shape)
cleaned_movie_df.to_csv(
    os.path.join(base_path, "data", "data_final", "cleaned_movie_data.csv"), index=False
)
