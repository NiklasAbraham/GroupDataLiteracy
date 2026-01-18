from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))
from src.utils.data_utils import *

data_path = os.path.join(base_path, "data", "data_final")
csv_path = os.path.join(data_path, "final_dataset.csv")
df = load_final_data_with_embeddings(csv_path, data_path, verbose=False)
embeddings = np.stack(df.embedding.tolist())
pairwise_cos_distance = cdist(embeddings, embeddings, metric="cosine")
with open("pairwise_cos_dist.npy", "wb") as f:
    np.save(f, pairwise_cos_distance)
