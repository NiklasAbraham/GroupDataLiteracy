# Plot Twists Over Time: How Movie Stories Have Changed Over 95 Years

Group 3 project for ML4102 Data Literacy at the University of Tübingen.

## Abstract

We analyze semantic evolution in cinema by embedding movie plot summaries from 1930 to 2024 into a unified semantic space. Using distance distributions, novelty scores, and statistical tests, we quantify how genres and thematic clusters shift over time. Our analysis reveals periods of semantic stability and reorganization, providing quantitative measures of cultural change in narrative structures across nearly a century of cinema.

## Project Workflow

The project follows a clear data flow through the folder structure:

### 1. Data Collection (`src/aaa_data_pipline/`)

**Flow**: `001_data_pipeline.py` → `api/` → `embedding/` → `data/data_final/`

- **`001_data_pipeline.py`**: Main orchestrator that runs the complete pipeline
- **`api/`**: External data sources
  - `wikidata_handler.py`: Fetches movie metadata (1930-2024)
  - `moviedb_handler.py`: Enriches with TMDb ratings and popularity
  - `wikipedia_handler.py`: Retrieves plot summaries
- **`003_imdb_ratings_addiction.py`**: Adds IMDb ratings data
- **`004_data_cleaning.py`**: Filters and cleans the dataset (entropy-based filtering, removes non-features, etc.)
- **`embedding/`**: Generates BGE-M3 embeddings
  - `embedding.py`: EmbeddingService for parallel GPU encoding
  - `models/`: Strategy pattern for different embedding models

**Output**: Raw data → `data/data_final/wikidata_movies_YYYY.csv` and `movie_embeddings_YYYY_*.npy`

### 2. Data Processing (`src/utils/`)

**Flow**: `data_utils.py` → Consolidated files in `data/data_final/`

- **`data_utils.py`**: Utilities for loading and processing
  - `load_final_data_with_embeddings()`: Primary function to load complete dataset
  - `cluster_genres()`: Genre normalization and clustering
  - Consolidates per-year files into `final_dataset.csv` and `final_dense_embeddings.npy`

**Output**: Consolidated dataset ready for analysis

### 3. Analysis (`src/aab_analysis/`)

**Flow**: Load data → Run analyses → Generate visualizations

- **`genre_classifier.py`**: Clusters 975 raw genres into 20 coherent categories
- **`genre_drift_utils.py`**: Analyzes genre evolution over time
- **`stats_data.py`**: Statistical analysis (distance distributions, KS tests)
- **`gaussian_fit.py`**: Gaussian distribution fitting for distance analysis
- **`projection_analysis.ipynb`**: PCA analysis with concept space projection
- **`concept_extraction.py`**: Maps plot nouns to semantic concept space
- **`chunking/`**: Experimental framework comparing embedding aggregation methods

**Output**: Analysis results and visualizations in `figures/`

### 4. Concept Space (`data/concept_space/`)

- Pre-computed WordNet-based concept vocabulary (~20,000 nouns)
- Used for semantic mapping in concept extraction and PCA analysis

## Project Structure

```
GroupDataLiteracy/
├── data/
│   ├── data_final/                # Processed data (per-year + consolidated)
│   │   ├── final_dataset.csv      # Consolidated movie dataset
│   │   ├── final_dense_embeddings.npy
│   │   ├── wikidata_movies_YYYY.csv
│   │   └── movie_embeddings_YYYY_*.npy
│   └── concept_space/             # Concept vocabulary for semantic mapping
│
├── src/
│   ├── aaa_data_pipline/          # STEP 1: Data collection
│   │   ├── 001_data_pipeline.py   # Main pipeline
│   │   ├── 002_cutoff_distance_method.py
│   │   ├── 003_imdb_ratings_addiction.py
│   │   ├── 004_data_cleaning.py
│   │   ├── api/                   # External APIs
│   │   └── embedding/             # Embedding generation
│   │
│   ├── utils/                     # STEP 2: Data utilities
│   │   └── data_utils.py          # Loading and consolidation
│   │
│   └── aab_analysis/              # STEP 3: Analysis
│       ├── genre_classifier.py
│       ├── genre_drift_utils.py
│       ├── stats_data.py
│       ├── gaussian_fit.py
│       ├── projection_analysis.ipynb
│       ├── concept_extraction.py
│       └── chunking/               # Embedding experiments
│
├── report/                        # LaTeX report
├── figures/                       # Analysis visualizations
└── tests/                         # Test suite
```

## Environment Setup

```bash
# Create conda environment
conda create -n dataLiteracy python=3.11
conda activate dataLiteracy

# Install dependencies
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate dataLiteracy
```

## Quick Start

### 1. Run Data Pipeline

```bash
python src/aaa_data_pipline/001_data_pipeline.py
```

This collects data from Wikidata, TMDb, Wikipedia, generates embeddings, and saves to `data/data_final/`.

### 2. Load Data for Analysis

```python
from src.utils.data_utils import load_final_data_with_embeddings

df = load_final_data_with_embeddings()
# df contains movies with 'embedding' column and metadata
```

### 3. Run Analyses

- **Genre classification**: `python src/aab_analysis/genre_classifier.py`
- **Statistical analysis**: Use functions in `stats_data.py` and `gaussian_fit.py`
- **PCA analysis**: Open `projection_analysis.ipynb`
- **Concept extraction**: Use `concept_extraction.py` or `chunking/test_single_chunking.py`

## Key Methods

- **Novelty Score**: Minimal cosine distance to all prior movies (Faiss-based)
- **Distance Analysis**: Pairwise cosine distances (μ=0.5195, σ=0.0624)
- **KS Test**: Compare distance distributions across decades/genres
- **Spread Metrics**: L2 norm, Frobenius norm, Spectral norm per year
- **Genre Clustering**: 975 raw genres → 20 categories via k-means on embeddings

## Dataset

Final dataset: **161,533 movies** (1930-2024) with 81% average coverage across actors, directors, genres, and years.

## License

- Wikidata: CC0 1.0 (public domain)
- Wikipedia: CC BY-SA 4.0
- TMDb: CC BY-NC 4.0 (non-commercial research)
