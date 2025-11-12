# GroupDataLiteracy
Welcome to Group 3 of ML4102 Data Literacy. 

This repo serves to document our project implementation, data engineering as well as exploration.

## Movies
We are using the Movies dataset

### Idea List for Datasets

## Project Structure

The project is organized as follows:

```
GroupDataLiteracy/
├── data/                          # Data storage directory
│   ├── mock/                      # Mock/test datasets
│   │   ├── mock_movies_100.csv   # Sample movie dataset (100 entries)
│   │   └── wikidata_movies.csv   # Movies data from Wikidata
│   ├── wikidata_movies_YYYY.csv  # Movie data files per year (1950-2024)
│   ├── movie_embeddings_YYYY.npy # Generated embeddings per year
│   ├── movie_ids_YYYY.npy        # Movie IDs corresponding to embeddings
│   ├── data_statistics.csv       # Statistical analysis of the dataset
│   └── *.png                     # Visualization outputs
│
├── src/                           # Source code directory
│   ├── analysis/                  # Analysis notebooks and scripts
│   │   ├── verify_test.ipynb     # Verification and testing notebooks
│   │   ├── verify_csv.ipynb      # CSV data verification
│   │   ├── stats_analysis.ipynb  # Statistical analysis
│   │   ├── embedding_vs_feature_per_year.ipynb  # Embedding analysis
│   │   ├── stats_data.py         # Statistical data utilities
│   │   └── tokenizer.py          # Tokenization utilities
│   ├── embedding/                 # Embedding generation module
│   │   ├── embedding.py          # EmbeddingService class for parallel GPU encoding
│   │   ├── util_embeddings.py    # Utility functions for GPU verification and embedding validation
│   │   ├── load_embeddings.py    # Utilities for loading embeddings
│   │   └── models/               # Strategy pattern for embedding models
│   │       ├── base_strategy.py  # Base strategy interface
│   │       ├── sentence_transformer_strategy.py  # SentenceTransformer implementation
│   │       ├── flag_embedding_strategy.py  # Flag embedding implementation
│   │       └── strategy_factory.py  # Factory for creating strategies
│   ├── api/                       # API handlers for external data sources
│   │   ├── wikidata_handler.py   # Wikidata data collection
│   │   ├── moviedb_handler.py    # TMDb API integration for movie metadata
│   │   └── wikipedia_handler.py # Wikipedia plot retrieval
│   ├── data_pipeline.py          # Main data processing pipeline orchestrator
│   ├── data_utils.py             # Utility methods for data loading
│   └── checks.ipynb              # Data quality checks
|   └── data_utils.py             # Utility methods for data loading
│
├── report/                       # LaTeX report files
│   ├── report_template.tex       # Main LaTeX template
│   ├── bibliography.bib         # Bibliography references
│   ├── icml2025.sty             # ICML 2025 style file
│   └── icml2025.bst             # Bibliography style
│
├── .gitignore                    # Git ignore rules
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip requirements
├── LICENSE                       # Project license
└── README.md                     # This file
```

### Folder Structure Rationale

The project follows a clear separation of concerns:

- **`data/`**: Contains all data files, organized by year:
  - `mock/`: Sample datasets for testing and development
  - `wikidata_movies_YYYY.csv`: Movie metadata files organized by year (1950-2024)
  - `movie_embeddings_YYYY.npy`: Generated embeddings per year
  - `movie_ids_YYYY.npy`: Movie IDs corresponding to embeddings (for indexing)
  - Visualization outputs and statistical summaries

- **`src/`**: Contains all source code organized by functionality:
  - **`api/`**: External API integrations
    - `wikidata_handler.py`: Fetches movie data from Wikidata by year
    - `moviedb_handler.py`: Enriches data with TMDb popularity, votes, and ratings
    - `wikipedia_handler.py`: Retrieves movie plots from Wikipedia via sitelinks
  - **`embedding/`**: Core embedding generation module
    - `embedding.py`: Defines the `EmbeddingService` class that handles SentenceTransformer models and parallel encoding across multiple GPUs
    - `util_embeddings.py`: Utility functions for GPU setup verification and embedding validation
    - `load_embeddings.py`: Utilities for loading and verifying embeddings
    - `models/`: Strategy pattern implementation for different embedding models
  - **`analysis/`**: Notebooks and scripts for data analysis, verification, and visualization
  - **Root level scripts**:
    - `data_pipeline.py`: Main orchestrator that runs the complete pipeline (Wikidata → MovieDB → Wikipedia → Embeddings)
    - `data_utils.py`: Helper functions for loading movie data and embeddings

- **`report/`**: LaTeX files for the final project report

This structure allows for:
- Clear separation between data collection, processing, and analysis
- Modular code that can be imported and reused
- Easy navigation and maintenance
- Organized data storage with per-year organization for scalability
- Strategy pattern for flexible embedding model selection

## Environment Setup

### Create Conda Environment

```bash
conda create -n dataLiteracy python=3.11
conda activate dataLiteracy
```

### Generate Requirements Files

```bash
# Create requirements.txt
pip freeze > requirements.txt

# Create environment.yml
conda env export > environment.yml
```

### Install from Files

```bash
# From requirements.txt
pip install -r requirements.txt

# From environment.yml
conda env create -f environment.yml
conda activate dataLiteracy
```

## Data Pipeline

The main data processing pipeline is orchestrated by `src/data_pipeline.py`. This pipeline performs the following steps:

1. **Wikidata Collection**: Fetches movie metadata from Wikidata by year (1950-2024)
2. **MovieDB Enrichment**: Enriches data with TMDb popularity, votes, and ratings via Wikidata ID
3. **Wikipedia Plot Retrieval**: Retrieves movie plots from Wikipedia via sitelinks
4. **Embedding Generation**: Generates embeddings for movie plots using SentenceTransformer models (default: BAAI/bge-m3)

The pipeline is designed to be incremental and resumable:
- Only processes years where data doesn't exist (unless `force_refresh=True`)
- Skips movies that already have enriched data
- Verifies existing embeddings before regenerating

To run the pipeline:

```bash
cd src
python data_pipeline.py
```

Configuration can be modified in `data_pipeline.py`:
- `START_YEAR` and `END_YEAR`: Year range to process
- `MOVIES_PER_YEAR`: Number of movies to fetch per year
- `SKIP_*` flags: Skip specific pipeline steps
- `MODEL_NAME`: Embedding model to use
- `BATCH_SIZE`: Batch size for embedding generation
- `TARGET_DEVICES`: GPU devices for parallel processing

## Loading Data

Helper methods have been created in `src/data_utils.py`.

Use `load_movie_embeddings(data_dir, verbose=False)` to get an ordered array of embeddings and the respective ordered array of movie IDs. The function loads embeddings from per-year files (1950-2024) and concatenates them.

Use `load_movie_data(data_dir, verbose=False)` to get a pd.DataFrame of movie features. The embeddings can be joined if needed using the movie_id column.

Example usage:

```python
from src.data_utils import load_movie_embeddings, load_movie_data
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
embeddings, movie_ids = load_movie_embeddings(data_dir, verbose=True)
movie_data = load_movie_data(data_dir, verbose=True)
```