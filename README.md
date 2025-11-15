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
│   ├── concept_space/            # Concept vocabulary for semantic mapping
│   │   ├── concept_words.npy    # WordNet noun concept vocabulary
│   │   └── concept_vecs.npy     # Pre-computed concept embeddings
│   └── *.png                     # Visualization outputs
│
├── src/                           # Source code directory
│   ├── analysis/                  # Analysis notebooks and scripts
│   │   ├── verify_test.ipynb     # Verification and testing notebooks
│   │   ├── verify_csv.ipynb      # CSV data verification
│   │   ├── stats_analysis.ipynb  # Statistical analysis
│   │   ├── embedding_vs_feature_per_year.ipynb  # Embedding analysis
│   │   ├── stats_data.py         # Statistical data utilities
│   │   ├── tokenizer.py          # Tokenization utilities
│   │   └── chunking/             # Embedding bias-variance experiment
│   │       ├── chunk_base_class.py  # Abstract base class for chunking methods
│   │       ├── chunk_mean_pooling.py  # Mean pooling implementation
│   │       ├── chunk_no_chunking_cls_token.py  # CLS token implementation
│   │       ├── chunk_first_then_embed.py  # Chunk-first embedding
│   │       ├── chunk_late_chunking.py  # Late chunking implementation
│   │       ├── calculations.py  # Metrics and analysis functions
│   │       ├── manager.py  # Experiment orchestration
│   │       ├── test_experiment.py  # Simple test script
│   │       └── test_single_chunking.py  # Single movie concept extraction test
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
├── outputs/                      # Experiment outputs directory
│   └── experiment_*/            # Timestamped experiment results
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
    - `chunking/`: Experimental framework for comparing document embedding aggregation methods (bias-variance analysis)
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

## Embedding Bias-Variance Experiment

The `src/analysis/chunking/` module implements an experimental framework to quantitatively compare document embedding aggregation methods on the movie corpus.

### Methods Implemented

1. **MeanPooling** - Global mean of all token embeddings
2. **CLSToken** - Use final-layer CLS representation  
3. **ChunkFirstEmbed** - Classical early chunking: split text, embed chunks independently, pool afterwards
4. **LateChunking** - Embed full text once, then pool hidden states over overlapping windows

### Metrics Computed

1. **Length-Norm Correlation** - Pearson correlation between text length and embedding L2 norm
2. **Isotropy (PCA)** - Percentage of variance explained by first principal component
3. **Within-Film Variance** - Mean intra-film variance for multi-segment texts
4. **Between-Film Distance** - Mean cosine distance between random pairs of films
5. **Genre Clustering Quality** - Silhouette score using known genre labels
6. **Temporal Drift Stability** - Cosine shifts of anchor films across decades

### Running the Experiment

#### Quick Test (10 movies)

```bash
# Activate conda environment first
conda activate dataLiteracy

# Run test
cd /home/nab/GroupDataLiteracy
python src/analysis/chunking/test_experiment.py
```

#### Full Experiment

Edit configuration parameters in `src/analysis/chunking/manager.py`:
- `N_MOVIES`: Number of movies to process (default: 1000)
- `BATCH_SIZE`: Batch size for embedding processing (default: 128)
- `MODEL_NAME`: Embedding model to use (default: "BAAI/bge-m3")
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)

Then run:

```bash
# Activate conda environment
conda activate dataLiteracy

# Run experiment
cd /home/nab/GroupDataLiteracy
python src/analysis/chunking/manager.py
```

#### Using as a Module

```python
from analysis.chunking.manager import main

# Run experiment (uses configuration in manager.py)
metrics, embeddings = main()
```

### Output

The experiment generates the following outputs in `outputs/experiment_{timestamp}/`:

- `metrics.csv` - All computed metrics for each method
- `summary_table.csv` - Summary comparison table
- `{method}_embeddings.npy` - Embeddings for each method (MeanPooling, CLSToken, ChunkFirstEmbed, LateChunking)
- `length_norm_corr.png` - Combined length-norm correlation comparison plot
- `pca_isotropy.png` - Isotropy comparison
- `variance_boxplot.png` - Variance comparison
- `genre_silhouette.png` - Genre clustering quality
- `drift_stability.png` - Temporal drift visualization

### Batch Processing

All embedding operations use batch processing for efficiency. The system is designed to:
- Process all movies in batches (no individual GPU calls)
- Batch all chunks together for ChunkFirstEmbed
- Batch all texts together for LateChunking
- Batch all segments together for within-film variance computation

This ensures maximum GPU utilization and efficient processing of large datasets.

## Concept Extraction from Movie Plots

The `src/analysis/chunking/test_single_chunking.py` script implements a concept extraction pipeline that maps movie plot nouns to a fixed semantic concept space using embedding-based similarity.

### Overview

The concept extraction system:
1. Extracts noun lemmas from movie plots using spaCy POS tagging
2. Aggregates BGE-M3 lexical weights from subword tokens to word-level nouns
3. Maps nouns to a fixed concept space of ~20,000 common English nouns (WordNet, Zipf ≥ 4.0)
4. Uses embedding similarity to map unknown nouns to nearest concept anchors
5. Aggregates weights per concept to create a semantic signature

### Concept Space

The concept space is built from:
- **Source**: WordNet noun synsets
- **Filtering**: Zipf frequency ≥ 4.0 (very common words)
- **Size**: Top 20,000 most frequent nouns
- **Embedding**: BGE-small-en-v1.5 (L2-normalized)
- **Storage**: `data/concept_space/concept_words.npy` and `concept_vecs.npy`

The concept space is built once and reused for all movies, providing a consistent semantic basis for comparison.

### Running Concept Extraction

```bash
# Activate conda environment
conda activate dataLiteracy

# Run single movie test
cd /home/nab/GroupDataLiteracy
python src/analysis/chunking/test_single_chunking.py
```

### Configuration

Edit parameters in `test_single_chunking.py`:
- `zipf_threshold`: Minimum Zipf frequency for filtering lemmas (default: 4.0)
- `min_zipf`: Minimum Zipf for concept vocabulary (default: 4.0)
- `max_vocab`: Maximum concept vocabulary size (default: 20000)
- `top_k`: Number of top concepts to return (default: 30)

### Output

The script displays:
- Movie plot text
- Token count
- Embedding array shapes (colbert_vecs, window_embeddings, final_embedding)
- Top concepts ranked by aggregated lexical weights

### Mathematical Foundation

The embedding-based mapping works as follows:

1. **Normalize weights**: Convert lexical weights to probability distribution
2. **Split lemmas**: Known (direct lookup) vs. unknown (need embedding)
3. **Direct assignment**: Known lemmas map directly to concept indices
4. **Embedding-based mapping**: Unknown lemmas are embedded and mapped via cosine similarity to nearest concept
5. **Aggregation**: Sum weights per concept to get final scores
6. **Ranking**: Return top-K concepts by aggregated score

This provides a normalized, comparable representation of movies in a fixed semantic space, suitable for temporal drift analysis and cross-movie comparison.