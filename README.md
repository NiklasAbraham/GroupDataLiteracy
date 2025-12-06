# GroupDataLiteracy

Group 3 project for ML4102 Data Literacy. This repository contains a complete pipeline for analyzing temporal semantic drift in movie plots using embedding-based techniques. The project collects movie metadata and plot summaries from Wikidata, Wikipedia, and TMDb, generates semantic embeddings, and analyzes how movie content evolves over time through concept extraction, PCA analysis, and genre classification.

## Project Structure

The project is organized as follows:

```
GroupDataLiteracy/
├── data/                          # Data storage directory
│   ├── mock/                      # Mock/test datasets
│   │   ├── mock_movies_100.csv   # Sample movie dataset (100 entries)
│   │   └── wikidata_movies.csv   # Movies data from Wikidata
│   ├── data_final/                # Final processed data directory
│   │   ├── wikidata_movies_YYYY.csv  # Movie data files per year (1950-2024)
│   │   ├── movie_embeddings_YYYY_*.npy # Generated embeddings per year (with chunking suffix)
│   │   ├── movie_ids_YYYY_*.npy  # Movie IDs corresponding to embeddings
│   │   └── movie_lexical_weights_YYYY_*.npz # Lexical weights per year
│   ├── data_statistics.csv       # Statistical analysis of the dataset
│   ├── concept_space/            # Concept vocabulary for semantic mapping
│   │   ├── concept_words*.npy    # WordNet noun concept vocabulary (parameterized)
│   │   └── concept_vecs*.npy     # Pre-computed concept embeddings (parameterized)
│   └── *.png                     # Visualization outputs
│
├── src/                           # Source code directory
│   ├── analysis/                  # Analysis notebooks and scripts
│   │   ├── verify_test.ipynb     # Verification and testing notebooks
│   │   ├── verify_csv.ipynb      # CSV data verification
│   │   ├── stats_analysis.ipynb  # Statistical analysis
│   │   ├── embedding_vs_feature_per_year.py  # UMAP/TSNE visualization by year/genre
│   │   ├── genre_classifier.py   # Genre classification via clustering
│   │   ├── concept_extraction.py # Concept extraction from text using lexical weights
│   │   ├── example_concept_extraction.py  # Example usage of concept extraction
│   │   ├── stats_data.py         # Statistical data utilities
│   │   ├── tokenizer.py          # Tokenization utilities
│   │   ├── genre_description_df.csv  # Cached Wikipedia genre descriptions
│   │   └── chunking/             # Embedding bias-variance experiment
│   │       ├── chunk_base_class.py  # Abstract base class for chunking methods
│   │       ├── chunk_mean_pooling.py  # Mean pooling implementation
│   │       ├── chunk_no_chunking_cls_token.py  # CLS token implementation
│   │       ├── chunk_first_then_embed.py  # Chunk-first embedding
│   │       ├── chunk_late_chunking.py  # Late chunking implementation
│   │       ├── calculations.py  # Metrics and analysis functions
│   │       ├── manager.py  # Experiment orchestration
│   │       └── test_single_chunking.py  # Single movie concept extraction test
│   ├── embedding/                 # Embedding generation module
│   │   ├── embedding.py          # EmbeddingService class for parallel GPU encoding
│   │   ├── util_embeddings.py    # Utility functions for GPU verification and embedding validation
│   │   ├── load_embeddings.py    # Utilities for loading embeddings
│   │   └── models/               # Strategy pattern for embedding models
│   │       ├── base_strategy.py  # Base strategy interface
│   │       ├── sentence_transformer_strategy.py  # SentenceTransformer implementation
│   │       ├── flag_embedding_strategy.py  # Flag embedding implementation
│   │       ├── qwen3_strategy.py  # Qwen3 embedding implementation
│   │       └── strategy_factory.py  # Factory for creating strategies
│   ├── api/                       # API handlers for external data sources
│   │   ├── wikidata_handler.py   # Wikidata data collection
│   │   ├── moviedb_handler.py    # TMDb API integration for movie metadata
│   │   └── wikipedia_handler.py # Wikipedia plot retrieval
│   ├── concept_words/            # Concept space and PCA analysis
│   │   ├── concept_space.py      # Concept space vocabulary and embeddings
│   │   ├── concept_extraction_dense.py  # Dense concept extraction
│   │   ├── concept_extraction_sparse.py  # Sparse concept extraction
│   │   ├── pca_analysis.py       # PCA analysis with concept space projection
│   │   ├── example_pca_analysis.py  # Example PCA analysis script
│   │   └── example_concept_extraction_*.py  # Concept extraction examples
│   ├── data_pipeline.py          # Main data processing pipeline orchestrator
│   ├── data_utils.py             # Utility methods for data loading and genre clustering
│   ├── data_cleaning.py          # Data cleaning and filtering utilities
│   ├── genre_fix_mapping.json    # Genre mapping for clustering and normalization
│   ├── genre_fix_mapping_new.json  # Alternative genre mapping file
│   └── checks.ipynb              # Data quality checks
│
├── tests/                         # Test suite
│   ├── test_data_loading.py      # Comprehensive tests for data loading
│   └── README.md                 # Test documentation
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
  - `data_final/`: Final processed data directory containing:
    - `wikidata_movies_YYYY.csv`: Movie metadata files organized by year (1950-2024)
    - `movie_embeddings_YYYY_*.npy`: Generated embeddings per year (with chunking suffix, e.g., `_cls_token`)
    - `movie_ids_YYYY_*.npy`: Movie IDs corresponding to embeddings (for indexing)
    - `movie_lexical_weights_YYYY_*.npz`: Lexical weights per year for concept extraction
  - `concept_space/`: Pre-computed concept vocabulary and embeddings for semantic mapping
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
    - `genre_classifier.py`: Classifies movie genres by fetching Wikipedia descriptions, embedding them, and clustering into categories
    - `concept_extraction.py`: Extracts semantic concepts from text using lexical weights and WordNet/embedding-based concept spaces
    - `example_concept_extraction.py`: Example script demonstrating concept extraction usage
    - `embedding_vs_feature_per_year.py`: Creates UMAP/TSNE visualizations of embeddings colored by year and genre
    - `chunking/`: Experimental framework for comparing document embedding aggregation methods (bias-variance analysis)
  - **`concept_words/`**: Concept space vocabulary and PCA analysis tools
    - `concept_space.py`: Manages WordNet-based concept vocabulary and embeddings
    - `pca_analysis.py`: Performs PCA on movie embeddings with concept space projection for semantic interpretation
    - `concept_extraction_dense.py` and `concept_extraction_sparse.py`: Dense and sparse concept extraction implementations
  - **Root level scripts**:
    - `data_pipeline.py`: Main orchestrator that runs the complete pipeline (Wikidata → MovieDB → Wikipedia → Embeddings)
    - `data_utils.py`: Helper functions for loading movie data and embeddings
    - `data_cleaning.py`: Data cleaning and filtering utilities (filters non-movies, handles plot length limits, etc.)

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

The data processing pipeline (`src/data_pipeline.py`) collects and processes movie data for temporal semantic drift analysis. The pipeline executes four sequential steps:

1. **Wikidata Collection**: Queries Wikidata SPARQL endpoint to fetch movie metadata (titles, release years, genres, Wikidata IDs) for films released between 1950-2024. Filters for feature-length films and excludes TV series, short films, and other non-feature content.

2. **MovieDB Enrichment**: Enriches Wikidata entries with TMDb metadata via API lookup using Wikidata IDs. Retrieves popularity scores, vote counts, average ratings, and additional metadata to supplement the dataset for analysis.

3. **Wikipedia Plot Retrieval**: Fetches full plot summaries from Wikipedia using Wikidata sitelinks. Handles redirects, extracts main plot sections, and filters out plots that are too short or contain insufficient narrative content.

4. **Embedding Generation**: Generates dense and sparse embeddings for all retrieved plot texts using BAAI/bge-m3 model by default. Produces both CLS token embeddings and lexical weight matrices for concept extraction. Supports multiple chunking strategies and parallel GPU processing.

The pipeline is designed to be incremental and resumable:
- Only processes years where data doesn't exist (unless `force_refresh=True`)
- Skips movies that already have enriched data
- Verifies existing embeddings before regenerating

Run the pipeline by executing `src/data_pipeline.py`. The pipeline outputs all processed data to `data/data_final/` directory, organized by year and including consolidated files for efficient loading.

Configuration can be modified in `data_pipeline.py`:
- `START_YEAR` and `END_YEAR`: Year range to process
- `MOVIES_PER_YEAR`: Number of movies to fetch per year
- `SKIP_*` flags: Skip specific pipeline steps
- `MODEL_NAME`: Embedding model to use
- `BATCH_SIZE`: Batch size for embedding generation
- `TARGET_DEVICES`: GPU devices for parallel processing

## Loading Data

Data loading utilities are provided in `src/data_utils.py`. The project uses consolidated data files for efficient access:

- **`load_final_data_with_embeddings()`**: Loads the complete movie dataset with embeddings merged into a single DataFrame. This is the primary function for analysis workflows. The function loads from `final_dataset.csv` and `final_dense_embeddings.npy`, merges them on movie_id, and applies genre clustering.

- **`load_final_dense_embeddings()`**: Loads embeddings and movie IDs from consolidated NumPy files. Returns a tuple of (embeddings array, movie_ids array) for direct embedding operations.

- **`load_movie_embeddings()`**: Legacy function for loading embeddings from per-year files (1950-2024). Supports chunking suffixes like `_cls_token`, `_mean_pooling`, etc. for different embedding aggregation methods.

- **`load_movie_data()`**: Loads movie metadata DataFrame from per-year CSV files.

All functions default to `/home/nab/Niklas/GroupDataLiteracy/data/data_final/` as the data directory. The consolidated files (`final_dense_embeddings.npy`, `final_dataset.csv`) are generated by the pipeline and provide faster loading than per-year files.

## Embedding Bias-Variance Experiment

The `src/analysis/chunking/` module implements an experimental framework to quantitatively compare document embedding aggregation methods on the movie corpus. This experiment evaluates how different text chunking and pooling strategies affect embedding quality, isotropy, genre clustering, and temporal stability metrics across 75 years of movie data.

### Methods Implemented

1. **MeanPooling** - Global mean of all token embeddings
2. **CLSToken** - Use final-layer CLS representation  
3. **ChunkFirstEmbed** - Classical early chunking: split text, embed chunks independently, pool afterwards
   - Variants: `ChunkFirstEmbed_512_256`, `ChunkFirstEmbed_1024_512`, `ChunkFirstEmbed_2048_1024` (chunk_size_stride)
4. **LateChunking** - Embed full text once, then pool hidden states over overlapping windows
   - Variants: `LateChunking_512_256`, `LateChunking_1024_512`, `LateChunking_2048_1024`, `LateChunking_2048_512` (window_size_stride)
   - Non-overlapping variants: `LateChunking_512_0`, `LateChunking_1024_0`, `LateChunking_2048_0` (stride=0)

### Metrics Computed

1. **Length-Norm Correlation** - Pearson correlation between text length and embedding L2 norm
2. **Isotropy (PCA)** - Percentage of variance explained by first principal component
3. **Within-Film Variance** - Mean intra-film variance for multi-segment texts
4. **Between-Film Distance** - Mean cosine distance between random pairs of films
5. **Genre Clustering Quality** - Silhouette score using known genre labels
6. **Temporal Drift Stability** - Cosine shifts of anchor films across decades

### Running the Experiment

#### Full Experiment

Edit configuration parameters in `src/analysis/chunking/manager.py`:
- `N_MOVIES`: Number of movies to process (default: 5000)
- `BATCH_SIZE`: Batch size for embedding processing (default: 100)
- `MODEL_NAME`: Embedding model to use (default: "BAAI/bge-m3")
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `DATA_DIR`: Path to data directory (default: project root `data/` folder)

Then run:

```bash
# Activate conda environment
conda activate dataLiteracy

# Run experiment
cd /home/nab/GroupDataLiteracy
python src/analysis/chunking/manager.py
```

The experiment can also be imported as a module. The `main()` function returns a metrics DataFrame and embeddings dictionary for further analysis.

### Genre Processing

The experiment uses genre clustering via `cluster_genres()` from `data_utils.py`:
- Processes raw genre strings using `genre_fix_mapping.json` for normalization
- Maps genres to standardized categories (multi-label format, separated by `|`)
- Filters movies to those with valid processed genres
- Uses processed genres for genre clustering quality metrics

### Output

The experiment generates the following outputs in `outputs/experiment_{timestamp}/`:

- `metrics.csv` - All computed metrics for each method
- `summary_table.csv` - Summary comparison table
- `{method}_embeddings.npy` - Embeddings for each method variant (e.g., `MeanPooling_embeddings.npy`, `LateChunking_512_256_embeddings.npy`)
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
- Uses single GPU (`cuda:0`) by default for stability
- Aggressive memory management between methods to prevent OOM errors

This ensures maximum GPU utilization and efficient processing of large datasets.

## Concept Extraction from Movie Plots

The concept extraction pipeline maps movie plot nouns to a fixed semantic concept space, enabling quantitative analysis of thematic content evolution over time. This is implemented in `src/analysis/chunking/test_single_chunking.py` for single-movie analysis and supports batch processing for temporal drift studies.

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

The script `src/analysis/chunking/test_single_chunking.py` implements concept extraction for a single movie, demonstrating the full pipeline from plot text to concept signature. It processes one movie at a time for detailed inspection and debugging.

### Configuration

Edit parameters in `test_single_chunking.py`:
- `zipf_threshold`: Minimum Zipf frequency for filtering lemmas (default: 4.0)
- `min_zipf`: Minimum Zipf for concept vocabulary (default: 4.0)
- `max_vocab`: Maximum concept vocabulary size (default: 20000)
- `top_k`: Number of top concepts to return (default: 30)

### Output

The script processes a single movie plot and outputs:
- Movie plot text and token count
- Embedding array shapes at each processing stage (colbert_vecs, window_embeddings, final_embedding)
- Top-K concepts ranked by aggregated lexical weights from the concept space
- Concept space metadata (number of concepts loaded, filtering parameters)

### Mathematical Foundation

The embedding-based mapping works as follows:

1. **Normalize weights**: Convert lexical weights to probability distribution
2. **Split lemmas**: Known (direct lookup) vs. unknown (need embedding)
3. **Direct assignment**: Known lemmas map directly to concept indices
4. **Embedding-based mapping**: Unknown lemmas are embedded and mapped via cosine similarity to nearest concept
5. **Aggregation**: Sum weights per concept to get final scores
6. **Ranking**: Return top-K concepts by aggregated score

This provides a normalized, comparable representation of movies in a fixed semantic space, suitable for temporal drift analysis and cross-movie comparison.

## PCA Analysis with Concept Space Projection

The `src/concept_words/pca_analysis.py` module performs Principal Component Analysis on movie embeddings to identify latent semantic dimensions and projects concept words onto these dimensions for interpretability. This enables understanding of what semantic factors drive variation in movie plot embeddings over time.

The analysis pipeline:
1. Extracts embeddings from movie DataFrame and performs PCA with column-wise centering
2. Loads the WordNet-based concept space vocabulary and embeddings
3. Selects top-K most relevant concepts based on cosine similarity to movie embedding centroid
4. Projects concept vectors onto PCA principal directions
5. Identifies top concepts per principal component dimension
6. Generates visualization with movies colored by genre and concept words overlaid

The `run_pca_analysis()` function accepts a DataFrame with an 'embedding' column and returns PCA scores, eigenvalues, principal directions, the fitted PCA model, and the concept space. Key parameters include `n_components` (default: 2), `top_k_concepts` for concept selection (default: 30), `top_per_dimension` for displaying concepts per PC (default: 10), and concept space configuration (`min_zipf_vocab`, `max_vocab`, `concept_model`).

Outputs include explained variance statistics per component, ranked lists of top concepts per dimension with projection scores, and a 2D scatterplot visualization saved as PNG. The visualization overlays concept word positions on the PCA space, enabling interpretation of what semantic dimensions each principal component captures.

This analysis is used to study how semantic concepts evolve in movie plots across different time periods and genres, revealing patterns in cultural and thematic shifts reflected in film content.

## Genre Classification

The `src/analysis/genre_classifier.py` module provides functionality to classify and cluster movie genres using semantic embeddings.

### Overview

The genre classification system:
1. Extracts unique genres from the movie dataset
2. Fetches Wikipedia descriptions for each genre (cached in `genre_description_df.csv`)
3. Embeds genre descriptions using sentence transformers (default: Qwen/Qwen3-Embedding-0.6B)
4. Clusters genres using KMeans into main categories
5. Generates a mapping JSON file compatible with `data_utils.py`

### Running Genre Classification

Run `src/analysis/genre_classifier.py` to generate genre clusters. The script processes all unique genres in the dataset, fetches Wikipedia descriptions for each, embeds them, and performs KMeans clustering to group similar genres.

### Configuration

Edit parameters in `genre_classifier.py`:
- `n_clusters`: Number of clusters for KMeans (default: 15)
- `model_name`: Sentence transformer model name (default: 'Qwen/Qwen3-Embedding-0.6B')
- `descriptions_csv_path`: Path to cache genre descriptions
- `output_json_path`: Path to save output JSON mapping
- `apply_data_cleaning`: Whether to apply data cleaning (default: True)

### Output

The script generates:
- `genre_description_df.csv`: Cached Wikipedia descriptions for genres
- `genre_fix_mapping_new.json`: Genre mapping dictionary (genre → cluster label)

## Concept Extraction

The `src/analysis/concept_extraction.py` module provides functionality to extract semantic concepts from movie plots using lexical weights and a fixed concept space.

### Overview

The concept extraction system:
1. Extracts noun lemmas and their weights from lexical weights
2. Filters by Zipf frequency to remove obscure words
3. Maps to concepts using WordNet hypernyms or embedding similarity
4. Aggregates weights per concept to create semantic signatures

The module supports both dense and sparse concept extraction approaches. Dense extraction works directly with sentence embeddings, while sparse extraction uses lexical weights from BGE-M3 models for more granular word-level concept mapping.

### Configuration

The concept extraction supports:
- Parameterized concept space files based on Zipf threshold, vocabulary size, and model name
- Automatic concept space building if files don't exist
- Flexible filtering and mapping strategies

## Data Cleaning

The `src/data_cleaning.py` module provides utilities for cleaning and filtering the movie dataset:

- **Filter non-movies**: Removes entries that are not actual movies (e.g., TV series, short films, video games)
- **Plot length filtering**: Filters movies with plots exceeding maximum length
- **Genre filtering**: Optional filtering of movies with single genres

The cleaning functions are used by the genre classifier and can be applied independently.

## Testing

The `tests/` directory contains comprehensive tests for the codebase. See `tests/README.md` for detailed documentation.

Tests are run using pytest. The test suite validates:
- CSV file loading and validation
- Lexical weights loading
- Data alignment between different sources
- Data consistency and quality