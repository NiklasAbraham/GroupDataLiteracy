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
# Activate conda environment
conda activate dataLiteracy

# Run pipeline
cd /home/nab/Niklas/GroupDataLiteracy/src
python data_pipeline.py
```

Note: The pipeline outputs data to `data/data_final/` directory.

Configuration can be modified in `data_pipeline.py`:
- `START_YEAR` and `END_YEAR`: Year range to process
- `MOVIES_PER_YEAR`: Number of movies to fetch per year
- `SKIP_*` flags: Skip specific pipeline steps
- `MODEL_NAME`: Embedding model to use
- `BATCH_SIZE`: Batch size for embedding generation
- `TARGET_DEVICES`: GPU devices for parallel processing

## Loading Data

Helper methods have been created in `src/data_utils.py`.

Use `load_movie_embeddings(data_dir, verbose=False, chunking_suffix='_cls_token')` to get an ordered array of embeddings and the respective ordered array of movie IDs. The function loads embeddings from per-year files (1950-2024) and concatenates them. The `chunking_suffix` parameter specifies which embedding variant to load (e.g., `'_cls_token'`, `'_mean_pooling'`, or `''` for no suffix).

Use `load_movie_data(data_dir, verbose=False)` to get a pd.DataFrame of movie features. The embeddings can be joined if needed using the movie_id column.

Example usage:

```python
from src.data_utils import load_movie_embeddings, load_movie_data
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_final')
embeddings, movie_ids = load_movie_embeddings(data_dir, verbose=True, chunking_suffix='_cls_token')
movie_data = load_movie_data(data_dir, verbose=True)
```

Note: The data directory should point to `data/data_final/` which contains the final processed data files.

## Embedding Bias-Variance Experiment

The `src/analysis/chunking/` module implements an experimental framework to quantitatively compare document embedding aggregation methods on the movie corpus.

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

#### Using as a Module

```python
from analysis.chunking.manager import main

# Run experiment (uses configuration in manager.py)
metrics, embeddings = main()
```

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

### Example Output

Example output from running `test_single_chunking.py`:

```
Selected movie ID: Q28416981
Plot length: 6253 characters

================================================================================
PLOT TEXT:
================================================================================
The story focuses on two young talented chefs, Gao Tian Ci (Nicholas Tse), a southern-style Chinese chef, and Paul Ahn (Jung Yong-hwa), a Michelin-starred Korean chef trained in France. Both have reasons to climb the culinary ladder—When Tian Ci was ten years old, his father, Gao Feng (Anthony Wong), left him behind with his friend, Uncle Seven (Ge You), master chef of Seven Restaurant. When asked why, Gao Feng told Tian Ci that he has no talent as a cook and he would only take him back if he can prove to be a great cook. In reality, Gao Feng chose to pursue his culinary career over being a father and made the excuse that Tian Ci can't even make a decent bowl of noodles. Because of that, Tian Ci spent twenty years training to become a great chef under Uncle Seven. On the other side, Paul made a promise to his dying father that he would become a great cook. Through that journey, he became a highly successful chef in Europe and decides to run his own restaurant in Hong Kong, Stellar. However, this brings conflict between Paul and Tian Ci. In an old area of Hong Kong, Tian Ci is now an acclaimed chef at Seven. However, the Li Management Group arrives and starts buying various properties of the old sector, including developing Stellar for Paul. The opening of Stellar proves to be a threat to Tian Ci as these two chefs find themselves fighting for the best ingredients in the markets and maintaining their clientele. Stellar's fine haute cuisine represents a form of aggressive gentrification to the neighborhood and a threat to traditional Chinese cuisine. Their rivalry begins with a challenge for the best fish and the culinary masters agreed to face each other in a culinary duel. Tian Ci makes a traditional salt-baked duck while Paul makes a foie gras sorbet. While both tied in points, the judges declare Paul the victor, as his dish presentation was superior to Tian Ci's bland plating. Although the victory should've solidified Paul's abilities as a cook, things do not go as expected. It was while celebrating their victory that the manager of Li Group tells Paul he wants to replace him with Mei You (Michelle Bai), sous chef and girlfriend, reasoning that a woman is far more appealing on media than a man. Betrayed and confused, Paul tries to defend his position as head chef, but Mei You exposes his dark secret: Paul has problems tasting certain flavors, especially saltiness. To compensate, he would utilize a notebook containing all his recipes and have others test taste for him. To make matters worse, the manager announces that he and Mei You are romantically involved. Mei You explains to Paul that she never loved him and only sided with him to surpass him. Now that she can take the title as executive chef of Stellar, she doesn't need Paul anymore. Angry that he has lost everything, Paul leaves. Tian Ci bumps into Paul drinking at an event stadium. Both share their past and troubles, and find mutual respect for each other. Both have a common goal of reaching the culinary top and decided to team up. At Seven, the Li Group wants Uncle Seven to sign away his restaurant, but Uncle Seven refuses. Paul and Tian Ci then appear, announcing their partnership. Paul reminds the manager that as the winner of the competition, he is eligible to compete at the culinary championship, not the Li Group. Surprised by that technicality, the Li Group leaves, hypocritically calling Paul a traitor but not before Paul headbutted the manager in revenge. Tian Ci trains Paul in the ways of Chinese cooking as well as developing Paul's limited palate to help create something new for the competition. In Macau, at the Studio City Casino, attending the 7th International Culinary Competition, Tian Ci and Paul use both their culinary strengths to compete against four other great chefs. Whoever wins the competition will gain the chance to face the current god of Cookery, Gao Feng Ko. In this competition, they face a French team, Indian team, a Japanese chef, and Mei You. The French team make a roasted squab dish, the Indian team make a five-flavor curry, the Japanese chef makes koi nigiri, Mei You, impliying that she is in the final stage thanks to manager's bribery because she hadn't enrolled to be in the competition, makes an oyster dish with frozen foam, and the duo create a deconstructed mapo tofu. Paul used Tian Ci's sense of taste to help him determine the flavor of the ingredients and Tian Ci relies on Paul's knowledge of molecular science and culinary artistry to create a traditional dish with a modern design. The victory goes to Paul and Tian Ci. In defeat, a shameful Mei You is unable to bring herself to look at Paul's in the eyes for having betrayed him and now once again having to live under Paul's shadow. Before the final round, Paul points out only one chef can compete against Gao Feng and he realizes Tian Ci's desire to beat Gao Feng was a very personal one. Grateful he managed to make it this far with his condition, Paul gives Tian Ci the chance to face his father. In the final competition, Tian Ci finally faces his father, and the judges allow the two to cook anything they want as long as it is considered the highest expression of cooking. While Gao Feng begins cooking, Tian Ci is distracted by his thoughts about how Seven and the people in the neighborhood mean the most to him. Gao Feng angrily splashes water at his son's face, demanding that he focus and show him something. Gao Feng creates a beautiful artistic sugar display of molten lava with a single flower on top. Tian Ci cooks something far more personal: an interpretation of the original noodle dish that Gao Feng made all those years ago before abandoning Tian Ci. Before the judges can score the dish, Tian Ci gives the bowl of noodles to Gao Feng, who is moved as he remembers what the noodles represent. Acknowledging his skills as a chef, Gao Feng calls his son brilliant before Tian Ci walks off the stage. Gao Feng continues to emotionally eat his noodles, with the victor unclear. Some time has passed and the people at Seven are getting ready for a poon choi Chinese New Year party with the neighborhood's people. The movie ends with the staff of Seven announcing: We wish you all 2017 a happy rooster year!
================================================================================

Token count: 1475

Loaded concept space: 18852 concepts

================================================================================
TOP 30 CONCEPTS (WORDNET 10K CONCEPT SPACE, Zipf >= 4.0):
================================================================================
 1. chef                          : 0.159699
 2. competition                   : 0.049049
 3. manager                       : 0.040440
 4. father                        : 0.038061
 5. cook                          : 0.035920
 6. victory                       : 0.032845
 7. dish                          : 0.031582
 8. judge                         : 0.029551
 9. team                          : 0.023605
10. flavor                        : 0.019188
11. neighborhood                  : 0.017827
12. victor                        : 0.017512
13. cooking                       : 0.017109
14. people                        : 0.015481
15. threat                        : 0.014922
16. year                          : 0.014125
17. master                        : 0.013373
18. taste                         : 0.013083
19. son                           : 0.011803
20. bowl                          : 0.011665
21. revenge                       : 0.010659
22. ladder                        : 0.010458
23. top                           : 0.010082
24. story                         : 0.009451
25. restaurant                    : 0.009424
26. movie                         : 0.009415
27. defeat                        : 0.009407
28. secret                        : 0.008948
29. talent                        : 0.008523
30. conflict                      : 0.008394
================================================================================

colbert_vecs[0] shape: (1476, 1024)
window_embeddings shape: (5, 1024)
final_embedding_pre_norm shape: (1024,)
final_embedding shape: (1024,)
```

The output shows:
- The full movie plot text
- Token count (1475 tokens)
- Concept space size (18852 concepts loaded)
- Top 30 concepts extracted from the plot, ranked by aggregated lexical weights
- Embedding array shapes at various stages of processing

### Mathematical Foundation

The embedding-based mapping works as follows:

1. **Normalize weights**: Convert lexical weights to probability distribution
2. **Split lemmas**: Known (direct lookup) vs. unknown (need embedding)
3. **Direct assignment**: Known lemmas map directly to concept indices
4. **Embedding-based mapping**: Unknown lemmas are embedded and mapped via cosine similarity to nearest concept
5. **Aggregation**: Sum weights per concept to get final scores
6. **Ranking**: Return top-K concepts by aggregated score

This provides a normalized, comparable representation of movies in a fixed semantic space, suitable for temporal drift analysis and cross-movie comparison.

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

```bash
# Activate conda environment
conda activate dataLiteracy

# Run genre classification
cd /home/nab/GroupDataLiteracy
python src/analysis/genre_classifier.py
```

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

### Running Concept Extraction

See `src/analysis/example_concept_extraction.py` for a complete example:

```bash
# Activate conda environment
conda activate dataLiteracy

# Run example
cd /home/nab/GroupDataLiteracy
python src/analysis/example_concept_extraction.py
```

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

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loading.py -v

# Run specific test class
pytest tests/test_data_loading.py::TestCSVLoading -v
```

The tests validate:
- CSV file loading and validation
- Lexical weights loading
- Data alignment between different sources
- Data consistency and quality