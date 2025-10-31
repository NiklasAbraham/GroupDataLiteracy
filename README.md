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
│   └── movie_embeddings.npy      # Generated embeddings for movies
│
├── src/                           # Source code directory
│   ├── analysis/                  # Analysis notebooks and scripts
│   │   └── verify_test.ipynb     # Verification and testing notebooks
│   ├── embedding/                 # Embedding generation module
│   │   ├── embedding.py          # EmbeddingService class for parallel GPU encoding
│   │   └── factory.py            # Main pipeline script for generating embeddings
│   └── data_pipline.py           # Data processing pipeline
│
├── ansel_test.ipynb              # Testing notebooks
├── wikidata_vibecoded.ipynb      # Wikidata exploration notebook
├── wikipedia_example.py          # Wikipedia API usage examples
└── PresentationDeutscheBahn.md   # Project presentation notes
```

### Folder Structure Rationale

The project follows a clear separation of concerns:

- **`data/`**: Contains all data files, organized into subdirectories:
  - `mock/`: Sample datasets for testing and development
  - Generated files (like embeddings) are stored at the data root level for easy access

- **`src/`**: Contains all source code organized by functionality:
  - `embedding/`: Core embedding generation module
    - `embedding.py`: Defines the `EmbeddingService` class that handles SentenceTransformer models and parallel encoding across multiple GPUs
    - `factory.py`: Main execution script that orchestrates the embedding pipeline (loads data, initializes service, runs encoding, saves results)
  - `analysis/`: Notebooks and scripts for data analysis and verification
  - Root level scripts for data processing pipelines

- **Root level notebooks**: Various Jupyter notebooks for experimentation, testing, and data exploration

This structure allows for:
- Clear separation between data, source code, and exploratory work
- Easy navigation and maintenance
- Modular code that can be imported and reused
- Organized data storage with clear distinction between source data and generated artifacts

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
