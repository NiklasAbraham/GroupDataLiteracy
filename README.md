# Plot Twists Over Time: How Movie Stories Have Changed Over 95 Years

Group 3 project for ML4102 Data Literacy at the University of Tübingen.

## Authors

- **Ansel Cheung** (Matrikelnummer 7274374, MSc Machine Learning) - ansel-heng-yu.cheung@uni-tuebingen.de
- **Alessio Villa** (Matrikelnummer 7306912, MSc Computer Science) - alessio.villa@student.uni-tuebingen.de
- **Bartol Markovinović** (Matrikelnummer 7324790, MSc Machine Learning) - bartol.markovinovic@student.uni-tuebingen.de
- **Martín López de Ipiña** (Matrikelnummer 7293076, MSc Machine Learning) - martin.lopez-de-ipina-munoz@student.uni-tuebingen.de
- **Niklas Abraham** (Matrikelnummer 7307188, MSc Machine Learning) - niklas-sebastian.abraham@student.uni-tuebingen.de

## Project Goal

This project analyzes how movie narratives have evolved over nearly a century of cinema (1930-2024). A common sentiment suggests that the film industry is "running out of ideas," with movies becoming less creative and more similar over time. We investigate this claim by embedding movie plot summaries into a unified semantic space and using quantitative methods to measure temporal shifts in narrative structures.

**Key Research Questions:**
- How have movie plot structures evolved semantically over time?
- Are movies becoming less novel and more similar to previous works?
- Do specific genres or thematic clusters exhibit distinct temporal emergence patterns?

**Approach:**
We embed movie plot summaries from Wikipedia into a unified semantic space using the BGE-M3 model. Using distance distributions, novelty scores, and Kolmogorov-Smirnov tests, we quantify temporal shifts in narrative structures, revealing overall semantic stability alongside distinct emergence patterns in specific subgenres.

## Abstract

We analyze semantic evolution in cinema by embedding movie plot summaries from 1930 to 2024 into a unified semantic space. Using distance distributions, novelty scores, and statistical tests, we quantify how genres and thematic clusters shift over time. Our analysis reveals periods of semantic stability and reorganization, providing quantitative measures of cultural change in narrative structures across nearly a century of cinema.

## Lab Book

The lab book documenting our weekly progress, intermediate experiments, and workflow is available at:

**`lab_book/LabBook.pdf`**

This document contains weekly slides from our tutorials that served as both planning documents and progress records. While somewhat informal, the lab book provides a complete view of our project process, including many intermediate ideas and experiments that did not make it into the final report. It reflects the collaborative workflow and iterative development of our analysis methods.

## Project Structure

```
GroupDataLiteracy/
├── data/
│   ├── data_final/          # Processed movie data and embeddings
│   └── concept_space/       # Concept vocabulary for semantic mapping
│
├── src/
│   ├── aaa_data_pipline/    # Data collection and processing
│   ├── utils/               # Data loading utilities
│   └── aab_analysis/        # Analysis scripts and notebooks
│
├── report/                  # LaTeX report and figures
├── lab_book/               # Weekly progress documentation
└── figures/                # Analysis visualizations
```

**Main Components:**
- **Data Pipeline** (`src/aaa_data_pipline/`): Collects data from Wikidata, TMDb, Wikipedia, and generates BGE-M3 embeddings
- **Analysis** (`src/aab_analysis/`): Genre classification, novelty analysis, statistical tests, and visualizations
- **Report** (`report/`): Final LaTeX report with all findings

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
