# Concept Words Extraction Module

This module extracts semantic concepts from text using two complementary methods: sparse lexical weights and dense embeddings. Both methods use a shared concept space built from WordNet nouns filtered by Zipf frequency.

## Core Idea

The module identifies the most important semantic concepts in a text by:

1. **Building a Concept Space**: A vocabulary of concept words extracted from WordNet nouns, filtered by Zipf frequency to include only common, meaningful words. These words are embedded using SentenceTransformers to create a semantic space.

2. **Extracting Concepts**: Two methods map text to concepts:
   - **Sparse Method**: Uses lexical weights from BGE-M3 embeddings to identify important noun lemmas, then maps them to concepts
   - **Dense Method**: Uses dense embeddings directly and finds the most similar concepts via cosine similarity

3. **PCA Analysis**: Projects movie embeddings onto principal components and interprets dimensions using concept words

## Architecture

### `concept_space.py`
Core functionality for building and managing the concept space:
- `build_wordnet_concept_vocab()`: Extracts nouns from WordNet, filters by Zipf frequency and generic words
- `embed_and_save_concept_vocab()`: Embeds concept vocabulary and saves to disk
- `ConceptSpace`: Class for loading and querying the concept space
- `map_lemmas()`: Maps noun lemmas to concepts (used by sparse method)

### `concept_extraction_sparse.py`
Sparse method using lexical weights:
- `build_noun_lemma_weights()`: Projects BGE-M3 lexical weights onto noun lemmas
- `filter_by_zipf()`: Filters lemmas by Zipf frequency
- `extract_concepts_from_text()`: Main function for sparse concept extraction
- `extract_concepts_from_embedding_results()`: Convenience wrapper

### `concept_extraction_dense.py`
Dense method using embedding similarity:
- `extract_dense_embedding_from_results()`: Extracts dense embedding from embedding service results
- `extract_concepts_from_dense_embedding()`: Main function for dense concept extraction
- `extract_concepts_from_embedding_results()`: Convenience wrapper

### `pca_analysis.py`
PCA analysis with concept space interpretation:
- `perform_pca()`: Performs PCA on movie embeddings
- `select_relevant_concepts()`: Selects most relevant concepts from concept space
- `project_concept_words()`: Projects concept words onto PCA space
- `visualize_pca()`: Creates visualization of PCA results
- `run_pca_analysis()`: Complete pipeline for PCA analysis

## Usage

### Sparse Method

```python
from src.aab_analysis.concept_words import extract_concepts_from_embedding_results
from transformers import AutoTokenizer
import spacy

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
nlp = spacy.load("en_core_web_sm")

# Get embedding results from EmbeddingService
results = embedding_service.encode_corpus([text], batch_size=1)

# Extract concepts
concepts = extract_concepts_from_embedding_results(
    text=text,
    embedding_results=results,
    tokenizer=tokenizer,
    nlp=nlp,
    top_k=30,
    zipf_threshold=4.0
)
```

### Dense Method

```python
from src.aab_analysis.concept_words import extract_concepts_from_dense_embedding

# Get dense embedding
dense_embedding = extract_dense_embedding_from_results(results)

# Extract concepts
concepts = extract_concepts_from_dense_embedding(
    dense_embedding=dense_embedding,
    top_k=30
)
```

### Building Concept Space

```python
from src.aab_analysis.concept_words import build_wordnet_concept_vocab, embed_and_save_concept_vocab

vocab = build_wordnet_concept_vocab(
    min_zipf=4.0,
    max_vocab=20000,
    filter_verbs=True,
    filter_generic=True
)

embed_and_save_concept_vocab(
    vocab,
    output_dir="data/concept_space",
    model_name="BAAI/bge-small-en-v1.5",
    min_zipf=4.0,
    max_vocab=20000
)
```

### PCA Analysis

```python
from src.aab_analysis.concept_words.pca_analysis import run_pca_analysis

pc_scores, eigenvalues, principal_directions, pca_model, concept_space = run_pca_analysis(
    df=df_with_embeddings,
    n_components=10,
    top_k_concepts=5000,
    top_per_dimension=10
)
```

## File Structure

The concept space is stored as two numpy files:
- `concept_words_zipf{min_zipf}_vocab{max_vocab}_{model_name}.npy`: Array of concept words
- `concept_vecs_zipf{min_zipf}_vocab{max_vocab}_{model_name}.npy`: Array of concept embeddings

Default location: `data/concept_space/`

## Dependencies

- `sentence-transformers`: For embedding concept words
- `nltk`: For WordNet access
- `wordfreq`: For Zipf frequency filtering (optional, but recommended)
- `spacy`: For text processing (sparse method)
- `transformers`: For tokenization (sparse method)
- `numpy`, `pandas`, `scikit-learn`: For data processing and PCA

## Key Parameters

- `min_zipf_vocab`: Minimum Zipf frequency for building concept vocabulary (higher = more common words)
- `max_vocab`: Maximum vocabulary size for concept space
- `top_k`: Number of top concepts to return
- `zipf_threshold`: Minimum Zipf frequency for filtering lemmas (sparse method only)
- `concept_model`: SentenceTransformer model name for concept space (must match embedding model dimension)

## Notes

- The concept space is built once and reused for all extractions
- Both methods use the same concept space, ensuring consistency
- The sparse method requires lexical weights from BGE-M3 or similar models
- The dense method works with any embedding model that matches the concept space dimension
- Concept space files are automatically built if they don't exist (when `build_concept_space=True`)
