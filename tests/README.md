# Test Suite for GroupDataLiteracy

This directory contains tests for the GroupDataLiteracy codebase.

## Test Structure

### `test_data_loading.py`

Comprehensive tests for CSV reading and lexical weights loading functionality.

#### Test Classes

1. **TestCSVLoading**: Tests for CSV file loading
   - Data directory existence
   - Year file discovery
   - Basic CSV loading
   - Column validation
   - Year column validation
   - Movie ID validation
   - Data dimensions

2. **TestLexicalWeights**: Tests for lexical weights loading
   - Basic loading functionality
   - Dimension matching (token_indices, weights, movie_ids)
   - Data type validation
   - Non-empty data checks
   - Movie ID format validation

3. **TestDataAlignment**: Tests for alignment between different data sources
   - CSV and lexical weights alignment
   - Embeddings and lexical weights alignment
   - Embeddings and CSV alignment

4. **TestDataConsistency**: Tests for data consistency and quality
   - No duplicate movie IDs in CSV
   - No duplicate movie IDs in lexical weights
   - Valid weight values
   - Valid token indices

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_data_loading.py -v
```

### Run specific test class:
```bash
pytest tests/test_data_loading.py::TestCSVLoading -v
```

### Run specific test:
```bash
pytest tests/test_data_loading.py::TestCSVLoading::test_load_movie_data_basic -v
```

## Test Configuration

Tests use the following configuration:
- **Data Directory**: `data/data_final/`
- **Chunking Suffix**: `_cls_token` (default)

These can be modified in the test file if needed.

## Expected Data Structure

The tests expect the following data structure:

```
data/
  data_final/
    wikidata_movies_YYYY.csv          # CSV files per year
    movie_embeddings_YYYY_cls_token.npy      # Embeddings per year
    movie_ids_YYYY_cls_token.npy             # Movie IDs per year
    movie_lexical_weights_YYYY_cls_token.npz # Lexical weights per year
```

## Notes

- Some tests may skip if required data files are not found (e.g., lexical weights)
- Alignment tests check for subset relationships rather than exact matches, as not all movies with plots may have embeddings/lexical weights
- Tests validate dimensions, data types, and consistency across all data sources

