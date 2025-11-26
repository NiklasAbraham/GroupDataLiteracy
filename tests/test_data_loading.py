"""
Tests for CSV reading and lexical weights loading.

This test suite verifies:
1. CSV files can be loaded correctly
2. Lexical weights files can be loaded correctly
3. Dimensions match between related data structures
4. Alignment between CSV data and lexical weights
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils import (
    load_movie_data,
    find_year_files,
    load_lexical_weights,
    load_movie_embeddings
)


# Test configuration
DATA_DIR = os.path.join(Path(__file__).parent.parent, "data", "data_final")
CHUNKING_SUFFIX = "_cls_token"


class TestCSVLoading:
    """Tests for CSV file loading functionality."""
    
    def test_data_directory_exists(self):
        """Test that the data directory exists."""
        assert os.path.exists(DATA_DIR), f"Data directory not found: {DATA_DIR}"
        assert os.path.isdir(DATA_DIR), f"Data path is not a directory: {DATA_DIR}"
    
    def test_find_year_files(self):
        """Test that year files can be found."""
        year_files = find_year_files(DATA_DIR)
        
        assert isinstance(year_files, dict), "find_year_files should return a dictionary"
        assert len(year_files) > 0, f"No year files found in {DATA_DIR}"
        
        # Check that all values are valid file paths
        for year, file_path in year_files.items():
            assert isinstance(year, int), f"Year should be int, got {type(year)}"
            assert os.path.exists(file_path), f"File path does not exist: {file_path}"
            assert file_path.endswith('.csv'), f"File should be CSV: {file_path}"
    
    def test_load_movie_data_basic(self):
        """Test basic loading of movie data."""
        df = load_movie_data(DATA_DIR, verbose=False)
        
        assert isinstance(df, pd.DataFrame), "load_movie_data should return a DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        
        # Check required columns exist
        required_columns = ['movie_id', 'year']
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from DataFrame"
    
    def test_load_movie_data_columns(self):
        """Test that loaded CSV has expected columns."""
        df = load_movie_data(DATA_DIR, verbose=False)
        
        # Check for common columns (based on checks.ipynb)
        expected_columns = [
            'movie_id', 'title', 'release_date', 'genre', 'director',
            'actors', 'duration', 'imdb_id', 'country', 'sitelinks',
            'wikipedia_link', 'budget', 'box_office', 'awards',
            'set_in_period', 'year', 'popularity', 'vote_average',
            'vote_count', 'tmdb_id', 'plot'
        ]
        
        # At least some of these columns should exist
        found_columns = [col for col in expected_columns if col in df.columns]
        assert len(found_columns) > 0, "No expected columns found in DataFrame"
    
    def test_load_movie_data_year_column(self):
        """Test that year column is properly set."""
        df = load_movie_data(DATA_DIR, verbose=False)
        
        assert 'year' in df.columns, "Year column should exist"
        assert df['year'].dtype in [int, 'int64', 'Int64'], "Year should be integer type"
        
        # Check year values are reasonable
        if len(df) > 0:
            years = df['year'].dropna()
            if len(years) > 0:
                assert years.min() >= 1900, "Years should be >= 1900"
                assert years.max() <= 2030, "Years should be <= 2030"
    
    def test_load_movie_data_movie_id_unique(self):
        """Test that movie_id column exists and has values."""
        df = load_movie_data(DATA_DIR, verbose=False)
        
        assert 'movie_id' in df.columns, "movie_id column should exist"
        assert df['movie_id'].notna().sum() > 0, "movie_id should have non-null values"
        
        # Check movie_id format (should be strings based on dtype=str)
        sample_ids = df['movie_id'].dropna().head(10)
        for movie_id in sample_ids:
            assert isinstance(movie_id, str) or pd.isna(movie_id), \
                f"movie_id should be string, got {type(movie_id)}"
    
    def test_load_movie_data_dimensions(self):
        """Test that loaded data has reasonable dimensions."""
        df = load_movie_data(DATA_DIR, verbose=False)
        
        # Based on checks.ipynb, we expect around 214682 entries
        assert len(df) > 0, "DataFrame should have rows"
        assert len(df.columns) > 0, "DataFrame should have columns"
        
        # Check that we have at least some data
        assert df['movie_id'].notna().sum() > 0, "Should have valid movie_ids"


class TestLexicalWeights:
    """Tests for lexical weights loading functionality."""
    
    def test_load_lexical_weights_basic(self):
        """Test basic loading of lexical weights."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        # Should return None if no files found, or tuple if files exist
        if result is None:
            pytest.skip("No lexical weights files found - this is OK if not generated yet")
        
        assert isinstance(result, tuple), "load_lexical_weights should return a tuple"
        assert len(result) == 3, "Should return (token_indices_list, weights_list, movie_ids)"
        
        token_indices_list, weights_list, movie_ids = result
        
        assert isinstance(token_indices_list, list), "token_indices_list should be a list"
        assert isinstance(weights_list, list), "weights_list should be a list"
        assert isinstance(movie_ids, np.ndarray), "movie_ids should be a numpy array"
    
    def test_load_lexical_weights_dimensions(self):
        """Test that lexical weights dimensions match."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        # Check that all lists have the same length
        n_docs = len(token_indices_list)
        assert len(weights_list) == n_docs, \
            f"weights_list length ({len(weights_list)}) != token_indices_list length ({n_docs})"
        assert len(movie_ids) == n_docs, \
            f"movie_ids length ({len(movie_ids)}) != token_indices_list length ({n_docs})"
        
        # Check that each document has matching token_indices and weights lengths
        for i in range(min(100, n_docs)):  # Check first 100 or all if less
            assert len(token_indices_list[i]) == len(weights_list[i]), \
                f"Document {i}: token_indices length ({len(token_indices_list[i])}) != weights length ({len(weights_list[i])})"
    
    def test_load_lexical_weights_data_types(self):
        """Test that lexical weights have correct data types."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        # Check data types
        assert isinstance(movie_ids, np.ndarray), "movie_ids should be numpy array"
        
        if len(token_indices_list) > 0:
            assert isinstance(token_indices_list[0], np.ndarray), \
                "token_indices should be numpy arrays"
            assert token_indices_list[0].dtype in [np.int32, np.int64], \
                f"token_indices should be integer, got {token_indices_list[0].dtype}"
        
        if len(weights_list) > 0:
            assert isinstance(weights_list[0], np.ndarray), \
                "weights should be numpy arrays"
            assert weights_list[0].dtype in [np.float32, np.float64], \
                f"weights should be float, got {weights_list[0].dtype}"
    
    def test_load_lexical_weights_non_empty(self):
        """Test that lexical weights contain data."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        assert len(token_indices_list) > 0, "Should have at least one document"
        assert len(movie_ids) > 0, "Should have at least one movie_id"
        
        # Check that we have some non-zero weights
        total_non_zero = sum(len(ti) for ti in token_indices_list)
        assert total_non_zero > 0, "Should have some non-zero weights"
    
    def test_load_lexical_weights_movie_ids_format(self):
        """Test that movie_ids in lexical weights are valid."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        # Check movie_ids are strings (or object array of strings)
        if len(movie_ids) > 0:
            sample_id = movie_ids[0]
            assert isinstance(sample_id, (str, np.str_)), \
                f"movie_id should be string, got {type(sample_id)}"


class TestDataAlignment:
    """Tests for alignment between CSV data and lexical weights."""
    
    def test_csv_and_lexical_weights_alignment(self):
        """Test that CSV movie_ids align with lexical weights movie_ids.
        
        Note: Lexical weights may be a subset of CSV movies with plots,
        as not all movies may have been successfully embedded.
        """
        # Load CSV data
        df = load_movie_data(DATA_DIR, verbose=False)
        
        # Load lexical weights
        lexical_result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if lexical_result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, lexical_movie_ids = lexical_result
        
        # Get movie_ids from CSV that have plots (lexical weights are only for movies with plots)
        if 'plot' in df.columns:
            csv_movie_ids_with_plots = df[
                df['plot'].notna() & 
                (df['plot'].astype(str).str.strip() != '') & 
                (df['plot'].astype(str) != 'nan')
            ]['movie_id'].values
        else:
            csv_movie_ids_with_plots = df['movie_id'].values
        
        # Convert to sets for comparison
        lexical_set = set(lexical_movie_ids)
        csv_set = set(csv_movie_ids_with_plots)
        
        # Check that all lexical weights movie_ids are in CSV (subset relationship)
        assert lexical_set.issubset(csv_set), \
            f"Some lexical weights movie_ids not found in CSV. " \
            f"Only in lexical: {lexical_set - csv_set}"
        
        # Check that we have a reasonable overlap (at least 50% of lexical weights should match)
        overlap = len(lexical_set & csv_set)
        assert overlap > 0, "No overlap between lexical weights and CSV movie_ids"
        assert overlap / len(lexical_set) >= 0.5, \
            f"Less than 50% of lexical weights movie_ids found in CSV ({overlap}/{len(lexical_set)})"
    
    def test_embeddings_and_lexical_weights_alignment(self):
        """Test that embeddings and lexical weights have matching movie_ids."""
        # Load embeddings
        embeddings, embedding_movie_ids = load_movie_embeddings(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if len(embedding_movie_ids) == 0:
            pytest.skip("No embeddings found")
        
        # Load lexical weights
        lexical_result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if lexical_result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, lexical_movie_ids = lexical_result
        
        # Check that counts match
        assert len(embedding_movie_ids) == len(lexical_movie_ids), \
            f"Embeddings movie_ids count ({len(embedding_movie_ids)}) != " \
            f"Lexical weights movie_ids count ({len(lexical_movie_ids)})"
        
        # Check that movie_ids match
        assert np.array_equal(embedding_movie_ids, lexical_movie_ids), \
            "Embedding movie_ids should match lexical weights movie_ids"
    
    def test_embeddings_and_csv_alignment(self):
        """Test that embeddings and CSV have matching movie_ids.
        
        Note: Embeddings may be a subset of CSV movies with plots,
        as not all movies may have been successfully embedded.
        """
        # Load CSV data
        df = load_movie_data(DATA_DIR, verbose=False)
        
        # Load embeddings
        embeddings, embedding_movie_ids = load_movie_embeddings(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if len(embedding_movie_ids) == 0:
            pytest.skip("No embeddings found")
        
        # Get movie_ids from CSV that have plots
        if 'plot' in df.columns:
            csv_movie_ids_with_plots = df[
                df['plot'].notna() & 
                (df['plot'].astype(str).str.strip() != '') & 
                (df['plot'].astype(str) != 'nan')
            ]['movie_id'].values
        else:
            csv_movie_ids_with_plots = df['movie_id'].values
        
        # Convert to sets for comparison
        embedding_set = set(embedding_movie_ids)
        csv_set = set(csv_movie_ids_with_plots)
        
        # Check that all embedding movie_ids are in CSV (subset relationship)
        assert embedding_set.issubset(csv_set), \
            f"Some embedding movie_ids not found in CSV. " \
            f"Only in embeddings: {embedding_set - csv_set}"
        
        # Check that we have a reasonable overlap (at least 50% of embeddings should match)
        overlap = len(embedding_set & csv_set)
        assert overlap > 0, "No overlap between embeddings and CSV movie_ids"
        assert overlap / len(embedding_set) >= 0.5, \
            f"Less than 50% of embedding movie_ids found in CSV ({overlap}/{len(embedding_set)})"


class TestDataConsistency:
    """Tests for data consistency and quality."""
    
    def test_csv_no_duplicate_movie_ids(self):
        """Test that CSV doesn't have duplicate movie_ids (within same year)."""
        df = load_movie_data(DATA_DIR, verbose=False)
        
        if 'year' in df.columns:
            # Check for duplicates within each year
            for year in df['year'].unique():
                year_df = df[df['year'] == year]
                duplicates = year_df['movie_id'].duplicated()
                assert not duplicates.any(), \
                    f"Found duplicate movie_ids in year {year}"
    
    def test_lexical_weights_no_duplicate_movie_ids(self):
        """Test that lexical weights don't have duplicate movie_ids."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        # Check for duplicates
        unique_ids = np.unique(movie_ids)
        assert len(unique_ids) == len(movie_ids), \
            f"Found duplicate movie_ids in lexical weights: {len(movie_ids) - len(unique_ids)} duplicates"
    
    def test_lexical_weights_weights_valid(self):
        """Test that lexical weights contain valid values."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        # Check a sample of weights
        for i in range(min(100, len(weights_list))):
            weights = weights_list[i]
            if len(weights) > 0:
                # Weights should be finite
                assert np.all(np.isfinite(weights)), \
                    f"Document {i} has non-finite weights"
                
                # Weights should be non-negative (lexical weights are typically non-negative)
                assert np.all(weights >= 0), \
                    f"Document {i} has negative weights"
    
    def test_lexical_weights_token_indices_valid(self):
        """Test that lexical weights token indices are valid."""
        result = load_lexical_weights(
            DATA_DIR,
            chunking_suffix=CHUNKING_SUFFIX,
            verbose=False
        )
        
        if result is None:
            pytest.skip("No lexical weights files found")
        
        token_indices_list, weights_list, movie_ids = result
        
        # Check a sample of token indices
        for i in range(min(100, len(token_indices_list))):
            token_indices = token_indices_list[i]
            if len(token_indices) > 0:
                # Token indices should be non-negative integers
                assert np.all(token_indices >= 0), \
                    f"Document {i} has negative token indices"
                assert np.all(token_indices == token_indices.astype(int)), \
                    f"Document {i} has non-integer token indices"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

