#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Qwen3-Embedding-4B implementation.

This script tests that the Qwen3-Embedding model works correctly with
all chunking methods in the codebase.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from embedding.embedding import EmbeddingService
from analysis.chunking import (
    MeanPooling,
    CLSToken,
    ChunkFirstEmbed,
    LateChunking
)

# Test texts
TEST_TEXTS = [
    "This is a short test sentence.",
    "This is a longer test sentence that contains more words and should be processed correctly by the embedding model.",
    "The quick brown fox jumps over the lazy dog. " * 10,  # Longer text
    "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
]


def test_embedding_service():
    """Test that EmbeddingService can load Qwen3-Embedding-4B."""
    print("\n" + "="*80)
    print("Test 1: Loading Qwen3-Embedding-4B with EmbeddingService")
    print("="*80)
    
    try:
        model_name = "Qwen/Qwen3-Embedding-4B"
        embedding_service = EmbeddingService(model_name, target_devices=['cuda:0'] if os.environ.get('CUDA_VISIBLE_DEVICES') else None)
        print(f"✓ Successfully loaded {model_name}")
        print(f"✓ Strategy type: {type(embedding_service.strategy).__name__}")
        
        # Test encoding
        test_texts = ["Hello, world!"]
        results = embedding_service.encode_corpus(test_texts, batch_size=1)
        print(f"✓ Encoding successful. Keys: {list(results.keys())}")
        
        if 'dense' in results:
            dense_shape = results['dense'].shape
            print(f"✓ Dense embeddings shape: {dense_shape}")
        
        if 'colbert_vecs' in results:
            print(f"✓ Token-level embeddings available: {len(results['colbert_vecs'])} documents")
            if len(results['colbert_vecs']) > 0:
                token_shape = results['colbert_vecs'][0].shape if isinstance(results['colbert_vecs'][0], np.ndarray) else "list"
                print(f"✓ Token embeddings shape (first doc): {token_shape}")
        
        embedding_service.cleanup()
        return embedding_service
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cls_token(embedding_service):
    """Test CLSToken chunking method."""
    print("\n" + "="*80)
    print("Test 2: CLSToken Chunking Method")
    print("="*80)
    
    try:
        model_name = "Qwen/Qwen3-Embedding-4B"
        cls_token = CLSToken(embedding_service=embedding_service, model_name=model_name)
        
        # Test single embedding
        embedding = cls_token.embed(TEST_TEXTS[0])
        print(f"✓ Single embedding shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        
        # Test batch embedding
        embeddings = cls_token.embed_batch(TEST_TEXTS, batch_size=2)
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(TEST_TEXTS)}, {embedding.shape[0]})")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
        
        if np.allclose(norms, 1.0, rtol=1e-3):
            print("✓ All embeddings are properly normalized")
        else:
            print("⚠ Warning: Some embeddings are not unit-norm")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mean_pooling(embedding_service):
    """Test MeanPooling chunking method."""
    print("\n" + "="*80)
    print("Test 3: MeanPooling Chunking Method")
    print("="*80)
    
    try:
        model_name = "Qwen/Qwen3-Embedding-4B"
        mean_pooling = MeanPooling(embedding_service=embedding_service, model_name=model_name)
        
        # Test single embedding
        embedding = mean_pooling.embed(TEST_TEXTS[0])
        print(f"✓ Single embedding shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        
        # Test batch embedding
        embeddings = mean_pooling.embed_batch(TEST_TEXTS, batch_size=2)
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(TEST_TEXTS)}, {embedding.shape[0]})")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
        
        if np.allclose(norms, 1.0, rtol=1e-3):
            print("✓ All embeddings are properly normalized")
        else:
            print("⚠ Warning: Some embeddings are not unit-norm")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunk_first_embed(embedding_service):
    """Test ChunkFirstEmbed chunking method."""
    print("\n" + "="*80)
    print("Test 4: ChunkFirstEmbed Chunking Method")
    print("="*80)
    
    try:
        model_name = "Qwen/Qwen3-Embedding-4B"
        chunk_first = ChunkFirstEmbed(
            embedding_service=embedding_service,
            model_name=model_name,
            chunk_size=512,
            stride=256
        )
        
        # Test single embedding
        embedding = chunk_first.embed(TEST_TEXTS[2])  # Use longer text
        print(f"✓ Single embedding shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        
        # Test batch embedding
        embeddings = chunk_first.embed_batch(TEST_TEXTS, batch_size=2)
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(TEST_TEXTS)}, {embedding.shape[0]})")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
        
        if np.allclose(norms, 1.0, rtol=1e-3):
            print("✓ All embeddings are properly normalized")
        else:
            print("⚠ Warning: Some embeddings are not unit-norm")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_late_chunking(embedding_service):
    """Test LateChunking chunking method."""
    print("\n" + "="*80)
    print("Test 5: LateChunking Chunking Method")
    print("="*80)
    
    try:
        model_name = "Qwen/Qwen3-Embedding-4B"
        late_chunking = LateChunking(
            embedding_service=embedding_service,
            model_name=model_name,
            window_size=512,
            stride=256
        )
        
        # Test single embedding
        embedding = late_chunking.embed(TEST_TEXTS[2])  # Use longer text
        print(f"✓ Single embedding shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        
        # Test batch embedding
        embeddings = late_chunking.embed_batch(TEST_TEXTS, batch_size=2)
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(TEST_TEXTS)}, {embedding.shape[0]})")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
        
        if np.allclose(norms, 1.0, rtol=1e-3):
            print("✓ All embeddings are properly normalized")
        else:
            print("⚠ Warning: Some embeddings are not unit-norm")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Qwen3-Embedding-4B Implementation Test Suite")
    print("="*80)
    
    results = {}
    
    # Test 1: Load embedding service
    embedding_service = test_embedding_service()
    if embedding_service is None:
        print("\n✗ Failed to load embedding service. Cannot continue with other tests.")
        return
    
    # Test 2-5: Test all chunking methods
    results['CLSToken'] = test_cls_token(embedding_service)
    results['MeanPooling'] = test_mean_pooling(embedding_service)
    results['ChunkFirstEmbed'] = test_chunk_first_embed(embedding_service)
    results['LateChunking'] = test_late_chunking(embedding_service)
    
    # Cleanup
    embedding_service.cleanup()
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Please check the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


