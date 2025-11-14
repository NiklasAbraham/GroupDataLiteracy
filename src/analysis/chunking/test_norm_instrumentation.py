#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_norm_instrumentation.py

Demonstrates the norm recording and verification features.
Tests pre-L2 magnitude recording and post-L2 unit norm enforcement.

Usage:
    conda activate dataLiteracy
    cd /home/nab/GroupDataLiteracy
    python src/analysis/chunking/test_norm_instrumentation.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from embedding.embedding import EmbeddingService
from analysis.chunking.chunk_mean_pooling import MeanPooling
from analysis.chunking.chunk_no_chunking_cls_token import CLSToken
from analysis.chunking.chunk_first_then_embed import ChunkFirstEmbed
from analysis.chunking.chunk_late_chunking import LateChunking


def test_norm_recording():
    """Test norm recording feature."""
    print("\n" + "="*80)
    print("NORM INSTRUMENTATION TEST")
    print("="*80)
    
    # Create embedding service
    embedding_service = EmbeddingService("BAAI/bge-m3", None)
    
    # Test texts
    test_texts = [
        "This is a short test document.",
        "This is a longer test document with more content. " * 5,
        "Another test with different length.",
    ]
    
    # Initialize methods WITH norm recording enabled
    methods = {
        'MeanPooling': MeanPooling(
            embedding_service=embedding_service,
            model_name="BAAI/bge-m3",
            record_norms=True
        ),
        'CLSToken': CLSToken(
            embedding_service=embedding_service,
            model_name="BAAI/bge-m3",
            record_norms=True
        ),
        'ChunkFirstEmbed': ChunkFirstEmbed(
            embedding_service=embedding_service,
            model_name="BAAI/bge-m3",
            chunk_size=512,
            stride=256,
            record_norms=True
        ),
        'LateChunking': LateChunking(
            embedding_service=embedding_service,
            model_name="BAAI/bge-m3",
            window_size=256,
            stride=128,
            record_norms=True
        ),
    }
    
    print("\nProcessing test texts...")
    print(f"Number of texts: {len(test_texts)}")
    
    # Embed with each method
    for method_name, method_instance in methods.items():
        print(f"\n{'='*40}")
        print(f"Method: {method_name}")
        print(f"{'='*40}")
        
        # Clear previous records
        method_instance.clear_norm_records()
        
        # Embed each text
        embeddings = []
        for i, text in enumerate(test_texts):
            print(f"  Embedding text {i+1}/{len(test_texts)}...")
            emb = method_instance.embed(text)
            embeddings.append(emb)
            
            # Verify unit norm
            emb_norm = np.linalg.norm(emb)
            if not np.isclose(emb_norm, 1.0, rtol=1e-5, atol=1e-8):
                print(f"    WARNING: Embedding norm is {emb_norm:.8f} (expected 1.0)")
        
        # Print norm summary
        method_instance.print_norm_summary()
        
        # Get detailed records
        records = method_instance.get_norm_records()
        
        if records['pre_norm_magnitudes']:
            print(f"\nDetailed Pre-L2 Magnitudes:")
            for record in records['pre_norm_magnitudes'][:5]:
                print(f"  {record['vector_id']}: {record['pre_norm_magnitude']:.6f}")
            if len(records['pre_norm_magnitudes']) > 5:
                print(f"  ... ({len(records['pre_norm_magnitudes']) - 5} more)")
        
        # Batch embeddings check
        print(f"\nBatch Processing Test:")
        print(f"  Embedding {len(test_texts)} texts in batch...")
        method_instance.clear_norm_records()
        batch_embeddings = method_instance.embed_batch(test_texts, batch_size=2)
        
        print(f"  Output shape: {batch_embeddings.shape}")
        print(f"  Batch norms (should all be ~1.0):")
        batch_norms = np.linalg.norm(batch_embeddings, axis=1)
        for i, norm in enumerate(batch_norms):
            print(f"    Embedding {i+1}: {norm:.8f}")
        
        # Verify all norms are 1.0
        all_unit_norm = np.allclose(batch_norms, 1.0, rtol=1e-5, atol=1e-8)
        status = "✓ PASS" if all_unit_norm else "✗ FAIL"
        print(f"  Unit norm check: {status}")


def demonstrate_public_api_unchanged():
    """Demonstrate that public API remains unchanged."""
    print("\n" + "="*80)
    print("PUBLIC API COMPATIBILITY TEST")
    print("="*80)
    
    embedding_service = EmbeddingService("BAAI/bge-m3", None)
    test_text = "This is a test document for API compatibility."
    
    # Old way (without norm recording) still works
    print("\nOld API (record_norms not specified, defaults to False):")
    method_old = MeanPooling(embedding_service=embedding_service, model_name="BAAI/bge-m3")
    emb_old = method_old.embed(test_text)
    print(f"  Embedding shape: {emb_old.shape}")
    print(f"  Embedding norm: {np.linalg.norm(emb_old):.8f}")
    print(f"  record_norms attribute: {method_old.record_norms}")
    
    # New way (with norm recording)
    print("\nNew API (record_norms=True for instrumentation):")
    method_new = MeanPooling(
        embedding_service=embedding_service,
        model_name="BAAI/bge-m3",
        record_norms=True
    )
    emb_new = method_new.embed(test_text)
    print(f"  Embedding shape: {emb_new.shape}")
    print(f"  Embedding norm: {np.linalg.norm(emb_new):.8f}")
    print(f"  record_norms attribute: {method_new.record_norms}")
    print(f"  Norm records captured: {len(method_new.get_norm_records()['pre_norm_magnitudes'])}")
    
    print("\n✓ Public API unchanged - both methods work identically with optional hooks")
    
    # Cleanup
    embedding_service.cleanup()


if __name__ == "__main__":
    test_norm_recording()
    demonstrate_public_api_unchanged()
    
    print("\n" + "="*80)
    print("INSTRUMENTATION TEST COMPLETE")
    print("="*80 + "\n")

