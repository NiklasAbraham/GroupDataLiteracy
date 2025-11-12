#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
determine_max_tokens.py

Script to determine the maximum token/context window size of an embedding model.
This helps identify if the model is truncating inputs and what the actual limit is.

Usage:
    cd /home/nab/GroupDataLiteracy
    python src/embedding/determine_max_tokens.py [model_name]
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
# Script should be run from project root, so go up 3 levels from script location
# or detect if we're already at project root
script_path = Path(__file__).resolve()
if 'src/embedding' in str(script_path):
    # Script is in src/embedding/, go up 3 levels
    BASE_DIR = script_path.parent.parent.parent
else:
    # Assume we're at project root
    BASE_DIR = Path.cwd()

SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Import EmbeddingService
# Import directly from the module file to avoid package issues
import importlib.util
embedding_path = SRC_DIR / 'embedding' / 'embedding.py'
spec = importlib.util.spec_from_file_location("embedding_service", embedding_path)
embedding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_module)
EmbeddingService = embedding_module.EmbeddingService


def check_model_config(model_name: str):
    """Check model configuration for context window information."""
    print("="*80)
    print("1. MODEL CONFIGURATION CHECK")
    print("="*80)
    
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        # Common attributes that indicate context window size
        context_attrs = {
            'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
            'n_positions': getattr(config, 'n_positions', None),
            'max_seq_length': getattr(config, 'max_seq_length', None),
            'context_window_size': getattr(config, 'context_window_size', None),
            'max_length': getattr(config, 'max_length', None),
        }
        
        print(f"\nModel: {model_name}")
        print(f"Config type: {type(config).__name__}")
        print("\nContext window attributes:")
        found_any = False
        for attr, value in context_attrs.items():
            if value is not None:
                print(f"  {attr}: {value}")
                found_any = True
        
        if not found_any:
            print("  No standard context window attributes found in config")
            print("\n  Available config attributes:")
            for attr in dir(config):
                if not attr.startswith('_') and not callable(getattr(config, attr)):
                    try:
                        val = getattr(config, attr)
                        if isinstance(val, (int, str, bool, float)):
                            print(f"    {attr}: {val}")
                    except:
                        pass
        
        return context_attrs
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def check_tokenizer_max_length(model_name: str):
    """Check tokenizer's model_max_length."""
    print("\n" + "="*80)
    print("2. TOKENIZER MAX LENGTH CHECK")
    print("="*80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"\nTokenizer model_max_length: {tokenizer.model_max_length}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        
        # Check if there's a special token for padding
        if hasattr(tokenizer, 'pad_token'):
            print(f"Has pad_token: {tokenizer.pad_token is not None}")
        
        return tokenizer.model_max_length
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


def test_actual_encoding_length(model_name: str, max_test_tokens: int = 10000):
    """Test actual encoding by progressively increasing text length."""
    print("\n" + "="*80)
    print("3. ACTUAL ENCODING TEST")
    print("="*80)
    print("\nTesting actual encoding with progressively longer texts...")
    print("This will show if the model truncates and at what point.\n")
    
    try:
        # Initialize embedding service
        print(f"Initializing EmbeddingService for {model_name}...")
        embedding_service = EmbeddingService(model_name, None)
        
        # Load tokenizer for counting
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create test texts of increasing length
        base_text = "This is a test sentence. " * 10  # ~50 tokens
        base_tokens = len(tokenizer(base_text, add_special_tokens=False)["input_ids"])
        
        test_lengths = []
        # Test at various token counts
        target_tokens = [100, 256, 512, 768, 1024, 2048, 4096, 8192, max_test_tokens]
        
        for target in target_tokens:
            if target > max_test_tokens:
                continue
            # Calculate how many repetitions needed
            reps = max(1, target // base_tokens)
            test_text = base_text * reps
            actual_tokens = len(tokenizer(test_text, add_special_tokens=False)["input_ids"])
            test_lengths.append((actual_tokens, test_text))
        
        print(f"Testing {len(test_lengths)} different text lengths...\n")
        
        results = []
        for input_tokens, test_text in test_lengths:
            try:
                # Encode and check colbert_vecs length
                result = embedding_service.encode_corpus([test_text], batch_size=1)
                
                if 'colbert_vecs' in result:
                    colbert_vecs = result['colbert_vecs']
                    if isinstance(colbert_vecs, list) and len(colbert_vecs) > 0:
                        output_seq_len = len(colbert_vecs[0])
                    elif hasattr(colbert_vecs, 'shape'):
                        output_seq_len = colbert_vecs.shape[1] if len(colbert_vecs.shape) > 1 else colbert_vecs.shape[0]
                    else:
                        output_seq_len = None
                else:
                    output_seq_len = None
                
                # Check dense_vecs
                if 'dense_vecs' in result:
                    dense_shape = result['dense_vecs'].shape
                else:
                    dense_shape = None
                
                truncated = output_seq_len is not None and output_seq_len < input_tokens
                
                status = "✓" if not truncated else "✗ TRUNCATED"
                output_str = str(output_seq_len) if output_seq_len is not None else 'N/A'
                print(f"  Input: {input_tokens:5d} tokens → Output seq_len: {output_str:>5s} {status}")
                
                results.append({
                    'input_tokens': input_tokens,
                    'output_seq_len': output_seq_len,
                    'truncated': truncated,
                    'dense_shape': dense_shape
                })
                
                # If truncated, we found the limit
                if truncated and output_seq_len:
                    print(f"\n  → TRUNCATION DETECTED at {input_tokens} input tokens!")
                    print(f"  → Actual max sequence length: ~{output_seq_len} tokens")
                    break
                    
            except Exception as e:
                print(f"  Input: {input_tokens:5d} tokens → ERROR: {e}")
                results.append({
                    'input_tokens': input_tokens,
                    'error': str(e)
                })
        
        embedding_service.cleanup()
        
        # Summary
        print("\n" + "-"*80)
        print("SUMMARY:")
        print("-"*80)
        
        max_found = None
        for r in results:
            if r.get('output_seq_len') is not None:
                if max_found is None or r['output_seq_len'] > max_found:
                    max_found = r['output_seq_len']
        
        if max_found:
            print(f"\nMaximum sequence length observed: {max_found} tokens")
        else:
            print("\nCould not determine maximum sequence length from encoding tests")
        
        return results
        
    except Exception as e:
        print(f"Error during encoding test: {e}")
        import traceback
        traceback.print_exc()
        return []


def check_flagembedding_settings(model_name: str):
    """Check FlagEmbedding-specific settings if applicable."""
    print("\n" + "="*80)
    print("4. FLAGEMBEDDING-SPECIFIC CHECK")
    print("="*80)
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        import torch
        
        # Try to load model and check its attributes
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"\nLoading model on {device} to check settings...")
        
        model = BGEM3FlagModel(model_name, device=device, use_fp16=True)
        
        print("\nModel attributes related to length:")
        attrs_to_check = [
            'max_length',
            'max_seq_length',
            'model_max_length',
            'max_position_embeddings',
        ]
        
        found_attrs = False
        for attr in attrs_to_check:
            if hasattr(model, attr):
                val = getattr(model, attr)
                print(f"  model.{attr}: {val}")
                found_attrs = True
        
        if hasattr(model, 'model'):
            print("\nChecking model.model attributes:")
            for attr in attrs_to_check:
                if hasattr(model.model, attr):
                    val = getattr(model.model, attr)
                    print(f"  model.model.{attr}: {val}")
                    found_attrs = True
        
        if hasattr(model, 'tokenizer'):
            print(f"\nModel tokenizer model_max_length: {model.tokenizer.model_max_length}")
            found_attrs = True
        
        if not found_attrs:
            print("  No standard length attributes found")
            print("\n  Available model attributes:")
            for attr in dir(model):
                if not attr.startswith('_') and 'length' in attr.lower():
                    try:
                        val = getattr(model, attr)
                        if not callable(val):
                            print(f"    {attr}: {val}")
                    except:
                        pass
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    except ImportError:
        print("\nFlagEmbedding not available - skipping FlagEmbedding-specific check")
        return False
    except Exception as e:
        print(f"\nError checking FlagEmbedding settings: {e}")
        return False


def main():
    """Main function to determine max token size."""
    print("\n" + "="*80)
    print("MAXIMUM TOKEN/CONTEXT WINDOW SIZE DETERMINATION")
    print("="*80)
    
    # Default model
    model_name = "BAAI/bge-m3"
    
    # Allow command line override
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print(f"\nAnalyzing model: {model_name}\n")
    
    # 1. Check model config
    config_attrs = check_model_config(model_name)
    
    # 2. Check tokenizer
    tokenizer_max = check_tokenizer_max_length(model_name)
    
    # 3. Check FlagEmbedding if available
    flagembedding_available = check_flagembedding_settings(model_name)
    
    # 4. Test actual encoding
    encoding_results = test_actual_encoding_length(model_name, max_test_tokens=8192)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nModel: {model_name}")
    
    if tokenizer_max:
        print(f"Tokenizer model_max_length: {tokenizer_max}")
    
    max_seq_from_config = None
    for attr in ['max_position_embeddings', 'n_positions', 'max_seq_length']:
        if attr in config_attrs and config_attrs[attr]:
            max_seq_from_config = config_attrs[attr]
            print(f"Config {attr}: {max_seq_from_config}")
            break
    
    # Find actual max from encoding tests
    actual_max = None
    for r in encoding_results:
        if r.get('output_seq_len') is not None:
            if actual_max is None or r['output_seq_len'] > actual_max:
                actual_max = r['output_seq_len']
    
    if actual_max:
        print(f"\nActual max sequence length (from encoding test): {actual_max} tokens")
    
    print("\n" + "="*80)
    print("\nRECOMMENDATION:")
    if actual_max:
        print(f"  Use max_length = {actual_max} tokens for this model")
        if actual_max < 512:
            print(f"  WARNING: This is lower than expected! Check if truncation is happening.")
        elif actual_max >= 8192:
            print(f"  This model supports long contexts (>= 8192 tokens)")
        else:
            print(f"  This model supports medium contexts ({actual_max} tokens)")
    else:
        if tokenizer_max:
            print(f"  Use max_length = {tokenizer_max} tokens (from tokenizer)")
        elif max_seq_from_config:
            print(f"  Use max_length = {max_seq_from_config} tokens (from config)")
        else:
            print("  Could not determine max_length - check model documentation")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

