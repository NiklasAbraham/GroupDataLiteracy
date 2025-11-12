# -*- coding: utf-8 -*-
"""
test_experiment.py

Simple test script to run the embedding bias-variance experiment with 10 movies.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
# test_experiment.py is in src/analysis/chunking/
# So we go up 3 levels to get to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

# Simple import - should work now that SRC_DIR is in path
from analysis.chunking.manager import ExperimentManager

def main():
    """Run test experiment with 10 movies."""
    # Get data directory
    data_dir = BASE_DIR / 'data'
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please ensure the data directory exists with movie CSV files.")
        return
    
    print(f"Using data directory: {data_dir}")
    
    # Detect device
    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create experiment manager
    manager = ExperimentManager(
        data_dir=str(data_dir),
        output_dir=None,  # Will create timestamped directory
        model_name="BAAI/bge-m3",
        device=device,
        n_movies=10,
        random_seed=42
    )
    
    # Run experiment
    try:
        metrics, embeddings = manager.run_experiment()
        print("\nTest completed successfully!")
        return metrics, embeddings
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()

