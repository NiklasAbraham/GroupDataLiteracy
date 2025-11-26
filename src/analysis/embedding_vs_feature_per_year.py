"""
Embedding vs Feature Analysis Per Year

This script loads movie embeddings and metadata, processes genres using the genre filtering
from data_utils, and creates UMAP visualizations colored by year and genre.
"""

import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.manifold import TSNE

# Set up paths - navigate from src/analysis to data directory
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Fallback for notebooks - go up two directories from current working directory
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

# Add project root and src directory to path for imports
SRC_DIR = os.path.join(BASE_DIR, 'src')
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import functions from data_utils and data_cleaning
# data_cleaning uses 'from src.data_utils import ...' so we need BASE_DIR in path
from src.data_utils import cluster_genres, load_movie_data
from src.data_cleaning import clean_dataset

DATA_DIR = os.path.join(BASE_DIR, 'data', 'data_final')
START_YEAR = 1930
END_YEAR = 2024
# Chunking suffix (e.g., '_cls_token', '_mean_pooling', or '' for no chunking)
# If None, will auto-detect from existing files
CHUNKING_SUFFIX = None  # Set to '_cls_token' or other suffix, or None to auto-detect

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Year range: {START_YEAR} to {END_YEAR}")

# Auto-detect chunking suffix if not specified
if CHUNKING_SUFFIX is None:
    # Try to find a file to detect the suffix
    test_year = START_YEAR
    found_suffix = None
    # Try common suffixes
    for suffix in ['_cls_token', '_mean_pooling', '']:
        test_path = os.path.join(DATA_DIR, f'movie_embeddings_{test_year}{suffix}.npy')
        if os.path.exists(test_path):
            found_suffix = suffix
            break
    
    if found_suffix is not None:
        CHUNKING_SUFFIX = found_suffix
        print(f"Auto-detected chunking suffix: '{CHUNKING_SUFFIX}'")
    else:
        CHUNKING_SUFFIX = ''
        print("No chunking suffix detected, using default (no suffix)")
else:
    print(f"Using chunking suffix: '{CHUNKING_SUFFIX}'")

# Load all embeddings and corresponding movie IDs
all_embeddings = []
all_movie_ids = []
all_years = []

for year in range(START_YEAR, END_YEAR + 1):
    embeddings_path = os.path.join(DATA_DIR, f'movie_embeddings_{year}{CHUNKING_SUFFIX}.npy')
    movie_ids_path = os.path.join(DATA_DIR, f'movie_ids_{year}{CHUNKING_SUFFIX}.npy')
    
    if os.path.exists(embeddings_path) and os.path.exists(movie_ids_path):
        embeddings = np.load(embeddings_path)
        movie_ids = np.load(movie_ids_path)
        
        all_embeddings.append(embeddings)
        all_movie_ids.append(movie_ids)
        all_years.extend([year] * len(movie_ids))
        
        print(f"Loaded year {year}: {len(movie_ids)} movies")

# Check if any embeddings were loaded
if len(all_embeddings) == 0:
    raise ValueError(
        f"No embedding files found in {DATA_DIR} for years {START_YEAR}-{END_YEAR} "
        f"with suffix '{CHUNKING_SUFFIX}'. "
        f"Please check that embedding files exist (e.g., movie_embeddings_YYYY{CHUNKING_SUFFIX}.npy)"
    )

# Concatenate all embeddings
all_embeddings = np.vstack(all_embeddings)
all_movie_ids = np.concatenate(all_movie_ids)
all_years = np.array(all_years)

print(f"\nTotal movies: {len(all_movie_ids)}")
print(f"Embedding shape: {all_embeddings.shape}")

# Load movie metadata using data_utils function
print("Loading movie metadata using load_movie_data from data_utils...")
movie_data = load_movie_data(DATA_DIR, verbose=True)

if movie_data.empty:
    raise ValueError(f"No movie data found in {DATA_DIR}")

print(f"Loaded {len(movie_data)} movies from metadata files")

# Apply data cleaning from data_cleaning.py
print("\nApplying data cleaning from data_cleaning.py...")
movie_data = clean_dataset(movie_data, filter_single_genres=True)
print(f"Movies after cleaning: {len(movie_data)}")

# Fill NaN values in genre column with empty string (preprocess_genres will handle it)
if 'genre' in movie_data.columns:
    movie_data['genre'] = movie_data['genre'].fillna('')

# Apply genre clustering/filtering from data_utils
# This uses the genre_fix_mapping.json to map and clean genres
# Note: preprocess_genres expects genre_fix_mapping.json in current directory
# Temporarily change to src directory to find the file
original_cwd = os.getcwd()
try:
    os.chdir(SRC_DIR)
    print("\nProcessing genres using cluster_genres from data_utils...")
    print("Using genre mapping from genre_classifier.py (genre_fix_mapping.json or genre_fix_mapping_new.json)")
    movie_data = cluster_genres(movie_data)
finally:
    os.chdir(original_cwd)

# Create a mapping from movie_id to genre
movie_to_genre = {}
movie_to_title = {}

for idx, row in movie_data.iterrows():
    movie_id = row['movie_id']
    # Use the new_genre column created by cluster_genres
    new_genre = row.get('new_genre', 'Unknown')
    title = row['title']
    
    movie_to_title[movie_id] = title
    
    # Extract first genre if multiple genres are separated by |
    if pd.notna(new_genre) and new_genre.strip() and new_genre != 'Unknown':
        # Split by | and take the first genre (genres are separated by | after preprocessing)
        first_genre = new_genre.split('|')[0].strip()
        movie_to_genre[movie_id] = first_genre
    else:
        movie_to_genre[movie_id] = 'Unknown'

print(f"Loaded metadata for {len(movie_data)} movies")
print(f"Unique movie_ids with metadata: {len(movie_to_genre)}")
print(f"Unique processed genres: {len(set(movie_to_genre.values()))}")

# Filter embeddings to only include movies that are in the cleaned metadata
print("\nFiltering embeddings to match cleaned metadata...")
cleaned_movie_ids_set = set(movie_data['movie_id'].values)
mask = np.array([mid in cleaned_movie_ids_set for mid in all_movie_ids])
all_embeddings = all_embeddings[mask]
all_movie_ids = all_movie_ids[mask]
all_years = all_years[mask]

print(f"After filtering: {len(all_movie_ids)} movies with both embeddings and cleaned metadata")
print(f"Embedding shape: {all_embeddings.shape}")

# Filter out movies with "Unknown" genre
print("\nFiltering out movies with 'Unknown' genre...")
genre_mask = np.array([movie_to_genre.get(mid, 'Unknown') != 'Unknown' for mid in all_movie_ids])
n_unknown = np.sum(~genre_mask)
all_embeddings = all_embeddings[genre_mask]
all_movie_ids = all_movie_ids[genre_mask]
all_years = all_years[genre_mask]

print(f"Removed {n_unknown} movies with 'Unknown' genre")
print(f"After genre filtering: {len(all_movie_ids)} movies with known genres")
print(f"Embedding shape: {all_embeddings.shape}")

# Sample 5000 random points for visualization
n_samples = 8000
sample_indices = np.random.choice(len(all_movie_ids), size=n_samples, replace=False)

sampled_embeddings = all_embeddings[sample_indices]
sampled_movie_ids = all_movie_ids[sample_indices]
sampled_years = all_years[sample_indices]

# Extract genres for sampled movies
sampled_genres = [movie_to_genre.get(mid, 'Unknown') for mid in sampled_movie_ids]

print(f"Sampled {n_samples} movies for visualization")
print(f"Year range in sample: {sampled_years.min()} - {sampled_years.max()}")
print(f"Number of unique genres: {len(set(sampled_genres))}")

# Display genre distribution
genre_counts = pd.Series(sampled_genres).value_counts()
print(f"\nTop 20 genres in sample:")
print(genre_counts.head(20))

# Create UMAP reduction
print("Computing UMAP reduction...")
reducer = umap.UMAP(n_components=2, random_state=42)
umap_embedding = reducer.fit_transform(sampled_embeddings)

print(f"UMAP embedding shape: {umap_embedding.shape}")

# Plot 1: Colored by year with gradient
plt.figure(figsize=(14, 10))
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                     c=sampled_years, cmap='viridis', 
                     s=10, alpha=0.6, edgecolors='none')
plt.colorbar(scatter, label='Year')
plt.title('UMAP Visualization of Movie Embeddings Colored by Year', fontsize=16, fontweight='bold')
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'umap_by_year.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: umap_by_year.png")

# Plot 2: Colored by genre
plt.figure(figsize=(16, 12))

# Get unique genres and assign colors
unique_genres = sorted(set(sampled_genres))
n_genres = len(unique_genres)

# Use a colormap with enough colors
colors = plt.cm.tab20(np.linspace(0, 1, 20))
if n_genres > 20:
    # Cycle through colors if we have more than 20 genres
    color_map = {genre: colors[i % 20] for i, genre in enumerate(unique_genres)}
else:
    color_map = {genre: colors[i] for i, genre in enumerate(unique_genres)}

# Plot each genre with its color
for genre in unique_genres:
    mask = np.array(sampled_genres) == genre
    plt.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1], 
               c=[color_map[genre]], label=genre, s=10, alpha=0.6, edgecolors='none')

plt.title('UMAP Visualization of Movie Embeddings Colored by Genre', fontsize=16, fontweight='bold')
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'umap_by_genre.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: umap_by_genre.png")

# Create t-SNE reduction
print("Computing t-SNE reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
tsne_embedding = tsne.fit_transform(sampled_embeddings)

print(f"t-SNE embedding shape: {tsne_embedding.shape}")

# Plot 3: t-SNE colored by year with gradient
plt.figure(figsize=(14, 10))
scatter = plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                     c=sampled_years, cmap='viridis', 
                     s=10, alpha=0.6, edgecolors='none')
plt.colorbar(scatter, label='Year')
plt.title('t-SNE Visualization of Movie Embeddings Colored by Year', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'tsne_by_year.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: tsne_by_year.png")

# Plot 4: t-SNE colored by genre
plt.figure(figsize=(16, 12))

# Get unique genres and assign colors (reuse the same color map from UMAP plot)
unique_genres = sorted(set(sampled_genres))
n_genres = len(unique_genres)

# Use a colormap with enough colors
colors = plt.cm.tab20(np.linspace(0, 1, 20))
if n_genres > 20:
    # Cycle through colors if we have more than 20 genres
    color_map = {genre: colors[i % 20] for i, genre in enumerate(unique_genres)}
else:
    color_map = {genre: colors[i] for i, genre in enumerate(unique_genres)}

# Plot each genre with its color
for genre in unique_genres:
    mask = np.array(sampled_genres) == genre
    plt.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1], 
               c=[color_map[genre]], label=genre, s=10, alpha=0.6, edgecolors='none')

plt.title('t-SNE Visualization of Movie Embeddings Colored by Genre', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'tsne_by_genre.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: tsne_by_genre.png")

print("Analysis complete!")
print(f"\nSummary:")
print(f"  Total movies loaded: {len(all_movie_ids)}")
print(f"  Movies sampled: {n_samples}")
print(f"  Year range: {all_years.min()} - {all_years.max()}")
print(f"  Unique genres: {len(set(sampled_genres))}")
print(f"\nPlots saved to:")
print(f"  {os.path.join(DATA_DIR, 'umap_by_year.png')}")
print(f"  {os.path.join(DATA_DIR, 'umap_by_genre.png')}")
print(f"  {os.path.join(DATA_DIR, 'tsne_by_year.png')}")
print(f"  {os.path.join(DATA_DIR, 'tsne_by_genre.png')}")

