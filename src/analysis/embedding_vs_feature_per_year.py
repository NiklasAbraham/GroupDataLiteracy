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
from sklearn.decomposition import PCA

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

# Import functions from data_utils
from src.data_utils import load_final_dataset, load_final_dense_embeddings

DATA_DIR = os.path.join(BASE_DIR, 'data', 'data_final')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'data_final', 'final_dataset.csv')
START_YEAR = 1930
END_YEAR = 2024

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Year range: {START_YEAR} to {END_YEAR}")

# Load all embeddings and corresponding movie IDs from consolidated files
print("Loading embeddings from consolidated files...")
all_embeddings, all_movie_ids = load_final_dense_embeddings(DATA_DIR, verbose=True)

if len(all_movie_ids) == 0:
    raise ValueError(f"No embeddings found in {DATA_DIR}")

print(f"\nTotal movies with embeddings: {len(all_movie_ids)}")
print(f"Embedding shape: {all_embeddings.shape}")

# Load movie metadata from consolidated CSV
print("\nLoading movie metadata from consolidated CSV...")
movie_data = load_final_dataset(CSV_PATH, verbose=True)

if movie_data.empty:
    raise ValueError(f"No movie data found in {CSV_PATH}")

print(f"Loaded {len(movie_data)} movies from metadata file")

# Filter by year range if year column exists
if 'year' in movie_data.columns:
    movie_data = movie_data[
        (movie_data['year'] >= START_YEAR) & 
        (movie_data['year'] <= END_YEAR)
    ].copy()
    print(f"Filtered to {len(movie_data)} movies between {START_YEAR} and {END_YEAR}")

# Create a mapping from movie_id to genre using genre_cluster_names from CSV
movie_to_genre = {}
movie_to_title = {}

for idx, row in movie_data.iterrows():
    movie_id = row['movie_id']
    # Use the genre_cluster_names column which already exists in the CSV
    genre_cluster_names = row.get('genre_cluster_names', '')
    title = row['title']
    
    movie_to_title[movie_id] = title
    
    # Extract first genre if multiple genres are separated by comma
    if pd.notna(genre_cluster_names) and str(genre_cluster_names).strip() and str(genre_cluster_names).strip() != '':
        # Split by comma and take the first genre (genres are comma-separated in genre_cluster_names)
        first_genre = str(genre_cluster_names).split(',')[0].strip()
        movie_to_genre[movie_id] = first_genre
    else:
        movie_to_genre[movie_id] = 'Unknown'

print(f"Loaded metadata for {len(movie_data)} movies")
print(f"Unique movie_ids with metadata: {len(movie_to_genre)}")
print(f"Unique processed genres: {len(set(movie_to_genre.values()))}")

# Create year array for movies with embeddings by looking up year from metadata
print("\nCreating year mapping for embeddings...")
movie_id_to_year = dict(zip(movie_data['movie_id'], movie_data['year']))
all_years = np.array([movie_id_to_year.get(mid, -1) for mid in all_movie_ids])

# Filter embeddings to only include movies that are in the filtered metadata
print("Filtering embeddings to match metadata...")
movie_ids_set = set(movie_data['movie_id'].values)
mask = np.array([mid in movie_ids_set for mid in all_movie_ids])
all_embeddings = all_embeddings[mask]
all_movie_ids = all_movie_ids[mask]
all_years = all_years[mask]

# Remove any movies where year lookup failed (year == -1)
valid_year_mask = all_years != -1
all_embeddings = all_embeddings[valid_year_mask]
all_movie_ids = all_movie_ids[valid_year_mask]
all_years = all_years[valid_year_mask]

print(f"After filtering: {len(all_movie_ids)} movies with both embeddings and metadata")
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
n_samples = 15_000
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
plt.title(f'UMAP Visualization of Movie Embeddings Colored by Year of {START_YEAR} to {END_YEAR} with {n_samples} movies', fontsize=16, fontweight='bold')
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

plt.title(f'UMAP Visualization of Movie Embeddings Colored by Genre with {n_samples} movies', fontsize=16, fontweight='bold')
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
# Parameters adjusted to increase separation between groups:
# - early_exaggeration: Higher values (20-50) help separate groups in early iterations
# - perplexity: For 15k samples, 30-50 is reasonable. Lower values (20-25) create more local structure
# - learning_rate: Higher values (300-500) can help with separation
# - max_iter: More iterations help convergence
# - min_grad_norm: Lower values allow more fine-tuning
tsne = TSNE(n_components=2, random_state=42, perplexity=55, 
            early_exaggeration=30, learning_rate=500, 
            max_iter=2000, min_grad_norm=1e-7, verbose=1)
tsne_embedding = tsne.fit_transform(sampled_embeddings)

print(f"t-SNE embedding shape: {tsne_embedding.shape}")

# Plot 3: t-SNE colored by year with gradient
plt.figure(figsize=(14, 10))
scatter = plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                     c=sampled_years, cmap='viridis', 
                     s=10, alpha=0.6, edgecolors='none')
plt.colorbar(scatter, label='Year')
plt.title(f't-SNE Visualization of Movie Embeddings Colored by Year of {START_YEAR} to {END_YEAR} with {n_samples} movies', fontsize=16, fontweight='bold')
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

plt.title(f't-SNE Visualization of Movie Embeddings Colored by Genre with {n_samples} movies', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'tsne_by_genre.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: tsne_by_genre.png")

# Create PCA reduction
print("Computing PCA reduction...")
pca = PCA(n_components=20, random_state=42)
pca_embedding = pca.fit_transform(sampled_embeddings)

print(f"PCA embedding shape: {pca_embedding.shape}")
print(f"Explained variance ratio (first 20 components): {pca.explained_variance_ratio_[:20].sum():.4f}")

# Plot 5: PCA colored by year with gradient
plt.figure(figsize=(14, 10))
scatter = plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                     c=sampled_years, cmap='viridis', 
                     s=10, alpha=0.6, edgecolors='none')
plt.colorbar(scatter, label='Year')
plt.title(f'PCA Visualization of Movie Embeddings Colored by Year of {START_YEAR} to {END_YEAR} with {n_samples} movies', fontsize=16, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'pca_by_year.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: pca_by_year.png")

# Plot 6: PCA colored by genre
plt.figure(figsize=(16, 12))

# Get unique genres and assign colors (reuse the same color map from previous plots)
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
    plt.scatter(pca_embedding[mask, 0], pca_embedding[mask, 1], 
               c=[color_map[genre]], label=genre, s=10, alpha=0.6, edgecolors='none')

plt.title(f'PCA Visualization of Movie Embeddings Colored by Genre with {n_samples} movies', fontsize=16, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'pca_by_genre.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: pca_by_genre.png")

# Plot 7: PCA explained variance bar plot for first 20 dimensions
plt.figure(figsize=(14, 8))
explained_var = pca.explained_variance_ratio_[:20]
dimensions = range(1, 21)
bars = plt.bar(dimensions, explained_var * 100, alpha=0.7, color='steelblue', edgecolor='black')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance (%)', fontsize=12)
plt.title('PCA Explained Variance for First 20 Dimensions', fontsize=16, fontweight='bold')
plt.xticks(dimensions)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Add value labels on top of bars
for i, (bar, var) in enumerate(zip(bars, explained_var)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{var*100:.2f}%',
             ha='center', va='bottom', fontsize=8)

plt.savefig(os.path.join(DATA_DIR, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
# plt.show()

print("Saved: pca_explained_variance.png")

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
print(f"  {os.path.join(DATA_DIR, 'pca_by_year.png')}")
print(f"  {os.path.join(DATA_DIR, 'pca_by_genre.png')}")
print(f"  {os.path.join(DATA_DIR, 'pca_explained_variance.png')}")

